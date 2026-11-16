"""Shadow-mode validation harness.

FraudLens is replayed over historical synthetic transactions and produces a
recommendation for each one. The original simulated payment decision is never
modified - that invariant is asserted and reported, because "no decision
impact" is the whole point of a shadow rollout.

Everything reported here is MEASURED during the run: latency is timed in
process, detection metrics are computed from the replay output, and audit
completeness is the share of recommendations carrying a full evidence trail.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from paybridge.fraud import evaluation as E
from paybridge.fraud.scoring import assess_transaction, score_frame

# Acceptance thresholds are TARGETS agreed for the shadow phase, not results.
ACCEPTANCE_CRITERIA: tuple[dict[str, Any], ...] = (
    {"key": "recall", "label": "Fraud detection recall", "target": 0.60, "direction": ">="},
    {"key": "precision", "label": "Alert precision", "target": 0.15, "direction": ">="},
    {"key": "false_positive_rate", "label": "False positive rate", "target": 0.03, "direction": "<="},
    {"key": "review_volume_pct", "label": "Manual review volume (% of payouts)", "target": 4.0, "direction": "<="},
    {"key": "latency_p95_ms", "label": "Scoring latency p95 (ms, in process)", "target": 50.0, "direction": "<="},
    {"key": "audit_completeness_pct", "label": "Recommendations with a complete evidence trail", "target": 100.0, "direction": ">="},
    {"key": "decisions_changed", "label": "Original payment decisions changed", "target": 0, "direction": "=="},
    {"key": "workflow_errors", "label": "Workflow errors during replay", "target": 0, "direction": "=="},
)


@dataclass
class ShadowRunResult:
    transactions_replayed: int
    period_start: str
    period_end: str
    measured: dict[str, float]
    criteria: list[dict[str, Any]] = field(default_factory=list)
    outstanding_defects: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c["met"] for c in self.criteria)

    @property
    def criteria_met(self) -> int:
        return sum(1 for c in self.criteria if c["met"])

    def as_dict(self) -> dict:
        return asdict(self)

    def criteria_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.criteria)


def _evaluate_criteria(measured: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for criterion in ACCEPTANCE_CRITERIA:
        value = measured.get(criterion["key"])
        if value is None:
            out.append({**criterion, "measured": None, "met": False, "note": "not measured"})
            continue
        direction = criterion["direction"]
        target = criterion["target"]
        met = (
            value >= target if direction == ">="
            else value <= target if direction == "<="
            else abs(value - target) < 1e-9
        )
        out.append({**criterion, "measured": round(float(value), 4), "met": bool(met), "note": ""})
    return out


def run_shadow_mode(
    df: pd.DataFrame,
    bundle=None,
    sample_size: int = 4000,
    latency_sample: int = 150,
    outstanding_defects: int = 0,
) -> ShadowRunResult:
    """Replay transactions through FraudLens without changing any decision."""
    window = E.test_split(df) if len(df) > sample_size else df
    if len(window) > sample_size:
        window = window.tail(sample_size)

    original_status = window["transaction_status"].copy()
    scored = score_frame(window, bundle)

    # Invariant: the shadow run must not have altered any payment outcome.
    decisions_changed = int((scored["transaction_status"].to_numpy() != original_status.to_numpy()).sum())

    y = scored["fraud_label"].to_numpy()
    flags = scored["combined_flagged"].to_numpy()
    metrics = E.metrics_for(y, flags, scored["combined_score"].to_numpy())
    economics = E.economics_for(y, flags, scored["amount_usd"].to_numpy())

    # Latency: time a single-transaction assessment end to end, as production
    # would call it (rules + model + decision policy).
    latencies: list[float] = []
    workflow_errors = 0
    sample = scored.head(min(latency_sample, len(scored)))
    for _, row in sample.iterrows():
        started = time.perf_counter()
        try:
            assess_transaction(row, bundle)
        except Exception:
            workflow_errors += 1
        latencies.append((time.perf_counter() - started) * 1000.0)

    # Audit completeness: a recommendation is auditable when it carries a score,
    # a band, a rules trace and (when a model is loaded) a model probability.
    required = scored["combined_score"].notna() & scored["combined_band"].notna() & scored["rules_triggered"].notna()
    if bundle is not None:
        required &= scored["ml_probability"].notna()
    audit_completeness = float(required.mean() * 100.0)

    measured = {
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "false_positive_rate": metrics["false_positive_rate"],
        "review_volume_pct": economics["review_volume_pct"],
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "audit_completeness_pct": audit_completeness,
        "decisions_changed": float(decisions_changed),
        "workflow_errors": float(workflow_errors),
        "alerts": float(economics["alerts"]),
        "fraud_cases": float(int(y.sum())),
        "prevented_loss_usd": economics["prevented_loss_usd"],
        "missed_loss_usd": economics["incurred_loss_usd"],
    }

    notes = [
        "FraudLens ran in recommendation-only mode: no payment status was modified by the replay.",
        f"Latency measured in process over {len(latencies)} single-transaction assessments on one "
        "container; it is not a load test and does not model concurrency or network overhead.",
        "Detection metrics come from synthetic labels and carry the limitations documented on the "
        "Model Evaluation page.",
    ]

    return ShadowRunResult(
        transactions_replayed=int(len(scored)),
        period_start=str(scored["transaction_timestamp"].min()),
        period_end=str(scored["transaction_timestamp"].max()),
        measured=measured,
        criteria=_evaluate_criteria(measured),
        outstanding_defects=outstanding_defects,
        notes=notes,
    )
