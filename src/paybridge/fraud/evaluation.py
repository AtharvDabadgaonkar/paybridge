"""FraudLens evaluation harness: rules baseline vs ML vs combined.

Everything here is computed from the held-out chronological TEST split unless
a caller passes another frame. The false-positive comparison is made at a
MATCHED OPERATING POINT (equal or better recall than the rules baseline), so
the two systems are compared at comparable fraud-loss exposure rather than at
arbitrary thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

from paybridge.config import FRAUD_LOSS_RATE, MANUAL_REVIEW_COST_USD
from paybridge.fraud import features as F
from paybridge.fraud.model import ModelBundle
from paybridge.fraud.rules import DEFAULT_FLAG_THRESHOLD
from paybridge.fraud.scoring import ML_WEIGHT, RULE_OVERRIDE, RULES_WEIGHT

APPROACH_RULES = "Rules baseline"
APPROACH_ML = "ML only"
APPROACH_COMBINED = "Rules + ML (combined)"


@dataclass(frozen=True)
class ApproachResult:
    approach: str
    threshold: float | None
    metrics: dict[str, float]
    economics: dict[str, float]
    confusion: dict[str, int]

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {"approach": self.approach, "threshold": self.threshold}
        row.update(self.metrics)
        row.update(self.economics)
        row.update(self.confusion)
        return row


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def economics_for(
    y_true: np.ndarray,
    flags: np.ndarray,
    amounts: np.ndarray,
    loss_rate: float = FRAUD_LOSS_RATE,
    review_cost: float = MANUAL_REVIEW_COST_USD,
) -> dict[str, float]:
    """Modelled fraud-loss and review-cost economics for one operating point.

    Assumptions (documented in docs/UNIT_ECONOMICS.md):
      * A flagged fraudulent payout is stopped, preventing amount x loss_rate.
      * A missed fraudulent payout incurs amount x loss_rate.
      * Every alert (true or false) consumes one manual review at review_cost.
    """
    y_true = np.asarray(y_true).astype(bool)
    flags = np.asarray(flags).astype(bool)
    amounts = np.asarray(amounts, dtype=float)

    caught = y_true & flags
    missed = y_true & ~flags
    alerts = int(flags.sum())

    prevented = float(amounts[caught].sum() * loss_rate)
    incurred = float(amounts[missed].sum() * loss_rate)
    review = float(alerts * review_cost)
    no_system_loss = float(amounts[y_true].sum() * loss_rate)

    return {
        "prevented_loss_usd": prevented,
        "incurred_loss_usd": incurred,
        "review_cost_usd": review,
        "total_cost_usd": incurred + review,
        "no_system_loss_usd": no_system_loss,
        "net_benefit_usd": no_system_loss - (incurred + review),
        "alerts": alerts,
        "review_volume_pct": _safe_div(alerts, len(flags)) * 100.0,
    }


def metrics_for(y_true: np.ndarray, flags: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    flags = np.asarray(flags).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, flags, labels=[0, 1]).ravel()
    positives, negatives = tp + fn, tn + fp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, positives)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "pr_auc": float(average_precision_score(y_true, scores)) if scores is not None and positives else float("nan"),
        "false_positive_rate": _safe_div(fp, negatives),
        "false_negative_rate": _safe_div(fn, positives),
        "alert_rate_pct": _safe_div(tp + fp, len(y_true)) * 100.0,
    }


def confusion_dict(y_true: np.ndarray, flags: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).astype(int), np.asarray(flags).astype(int), labels=[0, 1]
    ).ravel()
    return {"true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)}


def approach_scores(
    df: pd.DataFrame,
    bundle: ModelBundle | None,
    rules_threshold: float = DEFAULT_FLAG_THRESHOLD,
    ml_threshold: float | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Score arrays and default flags for each of the three approaches."""
    rules_score = df["rules_score"].to_numpy(dtype=float)
    rules_flags = rules_score >= rules_threshold

    if bundle is None:
        ml_prob = np.zeros(len(df))
        threshold = 1.1
    else:
        ml_prob = bundle.predict_proba(df)
        threshold = float(bundle.threshold if ml_threshold is None else ml_threshold)

    combined_score = ML_WEIGHT * ml_prob + RULES_WEIGHT * rules_score
    combined_flags = (ml_prob >= threshold) | (rules_score >= RULE_OVERRIDE)

    return {
        APPROACH_RULES: {"score": rules_score, "flags": rules_flags, "threshold": rules_threshold},
        APPROACH_ML: {"score": ml_prob, "flags": ml_prob >= threshold, "threshold": threshold},
        APPROACH_COMBINED: {"score": combined_score, "flags": combined_flags, "threshold": threshold},
    }


def compare_approaches(
    df: pd.DataFrame,
    bundle: ModelBundle | None,
    rules_threshold: float = DEFAULT_FLAG_THRESHOLD,
    ml_threshold: float | None = None,
    loss_rate: float = FRAUD_LOSS_RATE,
    review_cost: float = MANUAL_REVIEW_COST_USD,
) -> list[ApproachResult]:
    """Evaluate all three approaches at their own operating points."""
    y = df["fraud_label"].to_numpy()
    amounts = df["amount_usd"].to_numpy(dtype=float)
    out: list[ApproachResult] = []
    for name, payload in approach_scores(df, bundle, rules_threshold, ml_threshold).items():
        flags = payload["flags"]
        out.append(
            ApproachResult(
                approach=name,
                threshold=float(payload["threshold"]),
                metrics=metrics_for(y, flags, payload["score"]),
                economics=economics_for(y, flags, amounts, loss_rate, review_cost),
                confusion=confusion_dict(y, flags),
            )
        )
    return out


def comparison_frame(results: list[ApproachResult]) -> pd.DataFrame:
    return pd.DataFrame([r.flat() for r in results])


# --------------------------------------------------------------------------
# Matched operating point
# --------------------------------------------------------------------------
def threshold_for_recall(scores: np.ndarray, y_true: np.ndarray, target_recall: float) -> float | None:
    """Lowest-alert threshold whose recall is at least ``target_recall``."""
    y_true = np.asarray(y_true).astype(bool)
    positives = int(y_true.sum())
    if positives == 0 or target_recall <= 0:
        return None
    candidates = np.unique(scores)[::-1]
    for t in candidates:
        recall = _safe_div(int((scores >= t)[y_true].sum()), positives)
        if recall >= target_recall:
            return float(t)
    return float(candidates[-1]) if candidates.size else None


def matched_point_comparison(
    df: pd.DataFrame,
    bundle: ModelBundle | None,
    rules_threshold: float = DEFAULT_FLAG_THRESHOLD,
    loss_rate: float = FRAUD_LOSS_RATE,
    review_cost: float = MANUAL_REVIEW_COST_USD,
) -> dict[str, Any]:
    """False-positive comparison at equal-or-better recall than the baseline.

    Returns the calculated reduction whatever its sign. A negative reduction
    means the combined system produced MORE false positives at matched recall
    and is reported as such.
    """
    y = df["fraud_label"].to_numpy()
    amounts = df["amount_usd"].to_numpy(dtype=float)
    scores = approach_scores(df, bundle, rules_threshold)

    baseline_flags = scores[APPROACH_RULES]["flags"]
    baseline_metrics = metrics_for(y, baseline_flags, scores[APPROACH_RULES]["score"])
    baseline_conf = confusion_dict(y, baseline_flags)
    target_recall = baseline_metrics["recall"]

    out: dict[str, Any] = {
        "baseline": {
            "approach": APPROACH_RULES,
            "threshold": rules_threshold,
            "recall": target_recall,
            **baseline_metrics,
            **baseline_conf,
            **economics_for(y, baseline_flags, amounts, loss_rate, review_cost),
        },
        "target_recall": target_recall,
        "comparisons": [],
        "n": int(len(df)),
        "positives": int(y.sum()),
    }

    for name in (APPROACH_ML, APPROACH_COMBINED):
        score = scores[name]["score"]
        matched_threshold = threshold_for_recall(score, y, target_recall)
        if matched_threshold is None:
            continue
        flags = score >= matched_threshold
        metrics = metrics_for(y, flags, score)
        conf = confusion_dict(y, flags)
        fp_reduction = baseline_conf["false_positives"] - conf["false_positives"]
        out["comparisons"].append(
            {
                "approach": name,
                "matched_threshold": matched_threshold,
                **metrics,
                **conf,
                **economics_for(y, flags, amounts, loss_rate, review_cost),
                "false_positive_reduction_abs": int(fp_reduction),
                "false_positive_reduction_pct": _safe_div(
                    fp_reduction, baseline_conf["false_positives"]
                )
                * 100.0,
                "alert_reduction_abs": int(
                    (baseline_conf["true_positives"] + baseline_conf["false_positives"])
                    - (conf["true_positives"] + conf["false_positives"])
                ),
            }
        )
    return out


# --------------------------------------------------------------------------
# Sweeps and slices
# --------------------------------------------------------------------------
def threshold_sweep(
    df: pd.DataFrame,
    scores: np.ndarray,
    steps: int = 60,
    loss_rate: float = FRAUD_LOSS_RATE,
    review_cost: float = MANUAL_REVIEW_COST_USD,
) -> pd.DataFrame:
    """Metrics and economics across a threshold grid (for interactive tuning)."""
    y = df["fraud_label"].to_numpy()
    amounts = df["amount_usd"].to_numpy(dtype=float)
    scores = np.asarray(scores, dtype=float)
    grid = np.unique(np.quantile(scores, np.linspace(0.60, 0.9995, steps)))
    rows = []
    for t in grid:
        flags = scores >= t
        rows.append(
            {
                "threshold": float(t),
                **metrics_for(y, flags),
                **economics_for(y, flags, amounts, loss_rate, review_cost),
            }
        )
    return pd.DataFrame(rows)


def corridor_performance(
    df: pd.DataFrame, flags: np.ndarray, min_transactions: int = 50
) -> pd.DataFrame:
    """Per-corridor precision/recall for one operating point."""
    tmp = df.copy()
    tmp["_flag"] = np.asarray(flags).astype(int)
    rows = []
    for corridor, grp in tmp.groupby("corridor"):
        if len(grp) < min_transactions:
            continue
        y = grp["fraud_label"].to_numpy()
        f = grp["_flag"].to_numpy()
        m = metrics_for(y, f)
        rows.append(
            {
                "corridor": corridor,
                "transactions": int(len(grp)),
                "fraud_cases": int(y.sum()),
                "alerts": int(f.sum()),
                "precision": m["precision"],
                "recall": m["recall"],
                "false_positive_rate": m["false_positive_rate"],
            }
        )
    return pd.DataFrame(rows).sort_values("transactions", ascending=False)


def score_distribution(df: pd.DataFrame, scores: np.ndarray, bins: int = 25) -> pd.DataFrame:
    """Risk-score histogram split by true label."""
    tmp = pd.DataFrame({"score": np.asarray(scores, dtype=float), "fraud": df["fraud_label"].to_numpy()})
    edges = np.linspace(0, max(float(tmp["score"].max()), 1e-6), bins + 1)
    tmp["bin"] = pd.cut(tmp["score"], bins=edges, include_lowest=True)
    out = (
        tmp.groupby(["bin", "fraud"], observed=True)
        .size()
        .reset_index(name="count")
        .assign(bin_mid=lambda d: d["bin"].apply(lambda b: float(b.mid)))
    )
    out["label"] = np.where(out["fraud"] == 1, "Fraudulent", "Legitimate")
    return out[["bin_mid", "label", "count"]]


def test_split(df: pd.DataFrame) -> pd.DataFrame:
    """The held-out chronological test split used for reported metrics."""
    return F.chronological_split(df).test
