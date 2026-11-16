"""FraudLens combined decision service (rules + ML + explanation).

Decision policy
---------------
combined_score = ML_WEIGHT * ml_probability + RULES_WEIGHT * rules_score
flagged        = ml_probability >= model_threshold  OR  rules_score >= RULE_OVERRIDE

The rule override exists so a deterministic, explainable red flag (for example
a large payout to a brand-new beneficiary during a velocity spike) can never be
silently suppressed by a confident model. High and critical outcomes always
require a human decision; FraudLens never auto-approves them.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from paybridge.config import CORRIDOR_BY_CODE
from paybridge.fraud import features as F
from paybridge.fraud.explain import ExplanationResult, explain_transaction
from paybridge.fraud.model import ModelBundle, ensure_model
from paybridge.fraud.rules import (
    DEFAULT_FLAG_THRESHOLD,
    RuleResult,
    evaluate_rules,
    evaluate_rules_frame,
    risk_band,
)

ML_WEIGHT = 0.60
RULES_WEIGHT = 0.40
RULE_OVERRIDE = 0.55          # rules score that forces a review on its own
HUMAN_REVIEW_BANDS = ("high", "critical")


@dataclass(frozen=True)
class FraudAssessment:
    transaction_id: str
    rules: RuleResult
    ml_probability: float | None
    ml_threshold: float | None
    ml_flagged: bool
    combined_score: float
    combined_band: str
    flagged: bool
    decision: str                       # auto_approve | review | hold
    requires_human_review: bool
    model_name: str | None = None
    explanation: ExplanationResult | None = None
    notes: tuple[str, ...] = field(default=())

    def investigation_questions(self) -> list[str]:
        questions = self.rules.investigation_questions()
        if self.ml_probability is not None and self.ml_flagged and not questions:
            questions.append(
                "No deterministic rule fired: confirm with the customer whether this payout "
                "matches an expected commercial pattern before approving."
            )
        return questions

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "rules_score": self.rules.score,
            "rules_band": self.rules.band,
            "rules_triggered": list(self.rules.codes),
            "ml_probability": self.ml_probability,
            "ml_threshold": self.ml_threshold,
            "ml_flagged": self.ml_flagged,
            "combined_score": self.combined_score,
            "combined_band": self.combined_band,
            "flagged": self.flagged,
            "decision": self.decision,
            "requires_human_review": self.requires_human_review,
            "model_name": self.model_name,
            "explanation_method": self.explanation.method if self.explanation else None,
        }


def combine_scores(rules_score: float, ml_probability: float | None) -> float:
    """Blend the rules score and the model probability into one 0..1 score."""
    if ml_probability is None:
        return float(np.clip(rules_score, 0.0, 1.0))
    return float(np.clip(ML_WEIGHT * ml_probability + RULES_WEIGHT * rules_score, 0.0, 1.0))


def decide(combined_band: str, flagged: bool) -> tuple[str, bool]:
    """Map a risk outcome onto an operational decision."""
    if combined_band == "critical":
        return "hold", True
    if combined_band == "high" or flagged:
        return "review", True
    return "auto_approve", False


def assess_transaction(
    row: Mapping[str, Any] | pd.Series,
    bundle: ModelBundle | None = None,
    weight_overrides: Mapping[str, float] | None = None,
    rules_threshold: float = DEFAULT_FLAG_THRESHOLD,
    ml_threshold: float | None = None,
    with_explanation: bool = False,
    background: pd.DataFrame | None = None,
) -> FraudAssessment:
    """Score one transaction through all three FraudLens layers."""
    data = dict(row) if not isinstance(row, pd.Series) else row.to_dict()
    rules = evaluate_rules(data, weight_overrides, rules_threshold)

    ml_probability: float | None = None
    explanation: ExplanationResult | None = None
    notes: list[str] = []
    model_name: str | None = None
    threshold: float | None = None

    if bundle is not None:
        frame = pd.DataFrame([data])
        try:
            ml_probability = float(bundle.predict_proba(frame)[0])
            threshold = float(bundle.threshold if ml_threshold is None else ml_threshold)
            model_name = bundle.model_name
            if with_explanation:
                explanation = explain_transaction(bundle, frame, background=background)
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"Model scoring unavailable ({type(exc).__name__}); rules-only assessment.")
            ml_probability = None
            threshold = None
    else:
        notes.append("No trained model loaded; this is a rules-only assessment.")

    ml_flagged = bool(ml_probability is not None and threshold is not None and ml_probability >= threshold)
    combined = combine_scores(rules.score, ml_probability)
    band = risk_band(combined)
    flagged = bool(ml_flagged or rules.score >= RULE_OVERRIDE or rules.flagged)
    decision, requires_human = decide(band, flagged)

    return FraudAssessment(
        transaction_id=str(data.get("transaction_id", "UNSAVED")),
        rules=rules,
        ml_probability=ml_probability,
        ml_threshold=threshold,
        ml_flagged=ml_flagged,
        combined_score=round(combined, 6),
        combined_band=band,
        flagged=flagged,
        decision=decision,
        requires_human_review=requires_human,
        model_name=model_name,
        explanation=explanation,
        notes=tuple(notes),
    )


def score_frame(
    df: pd.DataFrame,
    bundle: ModelBundle | None = None,
    weight_overrides: Mapping[str, float] | None = None,
    rules_threshold: float = DEFAULT_FLAG_THRESHOLD,
    ml_threshold: float | None = None,
    recompute_rules: bool = False,
) -> pd.DataFrame:
    """Score a whole frame. Returns the input with FraudLens columns appended."""
    out = df.copy()
    if recompute_rules or "rules_score" not in out.columns:
        rules_cols = evaluate_rules_frame(out, weight_overrides, rules_threshold)
        for col in rules_cols.columns:
            out[col] = rules_cols[col]

    if bundle is not None:
        out["ml_probability"] = bundle.predict_proba(out)
        threshold = float(bundle.threshold if ml_threshold is None else ml_threshold)
        out["ml_threshold"] = threshold
        out["ml_flagged"] = out["ml_probability"] >= threshold
    else:
        out["ml_probability"] = np.nan
        out["ml_threshold"] = np.nan
        out["ml_flagged"] = False

    ml = out["ml_probability"]
    out["combined_score"] = np.where(
        ml.notna(), ML_WEIGHT * ml.fillna(0.0) + RULES_WEIGHT * out["rules_score"], out["rules_score"]
    )
    out["combined_band"] = [risk_band(float(s)) for s in out["combined_score"]]
    out["combined_flagged"] = out["ml_flagged"] | (out["rules_score"] >= RULE_OVERRIDE) | out["rules_flagged"]
    out["requires_human_review"] = out["combined_band"].isin(HUMAN_REVIEW_BANDS) | out["combined_flagged"]
    return out


# --------------------------------------------------------------------------
# Payout-time scoring context
# --------------------------------------------------------------------------
def build_payout_context(
    business_id: str,
    amount_usd: float,
    corridor_code: str,
    rail_code: str,
    history: pd.DataFrame,
    new_beneficiary: bool = False,
    beneficiary_age_days: float | None = None,
    details_changed_days: float = 999.0,
    at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a payment-time feature row for a payout that does not exist yet.

    Behavioural features come from the business's own simulated history; the
    rest come from the payout being composed. No future information is used.
    """
    corridor = CORRIDOR_BY_CODE[corridor_code]
    moment = at or datetime.now()
    past = history[history["business_id"] == business_id]
    if "transaction_timestamp" in past.columns and len(past):
        past = past.sort_values("transaction_timestamp")
        last_ts = pd.Timestamp(past["transaction_timestamp"].iloc[-1])
        window_start = pd.Timestamp(moment) - pd.Timedelta(hours=24)
        velocity = int((past["transaction_timestamp"] >= window_start).sum())
        days_since = float((pd.Timestamp(moment) - last_ts).total_seconds() / 86_400.0)
        last_30 = past[past["transaction_timestamp"] >= pd.Timestamp(moment) - pd.Timedelta(days=30)]
        count_30d = int(len(last_30))
        avg_30d = float(last_30["amount_usd"].mean()) if len(last_30) else float(past["amount_usd"].mean())
        failed = int(
            (
                (past["transaction_timestamp"] >= pd.Timestamp(moment) - pd.Timedelta(days=7))
                & (past["transaction_status"].isin(["failed", "returned"]))
            ).sum()
        )
        account_age = int(past["account_age_days"].iloc[-1] + max(days_since, 0))
    else:
        velocity, days_since, count_30d, failed, account_age = 0, 999.0, 0, 0, 1
        avg_30d = float(amount_usd)

    avg_30d = max(avg_30d, 1.0)
    if beneficiary_age_days is None:
        beneficiary_age_days = 0.5 if new_beneficiary else 180.0

    return {
        "transaction_id": "SIMULATED-PAYOUT",
        "business_id": business_id,
        "amount_usd": float(amount_usd),
        "payment_rail": rail_code,
        "corridor": corridor.code,
        "currency": corridor.currency,
        "sender_country": corridor.sender_country,
        "recipient_country": corridor.recipient_country,
        "corridor_risk": corridor.corridor_risk,
        "account_age_days": int(account_age),
        "beneficiary_age_days": float(beneficiary_age_days),
        "transaction_velocity": int(velocity),
        "days_since_last_txn": float(min(days_since, 999.0)),
        "business_txn_count_30d": int(count_30d),
        "business_avg_amount_30d": float(avg_30d),
        "amount_vs_business_mean_ratio": float(amount_usd / avg_30d),
        "recent_failed_count": int(failed),
        "hour_of_day": int(moment.hour),
        "day_of_week": int(moment.weekday()),
        "is_weekend": int(moment.weekday() >= 5),
        "is_round_amount": int(float(amount_usd) % 1000 == 0),
        "account_details_changed_days": float(details_changed_days),
        "is_new_beneficiary": int(float(beneficiary_age_days) <= 7),
        "transaction_timestamp": pd.Timestamp(moment),
    }


def load_bundle_safe() -> ModelBundle | None:
    """Best-effort model load; the app must still work when training fails."""
    try:
        return ensure_model()
    except Exception:  # pragma: no cover - defensive
        return None
