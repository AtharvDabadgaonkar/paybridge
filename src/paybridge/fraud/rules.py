"""FraudLens Layer A - configurable rule-based fraud detection.

Rules are pure functions of features that are observable AT THE TIME of the
payment. No rule may read the fraud label, the final transaction status, the
realised settlement time or any investigation outcome.

The rule score is a bounded weighted sum:

    rules_score = min(1.0, sum(weights of triggered rules) / SCORE_CAP)

so the score is fully explainable: every point of it is attributable to a
named rule. Weights and the flag threshold are configurable at runtime.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from paybridge.config import RISK_BAND_THRESHOLDS

SCORE_CAP = 3.0
DEFAULT_FLAG_THRESHOLD = 0.40


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None:
            return default
        out = float(value)
        return default if math.isnan(out) else out
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RuleDefinition:
    code: str
    name: str
    category: str
    weight: float
    description: str
    investigation_question: str
    predicate: Callable[[Mapping[str, Any]], bool]

    def evaluate(self, row: Mapping[str, Any]) -> bool:
        try:
            return bool(self.predicate(row))
        except Exception:
            return False


RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        "R01_LARGE_AMOUNT",
        "Unusually large transaction",
        "amount",
        0.55,
        "Absolute payment amount exceeds $25,000.",
        "Does the business have a documented commercial reason for a payout of this size?",
        lambda r: _num(r, "amount_usd") > 25_000,
    ),
    RuleDefinition(
        "R02_AMOUNT_SPIKE",
        "Sudden transaction size change",
        "amount",
        0.65,
        "Amount is at least 5x the business's own 30-day average payout.",
        "What explains the step change in payout size versus this customer's own baseline?",
        lambda r: _num(r, "amount_vs_business_mean_ratio") >= 5.0,
    ),
    RuleDefinition(
        "R03_HIGH_VELOCITY",
        "High transaction velocity",
        "velocity",
        0.60,
        "Eight or more payouts initiated by this business in the preceding 24 hours.",
        "Is the burst of payouts consistent with a payroll or supplier run?",
        lambda r: _num(r, "transaction_velocity") >= 8,
    ),
    RuleDefinition(
        "R04_NEW_BENEFICIARY",
        "Newly added beneficiary",
        "beneficiary",
        0.50,
        "Beneficiary was added to the business's payout book within the last 3 days.",
        "Who authorised this beneficiary, and was the bank detail confirmed out of band?",
        lambda r: _num(r, "beneficiary_age_days", 999) <= 3,
    ),
    RuleDefinition(
        "R05_NEW_BENEFICIARY_LARGE",
        "Large payout to a new beneficiary",
        "beneficiary",
        0.75,
        "Payout above $10,000 to a beneficiary less than 7 days old.",
        "Was the first payout to this beneficiary expected to be this large?",
        lambda r: _num(r, "beneficiary_age_days", 999) <= 7 and _num(r, "amount_usd") > 10_000,
    ),
    RuleDefinition(
        "R06_NEW_ACCOUNT",
        "Unusual account behaviour on a young account",
        "account",
        0.45,
        "Business account is 30 days old or younger.",
        "Does activity on this young account match the volumes declared at onboarding?",
        lambda r: _num(r, "account_age_days", 999) <= 30,
    ),
    RuleDefinition(
        "R07_REPEATED_FAILURES",
        "Repeated payment failures",
        "behaviour",
        0.50,
        "Three or more failed or returned payouts in the last 7 days.",
        "Are the prior failures explained by bank detail errors or by probing behaviour?",
        lambda r: _num(r, "recent_failed_count") >= 3,
    ),
    RuleDefinition(
        "R08_ODD_HOUR",
        "Off-hours initiation",
        "behaviour",
        0.30,
        "Payment initiated between 00:00 and 05:59 in the sender's local time.",
        "Does this customer normally transact outside business hours?",
        lambda r: 0 <= _num(r, "hour_of_day", 12) <= 5,
    ),
    RuleDefinition(
        "R09_STRUCTURING",
        "Possible structuring pattern",
        "pattern",
        0.70,
        "Amount sits just below the $10,000 reporting threshold with 3+ payouts in 24h.",
        "Do the clustered sub-threshold payouts aggregate to a single commercial purpose?",
        lambda r: 8_500 <= _num(r, "amount_usd") < 10_000 and _num(r, "transaction_velocity") >= 3,
    ),
    RuleDefinition(
        "R10_HIGH_RISK_CORRIDOR",
        "Elevated-risk corridor",
        "geography",
        0.35,
        "Destination corridor carries a synthetic geographic risk weight of 0.70 or above.",
        "Is the destination market consistent with the customer's declared trading countries?",
        lambda r: _num(r, "corridor_risk") >= 0.70,
    ),
    RuleDefinition(
        "R11_DORMANT_REACTIVATION",
        "Dormant account reactivation",
        "behaviour",
        0.55,
        "No payout for 21+ days, then a payout 3x above the business average.",
        "What prompted the account to reactivate with an above-average payout?",
        lambda r: _num(r, "days_since_last_txn") >= 21
        and _num(r, "amount_vs_business_mean_ratio") >= 3.0,
    ),
    RuleDefinition(
        "R12_ROUND_AMOUNT",
        "Large round-number payout",
        "pattern",
        0.25,
        "Payout of $20,000 or more on an exact $1,000 boundary.",
        "Is the round amount supported by an invoice or contract?",
        lambda r: bool(_num(r, "is_round_amount")) and _num(r, "amount_usd") >= 20_000,
    ),
    RuleDefinition(
        "R13_DETAILS_CHANGED",
        "Recently modified account information",
        "account",
        0.60,
        "Business or beneficiary payout details were modified within the last 7 days.",
        "Was the recent change to payout details verified with a known contact?",
        lambda r: _num(r, "account_details_changed_days", 999) <= 7,
    ),
    RuleDefinition(
        "R14_SUSPICIOUS_COMBINATION",
        "Suspicious combined pattern",
        "pattern",
        0.80,
        "New beneficiary, elevated velocity and an above-baseline amount occurring together.",
        "Treat as a possible account takeover: confirm the request with the mandated signer.",
        lambda r: _num(r, "beneficiary_age_days", 999) <= 7
        and _num(r, "transaction_velocity") >= 5
        and _num(r, "amount_vs_business_mean_ratio") >= 2.0,
    ),
)

RULES_BY_CODE = {rule.code: rule for rule in RULES}
DEFAULT_WEIGHTS = {rule.code: rule.weight for rule in RULES}


def risk_band(score: float) -> str:
    """Map a 0..1 risk score onto a named risk band."""
    thresholds = RISK_BAND_THRESHOLDS
    if score >= thresholds["critical"]:
        return "critical"
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    return "low"


@dataclass(frozen=True)
class TriggeredRule:
    code: str
    name: str
    category: str
    weight: float
    description: str
    investigation_question: str


@dataclass(frozen=True)
class RuleResult:
    score: float
    band: str
    flagged: bool
    triggered: tuple[TriggeredRule, ...] = field(default=())

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(t.code for t in self.triggered)

    @property
    def weight_sum(self) -> float:
        return round(sum(t.weight for t in self.triggered), 6)

    def investigation_questions(self) -> list[str]:
        return [t.investigation_question for t in self.triggered]


def evaluate_rules(
    row: Mapping[str, Any],
    weight_overrides: Mapping[str, float] | None = None,
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
) -> RuleResult:
    """Evaluate every rule against one transaction feature mapping."""
    weights = dict(DEFAULT_WEIGHTS)
    if weight_overrides:
        weights.update({k: float(v) for k, v in weight_overrides.items() if k in weights})

    triggered: list[TriggeredRule] = []
    total = 0.0
    for rule in RULES:
        if rule.evaluate(row):
            weight = weights[rule.code]
            total += weight
            triggered.append(
                TriggeredRule(
                    code=rule.code,
                    name=rule.name,
                    category=rule.category,
                    weight=round(weight, 4),
                    description=rule.description,
                    investigation_question=rule.investigation_question,
                )
            )

    score = min(1.0, total / SCORE_CAP)
    triggered.sort(key=lambda t: t.weight, reverse=True)
    return RuleResult(
        score=round(score, 6),
        band=risk_band(score),
        flagged=score >= flag_threshold,
        triggered=tuple(triggered),
    )


def evaluate_rules_frame(
    df: pd.DataFrame,
    weight_overrides: Mapping[str, float] | None = None,
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
) -> pd.DataFrame:
    """Vectorised-by-row rule evaluation for a transaction frame.

    Returns a frame indexed like ``df`` with columns: rules_score, rules_band,
    rules_flagged, rules_triggered (comma separated codes), rules_n_triggered.
    """
    scores: list[float] = []
    bands: list[str] = []
    flags: list[bool] = []
    codes: list[str] = []
    counts: list[int] = []

    for row in df.to_dict("records"):
        result = evaluate_rules(row, weight_overrides, flag_threshold)
        scores.append(result.score)
        bands.append(result.band)
        flags.append(result.flagged)
        codes.append(",".join(result.codes))
        counts.append(len(result.triggered))

    return pd.DataFrame(
        {
            "rules_score": scores,
            "rules_band": bands,
            "rules_flagged": flags,
            "rules_triggered": codes,
            "rules_n_triggered": counts,
        },
        index=df.index,
    )
