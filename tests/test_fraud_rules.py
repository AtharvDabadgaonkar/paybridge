"""FraudLens Layer A: rule evaluation, scoring and configurability."""
from __future__ import annotations

import pandas as pd
import pytest

from paybridge.fraud.rules import (
    DEFAULT_FLAG_THRESHOLD,
    DEFAULT_WEIGHTS,
    RULES,
    RULES_BY_CODE,
    SCORE_CAP,
    evaluate_rules,
    evaluate_rules_frame,
    risk_band,
)

CLEAN_ROW = {
    "amount_usd": 800.0, "beneficiary_age_days": 400.0, "transaction_velocity": 1,
    "amount_vs_business_mean_ratio": 1.0, "hour_of_day": 14, "corridor_risk": 0.15,
    "account_age_days": 900, "recent_failed_count": 0, "days_since_last_txn": 3.0,
    "is_round_amount": 0, "account_details_changed_days": 999.0,
}

TRIGGERS = {
    "R01_LARGE_AMOUNT": {"amount_usd": 30_000.0},
    "R02_AMOUNT_SPIKE": {"amount_vs_business_mean_ratio": 6.0},
    "R03_HIGH_VELOCITY": {"transaction_velocity": 9},
    "R04_NEW_BENEFICIARY": {"beneficiary_age_days": 1.0},
    "R05_NEW_BENEFICIARY_LARGE": {"beneficiary_age_days": 5.0, "amount_usd": 15_000.0},
    "R06_NEW_ACCOUNT": {"account_age_days": 10},
    "R07_REPEATED_FAILURES": {"recent_failed_count": 4},
    "R08_ODD_HOUR": {"hour_of_day": 3},
    "R09_STRUCTURING": {"amount_usd": 9_200.0, "transaction_velocity": 4},
    "R10_HIGH_RISK_CORRIDOR": {"corridor_risk": 0.75},
    "R11_DORMANT_REACTIVATION": {"days_since_last_txn": 30.0, "amount_vs_business_mean_ratio": 4.0},
    "R12_ROUND_AMOUNT": {"is_round_amount": 1, "amount_usd": 25_000.0},
    "R13_DETAILS_CHANGED": {"account_details_changed_days": 2.0},
    "R14_SUSPICIOUS_COMBINATION": {"beneficiary_age_days": 2.0, "transaction_velocity": 6,
                                   "amount_vs_business_mean_ratio": 3.0},
}


def test_every_rule_has_a_trigger_case():
    assert set(TRIGGERS) == {rule.code for rule in RULES}


@pytest.mark.parametrize("code,overrides", sorted(TRIGGERS.items()))
def test_each_rule_fires_on_its_own_case(code, overrides):
    result = evaluate_rules({**CLEAN_ROW, **overrides})
    assert code in result.codes, f"{code} did not fire for {overrides}"


def test_clean_transaction_triggers_nothing():
    result = evaluate_rules(CLEAN_ROW)
    assert result.triggered == () and result.score == 0.0
    assert result.band == "low" and result.flagged is False


def test_score_is_the_bounded_weighted_sum():
    row = {**CLEAN_ROW, **TRIGGERS["R01_LARGE_AMOUNT"], **TRIGGERS["R08_ODD_HOUR"]}
    result = evaluate_rules(row)
    expected = min(1.0, sum(DEFAULT_WEIGHTS[c] for c in result.codes) / SCORE_CAP)
    assert result.score == pytest.approx(expected, abs=1e-5)  # score rounded to 6dp
    assert result.weight_sum == pytest.approx(sum(DEFAULT_WEIGHTS[c] for c in result.codes))


def test_score_never_exceeds_one():
    everything = dict(CLEAN_ROW)
    for overrides in TRIGGERS.values():
        everything.update(overrides)
    assert evaluate_rules(everything).score == 1.0


def test_weight_overrides_change_the_score():
    row = {**CLEAN_ROW, **TRIGGERS["R01_LARGE_AMOUNT"]}
    base = evaluate_rules(row).score
    boosted = evaluate_rules(row, {"R01_LARGE_AMOUNT": 1.5}).score
    assert boosted > base


def test_flag_threshold_is_configurable():
    row = {**CLEAN_ROW, **TRIGGERS["R08_ODD_HOUR"]}     # weight 0.30 -> score 0.10
    assert evaluate_rules(row, flag_threshold=0.05).flagged is True
    assert evaluate_rules(row, flag_threshold=DEFAULT_FLAG_THRESHOLD).flagged is False


@pytest.mark.parametrize(
    "score,expected", [(0.0, "low"), (0.24, "low"), (0.25, "medium"), (0.54, "medium"),
                       (0.55, "high"), (0.79, "high"), (0.80, "critical"), (1.0, "critical")]
)
def test_risk_band_boundaries(score, expected):
    assert risk_band(score) == expected


def test_missing_fields_do_not_raise():
    result = evaluate_rules({"amount_usd": 100.0})
    assert isinstance(result.score, float)


def test_malformed_values_are_tolerated():
    result = evaluate_rules({**CLEAN_ROW, "amount_usd": "not-a-number", "transaction_velocity": None})
    assert isinstance(result.score, float)


def test_rules_carry_investigation_guidance():
    result = evaluate_rules({**CLEAN_ROW, **TRIGGERS["R14_SUSPICIOUS_COMBINATION"]})
    questions = result.investigation_questions()
    assert questions and all(isinstance(q, str) and q.strip() for q in questions)


def test_frame_evaluation_matches_row_evaluation():
    rows = [CLEAN_ROW, {**CLEAN_ROW, **TRIGGERS["R03_HIGH_VELOCITY"]},
            {**CLEAN_ROW, **TRIGGERS["R09_STRUCTURING"]}]
    frame = evaluate_rules_frame(pd.DataFrame(rows))
    for i, row in enumerate(rows):
        assert frame.iloc[i]["rules_score"] == pytest.approx(evaluate_rules(row).score, abs=1e-9)
        assert frame.iloc[i]["rules_band"] == evaluate_rules(row).band


def test_triggered_rules_are_sorted_by_weight():
    everything = dict(CLEAN_ROW)
    for overrides in TRIGGERS.values():
        everything.update(overrides)
    weights = [t.weight for t in evaluate_rules(everything).triggered]
    assert weights == sorted(weights, reverse=True)


def test_rules_use_only_payment_time_features():
    """No rule may reference a post-outcome or label-derived field."""
    from paybridge.fraud.features import EXCLUDED_COLUMNS

    forbidden = EXCLUDED_COLUMNS - {"risk_score"}
    poisoned = {**CLEAN_ROW, **{col: 1 for col in forbidden if col != "transaction_status"}}
    poisoned["fraud_label"] = 1
    assert evaluate_rules(poisoned).codes == evaluate_rules(CLEAN_ROW).codes
