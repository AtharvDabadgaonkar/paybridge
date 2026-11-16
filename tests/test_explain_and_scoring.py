"""FraudLens Layer C and the combined decision service."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paybridge.fraud import explain as X
from paybridge.fraud import scoring as S
from paybridge.fraud.model import ensure_model


@pytest.fixture(scope="module")
def bundle():
    return ensure_model()


def test_explanation_returns_ranked_contributions(bundle, transactions):
    row = transactions.head(1)
    result = X.explain_transaction(bundle, row, background=transactions.sample(100, random_state=1))
    assert result.contributions
    magnitudes = [abs(c.contribution) for c in result.top(5)]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(c.direction in {"increases risk", "reduces risk"} for c in result.contributions)


def test_explanation_reports_its_method(bundle, transactions):
    result = X.explain_transaction(bundle, transactions.head(1))
    assert result.method in {X.METHOD_SHAP_TREE, X.METHOD_SHAP_LINEAR, X.METHOD_FALLBACK}
    assert result.is_shap == (result.method != X.METHOD_FALLBACK)


def test_fallback_is_used_and_labelled_when_shap_is_unavailable(bundle, transactions, monkeypatch):
    monkeypatch.setattr(X, "SHAP_AVAILABLE", False)
    monkeypatch.setattr(X, "_EXPLAINER_CACHE", {})
    result = X.explain_transaction(bundle, transactions.head(1),
                                   background=transactions.sample(100, random_state=2))
    assert result.method == X.METHOD_FALLBACK
    assert not result.is_shap
    assert "SHAP was unavailable" in result.note
    assert result.contributions


def test_explain_requires_exactly_one_row(bundle, transactions):
    with pytest.raises(ValueError):
        X.explain_transaction(bundle, transactions.head(3))


def test_global_importance_is_ranked(bundle, transactions):
    gi = X.global_importance(bundle, transactions.sample(150, random_state=5))
    assert list(gi["mean_abs_contribution"]) == sorted(gi["mean_abs_contribution"], reverse=True)
    assert gi["method"].nunique() == 1


def test_combine_scores_weights_both_layers():
    assert S.combine_scores(0.5, 0.5) == pytest.approx(0.5)
    assert S.combine_scores(1.0, 0.0) == pytest.approx(S.RULES_WEIGHT)
    assert S.combine_scores(0.0, 1.0) == pytest.approx(S.ML_WEIGHT)
    assert S.combine_scores(0.8, None) == pytest.approx(0.8)   # rules only


def test_rule_override_flags_without_the_model():
    row = {"amount_usd": 60_000.0, "beneficiary_age_days": 1.0, "transaction_velocity": 9,
           "amount_vs_business_mean_ratio": 8.0, "hour_of_day": 3, "corridor_risk": 0.75,
           "account_age_days": 10, "recent_failed_count": 4, "days_since_last_txn": 1.0,
           "is_round_amount": 0, "account_details_changed_days": 1.0}
    assessment = S.assess_transaction(row, bundle=None)
    assert assessment.rules.score >= S.RULE_OVERRIDE
    assert assessment.flagged and assessment.requires_human_review
    assert assessment.ml_probability is None
    assert any("rules-only" in note for note in assessment.notes)


@pytest.mark.parametrize(
    "band,flagged,decision,human",
    [("critical", False, "hold", True), ("high", False, "review", True),
     ("medium", True, "review", True), ("low", False, "auto_approve", False)],
)
def test_decision_policy(band, flagged, decision, human):
    assert S.decide(band, flagged) == (decision, human)


def test_high_risk_is_never_auto_approved(bundle, transactions):
    scored = S.score_frame(transactions.head(3000), bundle)
    high = scored[scored["combined_band"].isin(["high", "critical"])]
    assert high["requires_human_review"].all()


def test_score_frame_adds_every_expected_column(bundle, transactions):
    scored = S.score_frame(transactions.head(500), bundle)
    for col in ("ml_probability", "ml_flagged", "combined_score", "combined_band",
                "combined_flagged", "requires_human_review"):
        assert col in scored.columns
    assert scored["combined_score"].between(0, 1).all()


def test_score_frame_works_without_a_model(transactions):
    scored = S.score_frame(transactions.head(200), bundle=None)
    assert scored["ml_probability"].isna().all()
    assert (scored["combined_score"] == scored["rules_score"]).all()


def test_assessment_includes_investigation_questions(bundle, transactions):
    top = transactions.nlargest(1, "rules_score")
    assessment = S.assess_transaction(top.iloc[0], bundle)
    assert assessment.investigation_questions()


def test_payout_context_is_built_from_history_only(transactions):
    context = S.build_payout_context("BIZ-0001", 12_000.0, "US-MX", "USDC", transactions,
                                     new_beneficiary=True)
    assert context["amount_usd"] == 12_000.0
    assert context["corridor"] == "US-MX" and context["currency"] == "MXN"
    assert context["is_new_beneficiary"] == 1
    assert context["transaction_velocity"] >= 0
    assert "fraud_label" not in context and "transaction_status" not in context


def test_payout_context_for_an_unknown_business(transactions):
    context = S.build_payout_context("BIZ-9999", 500.0, "US-GB", "SWIFT", transactions)
    assert context["business_txn_count_30d"] == 0
    assert context["days_since_last_txn"] == 999.0


def test_new_beneficiary_raises_the_assessed_risk(bundle, transactions):
    base = S.assess_transaction(
        S.build_payout_context("BIZ-0001", 45_000.0, "US-NG", "SWIFT", transactions,
                               new_beneficiary=False), bundle)
    risky = S.assess_transaction(
        S.build_payout_context("BIZ-0001", 45_000.0, "US-NG", "SWIFT", transactions,
                               new_beneficiary=True), bundle)
    assert risky.combined_score > base.combined_score
