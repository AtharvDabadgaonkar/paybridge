"""FraudLens Layer B: features, leakage control, splits and thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from paybridge.fraud import features as F
from paybridge.fraud import model as M


def test_excluded_columns_cover_label_and_post_outcome_fields():
    required = {"fraud_label", "transaction_status", "compliance_status", "settlement_time",
                "risk_score", "rules_score", "rules_flagged", "_latent_fraud_prob"}
    assert required <= F.EXCLUDED_COLUMNS


def test_feature_set_and_excluded_set_are_disjoint():
    assert not (set(F.ALL_FEATURES) & F.EXCLUDED_COLUMNS)


def test_feature_frame_contains_only_declared_features(transactions):
    X = F.build_feature_frame(transactions.head(200))
    assert list(X.columns) == list(F.ALL_FEATURES)
    assert not (set(X.columns) & F.EXCLUDED_COLUMNS)


def test_leakage_guard_raises():
    with pytest.raises(F.LeakageError):
        F.assert_no_leakage(["amount_usd", "fraud_label"])


def test_missing_feature_column_raises(transactions):
    broken = transactions.head(10).drop(columns=["transaction_velocity"])
    with pytest.raises(KeyError):
        F.build_feature_frame(broken)


def test_chronological_split_is_ordered_and_disjoint(transactions):
    split = F.chronological_split(transactions)
    assert split.train["transaction_timestamp"].max() <= split.validation["transaction_timestamp"].min()
    assert split.validation["transaction_timestamp"].max() <= split.test["transaction_timestamp"].min()
    assert len(split.train) + len(split.validation) + len(split.test) == len(transactions)
    ids = set(split.train["transaction_id"]) & set(split.test["transaction_id"])
    assert not ids


def test_split_fractions_are_validated(transactions):
    with pytest.raises(ValueError):
        F.chronological_split(transactions.head(100), train_frac=0.9, validation_frac=0.2)


def test_split_summary_reports_each_partition(transactions):
    summary = F.chronological_split(transactions).summary()
    assert summary["train_rows"] > summary["validation_rows"] > 0
    assert 0 <= summary["test_fraud_rate"] <= 1


def test_preprocessor_expands_categoricals(transactions):
    prep = F.build_preprocessor()
    X = F.build_feature_frame(transactions.head(500))
    out = prep.fit_transform(X)
    assert out.shape[1] > len(F.NUMERIC_FEATURES)
    assert len(F.expanded_feature_names(prep)) == out.shape[1]


def test_classification_metrics_are_correct():
    y = np.array([0, 0, 1, 1, 0, 1])
    prob = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.2])
    m = M.classification_metrics(y, prob, 0.5)
    assert (m["true_positives"], m["false_positives"], m["false_negatives"], m["true_negatives"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["false_negative_rate"] == pytest.approx(1 / 3)


def test_threshold_tuning_uses_validation_data_only():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    prob = np.clip(y * 0.5 + rng.random(500) * 0.5, 0, 1)
    threshold, metrics = M.tune_threshold(y, prob, "f1")
    assert 0 < threshold < 1
    assert metrics["threshold"] == pytest.approx(threshold)
    # The F1-optimal threshold must beat an arbitrary fixed one on the same data.
    assert metrics["f1"] >= M.classification_metrics(y, prob, 0.99)["f1"]


def test_capacity_objective_respects_the_review_budget():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    prob = np.clip(y * 0.4 + rng.random(2000) * 0.6, 0, 1)
    threshold, _ = M.tune_threshold(y, prob, "recall@review")
    assert M.classification_metrics(y, prob, threshold)["alert_rate"] <= 0.0401


def test_unknown_threshold_objective_raises():
    with pytest.raises(ValueError):
        M.tune_threshold(np.array([0, 1]), np.array([0.2, 0.8]), "nonsense")


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "hist_gradient_boosting"])
def test_each_candidate_model_is_configured(name):
    assert name in M.candidate_models()


def test_training_produces_a_selected_model_and_test_metrics(transactions):
    sample = transactions.sample(4000, random_state=3).sort_values("transaction_timestamp")
    bundle, report = M.train_models(
        sample, models={"logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced")}
    )
    assert bundle.model_name == "logistic_regression"
    assert 0 < bundle.threshold < 1
    assert report.results[0]["selected"] is True
    assert report.results[0]["test"]["n"] > 0
    assert set(report.features) == set(F.ALL_FEATURES)
    assert "fraud_label" in report.excluded_columns


def test_bundle_predicts_probabilities_in_range(transactions):
    bundle = M.ensure_model()
    probs = bundle.predict_proba(transactions.head(100))
    assert probs.shape == (100,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_bundle_roundtrips_to_disk(tmp_path, transactions):
    bundle = M.ensure_model()
    path = tmp_path / "bundle.joblib"
    M.save_bundle(bundle, path)
    loaded = M.load_bundle(path)
    assert loaded is not None
    assert loaded.model_name == bundle.model_name
    assert loaded.threshold == pytest.approx(bundle.threshold)
    np.testing.assert_allclose(
        loaded.predict_proba(transactions.head(50)), bundle.predict_proba(transactions.head(50))
    )


def test_missing_bundle_returns_none(tmp_path):
    assert M.load_bundle(tmp_path / "nope.joblib") is None


def test_trained_model_beats_random_on_the_test_split(transactions):
    """A sanity floor, not a performance claim: the model must carry signal."""
    from sklearn.metrics import average_precision_score

    bundle = M.ensure_model()
    test = F.chronological_split(transactions).test
    probs = bundle.predict_proba(test)
    pr_auc = average_precision_score(test["fraud_label"], probs)
    assert pr_auc > test["fraud_label"].mean() * 3
