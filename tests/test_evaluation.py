"""Evaluation harness: metrics, economics and matched operating points."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paybridge.fraud import evaluation as E
from paybridge.fraud.model import ensure_model


@pytest.fixture(scope="module")
def bundle():
    return ensure_model()


@pytest.fixture(scope="module")
def test_frame(transactions):
    return E.test_split(transactions)


def test_metrics_match_a_hand_computed_case():
    y = np.array([1, 1, 0, 0, 0, 1])
    flags = np.array([1, 0, 1, 0, 0, 1])
    m = E.metrics_for(y, flags)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["alert_rate_pct"] == pytest.approx(50.0)


def test_confusion_dict_matches_metrics():
    y = np.array([1, 0, 0, 1])
    flags = np.array([1, 1, 0, 0])
    assert E.confusion_dict(y, flags) == {
        "true_negatives": 1, "false_positives": 1, "false_negatives": 1, "true_positives": 1
    }


def test_economics_follow_the_documented_model():
    y = np.array([1, 1, 0])
    flags = np.array([1, 0, 1])
    amounts = np.array([1_000.0, 2_000.0, 500.0])
    econ = E.economics_for(y, flags, amounts, loss_rate=0.5, review_cost=10.0)
    assert econ["prevented_loss_usd"] == pytest.approx(500.0)     # caught 1000 * 0.5
    assert econ["incurred_loss_usd"] == pytest.approx(1_000.0)    # missed 2000 * 0.5
    assert econ["review_cost_usd"] == pytest.approx(20.0)         # 2 alerts * 10
    assert econ["total_cost_usd"] == pytest.approx(1_020.0)
    assert econ["no_system_loss_usd"] == pytest.approx(1_500.0)
    assert econ["net_benefit_usd"] == pytest.approx(480.0)


def test_threshold_for_recall_finds_a_qualifying_point():
    y = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    threshold = E.threshold_for_recall(scores, y, 2 / 3)
    assert threshold is not None
    assert (scores >= threshold)[y.astype(bool)].sum() / y.sum() >= 2 / 3


def test_threshold_for_recall_handles_no_positives():
    assert E.threshold_for_recall(np.array([0.5, 0.2]), np.array([0, 0]), 0.5) is None


def test_all_three_approaches_are_evaluated(test_frame, bundle):
    results = E.compare_approaches(test_frame, bundle)
    names = {r.approach for r in results}
    assert names == {E.APPROACH_RULES, E.APPROACH_ML, E.APPROACH_COMBINED}
    for result in results:
        assert 0 <= result.metrics["precision"] <= 1
        assert 0 <= result.metrics["recall"] <= 1
        assert result.confusion["true_positives"] + result.confusion["false_negatives"] == int(
            test_frame["fraud_label"].sum()
        )


def test_rules_only_evaluation_works_without_a_model(test_frame):
    results = E.compare_approaches(test_frame, None)
    rules = next(r for r in results if r.approach == E.APPROACH_RULES)
    assert rules.metrics["recall"] > 0


def test_matched_point_actually_matches_recall(test_frame, bundle):
    matched = E.matched_point_comparison(test_frame, bundle)
    target = matched["target_recall"]
    assert matched["comparisons"]
    for comparison in matched["comparisons"]:
        assert comparison["recall"] >= target - 1e-9


def test_false_positive_reduction_is_consistent_with_counts(test_frame, bundle):
    matched = E.matched_point_comparison(test_frame, bundle)
    baseline_fp = matched["baseline"]["false_positives"]
    for comparison in matched["comparisons"]:
        expected_abs = baseline_fp - comparison["false_positives"]
        assert comparison["false_positive_reduction_abs"] == expected_abs
        assert comparison["false_positive_reduction_pct"] == pytest.approx(
            expected_abs / baseline_fp * 100, rel=1e-9
        )


def test_threshold_sweep_is_monotone_in_alert_volume(test_frame, bundle):
    scores = E.approach_scores(test_frame, bundle)[E.APPROACH_COMBINED]["score"]
    sweep = E.threshold_sweep(test_frame, scores, steps=25)
    assert len(sweep) > 5
    assert sweep["alerts"].is_monotonic_decreasing
    assert sweep["recall"].is_monotonic_decreasing


def test_corridor_performance_covers_major_corridors(test_frame, bundle):
    flags = E.approach_scores(test_frame, bundle)[E.APPROACH_COMBINED]["flags"]
    corridors = E.corridor_performance(test_frame, flags, min_transactions=50)
    assert len(corridors) >= 3
    assert (corridors["alerts"] <= corridors["transactions"]).all()


def test_score_distribution_splits_by_label(test_frame, bundle):
    scores = E.approach_scores(test_frame, bundle)[E.APPROACH_COMBINED]["score"]
    dist = E.score_distribution(test_frame, scores, bins=10)
    assert set(dist["label"]) <= {"Fraudulent", "Legitimate"}
    assert dist["count"].sum() == len(test_frame)


def test_test_split_is_the_final_chronological_slice(transactions):
    test = E.test_split(transactions)
    assert test["transaction_timestamp"].max() == transactions["transaction_timestamp"].max()
    assert len(test) < len(transactions)
