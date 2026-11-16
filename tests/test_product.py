"""Product analytics and experiment design."""
from __future__ import annotations

import pytest

from paybridge.product import experiments as X
from paybridge.product import metrics as M


def test_sample_size_matches_the_standard_formula():
    """Baseline 0.50, +10% relative effect, alpha 0.05, power 0.80."""
    n = X.sample_size_two_proportions(0.50, 0.10, 0.05, 0.80)
    assert 1400 <= n <= 1750, n


def test_sample_size_grows_as_the_effect_shrinks():
    big = X.sample_size_two_proportions(0.30, 0.20)
    small = X.sample_size_two_proportions(0.30, 0.05)
    assert small > big * 4


def test_sample_size_grows_with_power():
    assert X.sample_size_two_proportions(0.30, 0.10, power=0.95) > X.sample_size_two_proportions(
        0.30, 0.10, power=0.80
    )


def test_sample_size_validates_inputs():
    for baseline, mde in ((0.0, 0.1), (1.0, 0.1), (0.5, 0.0), (0.5, -0.2)):
        with pytest.raises(ValueError):
            X.sample_size_two_proportions(baseline, mde)


def test_runtime_days_rounds_up():
    assert X.runtime_days(100, 30.0) == 7       # 200 units / 30 per day -> 6.67 -> 7
    with pytest.raises(ValueError):
        X.runtime_days(100, 0)


def test_designs_are_complete_and_computed():
    designs = X.seed_designs()
    assert len(designs) == len(X.EXPERIMENT_SEEDS)
    for design in designs:
        assert design.hypothesis and design.primary_kpi and design.guardrails
        assert design.rollout_criteria
        assert design.sample_size_per_arm > 0 and design.runtime_days > 0
        assert design.sample_size_per_arm == X.sample_size_two_proportions(
            design.baseline_rate, design.mde_relative, design.alpha, design.power
        )


def test_activation_metrics_are_event_derived():
    from paybridge.data.loader import load_product_events

    events = load_product_events()
    result = M.activation_metrics(events)
    assert result["signups"] > 0
    assert 0 <= result["activation_rate_pct"] <= 100
    assert result["activated"] <= result["signups"]


def test_activation_handles_empty_events():
    import pandas as pd

    empty = pd.DataFrame(columns=["business_id", "event_name", "event_timestamp"])
    assert M.activation_metrics(empty)["signups"] == 0


def test_feature_adoption_is_bounded():
    from paybridge.data.loader import load_product_events

    adoption = M.feature_adoption(load_product_events())
    assert (adoption["adoption_pct"] <= 100).all()
    assert (adoption["businesses"] > 0).any()


def test_payment_product_kpis(transactions):
    kpis = M.payment_product_kpis(transactions)
    assert kpis["transactions"] == len(transactions)
    assert 0 <= kpis["payment_success_rate_pct"] <= 100
    assert kpis["avg_revenue_per_transaction_usd"] > kpis["cost_per_payout_usd"]
    assert kpis["contribution_margin_per_transaction_usd"] == pytest.approx(
        kpis["avg_revenue_per_transaction_usd"] - kpis["cost_per_payout_usd"], rel=1e-9
    )


def test_retention_is_computed_week_over_week():
    from paybridge.data.loader import load_product_events

    retention = M.weekly_retention(load_product_events())
    assert not retention.empty
    assert (retention["retention_pct"] <= 100).all()


def test_investigation_cycle_time_reports_unavailability():
    import pandas as pd

    result = M.investigation_cycle_time(pd.DataFrame())
    assert result["available"] is False and result["reason"]


def test_investigation_cycle_time_measures_real_decisions(temp_db):
    from paybridge.fraud import investigations as INV

    case_id = INV.create_case("TXN-000001", "BIZ-0001", 500.0, 0.9, "critical", db=temp_db)
    INV.record_decision(case_id, "reject", "analyst.chen", "fraudulent", db=temp_db)
    from paybridge.db import storage

    result = M.investigation_cycle_time(storage.audit_trail("investigation", path=temp_db))
    assert result["available"] is True and result["decided_cases"] == 1
