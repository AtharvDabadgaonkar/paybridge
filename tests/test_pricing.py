"""Financial calculations: fees, FX, margin and savings."""
from __future__ import annotations

import numpy as np
import pytest

from paybridge.config import CORRIDOR_BY_CODE, RAIL_BY_CODE, SWIFT_LIFTING_FEE_PER_HOP_USD
from paybridge.payments.pricing import (
    compare_rails,
    cost_components_vectorized,
    corridor_for,
    estimate_settlement_hours,
    quote_payment,
    savings_vs_swift,
)


def test_swift_cost_matches_the_documented_formula():
    amount = 10_000.0
    q = quote_payment("SWIFT", "MX", amount)
    rail = RAIL_BY_CODE["SWIFT"]
    corridor = CORRIDOR_BY_CODE["US-MX"]
    expected = (
        rail.customer_fixed_fee_usd
        + amount * rail.customer_variable_pct
        + amount * rail.customer_fx_markup_bps / 10_000
        + SWIFT_LIFTING_FEE_PER_HOP_USD * corridor.swift_hops
    )
    assert q.total_customer_cost == pytest.approx(expected, rel=1e-9)


def test_cost_components_sum_to_total():
    q = quote_payment("ACH_PARTNER", "IN", 7_500.0)
    assert q.fixed_fee + q.variable_fee + q.fx_markup_cost + q.lifting_fees == pytest.approx(
        q.total_customer_cost, rel=1e-9
    )


def test_platform_cost_components_sum_to_direct_cost():
    q = quote_payment("USDC", "NG", 20_000.0)
    assert (
        q.provider_fixed_cost + q.provider_variable_cost + q.partner_payout_cost
        + q.fx_spread_cost + q.lifting_fees
    ) == pytest.approx(q.platform_direct_cost, rel=1e-9)


def test_contribution_margin_is_revenue_minus_direct_cost():
    for rail in ("ACH_PARTNER", "SWIFT", "USDC"):
        q = quote_payment(rail, "BR", 12_000.0)
        assert q.contribution_margin == pytest.approx(
            q.platform_revenue - q.platform_direct_cost, rel=1e-9
        )


def test_fx_markup_reduces_the_offered_rate():
    q = quote_payment("SWIFT", "GB", 5_000.0)
    assert q.offered_rate < q.mid_market_rate
    expected_rate = q.mid_market_rate * (1 - RAIL_BY_CODE["SWIFT"].customer_fx_markup_bps / 10_000)
    assert q.offered_rate == pytest.approx(expected_rate, rel=1e-6)


def test_recipient_amount_uses_the_net_of_fees():
    amount = 3_000.0
    q = quote_payment("ACH_PARTNER", "PH", amount)
    net = amount - q.fixed_fee - q.variable_fee - q.lifting_fees
    assert q.recipient_amount_local == pytest.approx(net * q.offered_rate, rel=1e-6)


def test_only_swift_charges_correspondent_fees():
    assert quote_payment("SWIFT", "PH", 5_000).lifting_fees > 0
    assert quote_payment("ACH_PARTNER", "PH", 5_000).lifting_fees == 0
    assert quote_payment("USDC", "PH", 5_000).lifting_fees == 0


def test_vectorized_pricing_matches_quote_payment():
    amounts = np.array([120.0, 950.0, 7_500.0, 48_000.0, 310_000.0])
    corridors = ["US-MX", "US-IN", "US-NG", "US-GB", "US-DE"]
    for rail in ("ACH_PARTNER", "SWIFT", "USDC"):
        vector = cost_components_vectorized(rail, amounts, corridors)
        for i, (amount, corridor_code) in enumerate(zip(amounts, corridors)):
            corridor = CORRIDOR_BY_CODE[corridor_code]
            q = quote_payment(rail, corridor.recipient_country, float(amount))
            assert vector["total_customer_cost"].iloc[i] == pytest.approx(q.total_customer_cost, rel=1e-6)
            assert vector["platform_direct_cost"].iloc[i] == pytest.approx(q.platform_direct_cost, rel=1e-6)
            assert vector["contribution_margin"].iloc[i] == pytest.approx(q.contribution_margin, rel=1e-6)


def test_compare_rails_is_sorted_cheapest_first():
    quotes = [q for q in compare_rails("MX", 5_000.0) if q.eligible]
    costs = [q.total_customer_cost for q in quotes]
    assert costs == sorted(costs)


def test_savings_are_calculated_not_assumed():
    result = savings_vs_swift("MX", 5_000.0)
    baseline = quote_payment("SWIFT", "MX", 5_000.0).total_customer_cost
    for rail, payload in result.items():
        rail_cost = quote_payment(rail, "MX", 5_000.0).total_customer_cost
        # savings are rounded to 4 decimal places at the presentation boundary
        assert payload["savings_usd"] == pytest.approx(baseline - rail_cost, abs=1e-4)
        assert payload["savings_pct"] == pytest.approx((baseline - rail_cost) / baseline * 100, abs=1e-3)


def test_savings_compress_as_amount_grows():
    """SWIFT's fixed fees amortise, so the relative saving must fall with size."""
    small = savings_vs_swift("MX", 500.0)["USDC"]["savings_pct"]
    large = savings_vs_swift("MX", 250_000.0)["USDC"]["savings_pct"]
    assert small > large


def test_rail_eligibility_limits_are_enforced():
    assert not quote_payment("SWIFT", "MX", 50.0).eligible          # below SWIFT minimum
    assert not quote_payment("ACH_PARTNER", "MX", 400_000.0).eligible  # above ACH maximum
    assert quote_payment("USDC", "MX", 400_000.0).eligible


def test_settlement_estimates_are_ordered_and_positive():
    corridor = CORRIDOR_BY_CODE["US-MX"]
    usdc = estimate_settlement_hours(RAIL_BY_CODE["USDC"], corridor)
    ach = estimate_settlement_hours(RAIL_BY_CODE["ACH_PARTNER"], corridor)
    swift = estimate_settlement_hours(RAIL_BY_CODE["SWIFT"], corridor)
    assert 0 < usdc[0] < ach[0] < swift[0]
    assert all(p90 > p50 for p50, p90 in (usdc, ach, swift))


def test_invalid_inputs_raise():
    with pytest.raises(KeyError):
        quote_payment("CARRIER_PIGEON", "MX", 100.0)
    with pytest.raises(ValueError):
        quote_payment("SWIFT", "MX", -10.0)
    with pytest.raises(KeyError):
        corridor_for("ZZ")


def test_portfolio_savings_are_computed_from_records(transactions):
    from paybridge.analytics.kpis import modelled_savings

    result = modelled_savings(transactions.head(2000))
    assert result["transactions_repriced"] == 2000
    assert result["savings_usd"] == pytest.approx(
        result["baseline_cost"] - result["actual_cost"], rel=1e-9
    )
