"""Deterministic cross-border payout pricing and unit economics.

Cost model (all figures are documented modelling assumptions, see
docs/UNIT_ECONOMICS.md):

Customer price (what the business pays PayBridge)
    fixed_fee          flat per-payout fee for the rail
    variable_fee       amount_usd * customer_variable_pct
    fx_markup_cost     amount_usd * customer_fx_markup_bps / 10_000
    lifting_fees       SWIFT only: per-correspondent-hop pass-through fee
    ------------------------------------------------------------------
    total_customer_cost = sum of the above

Platform direct cost (what PayBridge pays its providers)
    provider_fixed_cost     rail origination cost
    provider_variable_cost  amount_usd * provider_variable_pct
    partner_payout_cost     local payout partner fee (corridor specific)
    fx_spread_cost          amount_usd * corridor wholesale spread bps
    lifting_fees            pass-through, identical to the customer charge
    ------------------------------------------------------------------
    platform_direct_cost = sum of the above

contribution_margin = total_customer_cost - platform_direct_cost
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from paybridge.config import (
    CORRIDOR_BY_CODE,
    CORRIDORS,
    FX_MID_RATES,
    RAIL_BY_CODE,
    RAILS,
    SWIFT_LIFTING_FEE_PER_HOP_USD,
    Corridor,
    RailConfig,
)

BPS = 10_000.0

# Share of the corridor's local partner payout fee borne by each rail.
# SWIFT pays no local partner (the beneficiary bank handles the last mile and
# charges correspondent lifting fees instead).
RAIL_PARTNER_COST_FACTOR: dict[str, float] = {
    "ACH_PARTNER": 1.0,
    "SWIFT": 0.0,
    "USDC": 0.7,
}


def corridor_for(recipient_country: str, sender_country: str = "US") -> Corridor:
    """Return the corridor config for a sender/recipient country pair."""
    code = f"{sender_country}-{recipient_country}"
    if code in CORRIDOR_BY_CODE:
        return CORRIDOR_BY_CODE[code]
    for corridor in CORRIDORS:
        if corridor.recipient_country == recipient_country:
            return corridor
    raise KeyError(f"No configured corridor for {sender_country} -> {recipient_country}")


def estimate_settlement_hours(rail: RailConfig, corridor: Corridor) -> tuple[float, float]:
    """Deterministic (p50, p90) settlement estimate in hours for a quote.

    Corridor friction adds a modelled delay proportional to corridor risk; the
    effect is larger on bank-intermediated rails than on stablecoin settlement.
    """
    friction = 1.0 + corridor.corridor_risk * (0.10 if rail.code == "USDC" else 0.35)
    hop_penalty = (corridor.swift_hops - 1.0) * 10.0 if rail.code == "SWIFT" else 0.0
    p50 = rail.settlement_hours_p50 * friction + hop_penalty
    p90 = rail.settlement_hours_p90 * friction + hop_penalty * 1.8
    return round(p50, 2), round(p90, 2)


@dataclass(frozen=True)
class Quote:
    """A fully decomposed payout quote for one rail on one corridor."""

    rail: str
    rail_label: str
    corridor: str
    currency: str
    amount_usd: float
    # customer price
    fixed_fee: float
    variable_fee: float
    fx_markup_cost: float
    lifting_fees: float
    total_customer_cost: float
    effective_cost_pct: float
    # fx
    mid_market_rate: float
    offered_rate: float
    recipient_amount_local: float
    # platform economics
    provider_fixed_cost: float
    provider_variable_cost: float
    partner_payout_cost: float
    fx_spread_cost: float
    platform_direct_cost: float
    platform_revenue: float
    contribution_margin: float
    contribution_margin_pct: float
    # operations
    settlement_hours_p50: float
    settlement_hours_p90: float
    expected_success_rate: float
    eligible: bool
    ineligible_reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def quote_payment(
    rail_code: str,
    recipient_country: str,
    amount_usd: float,
    sender_country: str = "US",
) -> Quote:
    """Price one payout on one rail. Raises KeyError for an unknown rail."""
    if rail_code not in RAIL_BY_CODE:
        raise KeyError(f"Unknown payment rail: {rail_code}")
    if amount_usd <= 0:
        raise ValueError("amount_usd must be positive")

    rail = RAIL_BY_CODE[rail_code]
    corridor = corridor_for(recipient_country, sender_country)

    # ---- customer price ------------------------------------------------
    fixed_fee = rail.customer_fixed_fee_usd
    variable_fee = amount_usd * rail.customer_variable_pct
    fx_markup_cost = amount_usd * rail.customer_fx_markup_bps / BPS
    lifting_fees = (
        SWIFT_LIFTING_FEE_PER_HOP_USD * corridor.swift_hops if rail.code == "SWIFT" else 0.0
    )
    total_customer_cost = fixed_fee + variable_fee + fx_markup_cost + lifting_fees

    # ---- FX ------------------------------------------------------------
    mid = FX_MID_RATES[corridor.currency]
    offered_rate = mid * (1.0 - rail.customer_fx_markup_bps / BPS)
    net_to_convert = max(amount_usd - fixed_fee - variable_fee - lifting_fees, 0.0)
    recipient_amount_local = net_to_convert * offered_rate

    # ---- platform direct cost ------------------------------------------
    provider_fixed_cost = rail.provider_fixed_cost_usd
    provider_variable_cost = amount_usd * rail.provider_variable_pct
    partner_payout_cost = corridor.partner_payout_fee_usd * RAIL_PARTNER_COST_FACTOR[rail.code]
    fx_spread_cost = amount_usd * corridor.fx_wholesale_spread_bps / BPS
    platform_direct_cost = (
        provider_fixed_cost
        + provider_variable_cost
        + partner_payout_cost
        + fx_spread_cost
        + lifting_fees
    )

    platform_revenue = total_customer_cost
    contribution_margin = platform_revenue - platform_direct_cost

    p50, p90 = estimate_settlement_hours(rail, corridor)
    eligible = rail.eligible_for(corridor.code, amount_usd)
    reason = ""
    if not eligible:
        if amount_usd < rail.min_amount_usd:
            reason = f"Below {rail.code} minimum of ${rail.min_amount_usd:,.2f}"
        elif amount_usd > rail.max_amount_usd:
            reason = f"Above {rail.code} maximum of ${rail.max_amount_usd:,.2f}"
        else:
            reason = f"{rail.code} is not enabled for corridor {corridor.code}"

    return Quote(
        rail=rail.code,
        rail_label=rail.label,
        corridor=corridor.code,
        currency=corridor.currency,
        amount_usd=round(amount_usd, 2),
        fixed_fee=round(fixed_fee, 4),
        variable_fee=round(variable_fee, 4),
        fx_markup_cost=round(fx_markup_cost, 4),
        lifting_fees=round(lifting_fees, 4),
        total_customer_cost=round(total_customer_cost, 4),
        effective_cost_pct=round(total_customer_cost / amount_usd * 100.0, 4),
        mid_market_rate=mid,
        offered_rate=round(offered_rate, 6),
        recipient_amount_local=round(recipient_amount_local, 2),
        provider_fixed_cost=round(provider_fixed_cost, 4),
        provider_variable_cost=round(provider_variable_cost, 4),
        partner_payout_cost=round(partner_payout_cost, 4),
        fx_spread_cost=round(fx_spread_cost, 4),
        platform_direct_cost=round(platform_direct_cost, 4),
        platform_revenue=round(platform_revenue, 4),
        contribution_margin=round(contribution_margin, 4),
        contribution_margin_pct=round(
            contribution_margin / platform_revenue * 100.0 if platform_revenue else 0.0, 4
        ),
        settlement_hours_p50=p50,
        settlement_hours_p90=p90,
        expected_success_rate=rail.success_rate,
        eligible=eligible,
        ineligible_reason=reason,
    )


def compare_rails(
    recipient_country: str,
    amount_usd: float,
    sender_country: str = "US",
    eligible_only: bool = False,
) -> list[Quote]:
    """Quote every configured rail, cheapest eligible rail first."""
    quotes = [
        quote_payment(rail.code, recipient_country, amount_usd, sender_country)
        for rail in RAILS
    ]
    if eligible_only:
        quotes = [q for q in quotes if q.eligible]
    return sorted(quotes, key=lambda q: (not q.eligible, q.total_customer_cost))


def savings_vs_swift(
    recipient_country: str,
    amount_usd: float,
    sender_country: str = "US",
) -> dict[str, dict[str, float]]:
    """Modelled payout-cost saving of each rail against the SWIFT baseline.

    Returns a mapping of rail code -> {baseline_cost, rail_cost, savings_usd,
    savings_pct}. A negative saving means the rail is MORE expensive than
    SWIFT at this amount, and is reported as calculated.
    """
    baseline = quote_payment("SWIFT", recipient_country, amount_usd, sender_country)
    out: dict[str, dict[str, float]] = {}
    for quote in compare_rails(recipient_country, amount_usd, sender_country):
        if quote.rail == "SWIFT":
            continue
        savings = baseline.total_customer_cost - quote.total_customer_cost
        out[quote.rail] = {
            "baseline_cost": baseline.total_customer_cost,
            "rail_cost": quote.total_customer_cost,
            "savings_usd": round(savings, 4),
            "savings_pct": round(
                savings / baseline.total_customer_cost * 100.0
                if baseline.total_customer_cost
                else 0.0,
                4,
            ),
            "eligible": float(quote.eligible),
        }
    return out
