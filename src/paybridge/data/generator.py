"""Reproducible synthetic dataset generator for PayBridge.

Design notes
------------
*   Every figure is synthetic. No real customer, bank, beneficiary or
    sanctioned party is represented.
*   Transactions are simulated SEQUENTIALLY per business so that behavioural
    features (velocity, rolling averages, days since last payout, recent
    failures) are computed from PAST events only. This is the primary control
    against target leakage.
*   Fraud labels are sampled from a latent logistic risk model driven by the
    same causal risk factors an analyst would observe (new beneficiary,
    velocity, amount deviation, young account, corridor risk, off-hours,
    recent failures, recently changed payout details). The intercept is
    calibrated by bisection so the realised fraud rate matches
    ``config.BASE_FRAUD_RATE``.
*   ``risk_score`` on the dataset is the LEGACY RULES BASELINE score produced
    by :mod:`paybridge.fraud.rules`. It is a function of observable features
    only and is never derived from the fraud label. It is excluded from the ML
    feature set so that the rules layer and the ML layer stay comparable.
*   Post-outcome fields (``transaction_status``, ``settlement_time``,
    ``compliance_status``, ``fraud_label``) are never used as model features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from paybridge import config
from paybridge.config import CORRIDORS, RAIL_BY_CODE
from paybridge.fraud.rules import evaluate_rules_frame
from paybridge.payments.pricing import quote_payment

# --------------------------------------------------------------------------
# Fictional reference vocabularies
# --------------------------------------------------------------------------
_PREFIX = (
    "Northwind", "Harborline", "Cedarpoint", "Blue Meridian", "Vantage", "Lakeshore",
    "Ironvale", "Summit Row", "Greyfield", "Clearwater", "Oakstone", "Brightpath",
    "Copperline", "Stonebridge", "Westmark", "Trailhead", "Lantern", "Foxglove",
    "Pinebrook", "Silverway", "Redwood Gate", "Beacon Hill", "Kestrel", "Juniper",
    "Anchorpoint", "Mapleway", "Granite Bay", "Hollowpark", "Elmcourt", "Novaline",
)
_SUFFIX = (
    "Logistics", "Trading Co", "Manufacturing", "Agritech", "Textiles", "Components",
    "Freight", "Foods", "Solutions", "Studios", "Labs", "Technologies", "Imports",
    "Exports", "Partners", "Industries", "Supply", "Services", "Works", "Holdings",
)
_INDUSTRIES = (
    "Logistics & Freight", "Apparel & Textiles", "Electronics Manufacturing",
    "Agriculture & Food", "Business Process Outsourcing", "Software & IT Services",
    "Construction Materials", "Marketplace & E-commerce", "Professional Services",
    "Medical Supplies",
)
_SENDER_STATES = ("CA", "TX", "NY", "IL", "FL", "WA", "GA", "MA", "CO", "NC")
_REVIEWERS = ("analyst.rivera", "analyst.okafor", "analyst.chen", "analyst.dubois", "analyst.patel")
_WATCHLIST_SOURCES = ("SYNTH-SANCTIONS-A", "SYNTH-SANCTIONS-B", "SYNTH-PEP", "SYNTH-ADVERSE-MEDIA")

_VOLUME_TIERS = ("small", "mid", "enterprise")
_TIER_WEIGHTS = (0.50, 0.35, 0.15)
_TIER_DAILY_RATE = {"small": 0.55, "mid": 1.45, "enterprise": 3.30}
_TIER_AMOUNT_MU = {"small": 6.6, "mid": 7.6, "enterprise": 8.7}   # log-space mean
_TIER_AMOUNT_SIGMA = {"small": 0.95, "mid": 1.00, "enterprise": 1.10}

# Hour-of-day initiation weights (sender local time, business-hours skewed).
_HOUR_WEIGHTS = np.array(
    [0.6, 0.4, 0.3, 0.3, 0.4, 0.7, 1.4, 3.0, 6.0, 8.5, 9.5, 9.0,
     8.0, 8.5, 9.0, 8.5, 7.0, 5.0, 3.2, 2.2, 1.6, 1.3, 1.0, 0.8]
)
_HOUR_WEIGHTS = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()

# Latent fraud model coefficients (synthetic ground truth).
FRAUD_COEFFICIENTS = {
    "intercept": -5.20,
    "new_beneficiary_3d": 1.40,
    "new_beneficiary_large": 0.95,
    "velocity": 0.85,
    "young_account": 0.90,
    "amount_deviation": 1.15,
    "odd_hour": 0.70,
    "corridor_risk": 1.00,
    "recent_failures": 0.80,
    "structuring": 0.90,
    "details_changed": 0.65,
    "noise_sd": 0.40,
}

COMPROMISE_EPISODE_SHARE = 0.045   # share of businesses with an ATO-style episode


@dataclass
class GeneratedDataset:
    transactions: pd.DataFrame
    businesses: pd.DataFrame
    compliance_cases: pd.DataFrame
    product_events: pd.DataFrame
    watchlist: pd.DataFrame
    manifest: dict

    def summary(self) -> dict:
        return {
            "transactions": len(self.transactions),
            "businesses": len(self.businesses),
            "compliance_cases": len(self.compliance_cases),
            "product_events": len(self.product_events),
            "watchlist_entries": len(self.watchlist),
            "fraud_rate": float(self.transactions["fraud_label"].mean()),
        }


# --------------------------------------------------------------------------
# Businesses
# --------------------------------------------------------------------------
def _generate_businesses(rng: np.random.Generator, n: int, end: datetime) -> pd.DataFrame:
    names: list[str] = []
    used: set[str] = set()
    while len(names) < n:
        name = f"{rng.choice(_PREFIX)} {rng.choice(_SUFFIX)}"
        if name in used:
            name = f"{name} {len(names) + 1}"
        used.add(name)
        names.append(name)

    tiers = rng.choice(_VOLUME_TIERS, size=n, p=_TIER_WEIGHTS)
    # Account age at the end of the window: a tail of young accounts exists so
    # the "young account" risk factor is actually exercised.
    account_age = np.clip(rng.gamma(shape=2.2, scale=190.0, size=n), 3, 2_200).astype(int)

    rows = []
    for i in range(n):
        tier = str(tiers[i])
        created = end - timedelta(days=int(account_age[i]))
        n_corridors = int(rng.integers(1, 4))
        corridor_codes = list(
            rng.choice(
                [c.code for c in CORRIDORS],
                size=n_corridors,
                replace=False,
                p=np.array([c.weight for c in CORRIDORS]) / sum(c.weight for c in CORRIDORS),
            )
        )
        rows.append(
            {
                "business_id": f"BIZ-{i + 1:04d}",
                "legal_name": names[i],
                "country": "US",
                "state": str(rng.choice(_SENDER_STATES)),
                "industry": str(rng.choice(_INDUSTRIES)),
                "incorporation_date": (created - timedelta(days=int(rng.integers(90, 4_000)))).date().isoformat(),
                "account_created_date": created.date().isoformat(),
                "account_age_days_at_window_end": int(account_age[i]),
                "volume_tier": tier,
                "declared_monthly_volume_usd": float(
                    round(np.exp(_TIER_AMOUNT_MU[tier]) * rng.uniform(8, 30), 2)
                ),
                "n_beneficial_owners": int(rng.integers(1, 5)),
                "n_directors": int(rng.integers(1, 6)),
                "expected_corridors": ",".join(sorted(corridor_codes)),
                "amount_mu": float(_TIER_AMOUNT_MU[tier] + rng.normal(0, 0.28)),
                "amount_sigma": float(_TIER_AMOUNT_SIGMA[tier]),
                "daily_rate": float(_TIER_DAILY_RATE[tier] * rng.uniform(0.6, 1.5)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------
def _sample_timestamps(
    rng: np.random.Generator, n: int, start: datetime, end: datetime
) -> list[datetime]:
    """Sample n initiation timestamps, business-hours and weekday weighted."""
    if n <= 0:
        return []
    span_days = max((end - start).days, 1)
    out: list[datetime] = []
    while len(out) < n:
        day_offsets = rng.integers(0, span_days, size=n - len(out))
        hours = rng.choice(24, size=len(day_offsets), p=_HOUR_WEIGHTS)
        minutes = rng.integers(0, 60, size=len(day_offsets))
        for d, h, m in zip(day_offsets, hours, minutes):
            ts = start + timedelta(days=int(d), hours=int(h), minutes=int(m))
            if ts.weekday() >= 5 and rng.random() < 0.70:
                continue  # weekday-skewed commercial payout behaviour
            out.append(ts)
            if len(out) == n:
                break
    out.sort()
    return out


def _choose_rail(rng: np.random.Generator, corridor_code: str, amount: float) -> str:
    """Pick a rail with amount- and corridor-aware preferences."""
    weights = {"ACH_PARTNER": 0.44, "SWIFT": 0.30, "USDC": 0.26}
    if amount >= 100_000:
        weights = {"ACH_PARTNER": 0.30, "SWIFT": 0.55, "USDC": 0.15}
    elif amount < 1_000:
        weights = {"ACH_PARTNER": 0.48, "SWIFT": 0.10, "USDC": 0.42}
    if corridor_code in {"US-NG", "US-PH"}:
        weights["USDC"] += 0.08
        weights["SWIFT"] -= 0.08
    eligible = {
        code: max(w, 0.01)
        for code, w in weights.items()
        if RAIL_BY_CODE[code].eligible_for(corridor_code, amount)
    }
    if not eligible:
        return "SWIFT"
    codes = list(eligible)
    probs = np.array([eligible[c] for c in codes], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(codes, p=probs))


def _generate_transactions(
    rng: np.random.Generator,
    businesses: pd.DataFrame,
    target_transactions: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    corridor_by_code = {c.code: c for c in CORRIDORS}

    # Scale per-business rates so the expected total matches the target.
    active_days = np.array(
        [
            max((end - max(start, datetime.fromisoformat(row.account_created_date))).days, 1)
            for row in businesses.itertuples()
        ],
        dtype=float,
    )
    rates = businesses["daily_rate"].to_numpy(dtype=float)
    expected = float((rates * active_days).sum())
    scale = target_transactions / expected if expected else 1.0

    compromised = set(
        rng.choice(
            businesses["business_id"].to_numpy(),
            size=max(int(len(businesses) * COMPROMISE_EPISODE_SHARE), 1),
            replace=False,
        )
    )

    records: list[dict] = []
    txn_counter = 0

    for idx, biz in enumerate(businesses.itertuples()):
        created = datetime.fromisoformat(biz.account_created_date)
        window_start = max(start, created)
        n_txn = int(rng.poisson(max(biz.daily_rate * scale * active_days[idx], 0.2)))
        if n_txn == 0:
            continue
        timestamps = _sample_timestamps(rng, n_txn, window_start, end)
        if not timestamps:
            continue

        corridor_codes = biz.expected_corridors.split(",")

        # Beneficiary book: seeded before the window, grows over time.
        n_seed = int(rng.integers(2, 9))
        beneficiaries = [
            {
                "id": f"BEN-{biz.business_id[-4:]}-{j:03d}",
                "created": created + timedelta(days=float(rng.uniform(0, max((window_start - created).days, 1)))),
                "corridor": str(rng.choice(corridor_codes)),
            }
            for j in range(n_seed)
        ]

        # Payout-detail modification event (most businesses never change them).
        details_changed_at: datetime | None = None
        if rng.random() < 0.22:
            details_changed_at = window_start + timedelta(
                days=float(rng.uniform(0, max((end - window_start).days, 1)))
            )

        # Account-takeover style episode for a small share of businesses.
        episode: tuple[datetime, datetime] | None = None
        if biz.business_id in compromised and (end - window_start).days > 6:
            onset = window_start + timedelta(
                days=float(rng.uniform(2, max((end - window_start).days - 3, 3)))
            )
            episode = (onset, onset + timedelta(days=float(rng.uniform(1.0, 3.0))))

        prior_ts: list[datetime] = []
        prior_amounts: list[float] = []
        prior_failed_ts: list[datetime] = []

        for ts in timestamps:
            in_episode = bool(episode and episode[0] <= ts <= episode[1])

            # ---- causal behavioural features (past events only) ----------
            velocity = sum(1 for t in prior_ts if (ts - t) <= timedelta(hours=24))
            days_since_last = (
                (ts - prior_ts[-1]).total_seconds() / 86_400.0 if prior_ts else 999.0
            )
            win30 = [
                a for t, a in zip(prior_ts, prior_amounts) if (ts - t) <= timedelta(days=30)
            ]
            count_30d = len(win30)
            avg_30d = float(np.mean(win30)) if win30 else float(np.exp(biz.amount_mu))
            recent_failed = sum(1 for t in prior_failed_ts if (ts - t) <= timedelta(days=7))

            # ---- amount --------------------------------------------------
            mu = biz.amount_mu + (rng.uniform(0.6, 1.4) if in_episode else 0.0)
            amount = float(np.exp(rng.normal(mu, biz.amount_sigma)))
            amount = float(np.clip(amount, 25.0, 900_000.0))
            if rng.random() < 0.06:                     # round-number payouts
                amount = float(round(amount / 1_000.0) * 1_000.0) or 1_000.0
            if in_episode and rng.random() < 0.25:      # sub-threshold clustering
                amount = float(rng.uniform(8_600, 9_950))
            amount = round(amount, 2)

            # ---- beneficiary ---------------------------------------------
            add_new = rng.random() < (0.55 if in_episode else 0.08) or not beneficiaries
            if add_new:
                beneficiary = {
                    "id": f"BEN-{biz.business_id[-4:]}-{len(beneficiaries):03d}",
                    "created": ts - timedelta(hours=float(rng.uniform(0.5, 60.0))),
                    "corridor": str(rng.choice(corridor_codes)),
                }
                beneficiaries.append(beneficiary)
            else:
                beneficiary = beneficiaries[int(rng.integers(0, len(beneficiaries)))]

            corridor = corridor_by_code[
                beneficiary["corridor"] if beneficiary["corridor"] in corridor_by_code else corridor_codes[0]
            ]
            rail_code = _choose_rail(rng, corridor.code, amount)
            rail = RAIL_BY_CODE[rail_code]
            quote = quote_payment(rail_code, corridor.recipient_country, amount)

            beneficiary_age_days = max((ts - beneficiary["created"]).total_seconds() / 86_400.0, 0.0)
            account_age_days = max((ts - created).days, 0)
            details_changed_days = (
                (ts - details_changed_at).total_seconds() / 86_400.0
                if details_changed_at and details_changed_at <= ts
                else 999.0
            )

            # Provisional technical failure, independent of the fraud label.
            # Compromise episodes include probing failures (bad bank details).
            fail_p = (1.0 - rail.success_rate) * (3.2 if in_episode else 1.0)
            provisional_failed = bool(rng.random() < fail_p)

            # Realised settlement time (post-outcome; never a model feature).
            if provisional_failed:
                settlement_hours = float("nan")
            else:
                settlement_hours = float(
                    np.clip(
                        rng.lognormal(np.log(max(quote.settlement_hours_p50, 0.2)), 0.45),
                        0.1,
                        400.0,
                    )
                )

            txn_counter += 1
            records.append(
                {
                    "transaction_id": f"TXN-{txn_counter:06d}",
                    "business_id": biz.business_id,
                    "beneficiary_id": beneficiary["id"],
                    "sender_country": corridor.sender_country,
                    "recipient_country": corridor.recipient_country,
                    "corridor": corridor.code,
                    "currency": corridor.currency,
                    "amount_usd": amount,
                    "payment_rail": rail_code,
                    "transaction_timestamp": ts,
                    "processing_fee": round(quote.fixed_fee + quote.variable_fee + quote.lifting_fees, 4),
                    "fx_markup": quote.fx_markup_cost,
                    "total_cost_usd": quote.total_customer_cost,
                    "platform_direct_cost": quote.platform_direct_cost,
                    "contribution_margin": quote.contribution_margin,
                    "recipient_amount_local": quote.recipient_amount_local,
                    "settlement_time": settlement_hours,
                    "settlement_hours_p50": quote.settlement_hours_p50,
                    # --- features observable at payment time ---------------
                    "account_age_days": int(account_age_days),
                    "beneficiary_age_days": round(beneficiary_age_days, 3),
                    "transaction_velocity": int(velocity),
                    "days_since_last_txn": round(min(days_since_last, 999.0), 3),
                    "business_txn_count_30d": int(count_30d),
                    "business_avg_amount_30d": round(avg_30d, 2),
                    "amount_vs_business_mean_ratio": round(amount / max(avg_30d, 1.0), 4),
                    "recent_failed_count": int(recent_failed),
                    "hour_of_day": int(ts.hour),
                    "day_of_week": int(ts.weekday()),
                    "is_weekend": int(ts.weekday() >= 5),
                    "is_round_amount": int(amount % 1_000 == 0),
                    "account_details_changed_days": round(min(details_changed_days, 999.0), 3),
                    "corridor_risk": corridor.corridor_risk,
                    "is_new_beneficiary": int(beneficiary_age_days <= 7),
                    # --- internal simulation bookkeeping -------------------
                    "_provisional_failed": provisional_failed,
                    "_in_episode": int(in_episode),
                }
            )

            prior_ts.append(ts)
            prior_amounts.append(amount)
            if provisional_failed:
                prior_failed_ts.append(ts)

    df = pd.DataFrame(records)
    df = df.sort_values("transaction_timestamp").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Fraud labels
# --------------------------------------------------------------------------
def _fraud_logits(rng: np.random.Generator, df: pd.DataFrame) -> np.ndarray:
    c = FRAUD_COEFFICIENTS
    amount_dev = np.log1p(np.maximum(df["amount_vs_business_mean_ratio"].to_numpy(), 0.0))
    logits = (
        c["new_beneficiary_3d"] * (df["beneficiary_age_days"].to_numpy() <= 3)
        + c["new_beneficiary_large"]
        * ((df["beneficiary_age_days"].to_numpy() <= 7) & (df["amount_usd"].to_numpy() > 10_000))
        + c["velocity"] * np.minimum(df["transaction_velocity"].to_numpy() / 8.0, 1.6)
        + c["young_account"] * (df["account_age_days"].to_numpy() <= 30)
        + c["amount_deviation"] * np.minimum(amount_dev / 2.0, 1.6)
        + c["odd_hour"] * (df["hour_of_day"].to_numpy() <= 5)
        + c["corridor_risk"] * df["corridor_risk"].to_numpy()
        + c["recent_failures"] * np.minimum(df["recent_failed_count"].to_numpy() / 3.0, 1.5)
        + c["structuring"]
        * (
            (df["amount_usd"].to_numpy() >= 8_500)
            & (df["amount_usd"].to_numpy() < 10_000)
            & (df["transaction_velocity"].to_numpy() >= 3)
        )
        + c["details_changed"] * (df["account_details_changed_days"].to_numpy() <= 7)
        + rng.normal(0.0, c["noise_sd"], size=len(df))
    )
    return logits


def _calibrate_labels(
    rng: np.random.Generator, df: pd.DataFrame, base_rate: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample fraud labels, bisecting the intercept to hit the target rate."""
    logits = _fraud_logits(rng, df)
    lo, hi = -14.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(logits + mid)))))
        if rate > base_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2.0
    probs = 1.0 / (1.0 + np.exp(-(logits + intercept)))
    labels = (rng.random(len(df)) < probs).astype(int)
    return labels, probs, intercept


# --------------------------------------------------------------------------
# Statuses derived AFTER labelling (post-outcome fields)
# --------------------------------------------------------------------------
def _assign_statuses(rng: np.random.Generator, df: pd.DataFrame, end: datetime) -> pd.DataFrame:
    n = len(df)
    status = np.where(df["_provisional_failed"].to_numpy(), "failed", "completed").astype(object)

    # Technical failures split between hard failure and funds returned.
    failed_idx = np.flatnonzero(df["_provisional_failed"].to_numpy())
    returned = failed_idx[rng.random(len(failed_idx)) < 0.30]
    status[returned] = "returned"

    fraud = df["fraud_label"].to_numpy().astype(bool)
    flagged = df["rules_flagged"].to_numpy().astype(bool)
    settled = status == "completed"

    # Caught fraud: flagged fraudulent payouts are held or returned.
    caught = settled & fraud & flagged & (rng.random(n) < 0.72)
    status[caught & (rng.random(n) < 0.55)] = "returned"
    status[np.flatnonzero(caught & (status == "completed"))] = "under_review"

    # False positives: a share of flagged legitimate payouts is held for review.
    fp_hold = settled & (~fraud) & flagged & (rng.random(n) < 0.14) & (status == "completed")
    status[np.flatnonzero(fp_hold)] = "under_review"

    # In-flight payouts initiated in the final 24 hours.
    recent = (df["transaction_timestamp"] >= (end - timedelta(hours=24))).to_numpy()
    pend = recent & (status == "completed") & (rng.random(n) < 0.45)
    status[np.flatnonzero(pend)] = "pending"

    df = df.copy()
    df["transaction_status"] = status
    df.loc[df["transaction_status"].isin(["failed", "returned"]), "settlement_time"] = np.nan

    compliance = np.full(n, "cleared", dtype=object)
    compliance[df["transaction_status"].to_numpy() == "under_review"] = "pending_review"
    crit = (df["rules_band"].to_numpy() == "critical") & (
        df["transaction_status"].to_numpy() == "under_review"
    )
    compliance[crit] = "escalated"
    compliance[(df["transaction_status"].to_numpy() == "returned") & flagged] = "blocked"
    df["compliance_status"] = compliance
    return df


# --------------------------------------------------------------------------
# Compliance cases
# --------------------------------------------------------------------------
def _generate_watchlist(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(420):
        name = f"{rng.choice(_PREFIX)} {rng.choice(_SUFFIX)}"
        rows.append(
            {
                "entry_id": f"WL-{i + 1:04d}",
                "entity_name": name,
                "entity_type": str(rng.choice(["entity", "individual"], p=[0.7, 0.3])),
                "list_source": str(rng.choice(_WATCHLIST_SOURCES)),
                "country": str(rng.choice(["MX", "NG", "PH", "BR", "IN", "GB", "SG", "DE"])),
                "is_fictional": True,
            }
        )
    return pd.DataFrame(rows)


def _generate_compliance_cases(
    rng: np.random.Generator, businesses: pd.DataFrame, end: datetime
) -> pd.DataFrame:
    statuses = ("approved", "rejected", "more_info_required", "review", "screening", "submitted", "draft")
    probs = (0.72, 0.05, 0.08, 0.06, 0.04, 0.035, 0.015)
    rows = []
    for i, biz in enumerate(businesses.itertuples()):
        created = datetime.fromisoformat(biz.account_created_date)
        submitted_at = created - timedelta(days=float(rng.uniform(0.2, 6.0)))
        status = str(rng.choice(statuses, p=probs))
        screening_hits = int(rng.choice([0, 1, 2, 3], p=[0.80, 0.13, 0.05, 0.02]))
        ubo_complete = bool(rng.random() < 0.93)
        docs_complete = bool(rng.random() < 0.90)
        straightforward = screening_hits == 0 and ubo_complete and docs_complete and biz.volume_tier != "enterprise"

        if straightforward:
            duration = float(np.clip(rng.lognormal(np.log(4.2), 0.55), 1.2, 60))
        else:
            duration = float(np.clip(rng.lognormal(np.log(26.0), 0.70), 4.0, 600))

        manual_review = (not straightforward) or rng.random() < 0.10
        if status in {"approved", "rejected", "more_info_required"}:
            decided_at = submitted_at + timedelta(minutes=duration)
        else:
            decided_at = None
            duration = float("nan")

        risk_class = (
            "high"
            if screening_hits >= 2 or biz.industry in {"Marketplace & E-commerce"} and screening_hits
            else "medium"
            if screening_hits == 1 or biz.volume_tier == "enterprise"
            else "low"
        )
        rows.append(
            {
                "case_id": f"KYB-{i + 1:04d}",
                "business_id": biz.business_id,
                "legal_name": biz.legal_name,
                "onboarding_status": status,
                "risk_classification": risk_class,
                "screening_hits": screening_hits,
                "ubo_checklist_complete": ubo_complete,
                "documents_complete": docs_complete,
                "manual_review_required": bool(manual_review),
                "submitted_at": submitted_at,
                "decided_at": decided_at,
                "duration_minutes": round(duration, 2) if duration == duration else float("nan"),
                "reviewer": str(rng.choice(_REVIEWERS)) if manual_review else "",
                "decision_notes": "",
                "is_straightforward": bool(straightforward),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Product analytics events
# --------------------------------------------------------------------------
_EVENT_CATALOG = (
    ("dashboard_viewed", 0.26),
    ("payout_created", 0.18),
    ("rail_comparison_viewed", 0.14),
    ("fraud_alert_viewed", 0.11),
    ("investigation_opened", 0.07),
    ("investigation_decided", 0.06),
    ("copilot_query", 0.06),
    ("report_exported", 0.05),
    ("compliance_case_reviewed", 0.04),
    ("experiment_enrolled", 0.03),
)


def _generate_product_events(
    rng: np.random.Generator, businesses: pd.DataFrame, start: datetime, end: datetime
) -> pd.DataFrame:
    names = [e[0] for e in _EVENT_CATALOG]
    weights = np.array([e[1] for e in _EVENT_CATALOG])
    weights = weights / weights.sum()

    rows: list[dict] = []
    counter = 0
    for biz in businesses.itertuples():
        created = datetime.fromisoformat(biz.account_created_date)
        window_start = max(start, created)
        if window_start >= end:
            continue
        n_users = int(rng.integers(1, 5))
        users = [f"{biz.business_id}-U{u + 1}" for u in range(n_users)]
        active_days = max((end - window_start).days, 1)

        # Lifecycle events
        counter += 1
        rows.append(
            {
                "event_id": f"EVT-{counter:07d}",
                "business_id": biz.business_id,
                "user_id": users[0],
                "event_name": "account_created",
                "event_timestamp": created,
                "session_id": f"S-{counter:07d}",
            }
        )
        activated = rng.random() < 0.78
        if activated:
            counter += 1
            rows.append(
                {
                    "event_id": f"EVT-{counter:07d}",
                    "business_id": biz.business_id,
                    "user_id": users[0],
                    "event_name": "first_payout_created",
                    "event_timestamp": created + timedelta(days=float(rng.uniform(0.2, 21.0))),
                    "session_id": f"S-{counter:07d}",
                }
            )

        intensity = {"small": 20.0, "mid": 55.0, "enterprise": 130.0}[biz.volume_tier]
        n_events = int(rng.poisson(intensity * min(active_days, 90) / 30.0))
        if n_events <= 0:
            continue
        ev_names = rng.choice(names, size=n_events, p=weights)
        offsets = rng.random(n_events) * active_days
        for name, off in zip(ev_names, offsets):
            counter += 1
            user = users[int(rng.integers(0, n_users))]
            rows.append(
                {
                    "event_id": f"EVT-{counter:07d}",
                    "business_id": biz.business_id,
                    "user_id": user,
                    "event_name": str(name),
                    "event_timestamp": window_start + timedelta(days=float(off)),
                    "session_id": f"S-{counter:07d}",
                }
            )

    df = pd.DataFrame(rows).sort_values("event_timestamp").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def generate_dataset(
    seed: int = config.RANDOM_SEED,
    n_businesses: int = config.N_BUSINESSES,
    target_transactions: int = config.TARGET_TRANSACTIONS,
    sim_days: int = config.SIM_DAYS,
    end_date: str = config.SIM_END_DATE,
    base_fraud_rate: float = config.BASE_FRAUD_RATE,
) -> GeneratedDataset:
    """Generate the full synthetic dataset. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    end = datetime.fromisoformat(end_date)
    start = end - timedelta(days=sim_days)

    businesses = _generate_businesses(rng, n_businesses, end)
    txns = _generate_transactions(rng, businesses, target_transactions, start, end)

    labels, probs, intercept = _calibrate_labels(rng, txns, base_fraud_rate)
    txns["fraud_label"] = labels
    txns["_latent_fraud_prob"] = np.round(probs, 6)

    # Legacy rules baseline score (observable features only, no label input).
    rule_cols = evaluate_rules_frame(txns)
    txns = pd.concat([txns, rule_cols], axis=1)
    txns["risk_score"] = txns["rules_score"]

    txns = _assign_statuses(rng, txns, end)
    txns = txns.drop(columns=["_provisional_failed"])

    watchlist = _generate_watchlist(rng)
    compliance = _generate_compliance_cases(rng, businesses, end)
    events = _generate_product_events(rng, businesses, start, end)

    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "seed": seed,
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": sim_days},
        "row_counts": {
            "transactions": int(len(txns)),
            "businesses": int(len(businesses)),
            "compliance_cases": int(len(compliance)),
            "product_events": int(len(events)),
            "watchlist": int(len(watchlist)),
        },
        "fraud": {
            "target_base_rate": base_fraud_rate,
            "realised_rate": float(txns["fraud_label"].mean()),
            "calibrated_intercept": round(float(intercept), 6),
            "coefficients": FRAUD_COEFFICIENTS,
            "compromise_episode_share": COMPROMISE_EPISODE_SHARE,
        },
        "assumptions": [
            "All entities, payments and watchlist entries are fictional.",
            "Behavioural features are computed from prior transactions only.",
            "risk_score is the rules baseline score, not a label-derived value.",
            "transaction_status, settlement_time and compliance_status are "
            "post-outcome fields and are excluded from ML features.",
            "Rail fee and FX assumptions are documented in docs/UNIT_ECONOMICS.md.",
            "Technical failure rates follow each rail's configured success rate; "
            "compromise episodes raise failure probability to simulate probing.",
        ],
        "leakage_controls": [
            "fraud_label",
            "transaction_status",
            "compliance_status",
            "settlement_time",
            "risk_score",
            "rules_score",
            "rules_band",
            "rules_flagged",
            "rules_triggered",
            "_latent_fraud_prob",
            "_in_episode",
        ],
    }

    return GeneratedDataset(
        transactions=txns,
        businesses=businesses,
        compliance_cases=compliance,
        product_events=events,
        watchlist=watchlist,
        manifest=manifest,
    )


def write_dataset(dataset: GeneratedDataset, also_csv: bool = True) -> dict:
    """Persist the dataset to the configured data directory."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.transactions.to_parquet(config.TRANSACTIONS_FILE, index=False)
    dataset.businesses.to_parquet(config.BUSINESSES_FILE, index=False)
    dataset.compliance_cases.to_parquet(config.COMPLIANCE_FILE, index=False)
    dataset.product_events.to_parquet(config.EVENTS_FILE, index=False)
    dataset.watchlist.to_parquet(config.WATCHLIST_FILE, index=False)
    if also_csv:
        dataset.transactions.to_csv(config.TRANSACTIONS_CSV, index=False)
    config.DATASET_MANIFEST.write_text(json.dumps(dataset.manifest, indent=2))
    return dataset.manifest
