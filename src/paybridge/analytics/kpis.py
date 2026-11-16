"""Deterministic KPI and aggregate computation over the synthetic dataset.

Every figure shown in the UI and every figure quoted by the AI copilot is
produced here from the actual records. Nothing in this module is hardcoded.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd

from paybridge.config import FRAUD_LOSS_RATE
from paybridge.payments.pricing import cost_components_vectorized

BASELINE_RAIL = "SWIFT"


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    date_range: tuple[date | datetime | None, date | datetime | None] | None = None,
    rails: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    corridors: Sequence[str] | None = None,
    business_ids: Sequence[str] | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    risk_bands: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply the dashboard filter set. Empty/None selections mean 'all'."""
    out = df
    if date_range and date_range[0] is not None and date_range[1] is not None:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        out = out[(out["transaction_timestamp"] >= start) & (out["transaction_timestamp"] < end)]
    if rails:
        out = out[out["payment_rail"].isin(list(rails))]
    if statuses:
        out = out[out["transaction_status"].isin(list(statuses))]
    if corridors:
        out = out[out["corridor"].isin(list(corridors))]
    if business_ids:
        out = out[out["business_id"].isin(list(business_ids))]
    if risk_bands:
        out = out[out["rules_band"].isin(list(risk_bands))]
    if min_amount is not None:
        out = out[out["amount_usd"] >= min_amount]
    if max_amount is not None:
        out = out[out["amount_usd"] <= max_amount]
    return out


# --------------------------------------------------------------------------
# Modelled savings
# --------------------------------------------------------------------------
def modelled_savings(df: pd.DataFrame, baseline_rail: str = BASELINE_RAIL) -> dict[str, float]:
    """Modelled payout-cost saving of the realised rail mix vs an all-SWIFT mix.

    MODELLED, NOT REALISED: this is a counterfactual computed by repricing each
    synthetic transaction on the baseline rail under the documented fee
    assumptions in docs/UNIT_ECONOMICS.md. It is not booked savings.
    """
    if df.empty:
        return {
            "actual_cost": 0.0,
            "baseline_cost": 0.0,
            "savings_usd": 0.0,
            "savings_pct": 0.0,
            "transactions_repriced": 0,
        }
    baseline = cost_components_vectorized(
        baseline_rail, df["amount_usd"].to_numpy(), df["corridor"].tolist()
    )
    actual_cost = float(df["total_cost_usd"].sum())
    baseline_cost = float(baseline["total_customer_cost"].sum())
    savings = baseline_cost - actual_cost
    return {
        "actual_cost": actual_cost,
        "baseline_cost": baseline_cost,
        "savings_usd": savings,
        "savings_pct": (savings / baseline_cost * 100.0) if baseline_cost else 0.0,
        "transactions_repriced": int(len(df)),
    }


# --------------------------------------------------------------------------
# Executive KPIs
# --------------------------------------------------------------------------
def executive_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """Primary executive dashboard KPIs, computed from the filtered frame."""
    n = int(len(df))
    if n == 0:
        return {
            "total_volume_usd": 0.0,
            "transaction_count": 0,
            "successful_transactions": 0,
            "success_rate_pct": 0.0,
            "avg_transaction_fee_usd": 0.0,
            "avg_cost_pct_of_amount": 0.0,
            "avg_settlement_hours": 0.0,
            "median_settlement_hours": 0.0,
            "modelled_savings_usd": 0.0,
            "modelled_savings_pct": 0.0,
            "fraud_alert_rate_pct": 0.0,
            "transactions_under_review": 0,
            "total_revenue_usd": 0.0,
            "contribution_margin_usd": 0.0,
            "contribution_margin_pct": 0.0,
            "confirmed_fraud_count": 0,
            "confirmed_fraud_rate_pct": 0.0,
        }

    successful = int(df["is_successful"].sum())
    completed = df[df["transaction_status"] == "completed"]
    savings = modelled_savings(df)
    revenue = float(df["total_cost_usd"].sum())
    margin = float(df["contribution_margin"].sum())

    return {
        "total_volume_usd": float(df["amount_usd"].sum()),
        "transaction_count": n,
        "successful_transactions": successful,
        "success_rate_pct": successful / n * 100.0,
        "avg_transaction_fee_usd": float(df["total_cost_usd"].mean()),
        "avg_cost_pct_of_amount": float(df["cost_pct_of_amount"].mean()),
        "avg_settlement_hours": float(completed["settlement_time"].mean()) if len(completed) else 0.0,
        "median_settlement_hours": float(completed["settlement_time"].median()) if len(completed) else 0.0,
        "modelled_savings_usd": savings["savings_usd"],
        "modelled_savings_pct": savings["savings_pct"],
        "fraud_alert_rate_pct": float(df["rules_flagged"].mean() * 100.0),
        "transactions_under_review": int((df["transaction_status"] == "under_review").sum()),
        "total_revenue_usd": revenue,
        "contribution_margin_usd": margin,
        "contribution_margin_pct": (margin / revenue * 100.0) if revenue else 0.0,
        "confirmed_fraud_count": int(df["fraud_label"].sum()),
        "confirmed_fraud_rate_pct": float(df["fraud_label"].mean() * 100.0),
    }


# --------------------------------------------------------------------------
# Aggregates for visualisations
# --------------------------------------------------------------------------
def volume_timeseries(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "volume_usd", "transactions", "avg_fee_usd"])
    grp = (
        df.set_index("transaction_timestamp")
        .resample(freq)
        .agg(
            volume_usd=("amount_usd", "sum"),
            transactions=("transaction_id", "count"),
            avg_fee_usd=("total_cost_usd", "mean"),
        )
        .reset_index()
        .rename(columns={"transaction_timestamp": "period"})
    )
    return grp.fillna(0.0)


def corridor_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["corridor", "transactions", "volume_usd", "avg_cost_pct", "success_rate_pct"]
        )
    out = (
        df.groupby("corridor")
        .agg(
            transactions=("transaction_id", "count"),
            volume_usd=("amount_usd", "sum"),
            total_cost_usd=("total_cost_usd", "sum"),
            avg_cost_pct=("cost_pct_of_amount", "mean"),
            success_rate_pct=("is_successful", "mean"),
            fraud_rate_pct=("fraud_label", "mean"),
        )
        .reset_index()
    )
    out["success_rate_pct"] *= 100.0
    out["fraud_rate_pct"] *= 100.0
    return out.sort_values("volume_usd", ascending=False)


def rail_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["payment_rail", "transactions", "volume_usd", "share_pct"])
    out = (
        df.groupby("payment_rail")
        .agg(
            transactions=("transaction_id", "count"),
            volume_usd=("amount_usd", "sum"),
            avg_fee_usd=("total_cost_usd", "mean"),
            avg_cost_pct=("cost_pct_of_amount", "mean"),
            margin_usd=("contribution_margin", "sum"),
        )
        .reset_index()
    )
    out["share_pct"] = out["transactions"] / out["transactions"].sum() * 100.0
    return out.sort_values("transactions", ascending=False)


def settlement_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Realised settlement time distribution per rail (completed payouts only)."""
    completed = df[(df["transaction_status"] == "completed") & df["settlement_time"].notna()]
    if completed.empty:
        return pd.DataFrame(columns=["payment_rail", "p50_hours", "p90_hours", "mean_hours", "n"])
    out = (
        completed.groupby("payment_rail")["settlement_time"]
        .agg(
            p50_hours=lambda s: float(np.percentile(s, 50)),
            p90_hours=lambda s: float(np.percentile(s, 90)),
            mean_hours="mean",
            n="count",
        )
        .reset_index()
    )
    return out.sort_values("p50_hours")


def success_trend(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "success_rate_pct", "transactions"])
    out = (
        df.set_index("transaction_timestamp")
        .resample(freq)
        .agg(success_rate_pct=("is_successful", "mean"), transactions=("transaction_id", "count"))
        .reset_index()
        .rename(columns={"transaction_timestamp": "period"})
    )
    out["success_rate_pct"] = out["success_rate_pct"] * 100.0
    return out.dropna(subset=["success_rate_pct"])


def fraud_alerts_by_band(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["rules_band", "transactions", "alerts", "confirmed_fraud"])
    out = (
        df.groupby("rules_band")
        .agg(
            transactions=("transaction_id", "count"),
            alerts=("rules_flagged", "sum"),
            confirmed_fraud=("fraud_label", "sum"),
            volume_usd=("amount_usd", "sum"),
        )
        .reset_index()
    )
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    out["precision_pct"] = np.where(
        out["alerts"] > 0, out["confirmed_fraud"] / out["alerts"].replace(0, np.nan) * 100.0, 0.0
    )
    return out.sort_values("rules_band", key=lambda s: s.map(order))


def cost_breakdown_by_rail(df: pd.DataFrame) -> pd.DataFrame:
    """Customer-price composition per rail, in USD totals."""
    if df.empty:
        return pd.DataFrame(columns=["payment_rail", "component", "usd"])
    rows = []
    for rail, grp in df.groupby("payment_rail"):
        components = cost_components_vectorized(
            str(rail), grp["amount_usd"].to_numpy(), grp["corridor"].tolist()
        )
        for component in ("fixed_fee", "variable_fee", "fx_markup_cost", "lifting_fees"):
            rows.append(
                {
                    "payment_rail": rail,
                    "component": component,
                    "usd": float(components[component].sum()),
                }
            )
    return pd.DataFrame(rows)


def high_risk_trend(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "high_risk_transactions", "high_risk_rate_pct"])
    tmp = df.copy()
    tmp["is_high_risk"] = tmp["rules_band"].isin(["high", "critical"]).astype(int)
    out = (
        tmp.set_index("transaction_timestamp")
        .resample(freq)
        .agg(
            high_risk_transactions=("is_high_risk", "sum"),
            transactions=("transaction_id", "count"),
            under_review=("is_under_review", "sum"),
        )
        .reset_index()
        .rename(columns={"transaction_timestamp": "period"})
    )
    out["high_risk_rate_pct"] = np.where(
        out["transactions"] > 0, out["high_risk_transactions"] / out["transactions"] * 100.0, 0.0
    )
    return out


def top_risk_transactions(df: pd.DataFrame, n: int = 25, score_col: str = "rules_score") -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "transaction_id", "business_id", "corridor", "payment_rail", "amount_usd",
        "transaction_timestamp", score_col, "rules_band", "rules_triggered",
        "transaction_status", "fraud_label",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.nlargest(n, score_col)[cols]


def business_leaderboard(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("business_id")
        .agg(
            transactions=("transaction_id", "count"),
            volume_usd=("amount_usd", "sum"),
            fees_paid_usd=("total_cost_usd", "sum"),
            margin_usd=("contribution_margin", "sum"),
            alert_rate_pct=("rules_flagged", "mean"),
            success_rate_pct=("is_successful", "mean"),
        )
        .reset_index()
    )
    out["alert_rate_pct"] *= 100.0
    out["success_rate_pct"] *= 100.0
    return out.nlargest(n, "volume_usd")


def fraud_loss_exposure(df: pd.DataFrame, loss_rate: float = FRAUD_LOSS_RATE) -> dict[str, float]:
    """Modelled fraud loss exposure: settled fraudulent volume x loss rate."""
    if df.empty:
        return {"settled_fraud_volume_usd": 0.0, "modelled_loss_usd": 0.0, "loss_rate": loss_rate}
    settled_fraud = df[(df["fraud_label"] == 1) & (df["transaction_status"].isin(["completed", "pending"]))]
    volume = float(settled_fraud["amount_usd"].sum())
    return {
        "settled_fraud_volume_usd": volume,
        "modelled_loss_usd": volume * loss_rate,
        "loss_rate": loss_rate,
    }
