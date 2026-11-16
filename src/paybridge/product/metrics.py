"""Product management KPIs.

Adoption, activation and retention come from the synthetic PRODUCT EVENT
stream, not from payment records: payment data alone cannot support those
metrics, and deriving them from it would be misleading.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ACTIVATION_EVENT = "first_payout_created"
SIGNUP_EVENT = "account_created"

FEATURE_EVENTS: dict[str, str] = {
    "payout_created": "Create a payout",
    "rail_comparison_viewed": "Compare payment rails",
    "fraud_alert_viewed": "Review a fraud alert",
    "investigation_opened": "Open an investigation",
    "investigation_decided": "Decide an investigation",
    "copilot_query": "Ask the AI copilot",
    "report_exported": "Export a report",
    "compliance_case_reviewed": "Review a compliance case",
    "dashboard_viewed": "View the dashboard",
}


def activation_metrics(events: pd.DataFrame) -> dict[str, Any]:
    """Share of signed-up businesses reaching their first payout, and how fast."""
    signups = events[events["event_name"] == SIGNUP_EVENT][["business_id", "event_timestamp"]]
    activations = events[events["event_name"] == ACTIVATION_EVENT][["business_id", "event_timestamp"]]
    if signups.empty:
        return {"signups": 0, "activated": 0, "activation_rate_pct": 0.0,
                "median_days_to_activate": float("nan"), "activated_within_7_days_pct": 0.0}

    signups = signups.rename(columns={"event_timestamp": "signed_up_at"}).drop_duplicates("business_id")
    activations = (
        activations.rename(columns={"event_timestamp": "activated_at"})
        .sort_values("activated_at")
        .drop_duplicates("business_id")
    )
    merged = signups.merge(activations, on="business_id", how="left")
    merged["days_to_activate"] = (
        merged["activated_at"] - merged["signed_up_at"]
    ).dt.total_seconds() / 86_400.0
    activated = merged["activated_at"].notna()
    return {
        "signups": int(len(merged)),
        "activated": int(activated.sum()),
        "activation_rate_pct": float(activated.mean() * 100.0),
        "median_days_to_activate": float(merged.loc[activated, "days_to_activate"].median())
        if activated.any() else float("nan"),
        "activated_within_7_days_pct": float(
            (merged.loc[activated, "days_to_activate"] <= 7).sum() / len(merged) * 100.0
        ) if activated.any() else 0.0,
    }


def feature_adoption(events: pd.DataFrame) -> pd.DataFrame:
    """Share of active businesses that used each feature at least once."""
    active = events["business_id"].nunique()
    if not active:
        return pd.DataFrame(columns=["feature", "event_name", "businesses", "adoption_pct", "events"])
    rows = []
    for event_name, label in FEATURE_EVENTS.items():
        subset = events[events["event_name"] == event_name]
        rows.append(
            {
                "feature": label,
                "event_name": event_name,
                "businesses": int(subset["business_id"].nunique()),
                "adoption_pct": subset["business_id"].nunique() / active * 100.0,
                "events": int(len(subset)),
            }
        )
    return pd.DataFrame(rows).sort_values("adoption_pct", ascending=False)


def weekly_active_businesses(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["week", "active_businesses", "events"])
    return (
        events.groupby("week")
        .agg(active_businesses=("business_id", "nunique"), events=("event_id", "count"))
        .reset_index()
    )


def weekly_retention(events: pd.DataFrame, weeks: int = 8) -> pd.DataFrame:
    """Simple week-over-week return rate of active businesses."""
    weekly = events.groupby("week")["business_id"].apply(set).sort_index()
    rows = []
    previous_week = None
    previous_set: set[str] = set()
    for week, ids in weekly.items():
        if previous_week is not None and previous_set:
            retained = len(ids & previous_set)
            rows.append(
                {
                    "week": week,
                    "active": len(ids),
                    "retained_from_prior_week": retained,
                    "retention_pct": retained / len(previous_set) * 100.0,
                }
            )
        previous_week, previous_set = week, ids
    return pd.DataFrame(rows).tail(weeks)


def payment_product_kpis(transactions: pd.DataFrame) -> dict[str, Any]:
    """Payment-side product KPIs computed from the transaction records."""
    if transactions.empty:
        return {}
    n = len(transactions)
    revenue = float(transactions["total_cost_usd"].sum())
    direct_cost = float(transactions["platform_direct_cost"].sum())
    completed = transactions[transactions["transaction_status"] == "completed"]
    terminal = transactions[transactions["transaction_status"].isin(
        ["completed", "failed", "returned", "under_review"]
    )]
    return {
        "payout_completion_rate_pct": float(len(completed) / len(terminal) * 100.0) if len(terminal) else 0.0,
        "payment_success_rate_pct": float(transactions["is_successful"].mean() * 100.0),
        "cost_per_payout_usd": direct_cost / n,
        "avg_revenue_per_transaction_usd": revenue / n,
        "contribution_margin_per_transaction_usd": (revenue - direct_cost) / n,
        "manual_review_rate_pct": float(transactions["is_under_review"].mean() * 100.0),
        "alert_rate_pct": float(transactions["rules_flagged"].mean() * 100.0),
        "transactions": int(n),
        "active_businesses": int(transactions["business_id"].nunique()),
        "avg_payouts_per_business": float(n / transactions["business_id"].nunique()),
    }


def investigation_cycle_time(audit: pd.DataFrame) -> dict[str, Any]:
    """Time from case creation to decision, measured from the audit log.

    Returns ``available: False`` rather than a fabricated number when the demo
    session has not produced enough decisions to measure.
    """
    if audit.empty:
        return {"available": False, "reason": "No investigation audit records yet.", "decided_cases": 0}
    frame = audit.copy()
    frame["occurred_at"] = pd.to_datetime(frame["occurred_at"], format="mixed", utc=True)
    created = frame[frame["action"] == "create"].groupby("entity_id")["occurred_at"].min()
    decided = frame[frame["action"] == "decision"].groupby("entity_id")["occurred_at"].max()
    joined = pd.concat([created.rename("created"), decided.rename("decided")], axis=1).dropna()
    if joined.empty:
        return {
            "available": False,
            "reason": "No investigation has been decided in this demo session yet.",
            "decided_cases": 0,
        }
    minutes = (joined["decided"] - joined["created"]).dt.total_seconds() / 60.0
    return {
        "available": True,
        "decided_cases": int(len(minutes)),
        "median_minutes": float(minutes.median()),
        "mean_minutes": float(minutes.mean()),
        "p90_minutes": float(np.percentile(minutes, 90)),
    }


def usage_by_feature_over_time(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["week", "event_name", "events"])
    subset = events[events["event_name"].isin(FEATURE_EVENTS)]
    return (
        subset.groupby(["week", "event_name"]).size().reset_index(name="events")
    )
