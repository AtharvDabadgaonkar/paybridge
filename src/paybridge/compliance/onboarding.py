"""KYB onboarding workflow: status machine, decisions and audit trail.

The historical case list is part of the immutable synthetic dataset; every
decision a user makes in the app is stored in SQLite as an override, so the
dataset is never mutated and the full decision history stays auditable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from paybridge.db import storage

DRAFT = "draft"
SUBMITTED = "submitted"
SCREENING = "screening"
REVIEW = "review"
APPROVED = "approved"
REJECTED = "rejected"
MORE_INFO = "more_info_required"

TRANSITIONS: dict[str, tuple[str, ...]] = {
    DRAFT: (SUBMITTED,),
    SUBMITTED: (SCREENING, MORE_INFO),
    SCREENING: (REVIEW, APPROVED, REJECTED, MORE_INFO),
    REVIEW: (APPROVED, REJECTED, MORE_INFO),
    MORE_INFO: (SUBMITTED, REJECTED),
    APPROVED: (),
    REJECTED: (),
}

TERMINAL = tuple(s for s, nxt in TRANSITIONS.items() if not nxt)
DECIDED = (APPROVED, REJECTED)

# Product target, NOT a measured usability result.
ONBOARDING_TARGET_MINUTES = 5.0


class InvalidOnboardingTransition(ValueError):
    """Raised when an onboarding status change is not permitted."""


def can_transition(from_status: str, to_status: str) -> bool:
    return to_status in TRANSITIONS.get(from_status, ())


def allowed_next(from_status: str) -> tuple[str, ...]:
    return TRANSITIONS.get(from_status, ())


def record_decision(
    case_id: str,
    business_id: str,
    from_status: str,
    to_status: str,
    reviewer: str,
    note: str = "",
    db: Path | str | None = None,
) -> str:
    """Validate and persist an onboarding status change."""
    if not can_transition(from_status, to_status):
        raise InvalidOnboardingTransition(
            f"Cannot move onboarding case from '{from_status}' to '{to_status}'. "
            f"Allowed: {allowed_next(from_status) or 'none (terminal)'}"
        )
    storage.insert_row(
        "compliance_decisions",
        {
            "case_id": case_id,
            "business_id": business_id,
            "from_status": from_status,
            "to_status": to_status,
            "reviewer": reviewer,
            "note": note,
            "decided_at": storage.now_iso(),
        },
        path=db,
    )
    storage.log_audit(
        reviewer, "compliance_case", case_id,
        "status_change", {"from": from_status, "to": to_status, "note": note}, path=db,
    )
    return to_status


def decisions_for(case_id: str, db: Path | str | None = None) -> pd.DataFrame:
    return storage.query_df(
        "SELECT * FROM compliance_decisions WHERE case_id = ? ORDER BY decision_id",
        (case_id,),
        path=db,
    )


def all_decisions(db: Path | str | None = None) -> pd.DataFrame:
    return storage.query_df("SELECT * FROM compliance_decisions ORDER BY decision_id", path=db)


def apply_overrides(cases: pd.DataFrame, db: Path | str | None = None) -> pd.DataFrame:
    """Overlay in-app decisions onto the synthetic case list."""
    out = cases.copy()
    out["current_status"] = out["onboarding_status"]
    out["decided_in_app"] = False
    decisions = all_decisions(db)
    if decisions.empty:
        return out
    latest = decisions.sort_values("decision_id").groupby("case_id").tail(1)
    status_by_case = dict(zip(latest["case_id"], latest["to_status"]))
    reviewer_by_case = dict(zip(latest["case_id"], latest["reviewer"]))
    mask = out["case_id"].isin(status_by_case)
    out.loc[mask, "current_status"] = out.loc[mask, "case_id"].map(status_by_case)
    out.loc[mask, "decided_in_app"] = True
    out.loc[mask, "reviewer"] = out.loc[mask, "case_id"].map(reviewer_by_case)
    return out


def compliance_kpis(cases: pd.DataFrame) -> dict[str, Any]:
    """Onboarding KPIs computed from the (override-aware) case list."""
    status_col = "current_status" if "current_status" in cases.columns else "onboarding_status"
    n = int(len(cases))
    if n == 0:
        return {
            "total_cases": 0, "completion_rate_pct": 0.0, "approval_rate_pct": 0.0,
            "manual_review_rate_pct": 0.0, "screening_alert_rate_pct": 0.0,
            "median_duration_minutes": 0.0, "mean_duration_minutes": 0.0,
            "straightforward_median_minutes": 0.0, "pct_under_target": 0.0,
            "open_cases": 0, "target_minutes": ONBOARDING_TARGET_MINUTES,
        }

    decided = cases[cases[status_col].isin(DECIDED)]
    approved = cases[cases[status_col] == APPROVED]
    durations = cases["duration_minutes"].dropna()
    simple = cases[cases["is_straightforward"] == True]  # noqa: E712
    simple_durations = simple["duration_minutes"].dropna()

    return {
        "total_cases": n,
        "completion_rate_pct": len(decided) / n * 100.0,
        "approval_rate_pct": (len(approved) / len(decided) * 100.0) if len(decided) else 0.0,
        "manual_review_rate_pct": float(cases["manual_review_required"].mean() * 100.0),
        "screening_alert_rate_pct": float((cases["screening_hits"] > 0).mean() * 100.0),
        "median_duration_minutes": float(durations.median()) if len(durations) else 0.0,
        "mean_duration_minutes": float(durations.mean()) if len(durations) else 0.0,
        "straightforward_median_minutes": float(simple_durations.median()) if len(simple_durations) else 0.0,
        "pct_under_target": float((simple_durations < ONBOARDING_TARGET_MINUTES).mean() * 100.0)
        if len(simple_durations)
        else 0.0,
        "open_cases": int((~cases[status_col].isin(DECIDED)).sum()),
        "target_minutes": ONBOARDING_TARGET_MINUTES,
    }
