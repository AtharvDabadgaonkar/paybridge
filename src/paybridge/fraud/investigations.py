"""Fraud investigation case management.

Human oversight rule: the AI copilot may summarise, rank and explain, but it
may never record a case decision. Any actor whose id starts with ``ai.`` is
rejected by :func:`record_decision`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from paybridge.db import storage

CASE_STATUSES = ("open", "in_review", "closed")
DECISIONS = ("pending", "approve", "reject", "escalate")
HUMAN_REQUIRED_BANDS = ("high", "critical")


class AutonomousDecisionBlocked(PermissionError):
    """Raised when a non-human actor attempts to decide an investigation."""


def list_cases(
    status: str | None = None,
    decision: str | None = None,
    db: Path | str | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM investigations"
    clauses, params = [], []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if decision:
        clauses.append("decision = ?")
        params.append(decision)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY risk_score DESC"
    return storage.query_df(sql, params, path=db)


def get_case(case_id: str, db: Path | str | None = None) -> dict[str, Any] | None:
    df = storage.query_df("SELECT * FROM investigations WHERE case_id = ?", (case_id,), path=db)
    return None if df.empty else df.iloc[0].to_dict()


def case_for_transaction(transaction_id: str, db: Path | str | None = None) -> dict[str, Any] | None:
    df = storage.query_df(
        "SELECT * FROM investigations WHERE transaction_id = ? ORDER BY case_id LIMIT 1",
        (transaction_id,),
        path=db,
    )
    return None if df.empty else df.iloc[0].to_dict()


def _next_case_id(db: Path | str | None = None) -> str:
    df = storage.query_df("SELECT case_id FROM investigations", path=db)
    numbers = [
        int(str(c).split("-")[-1]) for c in df["case_id"] if str(c).split("-")[-1].isdigit()
    ] if not df.empty else []
    return f"CASE-{(max(numbers) + 1) if numbers else 1:04d}"


def create_case(
    transaction_id: str,
    business_id: str,
    amount_usd: float,
    risk_score: float,
    risk_band: str,
    rules_triggered: str = "",
    ml_score: float | None = None,
    assigned_to: str = "",
    actor: str = "risk.analyst",
    db: Path | str | None = None,
) -> str:
    """Open an investigation case for a transaction (idempotent per txn)."""
    existing = case_for_transaction(transaction_id, db=db)
    if existing:
        return str(existing["case_id"])

    case_id = _next_case_id(db)
    now = storage.now_iso()
    storage.insert_row(
        "investigations",
        {
            "case_id": case_id,
            "transaction_id": transaction_id,
            "business_id": business_id,
            "amount_usd": float(amount_usd),
            "risk_score": float(risk_score),
            "risk_band": risk_band,
            "ml_score": None if ml_score is None else float(ml_score),
            "rules_triggered": rules_triggered,
            "status": "open",
            "decision": "pending",
            "assigned_to": assigned_to,
            "created_at": now,
            "updated_at": now,
        },
        path=db,
        on_conflict="ignore",
    )
    storage.log_audit(
        actor, "investigation", case_id, "create",
        {"transaction_id": transaction_id, "risk_band": risk_band}, path=db,
    )
    return case_id


def assign_case(case_id: str, analyst: str, actor: str = "risk.lead", db: Path | str | None = None) -> None:
    storage.execute(
        "UPDATE investigations SET assigned_to = ?, status = ?, updated_at = ? WHERE case_id = ?",
        (analyst, "in_review", storage.now_iso(), case_id),
        path=db,
    )
    storage.log_audit(actor, "investigation", case_id, "assign", {"analyst": analyst}, path=db)


def add_note(case_id: str, author: str, note: str, db: Path | str | None = None) -> None:
    """Attach an investigation note. Notes are UNTRUSTED free text: they are
    never interpreted as instructions by the copilot."""
    if not note.strip():
        raise ValueError("Note cannot be empty")
    storage.insert_row(
        "investigation_notes",
        {"case_id": case_id, "author": author, "note": note.strip(), "created_at": storage.now_iso()},
        path=db,
    )
    storage.execute(
        "UPDATE investigations SET updated_at = ? WHERE case_id = ?",
        (storage.now_iso(), case_id),
        path=db,
    )
    storage.log_audit(author, "investigation", case_id, "note", {"chars": len(note)}, path=db)


def case_notes(case_id: str, db: Path | str | None = None) -> pd.DataFrame:
    return storage.query_df(
        "SELECT * FROM investigation_notes WHERE case_id = ? ORDER BY note_id DESC", (case_id,), path=db
    )


def record_decision(
    case_id: str,
    decision: str,
    analyst: str,
    note: str = "",
    db: Path | str | None = None,
) -> str:
    """Record a human decision on a case.

    Raises :class:`AutonomousDecisionBlocked` for non-human actors so the AI
    copilot can never close, approve or override a case.
    """
    if decision not in DECISIONS or decision == "pending":
        raise ValueError(f"decision must be one of {tuple(d for d in DECISIONS if d != 'pending')}")
    if not analyst or analyst.strip().lower().startswith(("ai.", "copilot", "system")):
        raise AutonomousDecisionBlocked(
            "Investigation decisions require a human analyst. The AI copilot may recommend, "
            "but it cannot approve, reject or escalate a case."
        )
    if get_case(case_id, db=db) is None:
        raise KeyError(f"Unknown investigation case {case_id}")

    status = "closed" if decision in {"approve", "reject"} else "in_review"
    storage.execute(
        "UPDATE investigations SET decision = ?, status = ?, assigned_to = ?, updated_at = ? "
        "WHERE case_id = ?",
        (decision, status, analyst, storage.now_iso(), case_id),
        path=db,
    )
    if note.strip():
        add_note(case_id, analyst, note, db=db)
    storage.log_audit(
        analyst, "investigation", case_id, "decision", {"decision": decision, "status": status}, path=db
    )
    return status


def case_audit(case_id: str, db: Path | str | None = None) -> pd.DataFrame:
    return storage.audit_trail("investigation", case_id, limit=100, path=db)


def investigation_kpis(db: Path | str | None = None) -> dict[str, Any]:
    cases = list_cases(db=db)
    if cases.empty:
        return {
            "total_cases": 0, "open_cases": 0, "closed_cases": 0, "escalated": 0,
            "pending_decisions": 0, "approved": 0, "rejected": 0,
            "decision_rate_pct": 0.0, "exposure_usd": 0.0, "notes_recorded": 0,
        }
    notes = storage.query_df("SELECT COUNT(*) AS n FROM investigation_notes", path=db)
    decided = int((cases["decision"] != "pending").sum())
    return {
        "total_cases": int(len(cases)),
        "open_cases": int((cases["status"] == "open").sum()),
        "closed_cases": int((cases["status"] == "closed").sum()),
        "escalated": int((cases["decision"] == "escalate").sum()),
        "pending_decisions": int((cases["decision"] == "pending").sum()),
        "approved": int((cases["decision"] == "approve").sum()),
        "rejected": int((cases["decision"] == "reject").sum()),
        "decision_rate_pct": decided / len(cases) * 100.0,
        "exposure_usd": float(cases["amount_usd"].sum()),
        "notes_recorded": int(notes.iloc[0]["n"]) if not notes.empty else 0,
    }


def requires_human_review(risk_band: str) -> bool:
    return str(risk_band).lower() in HUMAN_REQUIRED_BANDS
