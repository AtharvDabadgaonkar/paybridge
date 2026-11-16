"""Simulated payout lifecycle orchestration.

NOTHING in this module moves money. A "payout" is a row in SQLite plus an
auditable event history. The state machine mirrors how a real payout
operations flow is structured so the workflow logic can be tested.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paybridge.db import storage
from paybridge.payments.pricing import Quote, quote_payment

INITIAL_STATUS = "initiated"

VALID_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "initiated": ("screening", "rejected"),
    "screening": ("approved_for_payout", "under_review", "rejected"),
    "under_review": ("approved_for_payout", "rejected"),
    "approved_for_payout": ("in_transit", "rejected"),
    "in_transit": ("settled", "failed", "returned"),
    "settled": (),
    "rejected": (),
    "failed": ("in_transit",),      # a failed payout may be retried
    "returned": (),
}

TERMINAL_STATUSES = tuple(s for s, nxt in VALID_TRANSITIONS.items() if not nxt)
PAYOUT_STATUSES = tuple(VALID_TRANSITIONS)


class InvalidTransition(ValueError):
    """Raised when a payout status change is not permitted."""


@dataclass(frozen=True)
class PayoutRecord:
    payout_id: str
    reference: str
    status: str
    quote: Quote
    risk_score: float
    risk_band: str


def _reference(created: datetime) -> str:
    return f"PB-{created.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


def create_payout(
    business_id: str,
    recipient_country: str,
    amount_usd: float,
    rail_code: str,
    actor: str = "ops.user",
    business_name: str = "",
    risk_score: float = 0.0,
    risk_band: str = "low",
    ml_score: float | None = None,
    rules_triggered: str = "",
    db: Path | str | None = None,
) -> PayoutRecord:
    """Create a simulated payout and its first lifecycle event."""
    quote = quote_payment(rail_code, recipient_country, amount_usd)
    if not quote.eligible:
        raise InvalidTransition(f"Rail {rail_code} is not eligible: {quote.ineligible_reason}")

    created = datetime.now(timezone.utc)
    payout_id = f"PO-{uuid.uuid4().hex[:12]}"
    reference = _reference(created)
    ts = created.isoformat(timespec="seconds")

    storage.insert_row(
        "payouts",
        {
            "payout_id": payout_id,
            "reference": reference,
            "business_id": business_id,
            "business_name": business_name,
            "corridor": quote.corridor,
            "recipient_country": recipient_country,
            "currency": quote.currency,
            "amount_usd": quote.amount_usd,
            "payment_rail": quote.rail,
            "total_customer_cost": quote.total_customer_cost,
            "platform_direct_cost": quote.platform_direct_cost,
            "contribution_margin": quote.contribution_margin,
            "fx_markup_cost": quote.fx_markup_cost,
            "recipient_amount_local": quote.recipient_amount_local,
            "settlement_hours_p50": quote.settlement_hours_p50,
            "status": INITIAL_STATUS,
            "risk_score": float(risk_score),
            "risk_band": risk_band,
            "ml_score": None if ml_score is None else float(ml_score),
            "rules_triggered": rules_triggered,
            "created_at": ts,
            "updated_at": ts,
            "quote_json": json.dumps(quote.as_dict()),
        },
        path=db,
    )
    storage.insert_row(
        "payout_events",
        {
            "payout_id": payout_id,
            "from_status": None,
            "to_status": INITIAL_STATUS,
            "actor": actor,
            "note": "Simulated payout created",
            "occurred_at": ts,
        },
        path=db,
    )
    storage.log_audit(
        actor,
        "payout",
        payout_id,
        "create",
        {
            "reference": reference,
            "amount_usd": quote.amount_usd,
            "rail": quote.rail,
            "corridor": quote.corridor,
            "risk_band": risk_band,
        },
        path=db,
    )
    return PayoutRecord(payout_id, reference, INITIAL_STATUS, quote, float(risk_score), risk_band)


def get_payout(payout_id: str, db: Path | str | None = None) -> dict[str, Any] | None:
    df = storage.query_df("SELECT * FROM payouts WHERE payout_id = ?", (payout_id,), path=db)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def find_payout_by_reference(reference: str, db: Path | str | None = None) -> dict[str, Any] | None:
    df = storage.query_df("SELECT * FROM payouts WHERE reference = ?", (reference,), path=db)
    return None if df.empty else df.iloc[0].to_dict()


def list_payouts(limit: int = 500, db: Path | str | None = None) -> pd.DataFrame:
    return storage.query_df(
        "SELECT * FROM payouts ORDER BY created_at DESC LIMIT ?", (int(limit),), path=db
    )


def payout_events(payout_id: str, db: Path | str | None = None) -> pd.DataFrame:
    return storage.query_df(
        "SELECT * FROM payout_events WHERE payout_id = ? ORDER BY event_id",
        (payout_id,),
        path=db,
    )


def can_transition(from_status: str, to_status: str) -> bool:
    return to_status in VALID_TRANSITIONS.get(from_status, ())


def transition_payout(
    payout_id: str,
    to_status: str,
    actor: str = "ops.user",
    note: str = "",
    db: Path | str | None = None,
) -> str:
    """Move a payout to ``to_status``, validating the state machine."""
    record = get_payout(payout_id, db=db)
    if record is None:
        raise KeyError(f"Unknown payout {payout_id}")
    current = str(record["status"])
    if not can_transition(current, to_status):
        allowed = VALID_TRANSITIONS.get(current, ())
        raise InvalidTransition(
            f"Cannot move payout from '{current}' to '{to_status}'. Allowed: {allowed or 'none (terminal)'}"
        )

    ts = storage.now_iso()
    storage.execute(
        "UPDATE payouts SET status = ?, updated_at = ? WHERE payout_id = ?",
        (to_status, ts, payout_id),
        path=db,
    )
    storage.insert_row(
        "payout_events",
        {
            "payout_id": payout_id,
            "from_status": current,
            "to_status": to_status,
            "actor": actor,
            "note": note,
            "occurred_at": ts,
        },
        path=db,
    )
    storage.log_audit(
        actor, "payout", payout_id, "transition", {"from": current, "to": to_status, "note": note}, path=db
    )
    return to_status


def suggested_next_status(status: str, risk_band: str) -> str | None:
    """The next step a simulated payout would take under the configured policy.

    High and critical risk payouts are routed to human review; FraudLens never
    auto-approves them.
    """
    if status == "initiated":
        return "screening"
    if status == "screening":
        return "under_review" if risk_band in {"high", "critical"} else "approved_for_payout"
    if status == "under_review":
        return None            # requires an explicit human decision
    if status == "approved_for_payout":
        return "in_transit"
    if status == "in_transit":
        return "settled"
    if status == "failed":
        return "in_transit"
    return None


def advance_payout(
    payout_id: str, actor: str = "ops.user", note: str = "", db: Path | str | None = None
) -> str | None:
    """Advance one step along the policy path. Returns the new status, or None
    when the payout is terminal or awaiting a human decision."""
    record = get_payout(payout_id, db=db)
    if record is None:
        raise KeyError(f"Unknown payout {payout_id}")
    nxt = suggested_next_status(str(record["status"]), str(record.get("risk_band") or "low"))
    if nxt is None:
        return None
    return transition_payout(payout_id, nxt, actor=actor, note=note or "Simulated progression", db=db)
