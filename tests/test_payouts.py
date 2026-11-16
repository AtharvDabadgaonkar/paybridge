"""Payout workflow: creation, state machine and audit trail."""
from __future__ import annotations

import pytest

from paybridge.db import storage
from paybridge.payments import payouts as P


def _create(db, rail="USDC", amount=5_000.0, band="low"):
    return P.create_payout(
        business_id="BIZ-0001", recipient_country="MX", amount_usd=amount, rail_code=rail,
        business_name="Test Co", risk_score=0.1, risk_band=band, db=db,
    )


def test_create_payout_persists_quote_and_event(temp_db):
    record = _create(temp_db)
    assert record.reference.startswith("PB-")
    stored = P.get_payout(record.payout_id, db=temp_db)
    assert stored["status"] == "initiated"
    assert stored["total_customer_cost"] == pytest.approx(record.quote.total_customer_cost)
    events = P.payout_events(record.payout_id, db=temp_db)
    assert len(events) == 1 and events.iloc[0]["to_status"] == "initiated"


def test_create_payout_writes_audit_record(temp_db):
    record = _create(temp_db)
    trail = storage.audit_trail("payout", record.payout_id, path=temp_db)
    assert len(trail) == 1 and trail.iloc[0]["action"] == "create"


def test_reference_lookup(temp_db):
    record = _create(temp_db)
    assert P.find_payout_by_reference(record.reference, db=temp_db)["payout_id"] == record.payout_id


def test_ineligible_rail_is_rejected(temp_db):
    with pytest.raises(P.InvalidTransition):
        _create(temp_db, rail="SWIFT", amount=50.0)      # below the SWIFT minimum


def test_valid_transition_sequence(temp_db):
    record = _create(temp_db)
    for expected in ("screening", "approved_for_payout", "in_transit", "settled"):
        assert P.advance_payout(record.payout_id, db=temp_db) == expected
    assert P.advance_payout(record.payout_id, db=temp_db) is None     # terminal


def test_invalid_transitions_are_blocked(temp_db):
    record = _create(temp_db)
    with pytest.raises(P.InvalidTransition):
        P.transition_payout(record.payout_id, "settled", db=temp_db)   # cannot skip states
    P.transition_payout(record.payout_id, "screening", db=temp_db)
    with pytest.raises(P.InvalidTransition):
        P.transition_payout(record.payout_id, "initiated", db=temp_db)  # cannot go back


def test_terminal_states_accept_nothing(temp_db):
    record = _create(temp_db)
    P.transition_payout(record.payout_id, "rejected", db=temp_db)
    assert P.suggested_next_status("rejected", "low") is None
    with pytest.raises(P.InvalidTransition):
        P.transition_payout(record.payout_id, "screening", db=temp_db)


@pytest.mark.parametrize("band", ["high", "critical"])
def test_high_risk_payouts_route_to_human_review(temp_db, band):
    record = _create(temp_db, band=band)
    assert P.advance_payout(record.payout_id, db=temp_db) == "screening"
    assert P.advance_payout(record.payout_id, db=temp_db) == "under_review"
    # The engine stops: a human must approve or reject.
    assert P.advance_payout(record.payout_id, db=temp_db) is None


@pytest.mark.parametrize("band", ["low", "medium"])
def test_low_risk_payouts_are_not_held(temp_db, band):
    assert P.suggested_next_status("screening", band) == "approved_for_payout"


def test_failed_payout_may_be_retried(temp_db):
    record = _create(temp_db)
    for status in ("screening", "approved_for_payout", "in_transit"):
        P.transition_payout(record.payout_id, status, db=temp_db)
    P.transition_payout(record.payout_id, "failed", db=temp_db)
    assert P.transition_payout(record.payout_id, "in_transit", db=temp_db) == "in_transit"


def test_every_transition_is_audited(temp_db):
    record = _create(temp_db)
    P.advance_payout(record.payout_id, db=temp_db)
    P.advance_payout(record.payout_id, db=temp_db)
    trail = storage.audit_trail("payout", record.payout_id, path=temp_db)
    assert sorted(trail["action"]) == ["create", "transition", "transition"]
    events = P.payout_events(record.payout_id, db=temp_db)
    assert len(events) == 3


def test_unknown_payout_raises(temp_db):
    with pytest.raises(KeyError):
        P.transition_payout("PO-does-not-exist", "screening", db=temp_db)
