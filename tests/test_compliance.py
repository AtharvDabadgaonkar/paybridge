"""Compliance workflow: onboarding transitions, screening and KPIs."""
from __future__ import annotations

import pandas as pd
import pytest

from paybridge.compliance import onboarding as O
from paybridge.compliance import screening as S
from paybridge.db import storage


def test_onboarding_happy_path_transitions():
    chain = [(O.DRAFT, O.SUBMITTED), (O.SUBMITTED, O.SCREENING), (O.SCREENING, O.REVIEW),
             (O.REVIEW, O.APPROVED)]
    for src, dst in chain:
        assert O.can_transition(src, dst)


def test_invalid_onboarding_transitions_are_blocked():
    assert not O.can_transition(O.DRAFT, O.APPROVED)
    assert not O.can_transition(O.APPROVED, O.REVIEW)
    assert not O.can_transition(O.REJECTED, O.SUBMITTED)
    assert not O.can_transition(O.SCREENING, O.DRAFT)


def test_more_info_can_return_to_submitted():
    assert O.can_transition(O.REVIEW, O.MORE_INFO)
    assert O.can_transition(O.MORE_INFO, O.SUBMITTED)


def test_record_decision_rejects_invalid_transition(temp_db):
    with pytest.raises(O.InvalidOnboardingTransition):
        O.record_decision("KYB-0001", "BIZ-0001", O.APPROVED, O.DRAFT, "analyst.chen", db=temp_db)


def test_record_decision_persists_and_audits(temp_db):
    O.record_decision("KYB-0001", "BIZ-0001", O.REVIEW, O.APPROVED, "analyst.chen", "ok", db=temp_db)
    history = O.decisions_for("KYB-0001", db=temp_db)
    assert len(history) == 1 and history.iloc[0]["to_status"] == O.APPROVED
    trail = storage.audit_trail("compliance_case", "KYB-0001", path=temp_db)
    assert len(trail) == 1 and trail.iloc[0]["action"] == "status_change"


def test_overrides_replace_dataset_status(temp_db):
    cases = pd.DataFrame(
        [
            {"case_id": "KYB-0001", "business_id": "BIZ-0001", "onboarding_status": "review",
             "duration_minutes": 5.0, "is_straightforward": True, "manual_review_required": True,
             "screening_hits": 0, "legal_name": "X", "risk_classification": "low"},
        ]
    )
    O.record_decision("KYB-0001", "BIZ-0001", O.REVIEW, O.APPROVED, "analyst.chen", db=temp_db)
    merged = O.apply_overrides(cases, db=temp_db)
    assert merged.iloc[0]["current_status"] == O.APPROVED
    assert bool(merged.iloc[0]["decided_in_app"]) is True


def test_compliance_kpis_are_computed_from_the_cases():
    from paybridge.data.loader import load_compliance_cases

    cases = O.apply_overrides(load_compliance_cases())
    k = O.compliance_kpis(cases)
    assert k["total_cases"] == len(cases)
    decided = cases["current_status"].isin(O.DECIDED).sum()
    assert k["completion_rate_pct"] == pytest.approx(decided / len(cases) * 100)
    assert 0 <= k["approval_rate_pct"] <= 100
    assert k["target_minutes"] == O.ONBOARDING_TARGET_MINUTES


def test_compliance_kpis_handle_an_empty_frame():
    empty = pd.DataFrame(
        columns=["case_id", "onboarding_status", "duration_minutes", "is_straightforward",
                 "manual_review_required", "screening_hits"]
    )
    assert O.compliance_kpis(empty)["total_cases"] == 0


def test_screening_normalisation_ignores_legal_suffixes():
    assert S.normalise("Northwind Logistics Inc.") == "northwind logistics"
    assert S.similarity("Northwind Logistics", "Northwind Logistics LLC") > 0.95


def test_screening_finds_an_exact_match():
    watchlist = pd.DataFrame(
        [{"entry_id": "WL-1", "entity_name": "Harborline Trading Co", "entity_type": "entity",
          "list_source": "SYNTH-SANCTIONS-A", "country": "MX", "is_fictional": True}]
    )
    hits = S.screen_entity("Harborline Trading", watchlist)
    assert len(hits) == 1 and hits[0].match_score > 0.9 and hits[0].is_fictional


def test_screening_ignores_unrelated_names():
    watchlist = pd.DataFrame(
        [{"entry_id": "WL-1", "entity_name": "Harborline Trading Co", "entity_type": "entity",
          "list_source": "SYNTH-SANCTIONS-A", "country": "MX", "is_fictional": True}]
    )
    assert S.screen_entity("Pinebrook Medical Supplies", watchlist) == []


def test_screening_handles_an_empty_list_and_empty_name():
    assert S.screen_entity("Anything", pd.DataFrame()) == []
    assert S.screen_entity("", pd.DataFrame([{"entry_id": "1", "entity_name": "x", "entity_type": "e",
                                              "list_source": "s", "country": "MX"}])) == []


@pytest.mark.parametrize(
    "hits,tier,docs,ubo,expected",
    [
        (0, "mid", True, True, "low"),
        (1, "mid", True, True, "medium"),
        (2, "mid", True, True, "high"),
        (0, "enterprise", False, True, "high"),
        (0, "small", True, False, "medium"),
    ],
)
def test_onboarding_risk_classification(hits, tier, docs, ubo, expected):
    classification, reasons = S.classify_onboarding_risk(hits, tier, docs, ubo)
    assert classification == expected
    assert reasons


def test_manual_review_policy():
    assert S.requires_manual_review("low", 1) is True       # any hit needs a human
    assert S.requires_manual_review("high", 0) is True
    assert S.requires_manual_review("low", 0) is False
