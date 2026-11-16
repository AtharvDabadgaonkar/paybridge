"""Program management: milestones, RAID, gates, go/no-go and shadow mode."""
from __future__ import annotations

import pandas as pd
import pytest

from paybridge.db import seeds, storage
from paybridge.program import definitions as pdefs
from paybridge.program import program as P
from paybridge.program.shadow_mode import ACCEPTANCE_CRITERIA, run_shadow_mode


@pytest.fixture()
def seeded_db(temp_db):
    seeds.seed_all(temp_db)
    return temp_db


def test_seed_covers_every_workstream(seeded_db):
    milestones = P.get_milestones(seeded_db)
    assert set(milestones["workstream"]) == set(pdefs.WORKSTREAMS)
    assert len(milestones) == len(pdefs.MILESTONES)


def test_seeding_is_idempotent_and_preserves_edits(seeded_db):
    P.update_milestone("M-001", "at_risk", 42, "edited", db=seeded_db)
    seeds.seed_program(seeded_db)
    row = P.get_milestones(seeded_db).set_index("milestone_id").loc["M-001"]
    assert row["status"] == "at_risk" and row["pct_complete"] == 42


def test_milestone_update_persists_and_audits(seeded_db):
    P.update_milestone("M-010", "in_progress", 65, "reopened", db=seeded_db)
    row = P.get_milestones(seeded_db).set_index("milestone_id").loc["M-010"]
    assert row["status"] == "in_progress" and row["pct_complete"] == 65
    trail = storage.audit_trail("milestone", "M-010", path=seeded_db)
    assert len(trail) == 1


def test_milestone_percentage_is_clamped(seeded_db):
    P.update_milestone("M-011", "complete", 180, db=seeded_db)
    assert P.get_milestones(seeded_db).set_index("milestone_id").loc["M-011"]["pct_complete"] == 100


def test_invalid_milestone_status_raises(seeded_db):
    with pytest.raises(ValueError):
        P.update_milestone("M-001", "finished-ish", 100, db=seeded_db)


def test_workstream_summary_matches_the_rows(seeded_db):
    milestones = P.get_milestones(seeded_db)
    summary = P.workstream_summary(milestones)
    assert summary["milestones"].sum() == len(milestones)
    for row in summary.itertuples():
        subset = milestones[milestones["workstream"] == row.workstream]
        assert row.complete == int((subset["status"] == "complete").sum())


def test_raid_update_and_creation(seeded_db):
    P.update_raid("R-001", "mitigating", "medium", "New mitigation", db=seeded_db)
    item = P.get_raid(db=seeded_db).set_index("item_id").loc["R-001"]
    assert item["status"] == "mitigating" and item["severity"] == "medium"

    new_id = P.add_raid("issue", "Test issue", "Description", "System Testing", "high", "medium",
                        "QA", "Mitigate it", db=seeded_db)
    assert new_id.startswith("I-")
    assert new_id in set(P.get_raid(db=seeded_db)["item_id"])


def test_invalid_raid_inputs_raise(seeded_db):
    with pytest.raises(ValueError):
        P.update_raid("R-001", "not-a-status", db=seeded_db)
    with pytest.raises(KeyError):
        P.update_raid("R-999", "open", db=seeded_db)
    with pytest.raises(ValueError):
        P.add_raid("gossip", "t", "d", "System Testing", "high", "low", "QA", "m", db=seeded_db)


def test_blocking_issues_detect_high_severity_open_issues(seeded_db):
    before = len(P.open_blocking_issues(db=seeded_db))
    P.add_raid("issue", "Critical blocker", "d", "Pilot Launch", "critical", "high", "TPM", "m",
               db=seeded_db)
    assert len(P.open_blocking_issues(db=seeded_db)) == before + 1


def test_dependency_update(seeded_db):
    P.update_dependency("DEP-006", "satisfied", db=seeded_db)
    deps = P.get_dependencies(seeded_db).set_index("dependency_id")
    assert deps.loc["DEP-006"]["status"] == "satisfied"
    with pytest.raises(ValueError):
        P.update_dependency("DEP-006", "probably-fine", db=seeded_db)


def test_gates_start_unassessed(seeded_db):
    gates = P.get_gates(seeded_db)
    assert len(gates) == len(pdefs.RELEASE_GATES)
    assert (gates["status"] == "not_assessed").all()


def test_gate_assessment_without_evidence_stays_unassessed(seeded_db):
    gates = P.assess_gates(None, None, None, db=seeded_db)
    model_gates = gates[gates["gate_id"].isin(["G-01", "G-02", "G-03", "G-04"])]
    assert (model_gates["status"] == "not_assessed").all()
    # The human-review policy gate is verifiable from the code itself.
    assert gates.set_index("gate_id").loc["G-07"]["status"] == "pass"


def test_gate_assessment_uses_measured_metrics(seeded_db):
    metrics = {"recall": 0.75, "precision": 0.40, "false_positive_rate": 0.01}
    gates = P.assess_gates(metrics, None, None, db=seeded_db).set_index("gate_id")
    assert gates.loc["G-01"]["status"] == "pass"
    assert "0.750" in gates.loc["G-01"]["evidence"]

    failing = {"recall": 0.10, "precision": 0.02, "false_positive_rate": 0.20}
    gates = P.assess_gates(failing, None, None, db=seeded_db).set_index("gate_id")
    assert gates.loc["G-01"]["status"] == "fail"
    assert gates.loc["G-03"]["status"] == "fail"


def test_go_no_go_requires_every_gate():
    gates = pd.DataFrame({"gate_id": ["G-1", "G-2"], "status": ["pass", "pass"]})
    assert P.go_no_go(gates)["decision"] == "go"
    gates.loc[1, "status"] = "not_assessed"
    assert P.go_no_go(gates)["decision"] == "conditional_go"
    gates.loc[1, "status"] = "fail"
    assert P.go_no_go(gates)["decision"] == "no_go"
    assert P.go_no_go(pd.DataFrame())["decision"] == "no_go"


def test_qa_summary_never_invents_a_result(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "QA_RESULTS_FILE", tmp_path / "missing.json")
    summary = P.qa_summary()
    assert summary["available"] is False and summary["reason"]


def test_shadow_mode_changes_no_decision(transactions):
    from paybridge.fraud.model import ensure_model

    result = run_shadow_mode(transactions, ensure_model(), sample_size=800, latency_sample=20)
    assert result.measured["decisions_changed"] == 0
    assert result.measured["workflow_errors"] == 0
    assert result.transactions_replayed == 800


def test_shadow_mode_measures_every_criterion(transactions):
    from paybridge.fraud.model import ensure_model

    result = run_shadow_mode(transactions, ensure_model(), sample_size=600, latency_sample=15)
    assert len(result.criteria) == len(ACCEPTANCE_CRITERIA)
    assert all(c["measured"] is not None for c in result.criteria)
    assert result.criteria_met <= len(result.criteria)
    assert result.measured["audit_completeness_pct"] == 100.0
    assert result.measured["latency_p95_ms"] > 0


def test_shadow_mode_works_without_a_model(transactions):
    result = run_shadow_mode(transactions, None, sample_size=400, latency_sample=10)
    assert result.transactions_replayed == 400
    assert result.measured["decisions_changed"] == 0
