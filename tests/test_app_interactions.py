"""Interaction tests: the primary workflow on each page actually works."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _state(at: AppTest, key: str, default=None):
    """Read session state without raising when the key is absent."""
    try:
        return at.session_state[key]
    except Exception:
        return default


def _page(label: str, timeout: int = 300) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    at.sidebar.radio[0].set_value(label).run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_payout_can_be_created_from_the_ui():
    at = _page("💸  Payments & Payouts")
    before = _state(at, "last_payout_id")
    at.button(key="po_create").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    payout_id = _state(at, "last_payout_id")
    assert payout_id and payout_id != before

    from paybridge.payments.payouts import get_payout

    record = get_payout(payout_id)
    assert record is not None and record["status"] == "initiated"
    assert record["total_customer_cost"] > 0


def test_changing_the_rail_reprices_the_payout():
    at = _page("💸  Payments & Payouts")
    at.number_input(key="ue_amt").set_value(25_000.0).run()
    at.selectbox(key="ue_rail").set_value("SWIFT").run()
    assert not at.exception, [str(e) for e in at.exception]
    at.selectbox(key="ue_rail").set_value("USDC").run()
    assert not at.exception, [str(e) for e in at.exception]


def test_rule_weights_can_be_tuned_live():
    at = _page("🧠  FraudLens AI")
    at.slider(key="rl_thr").set_value(0.20).run()
    assert not at.exception, [str(e) for e in at.exception]


def test_analyst_can_record_an_investigation_decision():
    at = _page("🔎  Investigation Workbench")
    case_id = at.selectbox(key="wb_case").value
    at.selectbox(key="wb_analyst").set_value("analyst.chen").run()
    at.selectbox(key="wb_decision").set_value("escalate").run()
    at.text_area(key="wb_note").set_value("Escalated during an interaction test.").run()
    at.button(key="wb_record").click().run()
    assert not at.exception, [str(e) for e in at.exception]

    from paybridge.fraud import investigations as INV

    case = INV.get_case(case_id)
    assert case is not None and case["decision"] == "escalate"
    assert any(INV.case_audit(case_id)["action"] == "decision")


def test_compliance_reviewer_can_record_a_decision():
    at = _page("🛡️  Compliance Operations")
    # Narrow to cases still in review: terminal cases render no decision form,
    # which is itself the behaviour asserted by test_compliance.py.
    at.multiselect(key="cm_status").set_value(["review"]).run()
    case_id = at.selectbox(key="cm_case").value
    at.text_input(key="cm_note").set_value("Reviewed during an interaction test.").run()
    at.button(key="cm_submit").click().run()
    assert not at.exception, [str(e) for e in at.exception]

    from paybridge.compliance.onboarding import decisions_for

    assert len(decisions_for(case_id)) >= 1


def test_copilot_answers_without_an_api_key():
    at = _page("🤖  AI Copilot")
    at.text_input(key="cp_input").set_value("What are the highest-risk transactions?").run()
    at.button(key="cp_ask").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    rendered = " ".join(m.value for m in at.markdown)
    assert "exposure" in rendered.lower() or "highest-scoring" in rendered.lower()


def test_evaluation_threshold_slider_recomputes():
    at = _page("📈  Model Evaluation")
    slider = at.slider(key="ev_thr")
    slider.set_value(float(slider.value) * 1.05).run()
    assert not at.exception, [str(e) for e in at.exception]


def test_milestone_update_persists_from_the_ui():
    at = _page("🗂️  Program Command Center")
    at.selectbox(key="pm_ms").set_value("M-052").run()
    at.number_input(key="pm_pct").set_value(55).run()
    at.text_input(key="pm_notes").set_value("Updated during an interaction test.").run()
    at.button(key="pm_save").click().run()
    assert not at.exception, [str(e) for e in at.exception]

    from paybridge.program.program import get_milestones

    row = get_milestones().set_index("milestone_id").loc["M-052"]
    assert row["pct_complete"] == 55


def test_shadow_mode_runs_from_the_ui():
    at = _page("🗂️  Program Command Center", timeout=600)
    at.button(key="sm_run").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    result = _state(at, "shadow_result")
    assert result is not None and result.transactions_replayed > 0
    assert result.measured["decisions_changed"] == 0


def test_release_gates_can_be_assessed_from_the_ui():
    at = _page("🗂️  Program Command Center", timeout=600)
    at.button(key="rr_assess").click().run()
    assert not at.exception, [str(e) for e in at.exception]

    from paybridge.program.program import get_gates, go_no_go

    gates = get_gates()
    assert (gates["status"] != "not_assessed").any()
    assert go_no_go(gates)["decision"] in {"go", "conditional_go", "no_go"}


def test_experiment_calculator_responds():
    at = _page("📐  Product Analytics")
    at.slider(key="ex_mde").set_value(0.25).run()
    assert not at.exception, [str(e) for e in at.exception]
