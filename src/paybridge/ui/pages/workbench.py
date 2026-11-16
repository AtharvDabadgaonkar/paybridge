"""Module 5 - Fraud Investigation Workbench."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.data.loader import load_businesses
from paybridge.fraud import investigations as I
from paybridge.fraud.scoring import assess_transaction
from paybridge.ui import components as C
from paybridge.ui import theme
from paybridge.ui.services import background_sample, model_bundle, scored_transactions

ANALYSTS = ("analyst.rivera", "analyst.okafor", "analyst.chen", "analyst.dubois", "analyst.patel")


def _queue_kpis() -> None:
    k = I.investigation_kpis()
    C.kpi_cards(
        [
            {"label": "Cases in queue", "value": C.fmt_int(k["total_cases"]),
             "sub": f"{C.fmt_int(k['pending_decisions'])} awaiting a decision", "tone": "navy"},
            {"label": "Decision rate", "value": C.fmt_pct(k["decision_rate_pct"]),
             "sub": f"{C.fmt_int(k['closed_cases'])} closed · {C.fmt_int(k['escalated'])} escalated",
             "tone": "teal"},
            {"label": "Exposure under review", "value": C.fmt_usd(k["exposure_usd"], compact=True),
             "sub": "Total amount across open cases", "tone": "warning"},
            {"label": "Investigation notes", "value": C.fmt_int(k["notes_recorded"]),
             "sub": "Recorded in this demo session", "tone": "teal"},
        ],
        columns=4,
    )


def _tab_queue(scored: pd.DataFrame) -> None:
    _queue_kpis()
    cases = I.list_cases()
    if cases.empty:
        C.empty_state("The investigation queue is empty. Open a case from the FraudLens page.")
        return

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with f1:
        band = st.multiselect("Risk band", ["critical", "high", "medium", "low"], key="wb_band")
    with f2:
        status = st.multiselect("Case status", list(I.CASE_STATUSES), key="wb_status")
    with f3:
        decision = st.multiselect("Decision", list(I.DECISIONS), key="wb_dec")
    with f4:
        min_amount = st.number_input("Minimum amount (USD)", 0.0, 1_000_000.0, 0.0, 1000.0, key="wb_amt")

    view = cases
    if band:
        view = view[view["risk_band"].isin(band)]
    if status:
        view = view[view["status"].isin(status)]
    if decision:
        view = view[view["decision"].isin(decision)]
    if min_amount:
        view = view[view["amount_usd"] >= min_amount]

    if view.empty:
        C.empty_state("No cases match these filters.")
        return

    C.dataframe(
        view[["case_id", "transaction_id", "business_id", "amount_usd", "risk_score", "risk_band",
              "status", "decision", "assigned_to", "updated_at"]].rename(
            columns={"case_id": "Case", "transaction_id": "Transaction", "business_id": "Business",
                     "amount_usd": "Amount (USD)", "risk_score": "Risk score", "risk_band": "Band",
                     "status": "Status", "decision": "Decision", "assigned_to": "Assigned",
                     "updated_at": "Updated"}
        ),
        column_config={
            "Amount (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Risk score": st.column_config.NumberColumn(format="%.3f"),
        },
        height=330,
    )

    left, right = st.columns(2)
    with left:
        counts = view["risk_band"].value_counts().reset_index()
        counts.columns = ["band", "cases"]
        fig = px.bar(counts, x="band", y="cases", color="band",
                     color_discrete_map=theme.BAND_COLORS)
        fig.update_layout(title="Queue composition by risk band", showlegend=False,
                          xaxis_title="", yaxis_title="Cases")
        C.chart(fig, 270)
    with right:
        dec = view["decision"].value_counts().reset_index()
        dec.columns = ["decision", "cases"]
        fig = px.bar(dec, x="decision", y="cases")
        fig.update_traces(marker_color=theme.TEAL_500)
        fig.update_layout(title="Decisions recorded", xaxis_title="", yaxis_title="Cases")
        C.chart(fig, 270)


def _tab_case(scored: pd.DataFrame, businesses: pd.DataFrame) -> None:
    cases = I.list_cases()
    if cases.empty:
        C.empty_state("No investigation cases yet.")
        return

    labels = {
        row.case_id: f"{row.case_id} · {row.risk_band.upper()} · ${row.amount_usd:,.0f} · {row.decision}"
        for row in cases.itertuples()
    }
    case_id = st.selectbox("Case", list(labels), format_func=lambda c: labels[c], key="wb_case")
    case = I.get_case(case_id)
    if case is None:
        return

    txn_rows = scored[scored["transaction_id"] == case["transaction_id"]]
    if txn_rows.empty:
        C.notice("The underlying transaction is not present in the current dataset.", "warning", "⚠️")
        return
    txn = txn_rows.iloc[0]

    bundle = model_bundle()
    assessment = assess_transaction(txn, bundle, with_explanation=True, background=background_sample(300))

    tone = {"low": "success", "medium": "teal", "high": "warning", "critical": "danger"}[assessment.combined_band]
    C.kpi_cards(
        [
            {"label": "Case", "value": str(case["case_id"]),
             "sub": f"Status {case['status']} · decision {case['decision']}", "tone": "navy"},
            {"label": "Combined risk", "value": f"{assessment.combined_score:.3f}",
             "sub": f"Band {assessment.combined_band.upper()}", "tone": tone},
            {"label": "Amount at risk", "value": C.fmt_usd(float(txn["amount_usd"]), 2),
             "sub": f"{txn['corridor']} via {txn['payment_rail']}", "tone": "warning"},
            {"label": "Model probability",
             "value": f"{assessment.ml_probability:.3f}" if assessment.ml_probability is not None else "n/a",
             "sub": f"Threshold {assessment.ml_threshold:.3f}" if assessment.ml_threshold else "Rules only",
             "tone": "teal"},
        ],
        columns=4,
    )
    if I.requires_human_review(str(assessment.combined_band)):
        C.notice(
            "<b>Human decision required.</b> FraudLens recommends; it cannot approve, reject or "
            "escalate this case. Every decision below is attributed to a named analyst and audited.",
            "danger", "👤",
        )

    detail, signals = st.columns([1, 1.2])
    with detail:
        st.markdown("**Transaction**")
        C.key_values(
            [
                ("Transaction", txn["transaction_id"]),
                ("Initiated", str(txn["transaction_timestamp"])[:19]),
                ("Business", f"{txn['business_id']}"),
                ("Beneficiary", f"{txn['beneficiary_id']} ({float(txn['beneficiary_age_days']):.1f} days old)"),
                ("Route", f"{txn['corridor']} · {txn['payment_rail']} · {txn['currency']}"),
                ("Amount", C.fmt_usd(float(txn["amount_usd"]), 2)),
                ("Customer cost", C.fmt_usd(float(txn["total_cost_usd"]), 2)),
                ("Status", str(txn["transaction_status"]).replace("_", " ").title()),
                ("Compliance", str(txn["compliance_status"]).replace("_", " ").title()),
            ]
        )
        st.markdown("**Behavioural signals at payment time**")
        C.key_values(
            [
                ("24h velocity", int(txn["transaction_velocity"])),
                ("Days since previous payout", f"{float(txn['days_since_last_txn']):.2f}"),
                ("30-day payout count", int(txn["business_txn_count_30d"])),
                ("30-day average amount", C.fmt_usd(float(txn["business_avg_amount_30d"]), 2)),
                ("Amount vs own average", f"{float(txn['amount_vs_business_mean_ratio']):.2f}×"),
                ("Failures in last 7 days", int(txn["recent_failed_count"])),
                ("Account age", f"{int(txn['account_age_days'])} days"),
                ("Payout details changed", f"{float(txn['account_details_changed_days']):.0f} days ago"),
            ]
        )

    with signals:
        st.markdown("**Triggered rules**")
        if assessment.rules.triggered:
            for t in assessment.rules.triggered:
                st.markdown(f"{C.badge(t.code.split('_')[0], 'warning')} **{t.name}** — {t.description}",
                            unsafe_allow_html=True)
        else:
            st.caption("No deterministic rule fired.")

        exp = assessment.explanation
        if exp is not None:
            frame = exp.as_frame(7)
            fig = px.bar(frame.sort_values("contribution"), x="contribution", y="feature", orientation="h",
                         color=frame.sort_values("contribution")["contribution"] > 0,
                         color_discrete_map={True: theme.DANGER, False: theme.SUCCESS})
            fig.update_layout(title=f"Model drivers ({exp.method})", showlegend=False,
                              xaxis_title="Contribution", yaxis_title="")
            C.chart(fig, 290)
            if not exp.is_shap:
                st.caption(exp.note)

        st.markdown("**Suggested questions**")
        for q in assessment.investigation_questions()[:5]:
            st.markdown(f"- {q}")

    st.markdown("**Customer payout history**")
    history = scored[scored["business_id"] == txn["business_id"]].sort_values("transaction_timestamp")
    if len(history) > 1:
        fig = go.Figure()
        fig.add_scatter(
            x=history["transaction_timestamp"], y=history["amount_usd"], mode="markers",
            marker=dict(
                size=7,
                color=history["combined_score"],
                colorscale=[[0, theme.TEAL_200], [1, theme.DANGER]],
                showscale=True,
                colorbar=dict(title="Risk"),
            ),
            name="Payouts",
        )
        fig.add_scatter(
            x=[txn["transaction_timestamp"]], y=[txn["amount_usd"]], mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=theme.NAVY_800, width=3)),
            name="This payout",
        )
        fig.update_layout(title=f"Payout history for {txn['business_id']}", yaxis_title="Amount (USD)",
                          xaxis_title="")
        C.chart(fig, 300)
    C.dataframe(
        history.tail(12)[["transaction_id", "transaction_timestamp", "amount_usd", "corridor",
                          "payment_rail", "transaction_status", "combined_score", "combined_band"]]
        .sort_values("transaction_timestamp", ascending=False)
        .rename(columns={"transaction_id": "Transaction", "transaction_timestamp": "When",
                         "amount_usd": "Amount", "corridor": "Corridor", "payment_rail": "Rail",
                         "transaction_status": "Status", "combined_score": "Risk", "combined_band": "Band"}),
        column_config={"Amount": st.column_config.NumberColumn(format="$%.2f"),
                       "Risk": st.column_config.NumberColumn(format="%.3f")},
        height=220,
    )

    st.markdown("**Analyst decision**")
    d1, d2, d3 = st.columns([1.1, 1.1, 2.4])
    with d1:
        analyst = st.selectbox("Analyst", ANALYSTS, key="wb_analyst")
    with d2:
        decision = st.selectbox("Decision", ["approve", "reject", "escalate"], key="wb_decision")
    with d3:
        note = st.text_area("Investigation note", key="wb_note", height=80,
                            placeholder="Evidence reviewed, contacts made, rationale for the decision...")
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("Record decision", type="primary", key="wb_record"):
            try:
                status = I.record_decision(case_id, decision, analyst, note)
                st.success(f"Decision `{decision}` recorded by {analyst}. Case is now `{status}`.")
                st.rerun()
            except (ValueError, I.AutonomousDecisionBlocked) as exc:
                st.error(str(exc))
    with b2:
        if st.button("Add note only", key="wb_note_only"):
            try:
                I.add_note(case_id, analyst, note)
                st.success("Note recorded.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    ncol, acol = st.columns(2)
    with ncol:
        st.markdown("**Investigation notes**")
        notes = I.case_notes(case_id)
        if notes.empty:
            st.caption("No notes recorded on this case yet.")
        else:
            C.dataframe(
                notes[["created_at", "author", "note"]].rename(
                    columns={"created_at": "When", "author": "Author", "note": "Note"}
                ),
                height=220,
            )
            st.caption(
                "Notes are treated as untrusted free text: the AI copilot never follows instructions "
                "contained in them."
            )
    with acol:
        st.markdown("**Case audit trail**")
        trail = I.case_audit(case_id)
        if trail.empty:
            st.caption("No audit records for this case yet.")
        else:
            C.dataframe(
                trail[["occurred_at", "actor", "action", "detail"]].rename(
                    columns={"occurred_at": "When", "actor": "Actor", "action": "Action", "detail": "Detail"}
                ),
                height=220,
            )


def _tab_history() -> None:
    st.markdown("##### All investigation activity")
    from paybridge.db import storage

    trail = storage.audit_trail("investigation", limit=400)
    if trail.empty:
        C.empty_state("No investigation activity recorded yet.")
        return
    C.dataframe(
        trail[["occurred_at", "actor", "entity_id", "action", "detail"]].rename(
            columns={"occurred_at": "When", "actor": "Actor", "entity_id": "Case",
                     "action": "Action", "detail": "Detail"}
        ),
        height=460,
    )


def render() -> None:
    C.page_header(
        "Investigation Workbench",
        "A ranked alert queue, full case context, model explanations and auditable analyst decisions. "
        "High-risk cases always require a human.",
        eyebrow="Module 5 · Fraud operations",
    )
    C.synthetic_notice()

    scored = scored_transactions()
    businesses = load_businesses()
    tabs = st.tabs(["Alert queue", "Case detail", "Activity history"])
    with tabs[0]:
        _tab_queue(scored)
    with tabs[1]:
        _tab_case(scored, businesses)
    with tabs[2]:
        _tab_history()
    C.footer()
