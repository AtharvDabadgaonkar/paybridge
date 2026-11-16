"""Module 8 - Technical Program Management Command Center."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.data.loader import load_transactions
from paybridge.fraud import evaluation as E
from paybridge.program import definitions as pdefs
from paybridge.program import program as P
from paybridge.program.shadow_mode import run_shadow_mode
from paybridge.ui import components as C
from paybridge.ui import theme
from paybridge.ui.services import model_bundle

STATUS_COLORS = {
    "complete": theme.SUCCESS, "in_progress": theme.INFO, "at_risk": theme.WARNING,
    "blocked": theme.DANGER, "not_started": theme.NAVY_200,
}
GATE_TONE = {"pass": "success", "fail": "danger", "not_assessed": "neutral", "waived": "warning"}


def _tab_roadmap() -> None:
    milestones = P.get_milestones()
    summary = P.workstream_summary(milestones)

    C.kpi_cards(
        [
            {"label": "Milestones", "value": C.fmt_int(len(milestones)),
             "sub": f"{int((milestones['status'] == 'complete').sum())} complete", "tone": "navy"},
            {"label": "Programme completion",
             "value": C.fmt_pct((milestones["status"] == "complete").mean() * 100),
             "sub": f"Average progress {milestones['pct_complete'].mean():.0f}%", "tone": "teal"},
            {"label": "In progress", "value": C.fmt_int((milestones["status"] == "in_progress").sum()),
             "sub": "Active this period", "tone": "teal"},
            {"label": "At risk or blocked",
             "value": C.fmt_int(milestones["status"].isin(["at_risk", "blocked"]).sum()),
             "sub": "Require TPM attention",
             "tone": "danger" if milestones["status"].isin(["at_risk", "blocked"]).any() else "success"},
        ],
        columns=4,
    )

    timeline = milestones.dropna(subset=["start_date", "due_date"]).copy()
    timeline["start"] = pd.to_datetime(timeline["start_date"])
    timeline["finish"] = pd.to_datetime(timeline["due_date"])
    fig = px.timeline(
        timeline, x_start="start", x_end="finish", y="workstream", color="status",
        hover_name="milestone", color_discrete_map=STATUS_COLORS,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Programme roadmap by workstream", xaxis_title="", yaxis_title="")
    C.chart(fig, 420)

    C.section("Workstream progress")
    C.dataframe(
        summary.rename(columns={
            "workstream": "Workstream", "milestones": "Milestones", "complete": "Complete",
            "in_progress": "In progress", "at_risk": "At risk", "avg_pct_complete": "Avg % complete",
            "completion_pct": "Completion %",
        }),
        column_config={
            "Avg % complete": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Completion %": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )

    C.section("Update a milestone")
    m1, m2, m3 = st.columns([2.2, 1, 1])
    with m1:
        labels = {r.milestone_id: f"{r.milestone_id} · {r.milestone}" for r in milestones.itertuples()}
        milestone_id = st.selectbox("Milestone", list(labels), format_func=lambda m: labels[m], key="pm_ms")
    current = milestones[milestones["milestone_id"] == milestone_id].iloc[0]
    with m2:
        status = st.selectbox("Status", list(pdefs.MILESTONE_STATUSES),
                              index=list(pdefs.MILESTONE_STATUSES).index(current["status"]), key="pm_status")
    with m3:
        pct = st.number_input("% complete", 0, 100, int(current["pct_complete"]), 5, key="pm_pct")
    notes = st.text_input("Notes", value=str(current["notes"] or ""), key="pm_notes")
    if st.button("Save milestone", type="primary", key="pm_save"):
        P.update_milestone(milestone_id, status, int(pct), notes)
        st.success(f"{milestone_id} updated to `{status}` at {pct}%.")
        st.rerun()

    C.dataframe(
        milestones[["milestone_id", "workstream", "milestone", "owner_role", "due_date",
                    "status", "pct_complete", "notes"]].rename(
            columns={"milestone_id": "ID", "workstream": "Workstream", "milestone": "Milestone",
                     "owner_role": "Owner", "due_date": "Due", "status": "Status",
                     "pct_complete": "% complete", "notes": "Notes"}
        ),
        height=360,
    )


def _tab_raci() -> None:
    C.section("Responsibility matrix", pdefs.RACI_LEGEND)
    C.dataframe(pd.DataFrame(pdefs.RESPONSIBILITY_MATRIX).rename(columns={"activity": "Activity"}),
                height=400)

    C.section("Dependency tracker")
    deps = P.get_dependencies()
    C.dataframe(
        deps[["dependency_id", "description", "upstream", "downstream", "owner_role", "due_date", "status"]]
        .rename(columns={"dependency_id": "ID", "description": "Dependency", "upstream": "Upstream",
                         "downstream": "Downstream", "owner_role": "Owner", "due_date": "Due",
                         "status": "Status"}),
        height=300,
    )
    d1, d2 = st.columns([2, 1])
    with d1:
        dep_id = st.selectbox("Dependency", deps["dependency_id"].tolist(), key="pd_dep")
    with d2:
        dep_status = st.selectbox("Status", list(pdefs.DEPENDENCY_STATUSES), key="pd_status")
    if st.button("Update dependency", key="pd_save"):
        P.update_dependency(dep_id, dep_status)
        st.success(f"{dep_id} set to `{dep_status}`.")
        st.rerun()

    counts = deps["status"].value_counts().reset_index()
    counts.columns = ["status", "dependencies"]
    fig = px.bar(counts, x="status", y="dependencies")
    fig.update_traces(marker_color=theme.TEAL_500)
    fig.update_layout(title="Dependency status", xaxis_title="", yaxis_title="")
    C.chart(fig, 260)


def _tab_raid() -> None:
    raid = P.get_raid()
    C.kpi_cards(
        [
            {"label": "RAID items", "value": C.fmt_int(len(raid)), "tone": "navy"},
            {"label": "Open risks",
             "value": C.fmt_int(((raid["item_type"] == "risk") & (raid["status"] != "closed")).sum()),
             "tone": "warning"},
            {"label": "Open issues",
             "value": C.fmt_int(((raid["item_type"] == "issue") & (raid["status"] != "closed")).sum()),
             "tone": "danger"},
            {"label": "Blocking (high/critical issues)",
             "value": C.fmt_int(len(P.open_blocking_issues())), "tone": "danger"},
        ],
        columns=4,
    )

    type_filter = st.multiselect("Type", list(pdefs.RAID_TYPES), key="raid_type")
    view = raid[raid["item_type"].isin(type_filter)] if type_filter else raid
    C.dataframe(
        view[["item_id", "item_type", "title", "workstream", "severity", "likelihood",
              "owner_role", "status", "mitigation"]].rename(
            columns={"item_id": "ID", "item_type": "Type", "title": "Title", "workstream": "Workstream",
                     "severity": "Severity", "likelihood": "Likelihood", "owner_role": "Owner",
                     "status": "Status", "mitigation": "Mitigation"}
        ),
        height=340,
    )

    C.section("Update an item")
    u1, u2, u3 = st.columns([1.6, 1, 1])
    with u1:
        item_id = st.selectbox("Item", raid["item_id"].tolist(), key="raid_item")
    item = raid[raid["item_id"] == item_id].iloc[0]
    with u2:
        status = st.selectbox("Status", list(pdefs.RAID_STATUSES),
                              index=list(pdefs.RAID_STATUSES).index(item["status"]), key="raid_status")
    with u3:
        severity = st.selectbox("Severity", ["low", "medium", "high", "critical"],
                                index=["low", "medium", "high", "critical"].index(str(item["severity"]))
                                if str(item["severity"]) in ["low", "medium", "high", "critical"] else 1,
                                key="raid_sev")
    mitigation = st.text_area("Mitigation", value=str(item["mitigation"] or ""), height=70, key="raid_mit")
    if st.button("Save RAID item", type="primary", key="raid_save"):
        P.update_raid(item_id, status, severity, mitigation)
        st.success(f"{item_id} updated.")
        st.rerun()

    with st.expander("Log a new RAID item"):
        n1, n2, n3 = st.columns(3)
        with n1:
            new_type = st.selectbox("Type", list(pdefs.RAID_TYPES), key="raid_ntype")
            new_ws = st.selectbox("Workstream", list(pdefs.WORKSTREAMS), key="raid_nws")
        with n2:
            new_sev = st.selectbox("Severity", ["low", "medium", "high", "critical"], key="raid_nsev")
            new_like = st.selectbox("Likelihood", ["low", "medium", "high"], key="raid_nlike")
        with n3:
            new_owner = st.text_input("Owner role", value="TPM", key="raid_nowner")
        new_title = st.text_input("Title", key="raid_ntitle")
        new_desc = st.text_area("Description", height=70, key="raid_ndesc")
        new_mit = st.text_area("Mitigation", height=70, key="raid_nmit")
        if st.button("Add item", key="raid_add"):
            if not new_title.strip():
                st.error("A title is required.")
            else:
                item_id = P.add_raid(new_type, new_title, new_desc, new_ws, new_sev, new_like,
                                     new_owner, new_mit)
                st.success(f"Logged {item_id}.")
                st.rerun()

    heat = raid[raid["item_type"] == "risk"].groupby(["severity", "likelihood"]).size().reset_index(name="risks")
    if not heat.empty:
        pivot = heat.pivot(index="severity", columns="likelihood", values="risks").fillna(0)
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale=theme.SEQUENTIAL,
                        labels=dict(x="Likelihood", y="Severity", color="Risks"))
        fig.update_layout(title="Risk heat map")
        C.chart(fig, 280)


def _tab_shadow() -> None:
    C.section(
        "Shadow-mode validation",
        "FraudLens scores historical payouts and produces recommendations without changing any "
        "original payment decision.",
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        sample = st.select_slider("Transactions to replay", [1000, 2000, 4000, 8000], value=4000,
                                  key="sm_sample")
        run = st.button("Run shadow-mode replay", type="primary", key="sm_run")
    with col2:
        C.notice(
            "Acceptance criteria below are <b>targets agreed for the shadow phase</b>. The measured "
            "column is produced by the replay you run here — a criterion that is not met is reported "
            "as not met.", "modelled", "🎯",
        )

    if run or "shadow_result" not in st.session_state:
        with st.spinner("Replaying transactions through FraudLens..."):
            defects = len(P.open_blocking_issues())
            result = run_shadow_mode(load_transactions(), model_bundle(), sample_size=int(sample),
                                     outstanding_defects=defects)
        st.session_state["shadow_result"] = result
    result = st.session_state["shadow_result"]

    C.kpi_cards(
        [
            {"label": "Transactions replayed", "value": C.fmt_int(result.transactions_replayed),
             "sub": f"{result.period_start[:10]} → {result.period_end[:10]}", "tone": "navy"},
            {"label": "Criteria met", "value": f"{result.criteria_met} / {len(result.criteria)}",
             "sub": "Measured against the shadow-phase targets",
             "tone": "success" if result.passed else "warning"},
            {"label": "Scoring latency p95", "value": f"{result.measured['latency_p95_ms']:.1f} ms",
             "sub": f"p50 {result.measured['latency_p50_ms']:.1f} ms · in process, single container",
             "tone": "teal"},
            {"label": "Decisions changed", "value": C.fmt_int(result.measured["decisions_changed"]),
             "sub": "Shadow mode must change nothing",
             "tone": "success" if result.measured["decisions_changed"] == 0 else "danger"},
        ],
        columns=4,
    )

    criteria = result.criteria_frame()
    criteria["Result"] = criteria["met"].map({True: "MET", False: "NOT MET"})
    C.dataframe(
        criteria[["label", "direction", "target", "measured", "Result", "note"]].rename(
            columns={"label": "Criterion", "direction": "Test", "target": "Target",
                     "measured": "Measured", "note": "Note"}
        ),
    )
    if not result.passed:
        unmet = criteria[~criteria["met"]]["label"].tolist()
        C.notice(
            "<b>Not all shadow-phase criteria are met:</b> " + ", ".join(unmet)
            + ". This is the measured outcome on synthetic data and is reported as a blocker rather "
            "than waived.", "danger", "🚫",
        )
    else:
        C.notice("All shadow-phase acceptance criteria were met in this run.", "success", "✅")

    for note in result.notes:
        st.caption(f"• {note}")


def _tab_readiness() -> None:
    C.section("Release readiness checklist", "Each gate is assessed against measured evidence.")
    qa = P.qa_summary()
    if qa.get("available"):
        total = qa.get("total", 0)
        failed = qa.get("failed", 0) + qa.get("errors", 0)
        C.kpi_cards(
            [
                {"label": "Recorded test run", "value": f"{qa.get('passed', 0)} / {total} passed",
                 "sub": f"Finished {qa.get('finished_at', '')[:19]}",
                 "tone": "success" if failed == 0 else "danger"},
                {"label": "Failures / errors", "value": C.fmt_int(failed),
                 "sub": f"{qa.get('skipped', 0)} skipped",
                 "tone": "success" if failed == 0 else "danger"},
            ],
            columns=2,
        )
    else:
        C.notice(
            f"<b>No test run recorded in this environment.</b> {qa.get('reason', '')} Run "
            "<code>make test</code> to populate the quality-assurance evidence; gates that depend on it "
            "stay <em>not assessed</em> rather than being assumed to pass.",
            "warning", "🧪",
        )

    if st.button("Assess all gates against current evidence", type="primary", key="rr_assess"):
        with st.spinner("Collecting measured evidence..."):
            bundle = model_bundle()
            df = load_transactions()
            test = E.test_split(df)
            results = E.compare_approaches(test, bundle)
            combined = next((r for r in results if "Combined" in r.approach), None)
            matched = E.matched_point_comparison(test, bundle)
            shadow = st.session_state.get("shadow_result")
            P.assess_gates(combined.metrics if combined else None, matched, shadow)
        st.success("Gates reassessed from measured evidence.")
        st.rerun()

    gates = P.get_gates()
    display = gates[["gate_id", "gate_area", "criterion", "target", "status", "evidence",
                     "measurement_source", "updated_at"]].rename(
        columns={"gate_id": "Gate", "gate_area": "Area", "criterion": "Criterion", "target": "Target",
                 "status": "Status", "evidence": "Measured evidence",
                 "measurement_source": "Source", "updated_at": "Assessed"}
    )
    C.dataframe(display, height=430)

    decision = P.go_no_go(gates)
    tone = {"go": "success", "conditional_go": "warning", "no_go": "danger"}[decision["decision"]]
    C.notice(
        f"<b>Go / no-go recommendation: {decision['decision'].replace('_', ' ').upper()}</b><br>"
        f"{decision['reason']}<br>"
        f"{decision['passed']} of {decision['total']} gates passing.",
        tone, {"go": "✅", "conditional_go": "⚠️", "no_go": "🚫"}[decision["decision"]],
    )
    st.markdown(
        """
**Decision framework.** A gate passes only on measured evidence from this environment. Any gate
without evidence stays *not assessed* and caps the recommendation at **conditional go** — the
programme does not treat an unmeasured criterion as satisfied. Any failed gate is an automatic
**no go** until it is remediated or formally waived with a documented rationale.
"""
    )

    counts = gates["status"].value_counts().reset_index()
    counts.columns = ["status", "gates"]
    fig = px.bar(counts, x="status", y="gates", color="status",
                 color_discrete_map={"pass": theme.SUCCESS, "fail": theme.DANGER,
                                     "not_assessed": theme.NAVY_200, "waived": theme.WARNING})
    fig.update_layout(title="Release gate status", showlegend=False, xaxis_title="", yaxis_title="")
    C.chart(fig, 260)


def render() -> None:
    C.page_header(
        "Program Command Center",
        "Workstreams, milestones, dependencies, RAID log, shadow-mode validation and an "
        "evidence-based go/no-go framework.",
        eyebrow="Module 8 · Technical program management",
    )
    C.notice(
        "Milestone statuses describe the delivery of <b>this repository</b>. Work that is not "
        "implemented is marked as planned, never as delivered.", "info", "🗂️",
    )

    tabs = st.tabs(["Roadmap & milestones", "RACI & dependencies", "RAID log", "Shadow mode",
                    "Release readiness"])
    with tabs[0]:
        _tab_roadmap()
    with tabs[1]:
        _tab_raci()
    with tabs[2]:
        _tab_raid()
    with tabs[3]:
        _tab_shadow()
    with tabs[4]:
        _tab_readiness()
    C.footer()
