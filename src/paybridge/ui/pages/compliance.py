"""Module 3 - KYC/KYB and AML Compliance Operations (simulated)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.compliance import onboarding as O
from paybridge.compliance import screening as S
from paybridge.data.loader import load_businesses, load_compliance_cases, load_watchlist
from paybridge.db import storage
from paybridge.ui import components as C
from paybridge.ui import theme

REVIEWERS = ("compliance.analyst", "analyst.rivera", "analyst.okafor", "analyst.chen")


def _tab_overview(cases: pd.DataFrame) -> None:
    k = O.compliance_kpis(cases)
    C.kpi_cards(
        [
            {"label": "Onboarding cases", "value": C.fmt_int(k["total_cases"]),
             "sub": f"{C.fmt_int(k['open_cases'])} still open", "tone": "navy"},
            {"label": "Completion rate", "value": C.fmt_pct(k["completion_rate_pct"]),
             "sub": "Reached approved or rejected", "tone": "teal"},
            {"label": "Approval rate", "value": C.fmt_pct(k["approval_rate_pct"]),
             "sub": "Approved ÷ decided cases", "tone": "success"},
            {"label": "Manual review rate", "value": C.fmt_pct(k["manual_review_rate_pct"]),
             "sub": "Cases requiring a human reviewer", "tone": "warning"},
            {"label": "Screening alert rate", "value": C.fmt_pct(k["screening_alert_rate_pct"]),
             "sub": "Cases with one or more fictional-list matches", "tone": "warning"},
            {"label": "Median onboarding duration", "value": f"{k['median_duration_minutes']:.1f} min",
             "sub": f"Mean {k['mean_duration_minutes']:.1f} min · simulated workflow time", "tone": "teal"},
            {"label": "Straightforward cases (median)", "value": f"{k['straightforward_median_minutes']:.1f} min",
             "sub": f"No screening hit, complete documents", "tone": "teal"},
            {"label": "Straightforward under 5 min", "value": C.fmt_pct(k["pct_under_target"]),
             "sub": "Against the proposed <5 minute target",
             "tone": "success" if k["pct_under_target"] >= 80 else "warning"},
        ],
        columns=4,
    )
    C.notice(
        "<strong>Proposed target, not a validated result.</strong> The sub-five-minute onboarding goal is a "
        "product target. The figures above are measured against <em>simulated</em> workflow durations in the "
        "synthetic dataset — no moderated usability testing has been run, so the target remains unvalidated.",
        "modelled",
        "🎯",
    )

    left, right = st.columns([1.25, 1])
    with left:
        order = [O.DRAFT, O.SUBMITTED, O.SCREENING, O.REVIEW, O.MORE_INFO, O.APPROVED, O.REJECTED]
        counts = cases["current_status"].value_counts().reindex(order).fillna(0).reset_index()
        counts.columns = ["status", "cases"]
        fig = px.bar(counts, x="status", y="cases", color="status",
                     color_discrete_map={s: theme.STATUS_COLORS.get(s, theme.NAVY_400) for s in order})
        fig.update_layout(title="Onboarding pipeline by status", showlegend=False,
                          xaxis_title="", yaxis_title="Cases")
        C.chart(fig, 310)
    with right:
        risk = cases["risk_classification"].value_counts().reset_index()
        risk.columns = ["classification", "cases"]
        fig = go.Figure(
            go.Pie(labels=risk["classification"], values=risk["cases"], hole=0.55,
                   marker=dict(colors=[theme.BAND_COLORS.get(r, theme.NAVY_400) for r in risk["classification"]]))
        )
        fig.update_layout(title="Onboarding risk classification")
        C.chart(fig, 310)

    dur = cases.dropna(subset=["duration_minutes"]).copy()
    dur["Case type"] = dur["is_straightforward"].map({True: "Straightforward", False: "Complex"})
    fig = px.histogram(dur, x="duration_minutes", color="Case type", nbins=45, barmode="overlay",
                       color_discrete_map={"Straightforward": theme.TEAL_500, "Complex": theme.NAVY_400})
    fig.add_vline(x=O.ONBOARDING_TARGET_MINUTES, line_dash="dot", line_color=theme.DANGER,
                  annotation_text="5 min target")
    fig.update_layout(title="Simulated onboarding duration distribution",
                      xaxis_title="Minutes", yaxis_title="Cases")
    C.chart(fig, 300)


def _tab_queue(cases: pd.DataFrame, watchlist: pd.DataFrame, businesses: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns([1.3, 1.3, 2])
    with col1:
        status_filter = st.multiselect("Status", sorted(cases["current_status"].unique()), key="cm_status")
    with col2:
        risk_filter = st.multiselect("Risk classification", ["low", "medium", "high"], key="cm_risk")
    with col3:
        search = st.text_input("Search business name or case ID", key="cm_search")

    view = cases
    if status_filter:
        view = view[view["current_status"].isin(status_filter)]
    if risk_filter:
        view = view[view["risk_classification"].isin(risk_filter)]
    if search:
        needle = search.lower()
        view = view[
            view["legal_name"].str.lower().str.contains(needle)
            | view["case_id"].str.lower().str.contains(needle)
        ]

    if view.empty:
        C.empty_state("No onboarding cases match these filters.")
        return

    C.dataframe(
        view[["case_id", "legal_name", "current_status", "risk_classification", "screening_hits",
              "manual_review_required", "duration_minutes", "decided_in_app"]].rename(
            columns={"case_id": "Case", "legal_name": "Business", "current_status": "Status",
                     "risk_classification": "Risk", "screening_hits": "Screening hits",
                     "manual_review_required": "Manual review", "duration_minutes": "Duration (min)",
                     "decided_in_app": "Decided in app"}
        ),
        height=260,
    )

    case_id = st.selectbox("Select a case to review", view["case_id"].tolist(), key="cm_case")
    case = cases[cases["case_id"] == case_id].iloc[0]
    business = businesses[businesses["business_id"] == case["business_id"]]
    business_row = business.iloc[0] if len(business) else None

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("**Applicant**")
        pairs = [
            ("Case", case["case_id"]),
            ("Business", f"{case['legal_name']} ({case['business_id']})"),
            ("Current status", str(case["current_status"]).replace("_", " ").title()),
            ("Risk classification", str(case["risk_classification"]).upper()),
            ("Submitted", str(case["submitted_at"])[:19]),
        ]
        if business_row is not None:
            pairs += [
                ("Industry", business_row["industry"]),
                ("Incorporated", business_row["incorporation_date"]),
                ("Declared monthly volume", C.fmt_usd(business_row["declared_monthly_volume_usd"])),
                ("Beneficial owners", int(business_row["n_beneficial_owners"])),
                ("Declared corridors", business_row["expected_corridors"]),
            ]
        C.key_values(pairs)

        st.markdown("**Beneficial ownership & document checklist**")
        checks = [
            ("Certificate of incorporation", bool(case["documents_complete"])),
            ("Proof of business address", bool(case["documents_complete"])),
            ("Beneficial owners ≥25% identified", bool(case["ubo_checklist_complete"])),
            ("Director identity documents", bool(case["documents_complete"])),
            ("Source-of-funds declaration", bool(case["ubo_checklist_complete"])),
        ]
        st.markdown(
            "<div class='pb-kv'>"
            + "".join(f"<div>{'✅' if ok else '❌'} {label}</div>" for label, ok in checks)
            + "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("**Synthetic watchlist screening**")
        hits = S.screen_entity(str(case["legal_name"]), watchlist)
        if hits:
            C.dataframe(pd.DataFrame([h.as_dict() for h in hits]), height=150)
            C.notice(
                "Potential matches against the <b>fictional</b> PayBridge demo list. These are not real "
                "sanctions, PEP or adverse-media records.",
                "danger",
                "⚠️",
            )
        else:
            C.notice("No match above the configured similarity threshold on the fictional demo list.", "success", "✅")

        classification, reasons = S.classify_onboarding_risk(
            int(case["screening_hits"]),
            str(business_row["volume_tier"]) if business_row is not None else "mid",
            bool(case["documents_complete"]),
            bool(case["ubo_checklist_complete"]),
        )
        st.markdown(
            f"**Recomputed risk classification:** {C.badge(classification.upper(), classification)}<br>"
            + "<br>".join(f"• {r}" for r in reasons),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Policy: manual review required = "
            f"{S.requires_manual_review(classification, int(case['screening_hits']))}"
        )

    st.markdown("**Reviewer decision**")
    allowed = O.allowed_next(str(case["current_status"]))
    if not allowed:
        C.notice(
            f"`{case['current_status']}` is a terminal onboarding state — no further transition is permitted.",
            "info", "🔒",
        )
    else:
        d1, d2, d3 = st.columns([1.2, 1.2, 2.4])
        with d1:
            decision = st.selectbox("Move to", list(allowed), key="cm_decision",
                                    format_func=lambda s: s.replace("_", " ").title())
        with d2:
            reviewer = st.selectbox("Reviewer", REVIEWERS, key="cm_reviewer")
        with d3:
            note = st.text_input("Decision note", key="cm_note",
                                 placeholder="Rationale recorded in the audit trail")
        if st.button("Record decision", type="primary", key="cm_submit"):
            O.record_decision(
                case_id=str(case["case_id"]),
                business_id=str(case["business_id"]),
                from_status=str(case["current_status"]),
                to_status=decision,
                reviewer=reviewer,
                note=note,
            )
            st.success(f"Recorded: {case['current_status']} → {decision} by {reviewer}.")
            st.cache_data.clear()
            st.rerun()

    history = O.decisions_for(str(case["case_id"]))
    if not history.empty:
        st.markdown("**Case decision history**")
        C.dataframe(
            history[["decided_at", "from_status", "to_status", "reviewer", "note"]].rename(
                columns={"decided_at": "When", "from_status": "From", "to_status": "To",
                         "reviewer": "Reviewer", "note": "Note"}
            )
        )


def _tab_screening(watchlist: pd.DataFrame) -> None:
    st.markdown("##### Screening sandbox")
    C.notice(
        "Screens a name against PayBridge's <b>fictional</b> demo watchlist of "
        f"{len(watchlist):,} invented entities. This is a workflow simulation and performs no real "
        "sanctions, PEP or adverse-media checking.",
        "synthetic", "⚠️",
    )
    col1, col2 = st.columns([2.4, 1])
    with col1:
        name = st.text_input("Entity name to screen", value="Northwind Logistics", key="sc_name")
    with col2:
        threshold = st.slider("Match threshold", 0.60, 0.98, S.DEFAULT_MATCH_THRESHOLD, 0.01, key="sc_thr")

    hits = S.screen_entity(name, watchlist, threshold=threshold, max_hits=10)
    if hits:
        C.dataframe(pd.DataFrame([h.as_dict() for h in hits]))
    else:
        C.notice("No match at or above this threshold.", "success", "✅")

    st.markdown("##### Fictional watchlist composition")
    comp = watchlist.groupby(["list_source", "entity_type"]).size().reset_index(name="entries")
    fig = px.bar(comp, x="list_source", y="entries", color="entity_type", barmode="group")
    fig.update_layout(title="Demo list entries by source", xaxis_title="", yaxis_title="Entries")
    C.chart(fig, 280)


def _tab_audit() -> None:
    st.markdown("##### Compliance audit trail")
    trail = storage.audit_trail(entity_type="compliance_case", limit=300)
    if trail.empty:
        C.empty_state("No compliance decisions recorded yet in this demo session.")
        return
    C.dataframe(
        trail[["occurred_at", "actor", "entity_id", "action", "detail"]].rename(
            columns={"occurred_at": "When", "actor": "Actor", "entity_id": "Case",
                     "action": "Action", "detail": "Detail"}
        ),
        height=420,
    )


def render() -> None:
    C.page_header(
        "Compliance Operations",
        "Simulated KYB onboarding, fictional-watchlist screening and reviewer decisions, with a full "
        "audit trail. A workflow demonstration — not a regulatory compliance system.",
        eyebrow="Module 3 · Compliance",
    )
    C.notice(
        "<strong>Demonstration only.</strong> PayBridge performs no real identity verification, sanctions "
        "screening or regulatory reporting. The watchlist is fictional and generated by this project.",
        "danger", "⚠️",
    )

    cases = O.apply_overrides(load_compliance_cases())
    watchlist = load_watchlist()
    businesses = load_businesses()

    tabs = st.tabs(["Overview", "Case queue & review", "Screening sandbox", "Audit trail"])
    with tabs[0]:
        _tab_overview(cases)
    with tabs[1]:
        _tab_queue(cases, watchlist, businesses)
    with tabs[2]:
        _tab_screening(watchlist)
    with tabs[3]:
        _tab_audit()
    C.footer()
