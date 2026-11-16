"""Module 9 - Product metrics, analytics and experiment design."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.data.loader import load_product_events, load_transactions
from paybridge.db import storage
from paybridge.product import metrics as M
from paybridge.product.experiments import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    design_experiment,
    runtime_days,
    sample_size_two_proportions,
)
from paybridge.ui import components as C
from paybridge.ui import theme


def _tab_kpis(events: pd.DataFrame, transactions: pd.DataFrame) -> None:
    activation = M.activation_metrics(events)
    payment = M.payment_product_kpis(transactions)

    C.kpi_cards(
        [
            {"label": "Customer activation rate", "value": C.fmt_pct(activation["activation_rate_pct"]),
             "sub": f"{activation['activated']} of {activation['signups']} signed-up businesses "
                    "reached a first payout", "tone": "teal"},
            {"label": "Median time to activate",
             "value": f"{activation['median_days_to_activate']:.1f} d"
             if activation["median_days_to_activate"] == activation["median_days_to_activate"] else "-",
             "sub": f"{activation['activated_within_7_days_pct']:.0f}% activate within 7 days", "tone": "teal"},
            {"label": "Payout completion rate", "value": C.fmt_pct(payment["payout_completion_rate_pct"]),
             "sub": "Completed ÷ payouts reaching a terminal state", "tone": "success"},
            {"label": "Payment success rate", "value": C.fmt_pct(payment["payment_success_rate_pct"]),
             "sub": f"{C.fmt_int(payment['transactions'])} transactions", "tone": "success"},
            {"label": "Cost per payout", "value": C.fmt_usd(payment["cost_per_payout_usd"], 2),
             "sub": "Platform direct cost per transaction", "tone": "navy"},
            {"label": "Revenue per transaction",
             "value": C.fmt_usd(payment["avg_revenue_per_transaction_usd"], 2),
             "sub": f"Contribution {C.fmt_usd(payment['contribution_margin_per_transaction_usd'], 2)} per payout",
             "tone": "navy"},
            {"label": "Manual review rate", "value": C.fmt_pct(payment["manual_review_rate_pct"], 2),
             "sub": f"Alert rate {C.fmt_pct(payment['alert_rate_pct'], 2)}", "tone": "warning"},
            {"label": "Payouts per business", "value": f"{payment['avg_payouts_per_business']:.1f}",
             "sub": f"{C.fmt_int(payment['active_businesses'])} businesses transacting", "tone": "teal"},
        ],
        columns=4,
    )
    C.notice(
        "Activation, adoption and retention are computed from the synthetic <b>product event stream</b>, "
        "not inferred from payment records — payment data alone cannot support those metrics.",
        "modelled", "📐",
    )

    cycle = M.investigation_cycle_time(storage.audit_trail("investigation", limit=1000))
    if cycle.get("available"):
        C.kpi_cards(
            [
                {"label": "Median investigation time", "value": f"{cycle['median_minutes']:.1f} min",
                 "sub": f"p90 {cycle['p90_minutes']:.1f} min across {cycle['decided_cases']} decided cases",
                 "tone": "teal"},
            ],
            columns=1,
        )
    else:
        C.notice(
            f"<b>Investigation cycle time:</b> {cycle.get('reason')} Decide a case on the Investigation "
            "Workbench and this metric will be measured from the audit trail rather than estimated.",
            "info", "⏱️",
        )

    left, right = st.columns([1.3, 1])
    with left:
        weekly = M.weekly_active_businesses(events)
        fig = go.Figure()
        fig.add_bar(x=weekly["week"], y=weekly["events"], name="Events", marker_color=theme.NAVY_200)
        fig.add_scatter(x=weekly["week"], y=weekly["active_businesses"], name="Active businesses",
                        yaxis="y2", line=dict(color=theme.TEAL_500, width=2.5))
        fig.update_layout(title="Platform usage by week", yaxis=dict(title="Events"),
                          yaxis2=dict(title="Active businesses", overlaying="y", side="right", showgrid=False))
        C.chart(fig, 310)
    with right:
        retention = M.weekly_retention(events)
        fig = px.line(retention, x="week", y="retention_pct", markers=True)
        fig.update_traces(line=dict(color=theme.TEAL_500, width=2.5))
        fig.update_layout(title="Week-over-week return rate", yaxis_title="Retention (%)", xaxis_title="")
        C.chart(fig, 310)


def _tab_adoption(events: pd.DataFrame) -> None:
    adoption = M.feature_adoption(events)
    fig = px.bar(adoption.sort_values("adoption_pct"), x="adoption_pct", y="feature", orientation="h")
    fig.update_traces(marker_color=theme.TEAL_500)
    fig.update_layout(title="Feature adoption — share of active businesses using the feature",
                      xaxis_title="Adoption (%)", yaxis_title="")
    C.chart(fig, 380)

    C.dataframe(
        adoption.rename(columns={"feature": "Feature", "event_name": "Event", "businesses": "Businesses",
                                 "adoption_pct": "Adoption %", "events": "Events"}),
        column_config={"Adoption %": st.column_config.NumberColumn(format="%.1f%%")},
    )

    usage = M.usage_by_feature_over_time(events)
    fig = px.area(usage, x="week", y="events", color="event_name")
    fig.update_layout(title="Feature usage over time", xaxis_title="", yaxis_title="Events")
    C.chart(fig, 320)


def _tab_experiments(events: pd.DataFrame, transactions: pd.DataFrame) -> None:
    C.section("Experiment designs")
    C.notice(
        "These are <b>designs</b>. No experiment has been run against users, so no result is reported. "
        "Sample sizes and runtimes below are computed from the stated baseline, effect size, alpha and "
        "power.", "modelled", "🧪",
    )
    designs = storage.query_df("SELECT * FROM experiments ORDER BY experiment_id")
    if designs.empty:
        C.empty_state("No experiment designs are seeded in this environment.")
        return

    C.dataframe(
        designs[["experiment_id", "name", "primary_kpi", "baseline_rate", "mde_relative",
                 "sample_size_per_arm", "runtime_days", "status"]].rename(
            columns={"experiment_id": "ID", "name": "Experiment", "primary_kpi": "Primary KPI",
                     "baseline_rate": "Baseline", "mde_relative": "MDE (relative)",
                     "sample_size_per_arm": "Sample / arm", "runtime_days": "Runtime (days)",
                     "status": "Status"}
        ),
        column_config={
            "Baseline": st.column_config.NumberColumn(format="%.3f"),
            "MDE (relative)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    selected = st.selectbox("Experiment", designs["experiment_id"].tolist(),
                            format_func=lambda e: f"{e} · {designs.loc[designs.experiment_id == e, 'name'].iloc[0]}",
                            key="ex_pick")
    row = designs[designs["experiment_id"] == selected].iloc[0]
    left, right = st.columns([1.3, 1])
    with left:
        C.key_values(
            [
                ("Hypothesis", row["hypothesis"]),
                ("Primary KPI", row["primary_kpi"]),
                ("Guardrail metrics", row["guardrails"]),
                ("Baseline rate", f"{float(row['baseline_rate']):.3f}"),
                ("Minimum detectable effect", f"{float(row['mde_relative']) * 100:.0f}% relative"),
                ("Alpha / power", f"{float(row['alpha']):.2f} / {float(row['power']):.2f}"),
                ("Sample size per arm", f"{int(row['sample_size_per_arm']):,}"),
                ("Estimated runtime", f"{int(row['runtime_days'])} days"),
                ("Rollout criteria", row["rollout_criteria"]),
            ]
        )
    with right:
        observed = None
        if selected == "EXP-001":
            from paybridge.compliance.onboarding import apply_overrides, compliance_kpis
            from paybridge.data.loader import load_compliance_cases

            observed = compliance_kpis(apply_overrides(load_compliance_cases()))["completion_rate_pct"] / 100
            label = "Measured onboarding completion rate"
        elif selected == "EXP-002":
            rail_share = (transactions["payment_rail"] != "SWIFT").mean()
            observed, label = rail_share, "Measured non-SWIFT rail share"
        if observed is not None:
            C.kpi_cards(
                [
                    {"label": label, "value": f"{observed:.3f}",
                     "sub": f"Design baseline {float(row['baseline_rate']):.3f} — the design input and the "
                            "current synthetic measurement are shown side by side", "tone": "teal"},
                ],
                columns=1,
            )
        else:
            st.caption("No directly comparable baseline exists in the synthetic data for this design.")

    C.section("Sample size calculator")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        baseline = st.slider("Baseline rate", 0.01, 0.95, float(row["baseline_rate"]), 0.01, key="ex_base")
    with c2:
        mde = st.slider("Relative MDE", 0.02, 0.60, float(row["mde_relative"]), 0.01, key="ex_mde")
    with c3:
        alpha = st.select_slider("Alpha", [0.01, 0.05, 0.10], value=DEFAULT_ALPHA, key="ex_alpha")
    with c4:
        power = st.select_slider("Power", [0.70, 0.80, 0.90, 0.95], value=DEFAULT_POWER, key="ex_power")
    daily = st.number_input("Eligible units per day", 1.0, 5000.0, 50.0, 5.0, key="ex_daily")

    n = sample_size_two_proportions(baseline, mde, alpha, power)
    days = runtime_days(n, daily)
    C.kpi_cards(
        [
            {"label": "Sample size per arm", "value": C.fmt_int(n),
             "sub": f"Detecting {baseline:.3f} → {baseline * (1 + mde):.3f}", "tone": "navy"},
            {"label": "Total sample (2 arms)", "value": C.fmt_int(n * 2), "tone": "navy"},
            {"label": "Estimated runtime", "value": f"{days} days",
             "sub": f"At {daily:,.0f} eligible units per day", "tone": "teal"},
            {"label": "Assumptions", "value": f"α {alpha:.2f} · power {power:.2f}",
             "sub": "Two-sided two-proportion test", "tone": "teal"},
        ],
        columns=4,
    )

    curve = pd.DataFrame(
        [
            {"mde": m, "sample_per_arm": sample_size_two_proportions(baseline, m, alpha, power)}
            for m in [x / 100 for x in range(3, 61, 2)]
        ]
    )
    fig = px.line(curve, x="mde", y="sample_per_arm", log_y=True)
    fig.update_traces(line=dict(color=theme.TEAL_500, width=2.5))
    fig.add_vline(x=mde, line_dash="dot", line_color=theme.DANGER)
    fig.update_layout(title="Sample size versus detectable effect",
                      xaxis_title="Relative minimum detectable effect", yaxis_title="Sample per arm (log)")
    C.chart(fig, 320)


def render() -> None:
    C.page_header(
        "Product Analytics",
        "Activation, adoption, retention and unit economics for the platform, plus experiment designs "
        "with computed sample sizes.",
        eyebrow="Module 9 · Product management",
    )
    C.synthetic_notice()

    events = load_product_events()
    transactions = load_transactions()

    tabs = st.tabs(["Product KPIs", "Feature adoption", "Experiments"])
    with tabs[0]:
        _tab_kpis(events, transactions)
    with tabs[1]:
        _tab_adoption(events)
    with tabs[2]:
        _tab_experiments(events, transactions)
    C.footer()
