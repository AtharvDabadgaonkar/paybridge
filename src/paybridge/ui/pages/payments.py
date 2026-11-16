"""Module 2 - Cross-Border Payout Simulator and unit economics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.config import CORRIDORS, RAILS, SYNTHETIC_DATA_NOTICE
from paybridge.data.loader import load_businesses, load_transactions
from paybridge.fraud.scoring import assess_transaction, build_payout_context, load_bundle_safe
from paybridge.payments import payouts as payout_service
from paybridge.payments.pricing import compare_rails, quote_payment, savings_vs_swift
from paybridge.ui import components as C
from paybridge.ui import theme

AMOUNT_GRID = (250, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000)


def _rail_cards(quotes, recommended: str) -> None:
    cards = []
    for q in quotes:
        tone = "teal" if q.rail == recommended else "navy"
        if not q.eligible:
            tone = "danger"
        cards.append(
            {
                "label": q.rail.replace("_", " "),
                "value": C.fmt_usd(q.total_customer_cost, 2),
                "delta": "Recommended (lowest cost)" if q.rail == recommended and q.eligible else "",
                "delta_dir": "up" if q.rail == recommended else "flat",
                "sub": (
                    f"{C.fmt_pct(q.effective_cost_pct, 2)} of amount · settles ~{C.fmt_hours(q.settlement_hours_p50)} "
                    f"· recipient gets {q.recipient_amount_local:,.2f} {q.currency}"
                    if q.eligible
                    else f"Not available: {q.ineligible_reason}"
                ),
                "tone": tone,
            }
        )
    C.kpi_cards(cards, columns=min(len(cards), 3))


def _tab_create(transactions: pd.DataFrame, businesses: pd.DataFrame) -> None:
    st.markdown("##### 1. Compose the payout")
    col1, col2, col3, col4 = st.columns([2, 1.3, 1.2, 1.2])
    name_by_id = dict(zip(businesses["business_id"], businesses["legal_name"]))
    with col1:
        business_id = st.selectbox(
            "Business customer",
            list(name_by_id),
            format_func=lambda b: f"{name_by_id[b]} ({b})",
            key="po_biz",
        )
    with col2:
        corridor_code = st.selectbox(
            "Destination corridor",
            [c.code for c in CORRIDORS],
            format_func=lambda c: f"{c} · {next(x.currency for x in CORRIDORS if x.code == c)}",
            key="po_corridor",
        )
    corridor = next(c for c in CORRIDORS if c.code == corridor_code)
    with col3:
        amount = st.number_input(
            "Amount (USD)", min_value=10.0, max_value=500_000.0, value=7_500.0, step=250.0, key="po_amt"
        )
    with col4:
        st.text_input("Settlement currency", value=corridor.currency, disabled=True, key="po_ccy")

    opt1, opt2 = st.columns([1, 3])
    with opt1:
        new_beneficiary = st.checkbox("New beneficiary", value=False, key="po_newben")
    with opt2:
        st.caption(
            "A new beneficiary materially changes the FraudLens assessment below — "
            "toggle it to see the risk layers respond."
        )

    st.markdown("##### 2. Compare eligible routes")
    quotes = compare_rails(corridor.recipient_country, float(amount))
    eligible = [q for q in quotes if q.eligible]
    recommended = eligible[0].rail if eligible else ""
    _rail_cards(quotes, recommended)

    table = pd.DataFrame(
        [
            {
                "Rail": q.rail,
                "Total cost (USD)": q.total_customer_cost,
                "Cost %": q.effective_cost_pct,
                "Fixed fee": q.fixed_fee,
                "Variable fee": q.variable_fee,
                "FX markup": q.fx_markup_cost,
                "Correspondent fees": q.lifting_fees,
                f"Recipient gets ({q.currency})": q.recipient_amount_local,
                "Settles (p50)": q.settlement_hours_p50,
                "Platform margin": q.contribution_margin,
                "Eligible": q.eligible,
            }
            for q in quotes
        ]
    )
    C.dataframe(
        table,
        column_config={
            "Total cost (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Cost %": st.column_config.NumberColumn(format="%.2f%%"),
            "Fixed fee": st.column_config.NumberColumn(format="$%.2f"),
            "Variable fee": st.column_config.NumberColumn(format="$%.2f"),
            "FX markup": st.column_config.NumberColumn(format="$%.2f"),
            "Correspondent fees": st.column_config.NumberColumn(format="$%.2f"),
            "Settles (p50)": st.column_config.NumberColumn(format="%.1f h"),
            "Platform margin": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    savings = savings_vs_swift(corridor.recipient_country, float(amount))
    if savings:
        bits = [
            f"<b>{rail}</b>: {C.fmt_usd(v['savings_usd'], 2)} "
            f"({v['savings_pct']:.1f}%) {'cheaper' if v['savings_usd'] >= 0 else 'more expensive'} than SWIFT"
            for rail, v in savings.items()
        ]
        C.modelled_notice(
            "At this amount and corridor: " + "; ".join(bits)
            + ". Calculated from the documented fee assumptions, not from booked transactions."
        )

    st.markdown("##### 3. FraudLens pre-check")
    bundle = load_bundle_safe()
    rail_choice = st.selectbox(
        "Payment rail to use",
        [q.rail for q in quotes],
        index=[q.rail for q in quotes].index(recommended) if recommended else 0,
        format_func=lambda r: f"{r}{'' if any(q.rail == r and q.eligible for q in quotes) else '  (ineligible)'}",
        key="po_rail",
    )
    context = build_payout_context(
        business_id, float(amount), corridor_code, rail_choice, transactions,
        new_beneficiary=new_beneficiary,
    )
    assessment = assess_transaction(context, bundle)

    rcol1, rcol2 = st.columns([1, 1.6])
    with rcol1:
        C.kpi_cards(
            [
                {
                    "label": "Combined risk score",
                    "value": f"{assessment.combined_score:.3f}",
                    "sub": f"Band: {assessment.combined_band.upper()} · decision: {assessment.decision.replace('_', ' ')}",
                    "tone": {"low": "success", "medium": "teal", "high": "warning", "critical": "danger"}[
                        assessment.combined_band
                    ],
                },
                {
                    "label": "Model probability",
                    "value": f"{assessment.ml_probability:.3f}" if assessment.ml_probability is not None else "n/a",
                    "sub": (
                        f"Threshold {assessment.ml_threshold:.3f} · {assessment.model_name}"
                        if assessment.ml_threshold
                        else "No trained model loaded"
                    ),
                    "tone": "navy",
                },
            ],
            columns=1,
        )
    with rcol2:
        if assessment.rules.triggered:
            st.markdown(
                "**Triggered rules**<br>"
                + "<br>".join(
                    f"{C.badge(t.code.split('_')[0], 'warning')} <b>{t.name}</b> — {t.description}"
                    for t in assessment.rules.triggered
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown("**Triggered rules**<br>None — no deterministic red flag fired.", unsafe_allow_html=True)

    st.markdown("##### 4. Initiate the simulated payout")
    if assessment.requires_human_review:
        C.notice(
            "This payout would be routed to <b>human review</b> before release. FraudLens does not "
            "auto-approve high or critical risk payouts.",
            "danger" if assessment.combined_band == "critical" else "info",
            "🛑" if assessment.combined_band == "critical" else "👤",
        )
    can_create = any(q.rail == rail_choice and q.eligible for q in quotes)
    if not can_create:
        C.notice("The selected rail is not eligible for this amount or corridor.", "danger", "⚠️")

    if st.button("Create simulated payout", type="primary", disabled=not can_create,
                 key="po_create"):
        record = payout_service.create_payout(
            business_id=business_id,
            recipient_country=corridor.recipient_country,
            amount_usd=float(amount),
            rail_code=rail_choice,
            actor="ops.user",
            business_name=name_by_id[business_id],
            risk_score=assessment.combined_score,
            risk_band=assessment.combined_band,
            ml_score=assessment.ml_probability,
            rules_triggered=",".join(assessment.rules.codes),
        )
        st.session_state["last_payout_id"] = record.payout_id
        st.success(
            f"Simulated payout created. Reference **{record.reference}** · "
            f"{C.fmt_usd(record.quote.amount_usd, 2)} via {record.quote.rail} · status `{record.status}`. "
            "No money has moved — this is a simulation."
        )
        st.caption("Track and advance it on the **Payout tracker** tab.")


def _tab_economics() -> None:
    st.markdown("##### Unit economics calculator")
    col1, col2, col3 = st.columns(3)
    with col1:
        corridor_code = st.selectbox("Corridor", [c.code for c in CORRIDORS], key="ue_corr")
    with col2:
        amount = st.number_input("Amount (USD)", 10.0, 500_000.0, 10_000.0, 500.0, key="ue_amt")
    with col3:
        rail = st.selectbox("Rail", [r.code for r in RAILS], key="ue_rail")

    corridor = next(c for c in CORRIDORS if c.code == corridor_code)
    q = quote_payment(rail, corridor.recipient_country, float(amount))

    C.kpi_cards(
        [
            {"label": "Customer pays", "value": C.fmt_usd(q.total_customer_cost, 2),
             "sub": f"{C.fmt_pct(q.effective_cost_pct, 2)} of the payout", "tone": "navy"},
            {"label": "Platform direct cost", "value": C.fmt_usd(q.platform_direct_cost, 2),
             "sub": "Provider, partner, FX spread and pass-through fees", "tone": "teal"},
            {"label": "Contribution margin", "value": C.fmt_usd(q.contribution_margin, 2),
             "sub": f"{C.fmt_pct(q.contribution_margin_pct)} of revenue",
             "tone": "success" if q.contribution_margin > 0 else "danger"},
            {"label": "Recipient receives", "value": f"{q.recipient_amount_local:,.2f} {q.currency}",
             "sub": f"Offered rate {q.offered_rate:,.4f} vs mid {q.mid_market_rate:,.4f}", "tone": "teal"},
        ],
        columns=4,
    )

    left, right = st.columns([1.2, 1])
    with left:
        waterfall = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "total", "relative", "relative", "relative", "relative", "total"],
                x=["Fixed fee", "Variable fee", "FX markup", "Correspondent fees", "Customer pays",
                   "Provider fixed", "Provider variable", "Partner payout", "FX wholesale spread", "Contribution margin"],
                y=[q.fixed_fee, q.variable_fee, q.fx_markup_cost, q.lifting_fees, 0,
                   -q.provider_fixed_cost, -q.provider_variable_cost, -q.partner_payout_cost, -q.fx_spread_cost, 0],
                connector=dict(line=dict(color=theme.BORDER)),
                increasing=dict(marker=dict(color=theme.TEAL_500)),
                decreasing=dict(marker=dict(color=theme.NAVY_400)),
                totals=dict(marker=dict(color=theme.NAVY_800)),
            )
        )
        waterfall.update_layout(title=f"Revenue to margin — {rail} on {corridor_code}", yaxis_title="USD")
        C.chart(waterfall, 380)
    with right:
        C.key_values(
            [
                ("Rail", q.rail_label),
                ("Corridor", f"{q.corridor} ({q.currency})"),
                ("Amount", C.fmt_usd(q.amount_usd, 2)),
                ("Settlement p50 / p90", f"{C.fmt_hours(q.settlement_hours_p50)} / {C.fmt_hours(q.settlement_hours_p90)}"),
                ("Expected success rate", C.fmt_pct(q.expected_success_rate * 100, 1)),
                ("Platform revenue", C.fmt_usd(q.platform_revenue, 2)),
                ("Platform direct cost", C.fmt_usd(q.platform_direct_cost, 2)),
                ("Contribution margin", f"{C.fmt_usd(q.contribution_margin, 2)} ({C.fmt_pct(q.contribution_margin_pct)})"),
            ]
        )
        C.notice(
            "Fee, FX and partner-cost assumptions live in <code>config.py</code> and are documented "
            "in <code>docs/UNIT_ECONOMICS.md</code>. They are indicative modelling assumptions, "
            "not a provider rate card.",
            "modelled",
            "📐",
        )

    st.markdown("##### All rails at this amount and corridor")
    rows = [
        {
            "Rail": x.rail, "Customer pays": x.total_customer_cost, "Cost %": x.effective_cost_pct,
            "Direct cost": x.platform_direct_cost, "Margin": x.contribution_margin,
            "Margin %": x.contribution_margin_pct, "Settles (p50 h)": x.settlement_hours_p50,
            "Eligible": x.eligible,
        }
        for x in compare_rails(corridor.recipient_country, float(amount))
    ]
    C.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Customer pays": st.column_config.NumberColumn(format="$%.2f"),
            "Cost %": st.column_config.NumberColumn(format="%.2f%%"),
            "Direct cost": st.column_config.NumberColumn(format="$%.2f"),
            "Margin": st.column_config.NumberColumn(format="$%.2f"),
            "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


def _tab_sensitivity() -> None:
    st.markdown("##### Product hypothesis: can alternative rails cut payout cost by ~60% versus SWIFT?")
    C.modelled_notice(
        "The grid below is computed by repricing the same payout on every rail under the documented "
        "assumptions. The result is reported as calculated — including where it falls short of 60%."
    )
    rail = st.radio(
        "Rail to compare against the SWIFT baseline",
        [r.code for r in RAILS if r.code != "SWIFT"],
        horizontal=True,
        key="sens_rail",
    )

    rows = []
    for corridor in CORRIDORS:
        for amount in AMOUNT_GRID:
            result = savings_vs_swift(corridor.recipient_country, float(amount))
            if rail in result:
                rows.append(
                    {
                        "corridor": corridor.code,
                        "amount": amount,
                        "savings_pct": result[rail]["savings_pct"],
                        "savings_usd": result[rail]["savings_usd"],
                        "eligible": bool(result[rail]["eligible"]),
                    }
                )
    grid = pd.DataFrame(rows)
    eligible_grid = grid[grid["eligible"]]

    pivot = eligible_grid.pivot(index="corridor", columns="amount", values="savings_pct")
    fig = px.imshow(
        pivot,
        labels=dict(x="Payout amount (USD)", y="Corridor", color="Saving vs SWIFT (%)"),
        color_continuous_scale=theme.SEQUENTIAL,
        aspect="auto",
        text_auto=".0f",
    )
    fig.update_layout(title=f"Modelled payout-cost saving: {rail} vs SWIFT (%)")
    C.chart(fig, 380)

    left, right = st.columns([1.3, 1])
    with left:
        line = eligible_grid.groupby("amount")["savings_pct"].mean().reset_index()
        fig = px.line(line, x="amount", y="savings_pct", markers=True, log_x=True)
        fig.update_traces(line=dict(color=theme.TEAL_500, width=2.5))
        fig.add_hline(y=60, line_dash="dot", line_color=theme.DANGER,
                      annotation_text="60% hypothesis", annotation_position="top right")
        fig.update_layout(title="Average saving across corridors by payout size",
                          xaxis_title="Payout amount (USD, log scale)", yaxis_title="Saving vs SWIFT (%)")
        C.chart(fig, 320)
    with right:
        share = float((eligible_grid["savings_pct"] >= 60).mean() * 100)
        best = eligible_grid.loc[eligible_grid["savings_pct"].idxmax()]
        worst = eligible_grid.loc[eligible_grid["savings_pct"].idxmin()]
        C.kpi_cards(
            [
                {"label": "Scenarios at or above 60%", "value": C.fmt_pct(share),
                 "sub": f"of {len(eligible_grid)} eligible corridor × amount scenarios", "tone": "teal"},
                {"label": "Mean saving", "value": C.fmt_pct(eligible_grid['savings_pct'].mean()),
                 "sub": f"Best {best['savings_pct']:.0f}% ({best['corridor']}, {C.fmt_usd(best['amount'])}) · "
                        f"Worst {worst['savings_pct']:.0f}% ({worst['corridor']}, {C.fmt_usd(worst['amount'])})",
                 "tone": "navy"},
            ],
            columns=1,
        )
        st.caption(
            "Interpretation: the saving is largest on small and mid-size payouts, where SWIFT's fixed "
            "wire and correspondent fees dominate, and compresses as the amount grows and percentage-based "
            "FX costs take over."
        )

    C.dataframe(
        grid.pivot_table(index="corridor", columns="amount", values="savings_pct").round(1),
        hide_index=False,
    )


def _tab_tracker() -> None:
    st.markdown("##### Simulated payout tracker")
    df = payout_service.list_payouts(300)
    if df.empty:
        C.empty_state("No simulated payouts yet. Create one on the **Create payout** tab.")
        return

    C.kpi_cards(
        [
            {"label": "Payouts created", "value": C.fmt_int(len(df)), "tone": "navy"},
            {"label": "Total simulated volume", "value": C.fmt_usd(df["amount_usd"].sum(), compact=True), "tone": "teal"},
            {"label": "Awaiting human decision",
             "value": C.fmt_int((df["status"] == "under_review").sum()), "tone": "warning"},
            {"label": "Settled", "value": C.fmt_int((df["status"] == "settled").sum()), "tone": "success"},
        ],
        columns=4,
    )

    C.dataframe(
        df[["reference", "business_name", "corridor", "payment_rail", "amount_usd",
            "total_customer_cost", "risk_band", "status", "created_at"]].rename(
            columns={
                "reference": "Reference", "business_name": "Business", "corridor": "Corridor",
                "payment_rail": "Rail", "amount_usd": "Amount (USD)",
                "total_customer_cost": "Cost (USD)", "risk_band": "Risk", "status": "Status",
                "created_at": "Created",
            }
        ),
        column_config={
            "Amount (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Cost (USD)": st.column_config.NumberColumn(format="$%.2f"),
        },
        height=280,
    )

    options = dict(zip(df["payout_id"], df["reference"]))
    default_id = st.session_state.get("last_payout_id")
    index = list(options).index(default_id) if default_id in options else 0
    payout_id = st.selectbox("Select a payout", list(options), index=index, format_func=lambda p: options[p])
    record = payout_service.get_payout(payout_id)
    if record is None:
        return

    left, right = st.columns([1, 1.25])
    with left:
        C.key_values(
            [
                ("Reference", record["reference"]),
                ("Business", f"{record['business_name']} ({record['business_id']})"),
                ("Route", f"{record['corridor']} via {record['payment_rail']}"),
                ("Amount", C.fmt_usd(record["amount_usd"], 2)),
                ("Customer cost", C.fmt_usd(record["total_customer_cost"], 2)),
                ("Recipient receives", f"{record['recipient_amount_local']:,.2f} {record['currency']}"),
                ("Contribution margin", C.fmt_usd(record["contribution_margin"], 2)),
                ("Risk", f"{str(record['risk_band']).upper()} ({float(record['risk_score'] or 0):.3f})"),
                ("Status", str(record["status"]).replace("_", " ").title()),
            ]
        )
        nxt = payout_service.suggested_next_status(str(record["status"]), str(record["risk_band"] or "low"))
        if nxt:
            if st.button(f"Advance to `{nxt}`", type="primary", key="po_advance"):
                payout_service.advance_payout(payout_id, actor="ops.user")
                st.rerun()
        elif str(record["status"]) == "under_review":
            C.notice("Held for human decision. Approve or reject below.", "warning", "👤")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if st.button("Approve for payout", type="primary", key="po_appr"):
                    payout_service.transition_payout(payout_id, "approved_for_payout", "risk.analyst",
                                                     "Approved after manual review")
                    st.rerun()
            with dcol2:
                if st.button("Reject payout", key="po_rej"):
                    payout_service.transition_payout(payout_id, "rejected", "risk.analyst",
                                                     "Rejected after manual review")
                    st.rerun()
        else:
            C.notice(f"`{record['status']}` is a terminal state for this simulated payout.", "info", "✅")

    with right:
        events = payout_service.payout_events(payout_id)
        st.markdown("**Lifecycle audit trail**")
        C.dataframe(
            events[["occurred_at", "from_status", "to_status", "actor", "note"]].rename(
                columns={"occurred_at": "When", "from_status": "From", "to_status": "To",
                         "actor": "Actor", "note": "Note"}
            ),
            height=260,
        )


def render() -> None:
    C.page_header(
        "Payments & Payouts",
        "Compose a cross-border payout, compare every eligible rail on fully decomposed cost, "
        "and follow the simulated payout through its lifecycle.",
        eyebrow="Module 2 · Payout operations",
    )
    C.notice(f"<strong>Simulation only.</strong> {SYNTHETIC_DATA_NOTICE}", "synthetic", "⚠️")

    transactions = load_transactions()
    businesses = load_businesses()

    tabs = st.tabs(["Create payout", "Unit economics", "Savings sensitivity", "Payout tracker"])
    with tabs[0]:
        _tab_create(transactions, businesses)
    with tabs[1]:
        _tab_economics()
    with tabs[2]:
        _tab_sensitivity()
    with tabs[3]:
        _tab_tracker()
    C.footer()
