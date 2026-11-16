"""Module 1 - Executive Payment Dashboard."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.analytics import kpis
from paybridge.data.loader import load_businesses, load_transactions
from paybridge.ui import components as C
from paybridge.ui import theme
from paybridge.ui.filters import filter_panel


def _kpi_cards(k: dict) -> None:
    C.kpi_cards(
        [
            {"label": "Total payment volume", "value": C.fmt_usd(k["total_volume_usd"], compact=True),
             "sub": "Gross synthetic payout volume", "tone": "navy"},
            {"label": "Transactions", "value": C.fmt_int(k["transaction_count"]),
             "sub": f"{C.fmt_int(k['successful_transactions'])} completed", "tone": "navy"},
            {"label": "Payment success rate", "value": C.fmt_pct(k["success_rate_pct"]),
             "sub": "Completed ÷ all transactions",
             "tone": "success" if k["success_rate_pct"] >= 95 else "warning"},
            {"label": "Avg transaction fee", "value": C.fmt_usd(k["avg_transaction_fee_usd"], 2),
             "sub": f"{C.fmt_pct(k['avg_cost_pct_of_amount'], 2)} of amount on average", "tone": "teal"},
            {"label": "Avg settlement time", "value": C.fmt_hours(k["avg_settlement_hours"]),
             "sub": f"Median {C.fmt_hours(k['median_settlement_hours'])} · completed only", "tone": "teal"},
            {"label": "Modelled cost saving", "value": C.fmt_usd(k["modelled_savings_usd"], compact=True),
             "sub": f"{C.fmt_pct(k['modelled_savings_pct'])} vs an all-SWIFT counterfactual", "tone": "teal"},
            {"label": "Fraud alert rate", "value": C.fmt_pct(k["fraud_alert_rate_pct"], 2),
             "sub": f"{C.fmt_int(k['confirmed_fraud_count'])} labelled fraudulent in this synthetic set",
             "tone": "warning"},
            {"label": "Under review", "value": C.fmt_int(k["transactions_under_review"]),
             "sub": "Held pending a human decision", "tone": "warning"},
        ],
        columns=4,
    )


def render() -> None:
    C.page_header(
        "Executive Payment Overview",
        "Cost, settlement performance and risk posture across every simulated cross-border "
        "corridor. Every figure on this page is calculated from the filtered synthetic records.",
        eyebrow="Module 1 · Finance Operations",
    )
    C.synthetic_notice()

    transactions = load_transactions()
    businesses = load_businesses()

    with st.container():
        state = filter_panel(transactions, businesses, key_prefix="ov")

    df = kpis.apply_filters(
        transactions,
        date_range=state.date_range,
        rails=state.rails,
        statuses=state.statuses,
        corridors=state.corridors,
        business_ids=state.business_ids,
    )

    if df.empty:
        C.empty_state("No transactions match the current filters. Widen the date range or clear a filter.")
        return

    st.caption(
        f"Showing **{len(df):,}** of {len(transactions):,} transactions "
        f"({len(df) / len(transactions) * 100:.1f}% of the dataset)."
    )
    _kpi_cards(kpis.executive_kpis(df))
    C.modelled_notice(
        "Cost saving is a counterfactual: every filtered transaction is repriced on SWIFT using the "
        "documented fee assumptions and compared with its actual simulated rail cost. It is not booked savings."
    )

    # ---------------- Volume & success ----------------
    C.section("Payment volume and reliability")
    left, right = st.columns([1.55, 1])
    with left:
        ts = kpis.volume_timeseries(df, "D")
        fig = go.Figure()
        fig.add_bar(x=ts["period"], y=ts["volume_usd"], name="Volume (USD)",
                    marker_color=theme.TEAL_500, opacity=0.85)
        fig.add_scatter(x=ts["period"], y=ts["transactions"], name="Transactions",
                        yaxis="y2", mode="lines", line=dict(color=theme.NAVY_600, width=2))
        fig.update_layout(
            title="Daily payout volume and count",
            yaxis=dict(title="Volume (USD)"),
            yaxis2=dict(title="Transactions", overlaying="y", side="right", showgrid=False),
        )
        C.chart(fig, 330)
    with right:
        rails = kpis.rail_distribution(df)
        fig = go.Figure(
            go.Pie(
                labels=rails["payment_rail"], values=rails["transactions"], hole=0.58,
                marker=dict(colors=[theme.RAIL_COLORS.get(r, theme.TEAL_500) for r in rails["payment_rail"]]),
                textinfo="label+percent", sort=False,
            )
        )
        fig.update_layout(title="Payment rail mix", showlegend=False)
        C.chart(fig, 330)

    left, right = st.columns(2)
    with left:
        trend = kpis.success_trend(df, "D")
        fig = px.line(trend, x="period", y="success_rate_pct", markers=False)
        fig.update_traces(line=dict(color=theme.SUCCESS, width=2))
        fig.update_layout(title="Transaction success rate trend", yaxis_title="Success rate (%)",
                          xaxis_title="")
        C.chart(fig, 300)
    with right:
        settle = kpis.settlement_comparison(df)
        fig = go.Figure()
        fig.add_bar(x=settle["payment_rail"], y=settle["p50_hours"], name="p50 (hours)",
                    marker_color=theme.TEAL_500)
        fig.add_bar(x=settle["payment_rail"], y=settle["p90_hours"], name="p90 (hours)",
                    marker_color=theme.NAVY_400)
        fig.update_layout(title="Realised settlement time by rail", barmode="group",
                          yaxis_title="Hours", xaxis_title="")
        C.chart(fig, 300)

    # ---------------- Corridors & cost ----------------
    C.section("Corridor performance and cost composition")
    left, right = st.columns([1.3, 1])
    with left:
        corr = kpis.corridor_breakdown(df)
        fig = px.bar(corr, x="corridor", y="volume_usd", color="avg_cost_pct",
                     color_continuous_scale=theme.SEQUENTIAL,
                     labels={"avg_cost_pct": "Avg cost %"})
        fig.update_layout(title="Volume by corridor (shaded by average cost %)",
                          yaxis_title="Volume (USD)", xaxis_title="", coloraxis_colorbar=dict(title="Cost %"))
        C.chart(fig, 320)
    with right:
        cost = kpis.cost_breakdown_by_rail(df)
        label_map = {
            "fixed_fee": "Fixed fee", "variable_fee": "Variable fee",
            "fx_markup_cost": "FX markup", "lifting_fees": "Correspondent fees",
        }
        cost = cost.assign(component=cost["component"].map(label_map))
        fig = px.bar(cost, x="payment_rail", y="usd", color="component", barmode="stack")
        fig.update_layout(title="Customer cost composition by rail", yaxis_title="USD", xaxis_title="")
        C.chart(fig, 320)

    C.dataframe(
        corr.rename(
            columns={
                "corridor": "Corridor", "transactions": "Transactions", "volume_usd": "Volume (USD)",
                "total_cost_usd": "Fees (USD)", "avg_cost_pct": "Avg cost %",
                "success_rate_pct": "Success %", "fraud_rate_pct": "Labelled fraud %",
            }
        ),
        column_config={
            "Volume (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Fees (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Avg cost %": st.column_config.NumberColumn(format="%.2f%%"),
            "Success %": st.column_config.NumberColumn(format="%.1f%%"),
            "Labelled fraud %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    # ---------------- Risk ----------------
    C.section("Risk posture", "Alerts come from the FraudLens rules baseline evaluated on every transaction.")
    left, right = st.columns(2)
    with left:
        bands = kpis.fraud_alerts_by_band(df)
        fig = go.Figure()
        fig.add_bar(x=bands["rules_band"], y=bands["transactions"], name="Transactions",
                    marker_color=theme.NAVY_200)
        fig.add_bar(x=bands["rules_band"], y=bands["confirmed_fraud"], name="Labelled fraud",
                    marker_color=theme.DANGER)
        fig.update_layout(title="Transactions and labelled fraud by risk band", barmode="overlay",
                          yaxis_type="log", yaxis_title="Count (log scale)", xaxis_title="")
        C.chart(fig, 300)
    with right:
        hr = kpis.high_risk_trend(df, "D")
        fig = go.Figure()
        fig.add_scatter(x=hr["period"], y=hr["high_risk_transactions"], name="High + critical",
                        mode="lines", line=dict(color=theme.DANGER, width=2))
        fig.add_scatter(x=hr["period"], y=hr["under_review"], name="Held for review",
                        mode="lines", line=dict(color=theme.WARNING, width=2, dash="dot"))
        fig.update_layout(title="High-risk transaction trend", yaxis_title="Transactions", xaxis_title="")
        C.chart(fig, 300)

    C.section("Largest customers by volume")
    board = kpis.business_leaderboard(df, 15)
    name_by_id = dict(zip(businesses["business_id"], businesses["legal_name"]))
    board.insert(1, "business", board["business_id"].map(name_by_id))
    C.dataframe(
        board.rename(
            columns={
                "business_id": "ID", "business": "Business", "transactions": "Transactions",
                "volume_usd": "Volume (USD)", "fees_paid_usd": "Fees (USD)",
                "margin_usd": "Contribution (USD)", "alert_rate_pct": "Alert %",
                "success_rate_pct": "Success %",
            }
        ),
        column_config={
            "Volume (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Fees (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Contribution (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Alert %": st.column_config.NumberColumn(format="%.2f%%"),
            "Success %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        height=430,
    )
    C.footer()
