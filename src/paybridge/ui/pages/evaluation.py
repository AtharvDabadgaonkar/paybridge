"""Module 6 - Fraud model evaluation dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.config import FRAUD_LOSS_RATE, MANUAL_REVIEW_COST_USD
from paybridge.data.loader import load_transactions
from paybridge.fraud import evaluation as E
from paybridge.ui import components as C
from paybridge.ui import theme
from paybridge.ui.services import model_bundle


@st.cache_data(show_spinner="Evaluating FraudLens approaches...")
def _evaluate(loss_rate: float, review_cost: float) -> dict:
    df = load_transactions()
    bundle = model_bundle()
    test = E.test_split(df)
    results = E.compare_approaches(test, bundle, loss_rate=loss_rate, review_cost=review_cost)
    matched = E.matched_point_comparison(test, bundle, loss_rate=loss_rate, review_cost=review_cost)
    scores = E.approach_scores(test, bundle)
    return {
        "comparison": E.comparison_frame(results),
        "matched": matched,
        "test_rows": int(len(test)),
        "test_fraud": int(test["fraud_label"].sum()),
        "combined_scores": scores[E.APPROACH_COMBINED]["score"],
        "ml_scores": scores[E.APPROACH_ML]["score"],
        "rules_scores": scores[E.APPROACH_RULES]["score"],
        "combined_flags": scores[E.APPROACH_COMBINED]["flags"],
    }


def _confusion_figure(row: pd.Series) -> go.Figure:
    matrix = [[row["true_negatives"], row["false_positives"]], [row["false_negatives"], row["true_positives"]]]
    fig = px.imshow(
        matrix, text_auto=True, color_continuous_scale=theme.SEQUENTIAL,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
    )
    fig.update_layout(title=str(row["approach"]), coloraxis_showscale=False)
    return fig


def render() -> None:
    C.page_header(
        "Model Evaluation",
        "Rules baseline versus machine learning versus the combined engine, measured on the held-out "
        "chronological test split and compared at a matched operating point.",
        eyebrow="Module 6 · Risk evaluation",
    )
    C.synthetic_notice()

    bundle = model_bundle()
    if bundle is None:
        C.notice("No trained model is available, so only the rules baseline can be evaluated.", "warning", "⚠️")

    with st.expander("Economic assumptions used in the loss simulation", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            loss_rate = st.slider("Loss given undetected fraud (share of amount)", 0.1, 1.0,
                                  FRAUD_LOSS_RATE, 0.05, key="ev_loss")
        with c2:
            review_cost = st.slider("Cost of one manual review (USD)", 2.0, 60.0,
                                    MANUAL_REVIEW_COST_USD, 1.0, key="ev_cost")
        st.caption(
            "Both are modelling assumptions, not observed costs. They scale the fraud-loss and "
            "review-cost columns below."
        )

    data = _evaluate(loss_rate, review_cost)
    comparison = data["comparison"]
    matched = data["matched"]

    st.caption(
        f"Evaluated on the held-out test split: **{data['test_rows']:,} transactions** containing "
        f"**{data['test_fraud']} labelled fraud cases**. With this few positives, differences of a few "
        "cases move the metrics noticeably — treat small gaps as indicative."
    )

    C.section("Approach comparison at each system's own operating point")
    display = comparison[
        ["approach", "threshold", "precision", "recall", "f1", "pr_auc", "false_positive_rate",
         "false_negative_rate", "alerts", "review_volume_pct", "incurred_loss_usd",
         "review_cost_usd", "total_cost_usd", "net_benefit_usd"]
    ].rename(
        columns={
            "approach": "Approach", "threshold": "Threshold", "precision": "Precision",
            "recall": "Recall", "f1": "F1", "pr_auc": "PR-AUC",
            "false_positive_rate": "FPR", "false_negative_rate": "FNR", "alerts": "Alerts",
            "review_volume_pct": "Review volume %", "incurred_loss_usd": "Modelled loss (USD)",
            "review_cost_usd": "Review cost (USD)", "total_cost_usd": "Total cost (USD)",
            "net_benefit_usd": "Net benefit (USD)",
        }
    )
    C.dataframe(
        display,
        column_config={
            "Threshold": st.column_config.NumberColumn(format="%.3f"),
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.3f"),
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "FPR": st.column_config.NumberColumn(format="%.4f"),
            "FNR": st.column_config.NumberColumn(format="%.3f"),
            "Review volume %": st.column_config.NumberColumn(format="%.2f%%"),
            "Modelled loss (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Review cost (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Total cost (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Net benefit (USD)": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    cols = st.columns(len(comparison))
    for col, (_, row) in zip(cols, comparison.iterrows()):
        with col:
            C.chart(_confusion_figure(row), 260)

    left, right = st.columns(2)
    with left:
        melted = comparison.melt(id_vars="approach", value_vars=["precision", "recall", "f1"],
                                 var_name="metric", value_name="value")
        fig = px.bar(melted, x="metric", y="value", color="approach", barmode="group")
        fig.update_layout(title="Precision, recall and F1 by approach", xaxis_title="", yaxis_title="")
        C.chart(fig, 300)
    with right:
        fig = px.bar(comparison, x="approach", y=["incurred_loss_usd", "review_cost_usd"],
                     barmode="stack",
                     labels={"value": "USD", "variable": "Cost component"})
        fig.update_layout(title="Modelled cost: missed fraud loss + manual review", xaxis_title="")
        C.chart(fig, 300)

    # ---------------- Matched operating point ----------------
    C.section(
        "False-positive comparison at a matched operating point",
        "Each system is moved to the threshold where it matches the rules baseline's recall, so the "
        "comparison holds fraud-loss exposure roughly constant.",
    )
    baseline = matched["baseline"]
    cards = [
        {"label": "Baseline recall (rules only)", "value": f"{baseline['recall']:.3f}",
         "sub": f"{baseline['false_positives']} false positives at threshold {baseline['threshold']:.2f}",
         "tone": "navy"},
    ]
    for comp in matched["comparisons"]:
        reduction = comp["false_positive_reduction_pct"]
        cards.append(
            {
                "label": f"{comp['approach']} at matched recall",
                "value": f"{reduction:+.1f}%",
                "sub": (
                    f"False positives {baseline['false_positives']} → {comp['false_positives']} "
                    f"at recall {comp['recall']:.3f} (threshold {comp['matched_threshold']:.3f})"
                ),
                "tone": "success" if reduction > 0 else "danger",
            }
        )
    C.kpi_cards(cards, columns=len(cards))
    C.notice(
        "This is the calculated result on synthetic data, reported whatever its sign. A negative value "
        "would mean the system produced <em>more</em> false positives than the rules baseline at the "
        "same recall.", "modelled", "📐",
    )

    if matched["comparisons"]:
        comp_df = pd.DataFrame(matched["comparisons"])[
            ["approach", "matched_threshold", "precision", "recall", "false_positives",
             "true_positives", "alerts", "review_cost_usd", "incurred_loss_usd", "total_cost_usd",
             "false_positive_reduction_abs", "false_positive_reduction_pct"]
        ]
        C.dataframe(comp_df.round(4))

    # ---------------- Threshold explorer ----------------
    C.section("Detection threshold explorer", "Move the operating point and watch the trade-off move with it.")
    df = load_transactions()
    test = E.test_split(df)
    which = st.radio("Score to tune", ["Combined", "ML only", "Rules only"], horizontal=True, key="ev_which")
    score = {
        "Combined": data["combined_scores"], "ML only": data["ml_scores"], "Rules only": data["rules_scores"]
    }[which]
    sweep = E.threshold_sweep(test, score, loss_rate=loss_rate, review_cost=review_cost)

    chosen = st.slider("Threshold", float(sweep["threshold"].min()), float(sweep["threshold"].max()),
                       float(sweep["threshold"].median()), key="ev_thr")
    nearest = sweep.iloc[(sweep["threshold"] - chosen).abs().argmin()]
    C.kpi_cards(
        [
            {"label": "Precision", "value": f"{nearest['precision']:.3f}", "tone": "teal"},
            {"label": "Recall", "value": f"{nearest['recall']:.3f}", "tone": "teal"},
            {"label": "Alerts to review", "value": C.fmt_int(nearest["alerts"]),
             "sub": f"{nearest['review_volume_pct']:.2f}% of transactions", "tone": "warning"},
            {"label": "Modelled total cost", "value": C.fmt_usd(nearest["total_cost_usd"]),
             "sub": f"Net benefit {C.fmt_usd(nearest['net_benefit_usd'])}", "tone": "navy"},
        ],
        columns=4,
    )
    fig = go.Figure()
    fig.add_scatter(x=sweep["threshold"], y=sweep["precision"], name="Precision",
                    line=dict(color=theme.TEAL_500))
    fig.add_scatter(x=sweep["threshold"], y=sweep["recall"], name="Recall", line=dict(color=theme.NAVY_600))
    fig.add_scatter(x=sweep["threshold"], y=sweep["review_volume_pct"] / 100, name="Review volume (share)",
                    line=dict(color=theme.WARNING, dash="dot"))
    fig.add_vline(x=float(nearest["threshold"]), line_dash="dash", line_color=theme.DANGER)
    fig.update_layout(title=f"{which}: operating curve", xaxis_title="Threshold", yaxis_title="Rate")
    C.chart(fig, 320)

    left, right = st.columns(2)
    with left:
        fig = px.line(sweep, x="review_volume_pct", y="total_cost_usd", markers=False)
        fig.update_traces(line=dict(color=theme.NAVY_600, width=2))
        fig.update_layout(title="Modelled total cost vs manual review volume",
                          xaxis_title="Review volume (% of transactions)", yaxis_title="Total cost (USD)")
        C.chart(fig, 300)
    with right:
        dist = E.score_distribution(test, score)
        fig = px.bar(dist, x="bin_mid", y="count", color="label", barmode="overlay", log_y=True,
                     color_discrete_map={"Legitimate": theme.NAVY_200, "Fraudulent": theme.DANGER})
        fig.update_layout(title="Risk score distribution by true label", xaxis_title="Score",
                          yaxis_title="Transactions (log)")
        C.chart(fig, 300)

    C.section("Performance by corridor", "Combined engine at its configured operating point.")
    corridor = E.corridor_performance(test, data["combined_flags"])
    if corridor.empty:
        C.empty_state("Not enough transactions per corridor in the test split for a stable breakdown.")
    else:
        C.dataframe(
            corridor.rename(columns={
                "corridor": "Corridor", "transactions": "Transactions", "fraud_cases": "Fraud cases",
                "alerts": "Alerts", "precision": "Precision", "recall": "Recall",
                "false_positive_rate": "FPR",
            }),
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "FPR": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            "Corridors with very few fraud cases produce unstable precision and recall; read them as "
            "directional only."
        )

    C.section("Limitations of this evaluation")
    st.markdown(
        """
- **Synthetic generative process.** Fraud labels are sampled from a known logistic model driven by the
  same behavioural factors the detector observes. Real fraud is noisier, adversarial and
  non-stationary, so these figures are an optimistic upper bound, not a forecast of production
  performance.
- **Small positive class in the test window.** The chronological test split contains a limited number
  of labelled fraud cases; confidence intervals around precision and recall are wide.
- **No concept drift or adversarial adaptation.** Fraudsters respond to controls; this simulation has
  no feedback loop between detection and behaviour.
- **Label quality is perfect by construction.** Real investigations produce noisy, delayed and
  sometimes wrong labels.
- **Economics are modelled.** Loss-given-fraud and review cost are assumptions exposed above, not
  measured values.
- **Generalisation is unproven.** Nothing here establishes performance on real payment traffic. Shadow
  mode on real volume would be the next gate before any decisioning use.
"""
    )
    C.footer()
