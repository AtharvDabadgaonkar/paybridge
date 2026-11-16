"""Module 4 - FraudLens AI engine (rules + ML + explainability)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paybridge.fraud import investigations as I
from paybridge.fraud.explain import explain_transaction
from paybridge.fraud.rules import DEFAULT_FLAG_THRESHOLD, DEFAULT_WEIGHTS, RULES, evaluate_rules_frame
from paybridge.fraud.scoring import ML_WEIGHT, RULE_OVERRIDE, RULES_WEIGHT, assess_transaction
from paybridge.ui import components as C
from paybridge.ui import theme
from paybridge.ui.services import background_sample, model_bundle, scored_transactions


def _tab_engine(df: pd.DataFrame) -> None:
    bundle = model_bundle()
    st.markdown(
        """
<div class="pb-panel">
<b>Three-layer architecture.</b><br>
<b>Layer A — Rules.</b> 14 configurable, fully deterministic rules produce a bounded weighted score;
every point of that score is attributable to a named rule.<br>
<b>Layer B — Machine learning.</b> A supervised classifier trained on chronological splits using only
features observable at payment time.<br>
<b>Layer C — Explainability.</b> SHAP attributions for the model prediction, plus the triggered rules
and recommended investigation questions for the analyst.
</div>
""",
        unsafe_allow_html=True,
    )

    if bundle is None:
        C.notice("No trained model bundle is available — showing the rules layer only.", "warning", "⚠️")
    else:
        report = bundle.report
        test = next((r["test"] for r in report.results if r["selected"]), {})
        C.kpi_cards(
            [
                {"label": "Selected model", "value": bundle.model_name.replace("_", " ").title(),
                 "sub": report.selection_criterion, "tone": "navy"},
                {"label": "Decision threshold", "value": f"{bundle.threshold:.3f}",
                 "sub": f"Tuned on validation by {report.threshold_objective}", "tone": "teal"},
                {"label": "Test precision", "value": f"{test.get('precision', 0):.3f}",
                 "sub": f"Recall {test.get('recall', 0):.3f} · F1 {test.get('f1', 0):.3f}", "tone": "teal"},
                {"label": "Test PR-AUC", "value": f"{test.get('pr_auc', 0):.3f}",
                 "sub": f"FPR {test.get('false_positive_rate', 0):.4f} · {test.get('n', 0):,} test rows",
                 "tone": "teal"},
            ],
            columns=4,
        )
        st.caption(
            f"Chronological split — train {report.split_summary.get('train_rows', 0):,} rows "
            f"(to {report.split_summary.get('train_end', '')[:10]}), validation "
            f"{report.split_summary.get('validation_rows', 0):,}, test "
            f"{report.split_summary.get('test_rows', 0):,}. Metrics are measured on the held-out test split."
        )

    C.section("Combined decision policy")
    st.markdown(
        f"""
<div class="pb-panel pb-kv">
<b>combined_score</b> = {ML_WEIGHT:.2f} × model probability + {RULES_WEIGHT:.2f} × rules score<br>
<b>flagged</b> = model probability ≥ threshold <b>OR</b> rules score ≥ {RULE_OVERRIDE:.2f}<br>
<b>Human review</b> is mandatory for every <b>high</b> and <b>critical</b> outcome. FraudLens never
auto-approves a high-risk payout and never records an investigation decision by itself.
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])
    with left:
        counts = df["combined_band"].value_counts().reindex(["low", "medium", "high", "critical"]).fillna(0)
        fraud = df.groupby("combined_band")["fraud_label"].sum().reindex(counts.index).fillna(0)
        fig = go.Figure()
        fig.add_bar(x=counts.index, y=counts.values, name="Transactions",
                    marker_color=[theme.BAND_COLORS[b] for b in counts.index], opacity=0.85)
        fig.add_scatter(x=counts.index, y=fraud.values, name="Labelled fraud", mode="markers+lines",
                        marker=dict(color=theme.NAVY_800, size=9), yaxis="y2")
        fig.update_layout(title="Portfolio distribution by combined risk band", yaxis_type="log",
                          yaxis=dict(title="Transactions (log)"),
                          yaxis2=dict(title="Labelled fraud", overlaying="y", side="right", showgrid=False))
        C.chart(fig, 320)
    with right:
        if bundle is not None:
            from paybridge.fraud.explain import global_importance

            gi = global_importance(bundle, background_sample(300)).head(12)
            fig = px.bar(gi.sort_values("mean_abs_contribution"), x="mean_abs_contribution", y="feature",
                         orientation="h")
            fig.update_traces(marker_color=theme.TEAL_500)
            fig.update_layout(title=f"Global feature influence ({gi['method'].iloc[0]})",
                              xaxis_title="Mean |contribution|", yaxis_title="")
            C.chart(fig, 320)


def _tab_rules(df: pd.DataFrame) -> None:
    st.markdown("##### Rule library")
    st.caption(
        "Weights and the flag threshold are configurable. Adjusting them re-evaluates the rules layer "
        "on a live sample so the operational impact is visible immediately."
    )

    trigger_counts = {}
    fraud_counts = {}
    for rule in RULES:
        mask = df["rules_triggered"].fillna("").str.contains(rule.code, regex=False)
        trigger_counts[rule.code] = int(mask.sum())
        fraud_counts[rule.code] = int(df.loc[mask, "fraud_label"].sum())

    table = pd.DataFrame(
        [
            {
                "Code": r.code, "Rule": r.name, "Category": r.category,
                "Default weight": DEFAULT_WEIGHTS[r.code],
                "Triggered": trigger_counts[r.code],
                "Trigger rate %": trigger_counts[r.code] / len(df) * 100.0,
                "Labelled fraud in triggers": fraud_counts[r.code],
                "Hit precision %": (fraud_counts[r.code] / trigger_counts[r.code] * 100.0)
                if trigger_counts[r.code] else 0.0,
                "Description": r.description,
            }
            for r in RULES
        ]
    )
    C.dataframe(
        table,
        column_config={
            "Trigger rate %": st.column_config.NumberColumn(format="%.2f%%"),
            "Hit precision %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        height=430,
    )

    st.markdown("##### Tune the rules layer")
    sample_size = min(6000, len(df))
    sample = df.sample(sample_size, random_state=5)
    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        threshold = st.slider("Flag threshold (rules score)", 0.10, 0.95, DEFAULT_FLAG_THRESHOLD, 0.05,
                              key="rl_thr")
        overrides: dict[str, float] = {}
        with st.expander("Adjust individual rule weights", expanded=False):
            for rule in RULES:
                overrides[rule.code] = st.slider(
                    f"{rule.code} — {rule.name}", 0.0, 1.5, float(DEFAULT_WEIGHTS[rule.code]), 0.05,
                    key=f"w_{rule.code}",
                )
    recomputed = evaluate_rules_frame(sample, overrides or None, threshold)
    y = sample["fraud_label"].to_numpy()
    flags = recomputed["rules_flagged"].to_numpy()
    tp = int(((flags == 1) & (y == 1)).sum())
    fp = int(((flags == 1) & (y == 0)).sum())
    fn = int(((flags == 0) & (y == 1)).sum())

    with col_right:
        C.kpi_cards(
            [
                {"label": "Alert rate", "value": C.fmt_pct(flags.mean() * 100, 2),
                 "sub": f"{int(flags.sum()):,} alerts in a {sample_size:,}-transaction sample", "tone": "warning"},
                {"label": "Precision", "value": C.fmt_pct(tp / (tp + fp) * 100 if (tp + fp) else 0.0),
                 "sub": f"{tp} true / {fp} false positives", "tone": "teal"},
                {"label": "Recall", "value": C.fmt_pct(tp / (tp + fn) * 100 if (tp + fn) else 0.0),
                 "sub": f"{fn} fraudulent payouts missed", "tone": "navy"},
            ],
            columns=1,
        )
        st.caption("Measured live on the sample with your current weights — not a stored result.")


def _tab_score(df: pd.DataFrame) -> None:
    st.markdown("##### Score and explain a transaction")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        mode = st.radio("Pick a transaction", ["Highest combined risk", "Search by transaction ID"],
                        horizontal=True, key="fl_mode")
    if mode == "Search by transaction ID":
        with col2:
            txn_id = st.text_input("Transaction ID", value=str(df.iloc[0]["transaction_id"]), key="fl_txn")
        match = df[df["transaction_id"] == txn_id.strip()]
        if match.empty:
            C.empty_state(f"No transaction with ID `{txn_id}` in the synthetic dataset.")
            return
        row = match.iloc[0]
    else:
        top = df.nlargest(30, "combined_score")
        with col2:
            txn_id = st.selectbox(
                "Transaction", top["transaction_id"].tolist(),
                format_func=lambda t: f"{t} · {C.fmt_usd(float(top.loc[top.transaction_id == t, 'amount_usd'].iloc[0]))}",
                key="fl_top",
            )
        row = top[top["transaction_id"] == txn_id].iloc[0]

    bundle = model_bundle()
    assessment = assess_transaction(row, bundle, with_explanation=True, background=background_sample(300))

    C.kpi_cards(
        [
            {"label": "Combined risk score", "value": f"{assessment.combined_score:.3f}",
             "sub": f"Band {assessment.combined_band.upper()} · decision `{assessment.decision}`",
             "tone": {"low": "success", "medium": "teal", "high": "warning", "critical": "danger"}[assessment.combined_band]},
            {"label": "Rules score", "value": f"{assessment.rules.score:.3f}",
             "sub": f"{len(assessment.rules.triggered)} rule(s) triggered", "tone": "navy"},
            {"label": "Model probability",
             "value": f"{assessment.ml_probability:.3f}" if assessment.ml_probability is not None else "n/a",
             "sub": f"Threshold {assessment.ml_threshold:.3f}" if assessment.ml_threshold else "No model",
             "tone": "teal"},
            {"label": "Amount", "value": C.fmt_usd(float(row["amount_usd"]), 2),
             "sub": f"{row['corridor']} via {row['payment_rail']}", "tone": "navy"},
        ],
        columns=4,
    )

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("**Transaction detail**")
        C.key_values(
            [
                ("Transaction", row["transaction_id"]),
                ("Business", row["business_id"]),
                ("Initiated", str(row["transaction_timestamp"])[:19]),
                ("Beneficiary age", f"{float(row['beneficiary_age_days']):.1f} days"),
                ("Account age", f"{int(row['account_age_days'])} days"),
                ("24h velocity", int(row["transaction_velocity"])),
                ("Amount vs own 30d mean", f"{float(row['amount_vs_business_mean_ratio']):.2f}×"),
                ("Recent failures (7d)", int(row["recent_failed_count"])),
                ("Status", str(row["transaction_status"]).replace("_", " ").title()),
            ]
        )
        st.markdown("**Triggered rules (Layer A — deterministic)**")
        if assessment.rules.triggered:
            for t in assessment.rules.triggered:
                st.markdown(
                    f"{C.badge(f'w {t.weight:.2f}', 'warning')} **{t.name}** — {t.description}",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No deterministic rule fired for this transaction.")

    with right:
        exp = assessment.explanation
        st.markdown("**Model explanation (Layer C)**")
        if exp is None:
            C.notice("No model available, so no model-level explanation can be produced.", "warning", "⚠️")
        else:
            frame = exp.as_frame(8)
            fig = px.bar(
                frame.sort_values("contribution"), x="contribution", y="feature", orientation="h",
                color=frame.sort_values("contribution")["contribution"] > 0,
                color_discrete_map={True: theme.DANGER, False: theme.SUCCESS},
            )
            fig.update_layout(title=f"Top drivers — method: {exp.method}", showlegend=False,
                              xaxis_title="Contribution to risk", yaxis_title="")
            C.chart(fig, 320)
            if exp.is_shap:
                st.caption(
                    f"SHAP values from the trained model (base value {exp.base_value:.3f}). "
                    "Positive bars push this transaction towards fraud."
                )
            else:
                C.notice(exp.note, "warning", "ℹ️")

    st.markdown("**Recommended investigation questions**")
    questions = assessment.investigation_questions()
    if questions:
        for q in questions:
            st.markdown(f"- {q}")
    else:
        st.caption("No rule-derived questions: this transaction shows no deterministic red flag.")

    if assessment.requires_human_review:
        existing = I.case_for_transaction(str(row["transaction_id"]))
        if existing:
            C.notice(f"Investigation case <b>{existing['case_id']}</b> already exists for this transaction.",
                     "info", "📁")
        elif st.button("Open an investigation case", type="primary", key="fl_open_case"):
            case_id = I.create_case(
                transaction_id=str(row["transaction_id"]),
                business_id=str(row["business_id"]),
                amount_usd=float(row["amount_usd"]),
                risk_score=assessment.combined_score,
                risk_band=assessment.combined_band,
                rules_triggered=",".join(assessment.rules.codes),
                ml_score=assessment.ml_probability,
            )
            st.success(f"Opened case {case_id}. Work it on the **Investigation Workbench** page.")


def _tab_distribution(df: pd.DataFrame) -> None:
    st.markdown("##### Risk score distributions by true label")
    score_col = st.radio("Score", ["combined_score", "rules_score", "ml_probability"], horizontal=True,
                         key="fl_dist")
    if df[score_col].isna().all():
        C.empty_state("That score is unavailable (no trained model loaded).")
        return
    plot = df[[score_col, "fraud_label"]].dropna().copy()
    plot["Label"] = plot["fraud_label"].map({0: "Legitimate", 1: "Fraudulent"})
    fig = px.histogram(plot, x=score_col, color="Label", nbins=50, barmode="overlay",
                       histnorm="probability density",
                       color_discrete_map={"Legitimate": theme.NAVY_200, "Fraudulent": theme.DANGER})
    fig.update_layout(title=f"{score_col} density by label", xaxis_title=score_col, yaxis_title="Density")
    C.chart(fig, 340)

    stats = (
        plot.groupby("Label")[score_col]
        .agg(["count", "mean", "median", lambda s: s.quantile(0.9)])
        .rename(columns={"<lambda_0>": "p90"})
        .reset_index()
    )
    C.dataframe(stats.round(4))


def render() -> None:
    C.page_header(
        "FraudLens AI",
        "Explainable fraud detection in three layers: deterministic rules, a supervised model trained "
        "with chronological splits, and SHAP-based explanations for every scored payment.",
        eyebrow="Module 4 · Risk intelligence",
    )
    C.synthetic_notice()

    df = scored_transactions()
    tabs = st.tabs(["Engine overview", "Rule library", "Score a transaction", "Score distributions"])
    with tabs[0]:
        _tab_engine(df)
    with tabs[1]:
        _tab_rules(df)
    with tabs[2]:
        _tab_score(df)
    with tabs[3]:
        _tab_distribution(df)
    C.footer()
