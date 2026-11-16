"""Deterministic analytics behind every copilot answer.

Each builder takes the scored transaction frame and returns an
:class:`EvidencePack` containing only values computed here with Pandas/NumPy.
Each pack also carries a complete deterministic answer, which is what the user
sees whenever the LLM is unavailable or its output fails validation.
"""
from __future__ import annotations

import pandas as pd

from paybridge.analytics import kpis
from paybridge.copilot import intents as I
from paybridge.copilot.evidence import EvidencePack, wrap_untrusted
from paybridge.fraud.explain import explain_transaction
from paybridge.fraud.rules import evaluate_rules
from paybridge.fraud.scoring import assess_transaction

TOP_N = 10


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}"


def _base_pack(intent: str, question: str) -> EvidencePack:
    pack = EvidencePack(intent=intent, question=question)
    pack.warnings.append(
        "All figures come from a synthetic dataset; they are not real financial results."
    )
    return pack


# --------------------------------------------------------------------------
def explain_transaction_pack(
    question: str, transaction_id: str | None, df: pd.DataFrame, bundle=None, background=None
) -> EvidencePack:
    pack = _base_pack(I.EXPLAIN_TRANSACTION, question)
    if not transaction_id:
        pack.deterministic_answer = (
            "I need a transaction reference to explain a specific decision. Quote an identifier "
            "such as TXN-012345, or open the transaction on the FraudLens page and ask again."
        )
        pack.warnings.append("No transaction identifier supplied.")
        return pack

    rows = df[df["transaction_id"].str.upper() == transaction_id.upper()]
    if rows.empty:
        pack.deterministic_answer = (
            f"There is no transaction {transaction_id} in the loaded synthetic dataset, so I cannot "
            "explain it. Check the identifier and try again."
        )
        pack.warnings.append(f"Unknown transaction identifier {transaction_id}.")
        return pack

    row = rows.iloc[0]
    assessment = assess_transaction(row, bundle, with_explanation=bundle is not None, background=background)
    pack.transaction_ids.add(str(row["transaction_id"]))
    pack.business_ids.add(str(row["business_id"]))

    pack.add_fact("amount_usd", "Payout amount", float(row["amount_usd"]), "usd")
    pack.add_fact("combined_score", "Combined risk score", round(assessment.combined_score, 4))
    pack.add_fact("rules_score", "Rules score", round(assessment.rules.score, 4))
    if assessment.ml_probability is not None:
        pack.add_fact("ml_probability", "Model fraud probability", round(assessment.ml_probability, 4))
        pack.add_fact("ml_threshold", "Model decision threshold", round(assessment.ml_threshold or 0.0, 4))
    pack.add_fact("velocity", "Payouts in the previous 24 hours", int(row["transaction_velocity"]), "count")
    pack.add_fact("beneficiary_age_days", "Beneficiary age in days", round(float(row["beneficiary_age_days"]), 2))
    pack.add_fact("amount_ratio", "Amount versus the customer's 30-day average",
                  round(float(row["amount_vs_business_mean_ratio"]), 2))
    pack.add_fact("account_age_days", "Business account age in days", int(row["account_age_days"]), "count")
    pack.add_fact("recent_failed", "Failed payouts in the last 7 days", int(row["recent_failed_count"]), "count")

    if assessment.rules.triggered:
        pack.add_table(
            "triggered_rules",
            pd.DataFrame(
                [
                    {"code": t.code, "rule": t.name, "weight": t.weight, "why": t.description}
                    for t in assessment.rules.triggered
                ]
            ),
        )
        rule_text = "; ".join(f"{t.name} (weight {t.weight:.2f})" for t in assessment.rules.triggered)
    else:
        rule_text = "no deterministic rule fired"

    driver_text = ""
    if assessment.explanation is not None:
        top = assessment.explanation.top(5)
        pack.add_table(
            "model_drivers",
            pd.DataFrame(
                [
                    {"feature": c.feature, "contribution": round(c.contribution, 5), "direction": c.direction}
                    for c in top
                ]
            ),
        )
        pack.bullets.append(f"Model explanation method: {assessment.explanation.method}.")
        driver_text = ", ".join(f"{c.feature} ({c.direction})" for c in top[:3])

    pack.bullets.append(
        f"Operational decision under the configured policy: {assessment.decision.replace('_', ' ')}."
    )
    if assessment.requires_human_review:
        pack.bullets.append("This outcome requires a human decision; FraudLens cannot approve it.")

    ml_sentence = (
        f"The model scored it {assessment.ml_probability:.3f} against a {assessment.ml_threshold:.3f} threshold"
        if assessment.ml_probability is not None
        else "No trained model was loaded, so this is a rules-only assessment"
    )
    pack.deterministic_answer = (
        f"{row['transaction_id']} is a {_fmt_usd(float(row['amount_usd']))} payout on {row['corridor']} "
        f"via {row['payment_rail']}, scored {assessment.combined_score:.3f} ({assessment.combined_band}). "
        f"Rules layer: {rule_text}. {ml_sentence}. "
        + (f"Leading model drivers: {driver_text}. " if driver_text else "")
        + f"Recommended next checks: {' '.join(assessment.investigation_questions()[:2])}"
    ).strip()
    return pack


# --------------------------------------------------------------------------
def top_risk_pack(question: str, df: pd.DataFrame, n: int = TOP_N) -> EvidencePack:
    pack = _base_pack(I.TOP_RISK, question)
    score_col = "combined_score" if "combined_score" in df.columns else "rules_score"
    top = df.nlargest(n, score_col)
    pack.transaction_ids.update(top["transaction_id"].astype(str))
    pack.business_ids.update(top["business_id"].astype(str))

    pack.add_fact("n_reviewed", "Transactions evaluated", int(len(df)), "count")
    pack.add_fact("flagged", "Transactions currently flagged",
                  int(df.get("combined_flagged", df["rules_flagged"]).sum()), "count")
    pack.add_fact("top_score", "Highest risk score", round(float(top[score_col].iloc[0]), 4))
    pack.add_fact("top_exposure", "Combined amount of the listed payouts", float(top["amount_usd"].sum()), "usd")

    pack.add_table(
        "top_risk",
        top[["transaction_id", "business_id", "amount_usd", "corridor", "payment_rail",
             score_col, "combined_band" if "combined_band" in top.columns else "rules_band",
             "transaction_status"]].round(4),
        max_rows=n,
    )
    lead = top.iloc[0]
    pack.deterministic_answer = (
        f"The {n} highest-scoring payouts carry {_fmt_usd(float(top['amount_usd'].sum()))} of exposure. "
        f"The top case is {lead['transaction_id']} — {_fmt_usd(float(lead['amount_usd']))} on "
        f"{lead['corridor']} via {lead['payment_rail']}, score {float(lead[score_col]):.3f}. "
        f"Work the queue in score order; {int(df.get('combined_flagged', df['rules_flagged']).sum()):,} "
        f"of {len(df):,} evaluated payouts are currently flagged."
    )
    return pack


# --------------------------------------------------------------------------
def behaviour_change_pack(question: str, df: pd.DataFrame, days: int = 7) -> EvidencePack:
    pack = _base_pack(I.BEHAVIOUR_CHANGE, question)
    end = df["transaction_timestamp"].max()
    current_start = end - pd.Timedelta(days=days)
    prior_start = end - pd.Timedelta(days=days * 2)

    current = df[df["transaction_timestamp"] > current_start]
    prior = df[(df["transaction_timestamp"] > prior_start) & (df["transaction_timestamp"] <= current_start)]
    if current.empty or prior.empty:
        pack.deterministic_answer = "There is not enough history in the current selection to compare two periods."
        pack.warnings.append("Insufficient data for a period-over-period comparison.")
        return pack

    flag_col = "combined_flagged" if "combined_flagged" in df.columns else "rules_flagged"
    metrics = {
        "Transactions": (len(current), len(prior), "count"),
        "Volume": (float(current["amount_usd"].sum()), float(prior["amount_usd"].sum()), "usd"),
        "Average amount": (float(current["amount_usd"].mean()), float(prior["amount_usd"].mean()), "usd"),
        "Alert rate": (float(current[flag_col].mean() * 100), float(prior[flag_col].mean() * 100), "pct"),
        "Success rate": (float(current["is_successful"].mean() * 100), float(prior["is_successful"].mean() * 100), "pct"),
    }
    rows = []
    for label, (now, before, unit) in metrics.items():
        change = (now - before) / before * 100 if before else 0.0
        rows.append({"metric": label, "current_period": round(now, 4),
                     "prior_period": round(before, 4), "change_pct": round(change, 2)})
        pack.add_fact(f"{label.lower().replace(' ', '_')}_current", f"{label} (last {days} days)", round(now, 4), unit)
        pack.add_fact(f"{label.lower().replace(' ', '_')}_prior", f"{label} (previous {days} days)", round(before, 4), unit)
        pack.add_fact(f"{label.lower().replace(' ', '_')}_change", f"{label} change", round(change, 2), "pct")

    pack.add_table("period_comparison", pd.DataFrame(rows))
    movers = (
        current.groupby("business_id")["amount_usd"].sum()
        - prior.groupby("business_id")["amount_usd"].sum().reindex(
            current.groupby("business_id")["amount_usd"].sum().index, fill_value=0.0
        )
    ).nlargest(5)
    if len(movers):
        pack.add_table("largest_volume_increases",
                       movers.reset_index().rename(columns={"amount_usd": "volume_increase_usd"}).round(2))
        pack.business_ids.update(movers.index.astype(str))

    alert_change = rows[3]["change_pct"]
    pack.deterministic_answer = (
        f"Comparing the last {days} days with the {days} before: transactions moved from "
        f"{len(prior):,} to {len(current):,} ({rows[0]['change_pct']:+.1f}%), volume from "
        f"{_fmt_usd(rows[1]['prior_period'])} to {_fmt_usd(rows[1]['current_period'])} "
        f"({rows[1]['change_pct']:+.1f}%), and the alert rate from {rows[3]['prior_period']:.2f}% to "
        f"{rows[3]['current_period']:.2f}% ({alert_change:+.1f}%). Success rate moved "
        f"{rows[4]['change_pct']:+.1f}% to {rows[4]['current_period']:.2f}%."
    )
    return pack


# --------------------------------------------------------------------------
def feature_contribution_pack(
    question: str, df: pd.DataFrame, bundle=None, transaction_id: str | None = None, background=None
) -> EvidencePack:
    pack = _base_pack(I.FEATURE_CONTRIBUTION, question)
    if bundle is None:
        pack.deterministic_answer = (
            "No trained model is loaded, so no model-level feature attribution is available. The rules "
            "layer remains fully explainable: each triggered rule contributes its own weight to the score."
        )
        pack.warnings.append("No model bundle available.")
        return pack

    if transaction_id:
        rows = df[df["transaction_id"].str.upper() == transaction_id.upper()]
        if not rows.empty:
            exp = explain_transaction(bundle, rows.head(1), background=background)
            pack.transaction_ids.add(str(rows.iloc[0]["transaction_id"]))
            table = pd.DataFrame([c.as_dict() for c in exp.top(8)]).round(5)
            pack.add_table("transaction_drivers", table)
            pack.bullets.append(f"Explanation method: {exp.method}.")
            if not exp.is_shap:
                pack.warnings.append(exp.note)
            lead = exp.top(3)
            pack.deterministic_answer = (
                f"For {rows.iloc[0]['transaction_id']} the strongest model drivers are "
                + ", ".join(f"{c.feature} ({c.direction})" for c in lead)
                + f". Attribution method: {exp.method}."
            )
            return pack

    from paybridge.fraud.explain import global_importance

    gi = global_importance(bundle, df.sample(min(300, len(df)), random_state=4)).head(10)
    pack.add_table("global_feature_influence", gi.round(5))
    pack.add_fact("model_name", "Model", bundle.model_name)
    pack.bullets.append(f"Attribution method: {gi['method'].iloc[0]}.")
    pack.deterministic_answer = (
        "Across the portfolio the features with the largest average influence on the model are: "
        + ", ".join(f"{r.feature}" for r in gi.head(5).itertuples())
        + f". Attribution method: {gi['method'].iloc[0]}."
    )
    return pack


# --------------------------------------------------------------------------
def transaction_history_pack(
    question: str, df: pd.DataFrame, business_id: str | None, transaction_id: str | None = None
) -> EvidencePack:
    pack = _base_pack(I.TRANSACTION_HISTORY, question)
    if not business_id and transaction_id:
        rows = df[df["transaction_id"].str.upper() == transaction_id.upper()]
        if not rows.empty:
            business_id = str(rows.iloc[0]["business_id"])
    if not business_id:
        pack.deterministic_answer = (
            "Tell me which business to look at (for example BIZ-0042) or quote a transaction "
            "identifier and I will use the business behind it."
        )
        pack.warnings.append("No business identifier supplied.")
        return pack

    history = df[df["business_id"].str.upper() == business_id.upper()].sort_values("transaction_timestamp")
    if history.empty:
        pack.deterministic_answer = f"There is no activity for {business_id} in the loaded dataset."
        pack.warnings.append(f"Unknown business identifier {business_id}.")
        return pack

    flag_col = "combined_flagged" if "combined_flagged" in history.columns else "rules_flagged"
    pack.business_ids.add(business_id.upper())
    pack.transaction_ids.update(history.tail(8)["transaction_id"].astype(str))

    pack.add_fact("txn_count", "Payouts in the window", int(len(history)), "count")
    pack.add_fact("total_volume", "Total payout volume", float(history["amount_usd"].sum()), "usd")
    pack.add_fact("avg_amount", "Average payout", float(history["amount_usd"].mean()), "usd")
    pack.add_fact("max_amount", "Largest payout", float(history["amount_usd"].max()), "usd")
    pack.add_fact("alert_rate", "Alert rate", float(history[flag_col].mean() * 100), "pct")
    pack.add_fact("success_rate", "Success rate", float(history["is_successful"].mean() * 100), "pct")
    pack.add_fact("distinct_beneficiaries", "Distinct beneficiaries",
                  int(history["beneficiary_id"].nunique()), "count")
    pack.add_fact("corridors_used", "Corridors used", int(history["corridor"].nunique()), "count")

    pack.add_table(
        "recent_payouts",
        history.tail(8)[["transaction_id", "transaction_timestamp", "amount_usd", "corridor",
                         "payment_rail", "transaction_status"]].round(2),
    )
    pack.deterministic_answer = (
        f"{business_id.upper()} made {len(history):,} payouts totalling "
        f"{_fmt_usd(float(history['amount_usd'].sum()))} in the window, averaging "
        f"{_fmt_usd(float(history['amount_usd'].mean()))} with a largest payout of "
        f"{_fmt_usd(float(history['amount_usd'].max()))}. It used {history['corridor'].nunique()} corridor(s) "
        f"and {history['beneficiary_id'].nunique()} beneficiaries, with a {history[flag_col].mean() * 100:.2f}% "
        f"alert rate and a {history['is_successful'].mean() * 100:.2f}% success rate."
    )
    return pack


# --------------------------------------------------------------------------
def next_steps_pack(
    question: str, df: pd.DataFrame, transaction_id: str | None = None, bundle=None, notes: list[str] | None = None
) -> EvidencePack:
    pack = _base_pack(I.NEXT_STEPS, question)
    if notes:
        pack.untrusted_notes.extend(wrap_untrusted(n) for n in notes[:5])

    if transaction_id:
        rows = df[df["transaction_id"].str.upper() == transaction_id.upper()]
        if rows.empty:
            pack.deterministic_answer = f"There is no transaction {transaction_id} in the dataset."
            pack.warnings.append(f"Unknown transaction identifier {transaction_id}.")
            return pack
        row = rows.iloc[0]
        assessment = assess_transaction(row, bundle)
        pack.transaction_ids.add(str(row["transaction_id"]))
        pack.add_fact("combined_score", "Combined risk score", round(assessment.combined_score, 4))
        pack.add_fact("amount_usd", "Payout amount", float(row["amount_usd"]), "usd")
        questions = assessment.investigation_questions()
        pack.bullets.extend(questions)
        pack.deterministic_answer = (
            f"For {row['transaction_id']} (score {assessment.combined_score:.3f}, "
            f"{assessment.combined_band}): "
            + " ".join(f"{i + 1}) {q}" for i, q in enumerate(questions[:4]))
            + " A human analyst must record the decision; FraudLens only recommends."
        )
        return pack

    score_col = "combined_score" if "combined_score" in df.columns else "rules_score"
    queue = df.nlargest(5, score_col)
    pack.transaction_ids.update(queue["transaction_id"].astype(str))
    pack.add_fact("queue_exposure", "Exposure in the top five cases", float(queue["amount_usd"].sum()), "usd")
    pack.add_table("priority_queue",
                   queue[["transaction_id", "amount_usd", score_col, "corridor", "payment_rail"]].round(4))
    rule_counts = (
        df[df.get("combined_flagged", df["rules_flagged"])]["rules_triggered"]
        .fillna("").str.split(",").explode().replace("", pd.NA).dropna().value_counts().head(3)
    )
    if len(rule_counts):
        pack.add_table("most_common_triggers",
                       rule_counts.reset_index().set_axis(["rule_code", "alerts"], axis=1))
    pack.deterministic_answer = (
        f"Start with the five highest-scoring open payouts, carrying "
        f"{_fmt_usd(float(queue['amount_usd'].sum()))} of exposure, beginning with "
        f"{queue.iloc[0]['transaction_id']}. The most common triggers in the current alert population are "
        + (", ".join(f"{idx} ({val} alerts)" for idx, val in rule_counts.items()) if len(rule_counts) else "none")
        + ". Record every decision against a named analyst."
    )
    return pack


# --------------------------------------------------------------------------
def portfolio_kpi_pack(question: str, df: pd.DataFrame) -> EvidencePack:
    pack = _base_pack(I.PORTFOLIO_KPI, question)
    k = kpis.executive_kpis(df)
    units = {
        "total_volume_usd": ("Total payment volume", "usd"),
        "transaction_count": ("Transactions", "count"),
        "successful_transactions": ("Completed transactions", "count"),
        "success_rate_pct": ("Payment success rate", "pct"),
        "avg_transaction_fee_usd": ("Average transaction fee", "usd"),
        "avg_settlement_hours": ("Average settlement time in hours", ""),
        "modelled_savings_usd": ("Modelled cost saving versus an all-SWIFT mix", "usd"),
        "modelled_savings_pct": ("Modelled cost saving percentage", "pct"),
        "fraud_alert_rate_pct": ("Fraud alert rate", "pct"),
        "transactions_under_review": ("Transactions under review", "count"),
        "contribution_margin_usd": ("Contribution margin", "usd"),
        "confirmed_fraud_rate_pct": ("Labelled fraud rate", "pct"),
    }
    for key, (label, unit) in units.items():
        pack.add_fact(key, label, round(float(k[key]), 4), unit)

    pack.add_table("rail_mix", kpis.rail_distribution(df).round(3))
    pack.add_table("settlement_by_rail", kpis.settlement_comparison(df).round(2))
    pack.warnings.append(
        "The cost saving is a modelled counterfactual (every transaction repriced on SWIFT), not realised savings."
    )
    pack.deterministic_answer = (
        f"Across {k['transaction_count']:,} synthetic transactions worth "
        f"{_fmt_usd(k['total_volume_usd'])}, the payment success rate is {k['success_rate_pct']:.2f}% and the "
        f"average transaction fee is {_fmt_usd(k['avg_transaction_fee_usd'])} "
        f"({k['avg_cost_pct_of_amount']:.2f}% of amount). Average settlement time is "
        f"{k['avg_settlement_hours']:.1f} hours. The modelled saving versus an all-SWIFT mix is "
        f"{_fmt_usd(k['modelled_savings_usd'])} ({k['modelled_savings_pct']:.2f}%). "
        f"{k['fraud_alert_rate_pct']:.2f}% of payouts raise a rules alert and "
        f"{k['transactions_under_review']:,} are held under review."
    )
    return pack


def unknown_pack(question: str) -> EvidencePack:
    pack = _base_pack(I.UNKNOWN, question)
    pack.deterministic_answer = (
        "I did not recognise that as a question I can answer from the computed evidence. I can explain "
        "why a specific payment was flagged, rank the highest-risk transactions, compare this week with "
        "last week, break down which features drove a score, summarise a customer's payout history, "
        "suggest what to investigate next, or report portfolio KPIs."
    )
    pack.warnings.append("Intent not recognised; no analytics were run.")
    return pack


def build_evidence(
    question: str,
    df: pd.DataFrame,
    bundle=None,
    background=None,
    context_transaction_id: str | None = None,
    context_business_id: str | None = None,
    notes: list[str] | None = None,
) -> tuple[EvidencePack, I.Intent]:
    """Route a question to its deterministic analytics builder."""
    intent = I.detect_intent(question)
    txn = intent.primary_transaction or context_transaction_id
    biz = intent.primary_business or context_business_id

    if intent.name == I.EXPLAIN_TRANSACTION:
        pack = explain_transaction_pack(question, txn, df, bundle, background)
    elif intent.name == I.TOP_RISK:
        pack = top_risk_pack(question, df)
    elif intent.name == I.BEHAVIOUR_CHANGE:
        pack = behaviour_change_pack(question, df)
    elif intent.name == I.FEATURE_CONTRIBUTION:
        pack = feature_contribution_pack(question, df, bundle, txn, background)
    elif intent.name == I.TRANSACTION_HISTORY:
        pack = transaction_history_pack(question, df, biz, txn)
    elif intent.name == I.NEXT_STEPS:
        pack = next_steps_pack(question, df, txn, bundle, notes)
    elif intent.name == I.PORTFOLIO_KPI:
        pack = portfolio_kpi_pack(question, df)
    else:
        pack = unknown_pack(question)
    return pack, intent
