"""Module 7 - FraudLens AI Copilot (grounded investigation assistant)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from paybridge.copilot import client as llm_client
from paybridge.copilot.copilot import (
    MODE_DETERMINISTIC,
    MODE_LLM,
    MODE_REJECTED,
    MODE_UNAVAILABLE,
    ask,
)
from paybridge.copilot.intents import SUGGESTED_QUESTIONS
from paybridge.fraud import investigations as I
from paybridge.ui import components as C
from paybridge.ui.services import background_sample, model_bundle, scored_transactions

MODE_LABEL = {
    MODE_LLM: ("Claude explanation · validated", "success"),
    MODE_REJECTED: ("Claude output rejected · deterministic answer shown", "danger"),
    MODE_UNAVAILABLE: ("Deterministic answer · LLM layer unavailable", "info"),
    MODE_DETERMINISTIC: ("Deterministic answer", "info"),
}


def _architecture_panel() -> None:
    st.markdown(
        """
<div class="pb-panel pb-kv">
<b>Grounding pipeline.</b>
question → intent detection → <b>deterministic Python analytics</b> → allowlisted evidence pack →
LLM explanation (optional) → <b>claim validation</b> → response.<br>
The language model never calculates a figure. Every number it is allowed to write must already
exist in the evidence pack; anything else is rejected and the verified deterministic answer is
shown instead. The copilot recommends, it never decides.
</div>
""",
        unsafe_allow_html=True,
    )


def _render_response(response) -> None:
    label, tone = MODE_LABEL.get(response.mode, ("Answer", "info"))
    st.markdown(
        f"{C.badge(label, tone)} &nbsp; {C.badge('intent: ' + response.intent, 'neutral')} "
        f"&nbsp; {C.badge(f'confidence {response.confidence:.2f}', 'neutral')}",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='pb-panel'>{response.answer}</div>", unsafe_allow_html=True)

    if response.llm_error:
        C.notice(
            f"{response.llm_error} The deterministic analytics layer answered instead, so the figures "
            "above are still verified.",
            "info", "ℹ️",
        )
    if response.was_rejected and response.validation is not None:
        C.notice(
            "<b>Generated text rejected by claim validation.</b><br>"
            + "<br>".join(f"• {r}" for r in response.validation.reasons),
            "danger", "🛑",
        )
        with st.expander("Show the rejected model output (not used)"):
            st.code(response.rejected_text or "", language="text")

    with st.expander("Evidence pack used for this answer", expanded=False):
        facts = response.evidence_summary()
        if not facts.empty:
            C.dataframe(facts)
        for name, table in response.evidence.tables.items():
            st.markdown(f"**{name}**")
            C.dataframe(table)
        if response.evidence.bullets:
            st.markdown("**Observations**")
            for b in response.evidence.bullets:
                st.markdown(f"- {b}")
        if response.evidence.transaction_ids:
            st.caption("Citable transaction IDs: " + ", ".join(sorted(response.evidence.transaction_ids)))
        if response.evidence.untrusted_notes:
            st.markdown("**Untrusted content included as data only**")
            for note in response.evidence.untrusted_notes:
                st.code(note, language="text")
            st.caption("Analyst notes are never executed as instructions by the copilot.")
        if response.evidence.warnings:
            st.caption(" · ".join(response.evidence.warnings))
        if response.validation is not None and response.validation.claims:
            st.caption(
                f"Claim validation checked {response.validation.checked_count} numeric claim(s); "
                f"{len(response.validation.unsupported)} unsupported."
            )


def render() -> None:
    C.page_header(
        "FraudLens AI Copilot",
        "Ask about risk, cost and behaviour. Figures are computed deterministically in Python; the "
        "language model may only explain them, and anything it invents is rejected before you see it.",
        eyebrow="Module 7 · Analyst assistance",
    )
    C.synthetic_notice()
    _architecture_panel()

    available = llm_client.llm_available()
    if available:
        C.notice(
            "Claude API key detected — answers are narrated by the model and then validated against "
            "the evidence pack.", "success", "🤖",
        )
    else:
        reason = llm_client.unavailable_reason()
        detail = (
            "the Anthropic SDK is not installed" if reason == llm_client.ERROR_NO_SDK
            else "no ANTHROPIC_API_KEY is configured"
        )
        C.notice(
            f"<b>Deterministic mode</b> — {detail}. Every question is still answered from verified "
            "analytics: metrics, triggered rules, model scores and investigation guidance. No "
            "fabricated AI output is ever shown.",
            "info", "🧮",
        )

    df = scored_transactions()
    bundle = model_bundle()

    with st.expander("Scope the copilot to a case (optional)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            txn_context = st.text_input("Transaction ID", key="cp_txn", placeholder="TXN-010947")
        with c2:
            biz_context = st.text_input("Business ID", key="cp_biz", placeholder="BIZ-0042")
        with c3:
            case_ids = I.list_cases()["case_id"].tolist()
            case_choice = st.selectbox("Attach case notes from", ["(none)"] + case_ids, key="cp_case")

    notes: list[str] = []
    if case_choice and case_choice != "(none)":
        note_rows = I.case_notes(case_choice)
        notes = note_rows["note"].astype(str).tolist()
        if notes:
            st.caption(
                f"{len(notes)} analyst note(s) from {case_choice} will be supplied as untrusted data."
            )

    st.markdown("##### Ask a question")
    cols = st.columns(4)
    for i, question in enumerate(SUGGESTED_QUESTIONS[:4]):
        with cols[i]:
            if st.button(question, key=f"cp_sug_{i}", use_container_width=True):
                st.session_state["cp_question"] = question
    cols2 = st.columns(3)
    for i, question in enumerate(SUGGESTED_QUESTIONS[4:7]):
        with cols2[i]:
            if st.button(question, key=f"cp_sug2_{i}", use_container_width=True):
                st.session_state["cp_question"] = question

    question = st.text_input(
        "Your question",
        value=st.session_state.get("cp_question", "What are the highest-risk transactions right now?"),
        key="cp_input",
    )
    run_col, mode_col = st.columns([1, 3])
    with run_col:
        submitted = st.button("Ask the copilot", type="primary", key="cp_ask")
    with mode_col:
        force_deterministic = st.checkbox(
            "Force deterministic mode (skip the LLM even when a key is present)",
            value=False, key="cp_force",
        )

    if submitted or st.session_state.get("cp_last_question") != question:
        st.session_state["cp_last_question"] = question
        with st.spinner("Computing evidence..."):
            response = ask(
                question=question,
                df=df,
                bundle=bundle,
                background=background_sample(200),
                context_transaction_id=txn_context.strip() or None,
                context_business_id=biz_context.strip() or None,
                notes=notes or None,
                use_llm=not force_deterministic,
            )
        history = st.session_state.setdefault("cp_history", [])
        history.insert(0, {"question": question, "mode": response.mode, "answer": response.answer})
        st.session_state["cp_history"] = history[:8]
        _render_response(response)

    if st.session_state.get("cp_history"):
        with st.expander("Recent questions in this session"):
            C.dataframe(pd.DataFrame(st.session_state["cp_history"]))

    C.section("Safety behaviour you can test here")
    st.markdown(
        """
- Ask about a transaction that does not exist (for example `Why was TXN-999999 flagged?`) — the copilot
  states that there is no such record rather than inventing one.
- Attach case notes containing instructions — they are fenced as untrusted data and never obeyed.
- Numeric claims are checked against the evidence pack; unsupported figures cause the generated text
  to be discarded.
- The copilot cannot approve, reject or escalate a case; only a named analyst can, on the
  Investigation Workbench.
"""
    )
    C.footer()
