"""Copilot orchestrator.

    question -> intent -> deterministic analytics -> evidence pack
             -> (optional) LLM explanation -> claim validation -> response

The deterministic answer is computed for every question, so the copilot
degrades cleanly: no API key, an API failure or a failed validation all end in
a useful, verified answer rather than nothing or a fabrication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from paybridge.copilot import client as llm_client
from paybridge.copilot.analytics import build_evidence
from paybridge.copilot.evidence import EvidencePack
from paybridge.copilot.intents import Intent
from paybridge.copilot.prompts import SYSTEM_PROMPT, build_user_message
from paybridge.copilot.validator import ValidationResult, validate_response

MODE_LLM = "llm_verified"
MODE_DETERMINISTIC = "deterministic"
MODE_REJECTED = "deterministic_after_rejection"
MODE_UNAVAILABLE = "deterministic_no_llm"


@dataclass
class CopilotResponse:
    answer: str
    mode: str
    intent: str
    confidence: float
    evidence: EvidencePack
    validation: ValidationResult | None = None
    llm_error: str = ""
    rejected_text: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def used_llm(self) -> bool:
        return self.mode == MODE_LLM

    @property
    def was_rejected(self) -> bool:
        return self.mode == MODE_REJECTED

    def evidence_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"fact": f.label, "value": f.value, "unit": f.unit, "source": f.source}
                for f in self.evidence.facts
            ]
        )


def ask(
    question: str,
    df: pd.DataFrame,
    bundle=None,
    background: pd.DataFrame | None = None,
    context_transaction_id: str | None = None,
    context_business_id: str | None = None,
    notes: list[str] | None = None,
    use_llm: bool = True,
    completion_fn: Callable[..., llm_client.LLMResult] | None = None,
) -> CopilotResponse:
    """Answer a question with grounded evidence, optionally narrated by Claude."""
    pack, intent = build_evidence(
        question=question,
        df=df,
        bundle=bundle,
        background=background,
        context_transaction_id=context_transaction_id,
        context_business_id=context_business_id,
        notes=notes,
    )
    deterministic = pack.deterministic_answer or (
        "No evidence could be computed for that question."
    )

    if not use_llm:
        return CopilotResponse(deterministic, MODE_DETERMINISTIC, intent.name, intent.confidence, pack)

    complete = completion_fn or llm_client.complete
    if completion_fn is None and not llm_client.llm_available():
        reason = llm_client.unavailable_reason() or ""
        result = llm_client.LLMResult(False, error_type=reason)
        return CopilotResponse(
            deterministic, MODE_UNAVAILABLE, intent.name, intent.confidence, pack,
            llm_error=result.user_message,
        )

    if not pack.has_evidence():
        return CopilotResponse(
            deterministic, MODE_DETERMINISTIC, intent.name, intent.confidence, pack,
            notes=["No evidence was available, so the LLM layer was skipped."],
        )

    result = complete(SYSTEM_PROMPT, build_user_message(pack.to_prompt_block()))
    if not result.ok:
        return CopilotResponse(
            deterministic, MODE_UNAVAILABLE, intent.name, intent.confidence, pack,
            llm_error=result.user_message,
        )

    validation = validate_response(result.text, pack)
    if not validation.ok:
        return CopilotResponse(
            deterministic, MODE_REJECTED, intent.name, intent.confidence, pack,
            validation=validation, rejected_text=result.text,
            notes=["The generated explanation failed claim validation and was not shown."],
        )

    return CopilotResponse(result.text, MODE_LLM, intent.name, intent.confidence, pack, validation=validation)
