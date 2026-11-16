"""Deterministic intent detection for the copilot.

Keyword/pattern scoring - no model call. Intent detection must be
reproducible so the analytics layer behind each answer is predictable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EXPLAIN_TRANSACTION = "explain_transaction"
TOP_RISK = "top_risk_transactions"
BEHAVIOUR_CHANGE = "behaviour_change"
FEATURE_CONTRIBUTION = "feature_contribution"
TRANSACTION_HISTORY = "transaction_history"
NEXT_STEPS = "next_steps"
PORTFOLIO_KPI = "portfolio_kpi"
UNKNOWN = "unknown"

TXN_PATTERN = re.compile(r"\bTXN-\d{4,8}\b", re.IGNORECASE)
BIZ_PATTERN = re.compile(r"\bBIZ-\d{3,6}\b", re.IGNORECASE)

_KEYWORDS: dict[str, tuple[tuple[str, float], ...]] = {
    EXPLAIN_TRANSACTION: (
        ("why was", 3.0), ("why is", 3.0), ("why did", 2.5), ("flagged", 2.5),
        ("suspicious", 1.5), ("explain", 2.0), ("reason", 1.2), ("risk score", 1.0),
    ),
    TOP_RISK: (
        ("highest-risk", 3.0), ("highest risk", 3.0), ("riskiest", 3.0), ("top risk", 3.0),
        ("worst", 1.5), ("biggest risk", 2.5), ("most risky", 2.5), ("priorit", 1.5),
    ),
    BEHAVIOUR_CHANGE: (
        ("what changed", 3.0), ("changed in", 2.0), ("this week", 2.0), ("last week", 2.0),
        ("trend", 1.5), ("recent change", 2.5), ("shift", 1.2), ("compared with last", 2.0),
    ),
    FEATURE_CONTRIBUTION: (
        ("which features", 3.0), ("what features", 3.0), ("contribut", 2.0), ("drivers", 2.0),
        ("shap", 2.5), ("influenc", 1.5), ("important", 1.2),
    ),
    TRANSACTION_HISTORY: (
        ("history", 2.5), ("past transactions", 2.5), ("previous payouts", 2.5),
        ("behaviour of", 1.5), ("track record", 2.0), ("prior activity", 2.0),
    ),
    NEXT_STEPS: (
        ("what should", 3.0), ("next step", 3.0), ("investigate next", 3.0), ("recommend", 2.0),
        ("what do i do", 2.5), ("how should", 2.0), ("action", 1.2),
    ),
    PORTFOLIO_KPI: (
        ("success rate", 2.5), ("total volume", 2.5), ("how much", 1.5), ("average fee", 2.5),
        ("settlement time", 2.0), ("saving", 2.0), ("kpi", 2.0), ("overall", 1.2),
        ("portfolio", 1.5), ("how many transactions", 2.5), ("fraud rate", 2.0),
    ),
}


@dataclass(frozen=True)
class Intent:
    name: str
    confidence: float
    transaction_ids: tuple[str, ...] = ()
    business_ids: tuple[str, ...] = ()

    @property
    def primary_transaction(self) -> str | None:
        return self.transaction_ids[0] if self.transaction_ids else None

    @property
    def primary_business(self) -> str | None:
        return self.business_ids[0] if self.business_ids else None


def detect_intent(question: str) -> Intent:
    """Classify a question. Deterministic and side-effect free."""
    text = (question or "").lower().strip()
    txns = tuple(t.upper() for t in TXN_PATTERN.findall(question or ""))
    bizs = tuple(b.upper() for b in BIZ_PATTERN.findall(question or ""))

    if not text:
        return Intent(UNKNOWN, 0.0, txns, bizs)

    scores: dict[str, float] = {}
    for intent, keywords in _KEYWORDS.items():
        score = sum(weight for phrase, weight in keywords if phrase in text)
        if score:
            scores[intent] = score

    # A cited transaction id strongly implies a transaction-level question.
    if txns:
        scores[EXPLAIN_TRANSACTION] = scores.get(EXPLAIN_TRANSACTION, 0.0) + 2.0
        if any(w in text for w in ("history", "previous", "past", "before")):
            scores[TRANSACTION_HISTORY] = scores.get(TRANSACTION_HISTORY, 0.0) + 2.5
    if bizs:
        scores[TRANSACTION_HISTORY] = scores.get(TRANSACTION_HISTORY, 0.0) + 2.0

    if not scores:
        return Intent(UNKNOWN, 0.0, txns, bizs)

    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    return Intent(best, round(scores[best] / total, 4) if total else 0.0, txns, bizs)


SUGGESTED_QUESTIONS: tuple[str, ...] = (
    "Why was this payment flagged as suspicious?",
    "What are the highest-risk transactions right now?",
    "What changed in transaction behaviour this week?",
    "Which features contributed most to the risk score?",
    "What does the transaction history for this business show?",
    "What should the analyst investigate next?",
    "What is the overall payment success rate and average fee?",
)
