"""Claim validation: the backstop that stops the LLM inventing numbers.

Checks performed on every generated answer:
  1. Numeric claims - every number in the text must match a value in the
     evidence pack within tolerance.
  2. Transaction references - every TXN-xxxxxx must be one the pack supplied.
  3. Decision claims - the copilot may recommend, never decide or act.

Any failure causes the generated text to be rejected and the deterministic
answer to be shown instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from paybridge.copilot.evidence import EvidencePack

# Numbers the copilot may use without evidence: small integers that are almost
# always enumerations ("1)", "three checks"), and the digits inside identifiers
# (which are stripped before parsing).
SMALL_INT_ALLOWANCE = 10
RELATIVE_TOLERANCE = 0.015
ABSOLUTE_TOLERANCE = 0.01

_ID_PATTERN = re.compile(r"\b(?:TXN|BIZ|CASE|KYB|PO|BEN|EVT|WL|R\d{2})[-_][A-Za-z0-9]+", re.IGNORECASE)
_RULE_CODE_PATTERN = re.compile(r"\bR\d{2}_[A-Z_]+\b")
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>-)?\$?\s?"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<suffix>[KMB]\b|bn\b|%|percent)?",
    re.IGNORECASE,
)

_DECISION_CLAIM_PATTERN = re.compile(
    r"\b("
    r"i (?:have )?(?:approved|rejected|blocked|released|escalated|closed|cancelled|frozen)"
    r"|i (?:will|am going to) (?:approve|reject|block|release|escalate|close|cancel|freeze)"
    r"|(?:the )?(?:case|payment|payout|transaction) (?:has been|was) (?:approved|rejected|blocked|closed) by me"
    r"|i have (?:updated|changed) the (?:status|decision)"
    r"|decision recorded"
    r")\b",
    re.IGNORECASE,
)

_SUFFIX_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9}


@dataclass(frozen=True)
class NumericClaim:
    raw: str
    value: float
    supported: bool
    matched_value: float | None = None


@dataclass
class ValidationResult:
    ok: bool
    claims: list[NumericClaim] = field(default_factory=list)
    unsupported: list[NumericClaim] = field(default_factory=list)
    invalid_transaction_ids: list[str] = field(default_factory=list)
    decision_claims: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def checked_count(self) -> int:
        return len(self.claims)


def _strip_identifiers(text: str) -> str:
    text = _ID_PATTERN.sub(" ", text)
    text = _RULE_CODE_PATTERN.sub(" ", text)
    return _DATE_PATTERN.sub(" ", text)


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """Pull every numeric claim out of free text, normalising K/M/B suffixes."""
    out: list[tuple[str, float]] = []
    for match in _NUMBER_PATTERN.finditer(_strip_identifiers(text)):
        raw = match.group(0).strip()
        try:
            value = float(match.group("value").replace(",", ""))
        except ValueError:  # pragma: no cover - defensive
            continue
        suffix = (match.group("suffix") or "").lower()
        if suffix in _SUFFIX_SCALE:
            value *= _SUFFIX_SCALE[suffix]
        if match.group("sign"):
            value = -value
        out.append((raw, value))
    return out


def _matches(value: float, allowed: list[float]) -> float | None:
    for candidate in allowed:
        tolerance = max(ABSOLUTE_TOLERANCE, abs(candidate) * RELATIVE_TOLERANCE)
        if abs(value - candidate) <= tolerance:
            return candidate
    return None


def allowed_values(pack: EvidencePack) -> list[float]:
    """Evidence values plus the derived forms a writer may legitimately use."""
    base = pack.numeric_values()
    derived: list[float] = []
    for value in base:
        derived.append(round(value, 2))
        derived.append(round(value, 1))
        derived.append(round(value))
        if abs(value) >= 1000:
            derived.append(round(value / 1_000, 2))
            derived.append(round(value / 1_000_000, 2))
            derived.append(round(value / 1_000_000, 1))
        if 0 < abs(value) < 1:
            derived.append(round(value * 100, 2))   # ratio quoted as a percentage
    return base + derived


def validate_numbers(text: str, pack: EvidencePack) -> tuple[list[NumericClaim], list[NumericClaim]]:
    allowed = allowed_values(pack)
    claims: list[NumericClaim] = []
    unsupported: list[NumericClaim] = []
    for raw, value in extract_numbers(text):
        if float(value).is_integer() and abs(value) <= SMALL_INT_ALLOWANCE:
            continue                       # enumerations and ordinals
        matched = _matches(value, allowed)
        claim = NumericClaim(raw=raw, value=value, supported=matched is not None, matched_value=matched)
        claims.append(claim)
        if not claim.supported:
            unsupported.append(claim)
    return claims, unsupported


def validate_transaction_ids(text: str, pack: EvidencePack) -> list[str]:
    cited = {m.upper() for m in re.findall(r"\bTXN-\d{4,8}\b", text, flags=re.IGNORECASE)}
    allowed = {t.upper() for t in pack.transaction_ids}
    return sorted(cited - allowed)


def validate_decision_claims(text: str) -> list[str]:
    return [m.group(0) for m in _DECISION_CLAIM_PATTERN.finditer(text)]


def validate_response(text: str, pack: EvidencePack) -> ValidationResult:
    """Run every check. ``ok`` is False if any check fails."""
    if not text or not text.strip():
        return ValidationResult(ok=False, reasons=["The model returned an empty response."])

    claims, unsupported = validate_numbers(text, pack)
    invalid_ids = validate_transaction_ids(text, pack)
    decisions = validate_decision_claims(text)

    reasons: list[str] = []
    if unsupported:
        reasons.append(
            "Unsupported numeric claim(s) not present in the computed evidence: "
            + ", ".join(sorted({c.raw for c in unsupported}))
        )
    if invalid_ids:
        reasons.append("Reference to transaction id(s) not in the evidence pack: " + ", ".join(invalid_ids))
    if decisions:
        reasons.append(
            "The response claimed to take or record a decision, which the copilot is not permitted to do: "
            + "; ".join(sorted(set(decisions)))
        )

    return ValidationResult(
        ok=not reasons,
        claims=claims,
        unsupported=unsupported,
        invalid_transaction_ids=invalid_ids,
        decision_claims=decisions,
        reasons=reasons,
    )
