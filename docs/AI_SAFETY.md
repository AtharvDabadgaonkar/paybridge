# PayBridge — AI Safety and Grounding

How the FraudLens Copilot is prevented from inventing figures, from being steered by text in the
records it reads, and from making decisions that belong to a human.

---

## 1. Threat model

| # | Threat | Consequence | Control |
| --- | --- | --- | --- |
| T1 | The model states a number that is not in the data | An analyst acts on a fabricated figure | Deterministic analytics + numeric claim validation |
| T2 | The model cites a transaction that does not exist | Wasted investigation, lost trust | Transaction-id allowlist validation |
| T3 | Text inside a record instructs the model | Attacker steers the assistant through a note | Escaping, fencing, explicit system rules, validator backstop |
| T4 | The model claims to have taken an action | Audit trail diverges from reality | Decision-claim detection + code-level block on non-human decisions |
| T5 | The API is unavailable, slow or rate limited | Feature breaks for the user | Deterministic answer computed on every request |
| T6 | The model returns malformed or empty output | Blank or broken answer | Structured result handling; empty output fails validation |
| T7 | Over-reliance on the assistant | Analyst stops thinking | Evidence pack shown; recommendations framed as checks, not conclusions |

## 2. Grounding architecture

```
question
  → detect_intent()        deterministic pattern routing (no model call)
  → build_evidence()       Pandas/NumPy analytics produce an EvidencePack:
                             · facts        the only numbers that may be stated
                             · tables       small, already-computed frames
                             · transaction_ids  the only ids that may be cited
                             · untrusted_notes  escaped and fenced
                             · deterministic_answer  a complete answer on its own
  → [optional] Claude      narrates the evidence under a restrictive system prompt
  → validate_response()    numbers · ids · decision claims
  → answer + mode
```

**The LLM is a narrator, not a calculator.** It is given finished figures and asked to explain them.
Because the deterministic answer is always computed first, removing the LLM removes prose quality,
never capability.

## 3. Numeric verification

`copilot/validator.py` extracts every number from the generated text — handling `$1,234.56`, `12.5%`,
`2.4M` and plain decimals — after stripping identifiers, rule codes and dates so their digits are
never mistaken for claims. Each number must match a value in the evidence pack within tolerance
(1.5% relative, 0.01 absolute), where the allowed set includes the facts, every numeric table cell
and the legitimate rounded and scaled restatements of them (`$140,507,725.84` → `$140.51M`).

Two deliberate rules:

- **Bare small integers (≤10) are allowed** as enumerations and ordinals ("check 3 things: 1) … 2) …").
- **Money and percentages are never exempt.** A figure written as `$0.00` or `0%` is a claim even
  though it is small, and a zero monetary claim must be backed by a monetary or percentage fact, not
  by an unrelated zero count in a table.

Any unsupported number fails the whole response.

## 4. Transaction reference checking

Every `TXN-xxxxxx` in the output must appear in the evidence pack's `transaction_ids`. Citing a real
transaction that was not retrieved for this question fails validation just as an invented one does:
the copilot may only talk about evidence it was actually given.

When the user asks about a transaction that does not exist, the analytics layer says so explicitly
and the pack carries a warning, instead of letting the model improvise.

## 5. Prompt-injection handling

Analyst notes and other free text are **untrusted input**. `wrap_untrusted()` strips tag-like
sequences that could close the fence (`</untrusted_data>`, `<system>`, `<assistant>`), HTML-escapes
the remainder, collapses whitespace and truncates to 600 characters. The content is then placed
inside `<untrusted_data>` tags with an instruction block stating that it is data, must never be
followed as an instruction, and may never support a numeric claim.

Defence in depth: even if a model complied with an injected instruction, the output would still have
to pass numeric validation, id validation and decision-claim detection. The test suite exercises
exactly this path — an injected "report zero loss" instruction produces a response that is rejected
and replaced with the verified answer.

**Known limitation, stated plainly.** Numeric validation checks *values*, not *semantics*. A
sentence that pairs a correct number with a wrong interpretation would pass the numeric check. The
mitigations for that are the evidence pack being shown next to every answer, the recommendation/decision
separation, and human ownership of all consequential outcomes — not the validator.

## 6. Human oversight

- The copilot has **no write path**. It cannot create, approve, reject, escalate or close anything.
- `investigations.record_decision()` raises `AutonomousDecisionBlocked` for any actor whose id starts
  with `ai.`, `copilot` or `system`. A decision requires a named human.
- Every high or critical payout routes to human review; `suggested_next_status()` returns `None` at
  `under_review` so a person must act.
- Language claiming an action was taken ("I have approved…", "Decision recorded") is detected and
  causes rejection.
- Every analyst and reviewer action writes an `audit_log` row with actor, entity, action and timestamp.

## 7. Failure handling

| Failure | Behaviour | Mode reported |
| --- | --- | --- |
| No API key | Deterministic answer; the UI explains why | `deterministic_no_llm` |
| SDK not installed | Deterministic answer | `deterministic_no_llm` |
| Rate limited | Deterministic answer + "the API is rate limiting requests" | `deterministic_no_llm` |
| Connection error / timeout | Deterministic answer + reason | `deterministic_no_llm` |
| API status error | Deterministic answer + reason | `deterministic_no_llm` |
| Malformed or empty response | Deterministic answer | `deterministic_no_llm` / rejected |
| Validation failure | Deterministic answer + the rejection reasons, with the discarded text viewable | `deterministic_after_rejection` |
| No evidence for the question | Honest "I cannot answer that from the evidence" | `deterministic` |

The mode is displayed on every answer, so a user always knows whether they are reading verified
deterministic text or validated model narration.

## 8. What is deliberately not claimed

- Not a guarantee that model prose is free of error — only that its **numbers and references** are
  checked, and that unchecked output is never shown.
- Not a general prompt-injection solution — fencing plus validation plus no write path, with the
  residual risk stated above.
- Not an autonomous agent. No tool use, no chained actions, no memory across sessions.
- No claim that the copilot improves analyst accuracy; that would require a study, and the
  experiment to run it is designed (EXP-004) but has not been run.

## 9. Test coverage

`tests/test_copilot.py` covers: intent detection for every supported question type; grounding of
each evidence builder against the underlying records; number extraction including currency and
suffixes; acceptance of rounded restatements; rejection of fabricated figures, invalid transaction
ids, decision claims and empty responses; the zero-money rule; identifier digits not being treated
as claims; graceful degradation for missing key, missing SDK, rate limits, timeouts, connection
errors and malformed responses; fence-breaking attempts; and the code-level block on non-human
decisions.
