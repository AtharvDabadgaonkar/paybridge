# PayBridge — Interface Specification

PayBridge runs as a single Streamlit process, so this is **not** a deployed HTTP API. It specifies
the service interfaces the application actually implements in Python, expressed in the REST shape
they would take if exposed. Every operation listed below maps to real, tested code; the HTTP surface
itself is a Phase 2 roadmap item.

Conventions used below: amounts are USD decimals, timestamps are ISO-8601 UTC, and identifiers are
opaque strings. A real deployment would add authentication, idempotency keys, pagination and signed
webhooks — none of which exist today.

---

## 1. Quotes and payouts

### `POST /v1/quotes` — price a payout on every rail
Implemented by `payments.pricing.compare_rails`.

```json
Request  { "sender_country": "US", "recipient_country": "MX", "amount_usd": 5000.00 }
Response {
  "corridor": "US-MX", "currency": "MXN",
  "quotes": [{
    "rail": "USDC", "eligible": true,
    "fixed_fee": 1.00, "variable_fee": 30.00, "fx_markup_cost": 25.00, "lifting_fees": 0.00,
    "total_customer_cost": 56.00, "effective_cost_pct": 1.12,
    "mid_market_rate": 17.10, "offered_rate": 17.0145, "recipient_amount_local": 84226.76,
    "platform_direct_cost": 22.98, "contribution_margin": 33.02, "contribution_margin_pct": 58.96,
    "settlement_hours_p50": 1.57, "settlement_hours_p90": 7.25, "expected_success_rate": 0.981
  }]
}
```
Errors: `unknown_rail` (404), `invalid_amount` (422), `unsupported_corridor` (404).

### `POST /v1/payouts` — create a simulated payout
Implemented by `payments.payouts.create_payout`. Prices the payout, refuses an ineligible rail,
persists it in `initiated`, writes the first lifecycle event and an audit record.

```json
Request  { "business_id": "BIZ-0001", "recipient_country": "MX", "amount_usd": 5000.00,
           "payment_rail": "USDC", "risk_score": 0.21, "risk_band": "low" }
Response { "payout_id": "PO-ab12cd34ef56", "reference": "PB-20261116-A7F6B2",
           "status": "initiated", "quote": { ... } }
```
Errors: `rail_not_eligible` (422), `unknown_rail` (404).

### `GET /v1/payouts/{payout_id}` · `GET /v1/payouts?reference=` · `GET /v1/payouts`
`payouts.get_payout`, `find_payout_by_reference`, `list_payouts`.

### `POST /v1/payouts/{payout_id}/transitions` — move a payout
`payouts.transition_payout`. Validated against the state machine:

```
initiated → screening | rejected
screening → approved_for_payout | under_review | rejected
under_review → approved_for_payout | rejected        (human decision only)
approved_for_payout → in_transit | rejected
in_transit → settled | failed | returned
failed → in_transit                                   (retry)
settled, rejected, returned → terminal
```
Errors: `invalid_transition` (409) naming the allowed next states, `unknown_payout` (404).

### `POST /v1/payouts/{payout_id}/advance` — one policy step
`payouts.advance_payout`. Returns `null` at a terminal state or at `under_review`, where a human
decision is required: high and critical risk payouts are never advanced automatically.

### `GET /v1/payouts/{payout_id}/events`
`payouts.payout_events` — the full lifecycle trail with actor, note and timestamp.

## 2. Fraud scoring

### `POST /v1/risk/score` — score a transaction or a prospective payout
`fraud.scoring.assess_transaction`, with `build_payout_context` assembling payment-time features for
a payout that does not exist yet.

```json
Response {
  "transaction_id": "TXN-010947",
  "rules_score": 0.80, "rules_band": "critical",
  "rules_triggered": [{ "code": "R05_NEW_BENEFICIARY_LARGE", "name": "...", "weight": 0.75,
                        "description": "...", "investigation_question": "..." }],
  "ml_probability": 0.9397, "ml_threshold": 0.759, "ml_flagged": true,
  "combined_score": 0.8638, "combined_band": "critical",
  "decision": "hold", "requires_human_review": true,
  "model_name": "random_forest", "explanation_method": "shap_tree"
}
```

### `POST /v1/risk/explain` — SHAP contributions for one transaction
`fraud.explain.explain_transaction`. Always returns the attribution `method`; a deterministic
fallback is labelled as such and never presented as SHAP.

### `GET /v1/risk/rules` · `POST /v1/risk/rules/evaluate`
`fraud.rules.RULES`, `evaluate_rules` — the rule library with weights, and evaluation with optional
weight overrides and a configurable flag threshold.

### `GET /v1/risk/evaluation`
`fraud.evaluation.compare_approaches`, `matched_point_comparison` — rules vs ML vs combined on the
held-out test split, plus the matched-recall false-positive comparison.

## 3. Compliance workflow

### `GET /v1/compliance/cases` · `GET /v1/compliance/cases/{case_id}`
The onboarding case list with in-app decisions overlaid (`onboarding.apply_overrides`).

### `POST /v1/compliance/cases/{case_id}/decisions` — record a reviewer decision
`onboarding.record_decision`. Validated against:

```
draft → submitted
submitted → screening | more_info_required
screening → review | approved | rejected | more_info_required
review → approved | rejected | more_info_required
more_info_required → submitted | rejected
approved, rejected → terminal
```
Errors: `invalid_transition` (409). Every decision writes an audit record.

### `POST /v1/compliance/screening` — screen a name against the fictional list
`compliance.screening.screen_entity`. Returns matches with `match_score`, `list_source` and
`is_fictional: true`. **Simulation only — no real sanctions, PEP or identity checking.**

### `POST /v1/compliance/risk-classification`
`screening.classify_onboarding_risk` — deterministic classification with the reasons that produced it.

## 4. Investigation case management

### `GET /v1/investigations` · `GET /v1/investigations/{case_id}`
`investigations.list_cases`, `get_case` — filterable by status and decision.

### `POST /v1/investigations` — open a case
`investigations.create_case`. Idempotent per transaction: a second call returns the existing case.

### `POST /v1/investigations/{case_id}/notes` — add a note
`investigations.add_note`. Notes are **untrusted free text** and are never interpreted as
instructions by the copilot.

### `POST /v1/investigations/{case_id}/decisions` — record a decision
`investigations.record_decision`. `approve` and `reject` close the case; `escalate` keeps it in
review. **Any actor whose id starts with `ai.`, `copilot` or `system` is rejected with
`autonomous_decision_blocked` (403).**

### `GET /v1/investigations/{case_id}/audit`
`investigations.case_audit` — create, assign, note and decision records with actor and timestamp.

## 5. Copilot

### `POST /v1/copilot/ask`
`copilot.copilot.ask`.

```json
Request  { "question": "Why was TXN-010947 flagged?", "context": { "transaction_id": "TXN-010947" } }
Response { "answer": "...", "mode": "llm_verified | deterministic_after_rejection |
                       deterministic_no_llm | deterministic",
           "intent": "explain_transaction", "confidence": 1.0,
           "evidence": { "facts": [...], "tables": {...}, "transaction_ids": [...] },
           "validation": { "ok": true, "checked_numbers": 7, "unsupported": [],
                           "invalid_transaction_ids": [] } }
```
The response always carries an answer: with no API key, on any API failure, or when generated text
fails validation, the deterministic answer is returned and the mode says so.

## 6. Program management

`GET/POST /v1/program/milestones`, `/raid`, `/dependencies`, `/gates`, `POST /v1/program/gates/assess`,
`POST /v1/program/shadow-mode/run` — implemented by `program.program` and `program.shadow_mode`.
Gate assessment never marks an unmeasured criterion as passed; it stays `not_assessed`.

## 7. Not implemented

No HTTP server, authentication, authorisation, rate limiting, idempotency keys, pagination,
webhooks, API versioning or SDKs. Adding them is Phase 2 and would be a real service, not a wrapper
around a Streamlit process.
