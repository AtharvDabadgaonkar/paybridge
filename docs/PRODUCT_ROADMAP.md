# PayBridge — Product Roadmap

Everything under **MVP** is implemented in this repository and verifiable by running the app and the
test suite. Everything under v1.1 and Phase 2 is **proposed** and not built.

---

## MVP — delivered

| Area | Capability | Verification |
| --- | --- | --- |
| Data | Seeded 90-day synthetic dataset: 24,961 payouts, 200 businesses, 26,853 product events, 200 onboarding cases, 420 fictional watchlist entries | `scripts/generate_data.py`, `tests/test_data_quality.py` |
| Payments | Three rails, eight corridors, decomposed pricing, vectorised parity with the scalar engine | `tests/test_pricing.py` |
| Payments | Payout state machine, lifecycle events, reference tracking | `tests/test_payouts.py` |
| Analytics | Executive dashboard with live filtering and a SWIFT counterfactual | `tests/test_pricing.py::test_portfolio_savings_are_computed_from_records` |
| Compliance | KYB state machine, fictional-list screening, risk classification, audit trail | `tests/test_compliance.py` |
| FraudLens | 14 configurable rules; three ML candidates on chronological splits; SHAP + fallback | `tests/test_fraud_rules.py`, `tests/test_fraud_model.py`, `tests/test_explain_and_scoring.py` |
| FraudLens | Combined decision policy with a rule override and mandatory human review | `tests/test_explain_and_scoring.py` |
| Operations | Investigation workbench with notes, decisions and an audit trail | `tests/test_integration.py` |
| Evaluation | Rules vs ML vs combined at a matched operating point, loss simulation, corridor slices | `tests/test_evaluation.py` |
| AI | Grounded copilot: intent → analytics → evidence → LLM → validation, with full no-key fallback | `tests/test_copilot.py` |
| TPM | Milestones, RACI, dependencies, RAID, release gates, shadow mode, go/no-go | `tests/test_program.py` |
| PM | Activation, adoption, retention, unit economics, experiment designs | `tests/test_product.py` |

## Version 1.1 — proposed (next increment, no build started)

1. **Bulk payout upload.** CSV ingest with per-row validation, pre-flight cost summary and
   per-row risk scoring before submission. *Why:* payroll and supplier runs are the dominant real
   workflow and are currently one-at-a-time.
2. **Alert queue SLAs and routing.** Ageing, assignment rules and an escalation ladder, with a
   time-to-decision KPI. *Why:* the queue is ranked but unmanaged; ageing alerts are an operational risk.
3. **Saved views and scheduled exports.** Named filter sets per persona and a scheduled summary.
4. **Rule authoring UI.** Create and version rules in the product rather than in code, with a
   back-test against history before activation.
5. **Beneficiary book.** First-class beneficiary records with verification state and change history,
   so "new beneficiary" and "details changed" signals come from a real object rather than a field.
6. **Model monitoring.** Score drift, feature drift and a retraining trigger with a champion/challenger
   record.
7. **Experiment execution.** Assignment service, exposure logging and a results readout, so the
   designs on the Product Analytics page can actually be run.

## Phase 2 — proposed (strategic, dependency-heavy)

1. **Real payout provider integration.** ACH origination, SWIFT message handling and a licensed
   stablecoin on/off-ramp partner per corridor. Blocked on commercial agreements and licensing; this
   is the step that turns the simulation into a product.
2. **Licensed screening and identity vendors.** Replace the fictional watchlist with a licensed
   sanctions/PEP/adverse-media feed and a KYC vendor, with four-eyes adjudication.
3. **Multi-tenancy, auth and RBAC.** Organisation and role model, SSO, per-tenant data isolation,
   durable storage with backups.
4. **Public API and webhooks.** The surface described in `API_SPEC.md` implemented as a real service
   with idempotency keys, rate limits and signed webhooks.
5. **Shadow mode on live traffic.** Score real payments without affecting decisions until the gates
   in `PROGRAM_PLAN.md` pass on real data.
6. **FX treasury.** Pre-funded local balances and hedging to compress the FX component further.
7. **Case management depth.** Linked cases, network/graph views across beneficiaries, SAR-style
   reporting workflows.

## Explicit non-goals

Consumer remittances; card acquiring; lending or credit; tax filing; accounting-system replacement;
providing regulatory or investment advice.
