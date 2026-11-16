# PayBridge — Program Plan

The interactive version of this plan lives in the application under **Program Command Center**,
where milestones, RAID items, dependencies and gates are editable and persisted.

> Milestone statuses describe the delivery of **this repository**. Work that is not implemented is
> marked *planned* and never counted as delivered.

---

## 1. Workstreams

| # | Workstream | Outcome | Accountable |
| --- | --- | --- | --- |
| 1 | Product Discovery | Problem, personas, MVP scope, prioritisation | Product Manager |
| 2 | Payment Infrastructure | Rail economics, quoting, payout lifecycle, analytics | Engineering |
| 3 | Compliance Integration | Onboarding workflow, screening simulation, audit | Compliance Ops |
| 4 | FraudLens Development | Rules, features, models, explanations, copilot | Data Science |
| 5 | Risk Evaluation | Comparison harness, loss simulation, corridor review | Risk |
| 6 | System Testing | Automated suite, CI, integration and load testing | QA |
| 7 | Shadow-Mode Validation | Recommendation-only replay and evidence pack | TPM |
| 8 | Pilot Launch | Cohort, runbook, rollback, go/no-go | TPM |
| 9 | GA Planning | Pricing, support model, launch readiness | Product Manager |

## 2. Milestones

32 milestones are tracked in the application. Representative set:

| ID | Workstream | Milestone | Due | Status |
| --- | --- | --- | --- | --- |
| M-003 | Product Discovery | MVP scope lock and RICE prioritisation | 2026-02-28 | complete |
| M-010 | Payment Infrastructure | Rail economics model and quote engine | 2026-03-20 | complete |
| M-011 | Payment Infrastructure | Payout state machine and lifecycle audit | 2026-04-03 | complete |
| M-013 | Payment Infrastructure | Production provider integration | 2026-12-18 | **planned — not built** |
| M-021 | Compliance Integration | Synthetic watchlist screening simulation | 2026-05-15 | complete |
| M-023 | Compliance Integration | Licensed vendor screening contract | 2027-01-29 | **planned — not built** |
| M-032 | FraudLens | Model training, selection and threshold tuning | 2026-07-03 | complete |
| M-033 | FraudLens | SHAP explanation service | 2026-07-24 | complete |
| M-040 | Risk Evaluation | Baseline vs ML vs combined harness | 2026-07-31 | complete |
| M-043 | Risk Evaluation | Independent model validation review | 2026-12-11 | **planned — not built** |
| M-050 | System Testing | Automated test suite and CI pipeline | 2026-09-04 | complete |
| M-052 | System Testing | Load and latency testing at pilot volume | 2026-11-20 | in progress (35%) |
| M-060 | Shadow-Mode | Shadow scoring harness | 2026-10-02 | complete |
| M-062 | Shadow-Mode | Analyst workflow dry run with ops | 2026-11-27 | in progress (45%) |
| M-070 | Pilot Launch | Cohort selection and success criteria | 2026-11-27 | in progress (40%) |
| M-072 | Pilot Launch | Go / no-go review | 2026-12-23 | planned |

## 3. Critical path

```
Product Discovery
  → Payment Infrastructure (quote engine → payout lifecycle → analytics)
      → FraudLens (features → models → threshold → explanations)
          → Risk Evaluation (comparison harness → operating point)
              → System Testing (automated suite → load testing)
                  → Shadow-Mode Validation (replay → evidence pack)
                      → Pilot Launch (runbook → go/no-go)
                          → GA Planning
Compliance Integration runs in parallel and blocks GA, not the pilot.
```

The binding constraint today is **System Testing → Pilot Launch**: load and latency testing
(M-052, DEP-006) is incomplete, so no SLO can be committed in the pilot runbook.

## 4. Dependency tracker

| ID | Dependency | Upstream → Downstream | Status |
| --- | --- | --- | --- |
| DEP-001 | Rail economics before cost KPIs | Payment Infra → Discovery | satisfied |
| DEP-002 | Feature pipeline before training | FraudLens → Risk Evaluation | satisfied |
| DEP-003 | Model + threshold before comparison | FraudLens → Risk Evaluation | satisfied |
| DEP-004 | Persisted decisions before audit-completeness assessment | FraudLens → Shadow Mode | satisfied |
| DEP-005 | Shadow evidence before go/no-go | Shadow Mode → Pilot | satisfied |
| DEP-006 | Load testing before SLOs | System Testing → Pilot | **in progress — critical path** |
| DEP-007 | Licensed screening vendor before any real-data compliance launch | Compliance → GA | **blocked (external)** |
| DEP-008 | Provider integrations before GA pricing commitments | Payment Infra → GA | not started |

## 5. RAID log (summary)

**Risks**
- R-001 Synthetic fraud patterns may not generalise — *high/high*, open. Mitigation: treat all metrics as synthetic-only; require shadow mode on real traffic.
- R-002 False positives create operational load — *medium/high*, mitigating. Mitigation: tune on validation at a fixed loss constraint; track review volume as a guardrail.
- R-003 Stablecoin rail carries regulatory uncertainty — *high/medium*, open. Mitigation: corridor availability contingent on partner licensing.
- R-004 Ephemeral storage on the demo host — *low/high*, monitoring. Mitigation: documented; dataset regenerable; demo reset.
- R-005 LLM could state an unsupported number — *high/medium*, mitigating. Mitigation: deterministic analytics + claim validator.

**Assumptions**
- A-001 Rail fee and FX assumptions approximate market levels (not a provider rate card).
- A-002 Loss given undetected fraud = 85% of amount.
- A-003 Manual review costs $18 per case.

**Issues**
- I-001 No load-testing rig at pilot volume — open, blocks DEP-006 and gate G-06's generalisation.
- I-002 Onboarding duration target unvalidated — open; the claim stays labelled as proposed.

**Dependencies** — D-001 local payout partner coverage; D-002 licensed screening vendor.

## 6. Shadow-mode validation

FraudLens replays historical payouts and emits recommendations while the original simulated
decision is left untouched; the harness asserts and reports that invariant.

Measured on the most recent 4,000-transaction replay (reproduce with `python scripts/report.py`):

| Criterion | Direction | Target | Measured | Met |
| --- | --- | --- | --- | --- |
| Fraud detection recall | ≥ | 0.60 | 0.47 | **No** |
| Alert precision | ≥ | 0.15 | 0.18 | Yes |
| False positive rate | ≤ | 3.0% | 2.1% | Yes |
| Manual review volume | ≤ | 4.0% | 2.6% | Yes |
| Scoring latency p95 (in process) | ≤ | 50 ms | ~18 ms | Yes |
| Evidence-trail completeness | ≥ | 100% | 100% | Yes |
| Original payment decisions changed | = | 0 | 0 | Yes |
| Workflow errors | = | 0 | 0 | Yes |

**Seven of eight criteria are met. Recall is below target and is reported as a blocker, not waived.**
The remediation options on the table are: lower the decision threshold (costs review volume, which
is currently well inside budget), add rules for the miss patterns, or revisit the model family.

## 7. Release readiness and go/no-go

Fourteen gates across Model performance, Operations, Compliance, Engineering and Launch are assessed
in the app against measured evidence. The decision framework:

- **Any gate failing → no go.** No silent waivers; a waiver requires a documented rationale.
- **Any gate unassessed → conditional go at best.** An unmeasured criterion is never treated as passed.
- **All gates passing on measured evidence → go.**

Current position: with the test suite recorded and a shadow run completed, the model-recall gate
(G-01) fails on the same evidence as the shadow criterion above, so the recommendation is **no go**
for decisioning use. The honest reading is that PayBridge is ready to run in shadow and to support
analysts, and is not ready to make automated hold/release decisions.

## 8. Governance

Weekly delivery review (milestones, dependencies, RAID), fortnightly risk review (operating point,
review volume, model behaviour), and a launch review at each gate. Every analyst and reviewer
decision in the product writes an audit record, which is the evidence base for gate G-08.
