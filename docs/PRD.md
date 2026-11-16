# PayBridge — Product Requirements Document

**Product:** PayBridge — Cross-Border Payments Platform with AI Fraud Detection
**Core engine:** FraudLens AI
**Status:** MVP delivered as a working simulation (independent portfolio project)
**Owner:** Product Manager (single-owner project)
**Last updated:** regenerate figures with `python scripts/report.py`

> **Scope note.** PayBridge is an independent personal portfolio project. It is not affiliated with
> any employer, financial institution or payment processor. All data is synthetic; no real payment,
> screening or regulatory process is performed.

---

## 1. Product vision

Give finance and risk teams at mid-market businesses one place to move money across borders
cheaply and predictably, with fraud controls they can actually explain to an auditor.

Cross-border payouts today force a choice: the bank wire that always works but costs 3–10% on
smaller amounts and settles in days, or a patchwork of local providers that is cheaper but
operationally fragmented. Layered on top sits a fraud function that is either a rules engine
drowning operations in false positives, or a model nobody can explain. PayBridge addresses both
problems in one product surface: route each payout on its cheapest eligible rail, and score it with
an engine whose every decision decomposes into named rules and measured feature contributions.

## 2. Problem definition

| # | Problem | Evidence in the product |
| --- | --- | --- |
| P1 | High and opaque transaction + FX cost | Fully decomposed quote per rail (fixed, variable, FX markup, correspondent fees) |
| P2 | Unpredictable settlement times | Realised p50/p90 settlement per rail and corridor |
| P3 | Fragmented payout infrastructure | One payout workflow across three rails and eight corridors |
| P4 | Slow, manual KYB onboarding | Onboarding state machine with duration measurement |
| P5 | AML/sanctions screening obligations | Screening simulation with adjudication and audit trail |
| P6 | Fraud detection with excessive false positives | Matched-operating-point comparison of rules vs ML vs combined |
| P7 | Manual, undocumented investigation | Case workbench with evidence, notes and an immutable audit trail |
| P8 | No visibility into payment performance | Executive dashboard computed live from the records |
| P9 | Integration and release-management risk | Program Command Center: dependencies, RAID, gates, shadow mode |

## 3. Target users and personas

**Finance Operations Manager (primary buyer-user).** Owns payout cost and reliability.
*Needs:* cost per corridor, settlement predictability, failure diagnosis.
*Success:* can answer "what did cross-border cost us last month, and why" in under a minute.

**Risk & Fraud Analyst (primary daily user).** Works the alert queue.
*Needs:* ranked alerts, explainable scores, context about the customer, a defensible decision record.
*Success:* decides a case with evidence attached, without exporting anything to a spreadsheet.

**Compliance Analyst.** Owns onboarding and screening adjudication.
*Needs:* a clear status model, a checklist, screening results, and an audit trail.
*Success:* every decision is attributable, time-stamped and reversible only by a new decision.

**Product Manager.** Owns adoption and unit economics.
*Needs:* activation, feature adoption, retention, cost per payout, contribution margin, experiment design.
*Success:* can state which feature to build next and what evidence would prove it worked.

**Technical Program Manager.** Owns delivery.
*Needs:* milestones, dependencies, RAID, release readiness, a defensible go/no-go.
*Success:* the launch decision is made against measured evidence, not opinion.

## 4. Jobs to be done

- When I send money abroad, I want to see every route's true all-in cost, so I can stop overpaying for wires.
- When a payout is held, I want to know exactly which signals held it, so I can resolve it or release it fast.
- When I onboard a customer, I want a single guided flow with a clear decision record, so review is defensible.
- When I review the quarter, I want cost, reliability and risk in one view, so I can brief the CFO.
- When I prepare a launch, I want evidence against acceptance criteria, so go/no-go is not a judgement call.

## 5. Product goals and KPI framework

| Goal | KPI | Target | Measured today |
| --- | --- | --- | --- |
| Reduce payout cost | Modelled saving vs an all-SWIFT mix | Directional: material reduction | 46.1% portfolio-wide; 61.8–87.7% on US-MX at $500–$5,000 (modelled) |
| Improve settlement predictability | p50 settlement by rail | USDC < ACH-partner < SWIFT | 1.6 h / 31.3 h / 60.5 h (simulated) |
| Keep payouts reliable | Payment success rate | ≥ 95% | 96.4% (simulated) |
| Cut false positives | FP reduction vs rules at matched recall | > 0% | +20.9% (synthetic test split) |
| Keep review load affordable | Manual review volume | ≤ 4% of payouts | 1.2% at the configured operating point |
| Speed up onboarding | Straightforward onboarding duration | < 5 minutes (**proposed, unvalidated**) | 4.7 min median; 58.8% under target (simulated durations) |
| Make risk explainable | Share of alerts with a complete evidence trail | 100% | 100% in shadow replay |

Guardrails across all goals: payment success rate, fraud recall, audit completeness, review volume.

## 6. MVP scope

**In scope (delivered).** Synthetic data platform; rail pricing and unit economics; payout creation
and lifecycle; executive analytics; KYB onboarding and screening simulation; FraudLens rules, ML and
SHAP layers; investigation workbench; model evaluation; grounded AI copilot; program command center;
product analytics and experiment design.

**Explicitly out of scope (not built).** Real provider/bank/chain connectivity; real sanctions or
identity verification; multi-tenant auth and RBAC; durable multi-user storage; a public API service;
mobile clients; FX hedging; chargeback/dispute handling; production model monitoring and retraining.

## 7. Functional requirements

| ID | Requirement | Status |
| --- | --- | --- |
| FR-01 | Generate a reproducible synthetic dataset from a fixed seed, regenerable on demand | Delivered |
| FR-02 | Quote any payout on every configured rail with fully decomposed cost | Delivered |
| FR-03 | Enforce rail eligibility by amount and corridor | Delivered |
| FR-04 | Create a simulated payout, return a reference and track status transitions | Delivered |
| FR-05 | Reject invalid payout status transitions | Delivered |
| FR-06 | Route every high/critical risk payout to human review | Delivered |
| FR-07 | Compute executive KPIs from filtered records, never from constants | Delivered |
| FR-08 | Model payout-cost savings against a SWIFT counterfactual, labelled as modelled | Delivered |
| FR-09 | Run a KYB onboarding state machine with reviewer decisions and an audit trail | Delivered |
| FR-10 | Screen names against a fictional watchlist and classify onboarding risk | Delivered |
| FR-11 | Evaluate 14 configurable fraud rules producing an attributable score | Delivered |
| FR-12 | Train and compare ≥3 models on chronological splits with a tuned threshold | Delivered |
| FR-13 | Produce SHAP explanations with a labelled deterministic fallback | Delivered |
| FR-14 | Combine rules and ML under a documented decision policy | Delivered |
| FR-15 | Persist investigation notes and decisions with an audit trail | Delivered |
| FR-16 | Block any non-human actor from recording an investigation decision | Delivered |
| FR-17 | Compare rules vs ML vs combined at a matched operating point | Delivered |
| FR-18 | Answer analyst questions from deterministic analytics, with or without an API key | Delivered |
| FR-19 | Validate every generated numeric claim against the evidence pack | Delivered |
| FR-20 | Track milestones, dependencies, RAID and release gates, editable and persisted | Delivered |
| FR-21 | Run shadow-mode validation that changes no payment decision | Delivered |
| FR-22 | Report product adoption from an event stream, not inferred from payments | Delivered |
| FR-23 | Design experiments with computed sample size and runtime | Delivered |

## 8. Non-functional requirements

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-01 | Start with one documented command | Delivered (`make run`) |
| NFR-02 | Work with no paid API credentials | Delivered (copilot fallback) |
| NFR-03 | Single-process deployment, no separate backend | Delivered |
| NFR-04 | Degrade gracefully when optional dependencies are missing | Delivered (SHAP, Anthropic SDK) |
| NFR-05 | Secrets via environment variables only; none committed | Delivered |
| NFR-06 | Scoring latency acceptable for interactive use | Delivered: p95 ≈ 18 ms in process (not load tested) |
| NFR-07 | Automated tests covering data, finance, workflow, ML and AI safety | Delivered: 262 tests |
| NFR-08 | Synthetic and modelled figures labelled wherever displayed | Delivered |
| NFR-09 | Demo reset restores a clean state | Delivered |
| NFR-10 | Storage limitations documented, not overstated | Delivered |

## 9. User stories and acceptance criteria

**US-01 — Compare payout routes.** *As a Finance Operations Manager, I want to compare every
eligible rail for a payout so that I can pick the cheapest acceptable route.*
Given a business, corridor and amount, when I open the comparison, then every rail shows total cost,
cost as a percentage, fee decomposition, recipient amount and settlement estimate; ineligible rails
are shown with the reason; the cheapest eligible rail is marked recommended.

**US-02 — Create and track a payout.** *As a Finance Operations Manager, I want to initiate a
simulated payout and follow it.*
Given an eligible rail, when I create the payout, then I receive a reference, the payout starts in
`initiated`, each transition is recorded with actor and timestamp, and invalid transitions are refused.

**US-03 — Hold risky payouts.** *As a Risk Analyst, I want high-risk payouts stopped before release.*
Given a payout scored high or critical, when it reaches screening, then it moves to `under_review`
and cannot progress without an explicit human approval or rejection.

**US-04 — Understand an alert.** *As a Risk Analyst, I want to know why a payment was flagged.*
Given a flagged transaction, when I open it, then I see the combined score and band, every triggered
rule with its weight and rationale, the model probability against its threshold, the top feature
contributions with the attribution method named, and recommended investigation questions.

**US-05 — Decide a case.** *As a Risk Analyst, I want to record a decision with evidence.*
Given an open case, when I submit approve/reject/escalate with my name, then the decision, any note
and an audit record are persisted; a non-human actor attempting the same is refused.

**US-06 — Adjudicate onboarding.** *As a Compliance Analyst, I want to progress a KYB case.*
Given a case in `review`, when I record a decision, then only permitted transitions are offered, the
decision overrides the dataset status, and the audit trail shows who decided what and when.

**US-07 — Choose an operating point.** *As a Risk Lead, I want to see the cost of a threshold.*
Given the evaluation page, when I move the threshold, then precision, recall, review volume and
modelled total cost recompute from the test split.

**US-08 — Ask the copilot.** *As a Risk Analyst, I want to ask questions in plain language.*
Given any supported question, when I ask it, then the answer contains only figures computed
deterministically; with no API key the answer still arrives; any generated claim that is not in the
evidence pack is discarded and the verified answer is shown instead.

**US-09 — Run a launch review.** *As a TPM, I want an evidence-based go/no-go.*
Given the readiness page, when I assess the gates, then each gate shows measured evidence or stays
`not_assessed`, and an unassessed gate caps the recommendation at conditional go.

## 10. RICE prioritisation

Reach = share of the three operating personas touched per week (1–10). Impact: 3 massive / 2 high /
1 medium / 0.5 low. Confidence: evidence-based estimate. Effort: person-weeks.

| Feature | R | I | C | E | RICE | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Rail comparison + unit economics | 9 | 3 | 0.9 | 3 | **8.1** | MVP |
| FraudLens rules layer | 8 | 3 | 0.95 | 2 | **11.4** | MVP |
| FraudLens ML + threshold tuning | 8 | 3 | 0.8 | 4 | **4.8** | MVP |
| SHAP explanations | 7 | 2 | 0.8 | 2 | **5.6** | MVP |
| Investigation workbench | 7 | 3 | 0.9 | 3 | **6.3** | MVP |
| Executive dashboard | 9 | 2 | 0.95 | 2 | **8.6** | MVP |
| Grounded AI copilot | 6 | 2 | 0.7 | 4 | **2.1** | MVP (differentiator) |
| KYB onboarding + screening sim | 5 | 2 | 0.85 | 3 | **2.8** | MVP |
| Model evaluation dashboard | 4 | 3 | 0.9 | 2 | **5.4** | MVP |
| Program command center | 3 | 2 | 0.9 | 2 | **2.7** | MVP |
| Product analytics + experiments | 3 | 2 | 0.85 | 2 | **2.6** | MVP |
| Bulk payout upload | 6 | 2 | 0.7 | 3 | **2.8** | v1.1 |
| Alert queue SLAs and routing | 5 | 2 | 0.7 | 2 | **3.5** | v1.1 |
| Real provider integration | 9 | 3 | 0.4 | 20 | **0.54** | Phase 2 |
| Licensed screening vendor | 5 | 3 | 0.5 | 12 | **0.63** | Phase 2 |

The two highest-value items for a real business (provider integration, licensed screening) score
lowest on RICE for *this* project because effort and confidence are dominated by commercial
dependencies, not engineering. That is the honest reason they are simulated here.

## 11. Risks and trade-offs

| Risk | Impact | Mitigation / trade-off accepted |
| --- | --- | --- |
| Synthetic fraud is easier than real fraud | Metrics overstate real performance | Stated everywhere; shadow mode on real traffic is the gate before decisioning |
| Small positive class in the test window (50 cases) | Wide confidence intervals | Reported alongside every metric; no claim rests on a few cases |
| LLM could fabricate a figure | Loss of trust, bad decisions | Numbers only from deterministic analytics; validator rejects unsupported claims |
| Rules-only is more explainable; ML is more precise | Operational tension | Combined policy with a rule override so a deterministic red flag is never suppressed |
| Lower threshold catches more fraud but floods review | Analyst cost | Threshold tuned on validation; review volume is a tracked guardrail |
| Ephemeral demo storage | Lost demo state | Documented; demo reset provided; no durability claimed |
| Onboarding target unvalidated | Overstated claim | Labelled proposed; measured against simulated durations only |
