# PayBridge — Portfolio Case Study

**Independent personal project.** Not affiliated with any employer, financial institution or payment
processor. All data is synthetic; no real payment, screening or regulatory process is performed.
Every figure below is reproducible with `python scripts/report.py` and `pytest`.

---

## Problem statement

A mid-market business paying suppliers or contractors abroad faces two unrelated-looking problems
that are actually one operating problem. On the money side, a SWIFT wire costs 3–10% of a smaller
payout once wire, correspondent and FX-markup charges are counted, and settles in days with little
visibility. On the risk side, the fraud controls protecting those payouts are usually either a rules
engine that buries analysts in false positives, or a model nobody can explain to a reviewer. Both
end in the same place: payouts that cost too much and take too long, and a risk function that cannot
defend its decisions.

PayBridge is a working simulation of the platform that addresses both: route every payout on its
cheapest eligible rail with fully decomposed pricing, and score it with an engine whose every
decision breaks down into named rules and measured feature contributions.

## Product solution

Nine modules built as one Streamlit application over a seeded 90-day synthetic dataset of 24,961
cross-border payouts, 200 fictional businesses, 8 corridors and 3 payment rails:

1. **Executive overview** — volume, cost, settlement and risk KPIs, computed live from filtered records.
2. **Payments & payouts** — rail comparison, unit-economics calculator, savings sensitivity grid, payout lifecycle tracker.
3. **Compliance operations** — KYB onboarding state machine, fictional-watchlist screening, reviewer decisions, audit trail.
4. **FraudLens AI** — 14 configurable rules, a supervised model, SHAP explanations, a tunable rules layer.
5. **Investigation workbench** — ranked queue, case context, customer history, notes and auditable decisions.
6. **Model evaluation** — rules vs ML vs combined at a matched operating point, loss simulation, threshold explorer.
7. **AI copilot** — grounded assistant that computes nothing itself and validates everything it says.
8. **Program command center** — milestones, RACI, dependencies, RAID, shadow mode, evidence-based go/no-go.
9. **Product analytics** — activation, adoption, retention, unit economics, experiment designs with computed sample sizes.

## Technical architecture

Python 3.11 · Streamlit · Plotly · pandas/NumPy/SciPy · scikit-learn · SHAP · SQLite · Parquet ·
Pytest · GitHub Actions · optional Anthropic Claude API.

One process, strictly layered: `ui` → services (`payments`, `compliance`, `fraud`, `copilot`,
`program`, `product`, `analytics`) → `data`/`db` → `config`. No service module imports Streamlit,
so the entire business layer — pricing maths, state machines, the feature pipeline, the model
selection protocol and the copilot validator — is unit tested directly. The immutable synthetic
dataset lives in Parquet; everything a user changes lives in SQLite with an audit row per action.

## Verified results

**Payment economics (modelled under documented assumptions, not realised savings)**

| Scenario (US→MX) | USDC vs SWIFT | ACH + partner vs SWIFT |
| --- | --- | --- |
| $500 | 87.7% lower | 83.5% lower |
| $5,000 | 67.1% lower | 61.8% lower |
| $50,000 | 58.9% lower | 53.2% lower |
| Whole synthetic portfolio vs an all-SWIFT counterfactual | **46.1% lower** ($2.20M on $140.5M) | |

The ~60% hypothesis holds for small and mid-size payouts and decays as amounts grow, because
SWIFT's fixed wire and correspondent fees amortise while percentage-based costs do not. That
finding, not the headline, is the product insight.

**Fraud model (held-out chronological test split: 4,993 payouts, 50 labelled fraud cases)**

| Approach | Precision | Recall | F1 | PR-AUC | FPR | Review volume |
| --- | --- | --- | --- | --- | --- | --- |
| Rules baseline | 0.184 | 0.520 | 0.272 | 0.281 | 2.33% | 2.82% |
| ML only (random forest) | 0.455 | 0.400 | 0.426 | 0.341 | 0.49% | 0.88% |
| Rules + ML combined | 0.344 | 0.420 | 0.378 | **0.360** | 0.81% | 1.22% |

**At matched recall (0.520), the combined engine produced 91 false positives against the rules
baseline's 115 — a 20.9% reduction** (ML-only: 14.8%). Random forest was selected on validation
PR-AUC; the threshold (0.759) was tuned on validation and the metrics above are from the untouched
test split.

**Shadow-mode validation (4,000-payout replay):** 7 of 8 acceptance criteria met — 0 original
payment decisions changed, 100% evidence-trail completeness, 2.1% false-positive rate, 2.6% review
volume, ~18 ms p95 in-process scoring latency — **and recall at 0.47 against a 0.60 target, reported
as a blocking gate rather than waived.**

**Quality assurance:** 251 automated tests passing, covering data reproducibility, financial
calculations, workflow state machines, leakage controls, threshold tuning, explainability fallbacks,
copilot grounding and injection handling, program logic and end-to-end flows, plus a render test for
every page.

## Product management demonstrated

- Framed nine business problems into a scoped MVP with explicit non-goals, and prioritised with RICE
  — including documenting why the two highest-value real-world features score lowest here
  (commercial dependencies, not engineering).
- Wrote personas, JTBD, 23 functional and 10 non-functional requirements, and user stories with
  testable acceptance criteria that map one-to-one onto the test suite.
- Built a KPI framework with guardrails, and separated *targets* from *measurements* everywhere —
  the sub-five-minute onboarding goal ships labelled as proposed because only 58.8% of simulated
  straightforward cases meet it and no usability study has been run.
- Designed four experiments with hypotheses, primary KPIs, guardrails, computed sample sizes and
  rollout criteria — and reported no results, because none has been run.
- Chose an operating point as a business decision: the combined policy accepts lower precision than
  ML-only to keep the rules layer's deterministic red flags, and keeps review volume inside budget.

## TPM practices demonstrated

- Nine workstreams, 32 milestones, a documented critical path, and an 8-item dependency tracker that
  names the one blocking the pilot (load testing → SLOs).
- A live RAID log where risks carry mitigations and open high-severity issues automatically fail a
  release gate.
- A shadow-mode rollout that asserts and reports zero decision impact, measuring detection, latency,
  review volume and audit completeness against pre-agreed thresholds.
- A 14-gate release checklist with a decision rule that cannot be fudged: any failure is a no-go, and
  any gate without measured evidence caps the recommendation at conditional go. On current evidence
  the honest recommendation is **no-go for automated decisioning, go for shadow and analyst support**.

## Trade-offs and lessons learned

- **The first dataset was unlearnable.** My initial generative process produced a Bayes-optimal
  PR-AUC of 0.097 — the labels were mostly noise, so no model could have performed. Measuring the
  achievable ceiling before blaming the model changed the diagnosis from "tune harder" to "the data
  has no signal", and I rebuilt the risk structure so fraud concentrated in an identifiable
  subpopulation. Knowing the ceiling is as important as knowing the score.
- **Rules and ML are complementary, not competing.** ML-only had the best precision but missed cases
  the rules caught. Keeping a rule override in the combined policy costs precision and buys the
  guarantee that a deterministic red flag is never silently suppressed. That is a product decision,
  not a modelling one.
- **Comparing at a matched operating point is the only honest comparison.** Two systems at arbitrary
  thresholds can be made to tell any story; equalising recall before comparing false positives is
  what makes the 20.9% figure mean something.
- **Grounding an LLM is an architecture problem.** Prompting a model to "be accurate" does not work.
  Computing every figure deterministically, passing it an allowlist, and rejecting any output
  containing an unverifiable number does. The validator caught my own fabricated test cases,
  including a `$0.00` claim that an earlier version of the rule let through.
- **Reporting a failure is more credible than hiding it.** Recall misses the shadow-phase target, and
  the product says so on the page, in the gate and in this document. A portfolio project that only
  ever reports good news tells a reviewer nothing about judgement.
- **Synthetic results are an upper bound.** Labels generated from the factors the detector observes
  make the problem easier than reality. Stating that plainly is part of the work.

## Two ATS-friendly resume bullets

> Built **PayBridge**, a cross-border payments and AI fraud-intelligence platform in **Python,
> Streamlit, scikit-learn and SHAP**, simulating **24,961 payouts across 8 corridors and 3 payment
> rails**; the combined rules-plus-ML engine **reduced false positives 20.9% versus a rules-only
> baseline at matched recall** on a held-out chronological test split, and modelled **46–88% lower
> payout cost** than SWIFT depending on corridor and amount.

> Designed an **explainable fraud engine and a grounded LLM copilot** that computes every metric
> deterministically in Python and **validates all generated numeric claims against an evidence
> allowlist**, rejecting unsupported output; delivered with **251 passing automated tests**, CI, a
> leakage-controlled chronological training pipeline, and full functionality with **no paid API
> credentials**.
