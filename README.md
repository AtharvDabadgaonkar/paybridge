# PayBridge

**Cross-Border Payments Platform with AI Fraud Detection**
Core engine: **FraudLens AI** · Python · Streamlit · scikit-learn · SHAP · SQLite

> **Independent personal portfolio project.** Not affiliated with, endorsed by or built for any
> employer, financial institution or payment processor. **All data is synthetic.** No real payment
> is ever initiated, and no real sanctions, identity or regulatory verification is performed.

PayBridge simulates a B2B cross-border payment operations platform: price a payout across three
rails and eight corridors, run it through a KYB and screening workflow, score it with an explainable
fraud engine, investigate it in a case workbench, and manage the whole thing as a delivery
programme with evidence-based release gates.

---

## Quickstart

```bash
make install     # create .venv (Python 3.11) and install dependencies
make run         # start the app at http://localhost:8501
```

The first launch generates the synthetic dataset, creates the SQLite schema and seeds the demo.
Training the fraud model happens automatically on first use, or explicitly:

```bash
make data        # regenerate the seeded dataset
make train       # train + evaluate FraudLens, write the model bundle
make test        # run the automated test suite
make report      # print every headline figure, recomputed from the data
make reset       # wipe generated data/models and rebuild
```

Without `make`:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/streamlit run app.py
```

**No API key is required.** With `ANTHROPIC_API_KEY` set, the AI copilot adds an LLM explanation
layer on top of the deterministic analytics; without it, every feature still works.

## The application

| # | Module | What it does |
| --- | --- | --- |
| 1 | Executive Overview | Volume, cost, settlement and risk KPIs, computed live from filtered records |
| 2 | Payments & Payouts | Rail comparison, unit economics, savings sensitivity, payout lifecycle tracker |
| 3 | Compliance Operations | KYB onboarding workflow, fictional-watchlist screening, reviewer decisions, audit trail |
| 4 | FraudLens AI | 14 configurable rules, supervised model, SHAP explanations, live rule tuning |
| 5 | Investigation Workbench | Ranked alert queue, case context, customer history, auditable decisions |
| 6 | Model Evaluation | Rules vs ML vs combined at a matched operating point, loss simulation, threshold explorer |
| 7 | AI Copilot | Grounded assistant: deterministic analytics, validated claims, full no-key fallback |
| 8 | Program Command Center | Milestones, RACI, dependencies, RAID, shadow mode, go/no-go |
| 9 | Product Analytics | Activation, adoption, retention, unit economics, experiment designs |
| 10 | About PayBridge | Scope, what is and is not implemented, live environment status |

## Verified results

Reproduce all of these with `make report` and `make test`.

**Synthetic dataset (seed 42, 90 days):** 24,961 payouts · 200 businesses · 8 corridors · 3 rails ·
26,853 product events · 200 onboarding cases · 420 fictional watchlist entries · 1.53% fraud rate.

**Modelled payout cost vs SWIFT** (documented assumptions in `docs/UNIT_ECONOMICS.md`, *not*
realised savings):

| US→MX payout | USDC | ACH + local partner |
| --- | --- | --- |
| $500 | 87.7% lower | 83.5% lower |
| $5,000 | 67.1% lower | 61.8% lower |
| $50,000 | 58.9% lower | 53.2% lower |
| Portfolio vs all-SWIFT counterfactual | **46.1% lower** ($2.20M of $4.77M) | |

**Fraud detection** — held-out chronological test split, 4,993 payouts, 50 labelled fraud cases:

| Approach | Precision | Recall | F1 | PR-AUC | FPR | Review volume |
| --- | --- | --- | --- | --- | --- | --- |
| Rules baseline | 0.184 | 0.520 | 0.272 | 0.281 | 2.33% | 2.82% |
| ML only (random forest) | 0.455 | 0.400 | 0.426 | 0.341 | 0.49% | 0.88% |
| Rules + ML combined | 0.344 | 0.420 | 0.378 | **0.360** | 0.81% | 1.22% |

**At matched recall (0.520): 91 false positives for the combined engine vs 115 for rules-only — a
20.9% reduction.** Model selected on validation PR-AUC; threshold (0.759) tuned on validation;
metrics measured on the untouched test split.

**Shadow-mode validation (4,000-payout replay):** 7 of 8 criteria met — 0 payment decisions changed,
100% evidence-trail completeness, 2.1% FPR, 2.6% review volume, ~18 ms p95 in-process latency — and
**recall 0.47 against a 0.60 target, reported as a blocking gate rather than waived.**

**Quality assurance:** 262 automated tests passing (data quality, financial maths, workflow state
machines, leakage controls, threshold tuning, explainability, copilot grounding and prompt
injection, program logic, end-to-end flows, and a render test for all 10 pages).

## Repository structure

```
app.py                       Streamlit entry point and navigation shell
src/paybridge/
  config.py                  corridors, rail economics, thresholds, paths
  analytics/kpis.py          deterministic KPI layer (shared by UI and copilot)
  data/                      seeded synthetic generator + cached loaders
  db/                        SQLite schema, audit log, idempotent seeds
  payments/                  pricing engine, payout state machine
  compliance/                KYB onboarding, synthetic screening
  fraud/                     rules, features, model, explain, scoring, evaluation, investigations
  copilot/                   intents, analytics, evidence, validator, Anthropic client
  program/                   milestones, RAID, gates, shadow mode
  product/                   adoption metrics, experiment design
  ui/                        theme, components, filters, services, 10 page modules
scripts/                     generate_data.py · train_model.py · report.py
tests/                       262 tests across 13 modules
docs/                        PRD · roadmap · program plan · architecture · API spec ·
                             unit economics · model card · AI safety · case study
```

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`,
   Python 3.11.
3. Dependencies install from `requirements.txt`.
4. *Optional:* add `ANTHROPIC_API_KEY` under **App settings → Secrets** to enable the copilot's LLM
   layer. The app runs fully without it.
5. First boot generates the dataset and trains the model (about a minute), then caches both.

**Persistence limitation.** Streamlit Community Cloud containers are ephemeral. SQLite state —
simulated payouts, investigation decisions, compliance decisions, program edits — survives within a
running container and is lost when the app is rebuilt or recycled. The dataset regenerates
deterministically from seed 42, and **Demo controls → Reset demo state** restores a clean
environment. PayBridge makes no claim of durable multi-user storage.

## Honest scope

**Implemented:** everything in the module table above, as working, tested code.

**Not implemented:** real money movement of any kind; real sanctions, PEP, adverse-media or identity
screening; provider, bank, wallet, chain or exchange connectivity; multi-tenancy, auth or RBAC; a
deployed HTTP API; durable multi-user storage; production model monitoring or retraining.

**Read as modelled, not realised:** all cost savings (counterfactual repricing under documented
assumptions), all fraud-loss and review-cost figures (two stated assumptions), and all model metrics
(synthetic labels generated from the same factors the detector observes — an optimistic upper
bound, not a forecast of real-world performance).

**Targets vs measurements are kept apart.** The sub-five-minute onboarding goal is a *proposed
target*; 58.8% of simulated straightforward cases meet it and no usability testing has been run.
The 0.60 recall acceptance threshold is a *target*; measured recall is below it, and the release
gate fails accordingly.

## Documentation

[PRD](docs/PRD.md) · [Product roadmap](docs/PRODUCT_ROADMAP.md) · [Program plan](docs/PROGRAM_PLAN.md) ·
[Architecture](docs/ARCHITECTURE.md) · [API spec](docs/API_SPEC.md) ·
[Unit economics](docs/UNIT_ECONOMICS.md) · [Model card](docs/MODEL_CARD.md) ·
[AI safety](docs/AI_SAFETY.md) · [Case study](docs/RESUME_CASE_STUDY.md) ·
[Engineering guide](CLAUDE.md)

## Licence

Released for portfolio and educational purposes. The synthetic data, rate assumptions and watchlist
are fictional and must not be used for any real financial, compliance or risk decision.
