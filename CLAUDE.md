# CLAUDE.md - PayBridge

Guidance for Claude Code (and any other engineer) working in this repository.

## What this project is

PayBridge is an **independent personal portfolio project**: a B2B FinTech SaaS
simulation of a cross-border payment operations platform with an explainable AI
fraud engine (**FraudLens AI**). It is not affiliated with any employer,
financial institution or payment processor.

## Hard rules

1. **Synthetic data only.** Never add real customer, bank or sanctions data.
2. **Never initiate a real financial transaction.** No wallets, no keys, no
   exchange or chain connectivity, no provider API calls that move value.
3. **No real screening.** KYC/KYB/AML screening is a fictional-watchlist
   simulation and must be labelled as such in the UI.
4. **No hardcoded results.** Every KPI, model metric and cost saving must be
   computed from the synthetic records or from documented assumptions at run
   time. Never paste a metric into the UI, the docs or the README.
5. **Modelled is not realised.** Cost savings are *modelled under documented
   assumptions*; label them that way everywhere.
6. **Delivered vs proposed.** Documentation describes implemented behaviour as
   delivered and everything else as roadmap.
7. **Runs without paid APIs.** The Anthropic key is optional; every feature has
   a deterministic fallback.
8. **No target leakage.** See "Leakage controls" below.
9. **The app must stay runnable** after every change.

## Architecture

```
app.py                      Streamlit entry point + navigation shell
src/paybridge/
  config.py                 Paths, corridors, rail economics, thresholds
  data/generator.py         Reproducible synthetic dataset (seeded)
  data/loader.py            Cached loading + first-run generation
  db/storage.py             SQLite schema, audit log, generic helpers
  db/seeds.py               Program/experiment/investigation seed data
  payments/pricing.py       Quote + unit economics engine (pure functions)
  payments/payouts.py       Payout state machine + lifecycle events
  compliance/              KYB onboarding workflow + synthetic screening
  fraud/rules.py            Layer A - configurable rules
  fraud/features.py         Leakage-controlled feature pipeline
  fraud/model.py            Layer B - chronological train/val/test + selection
  fraud/explain.py          Layer C - SHAP + deterministic fallback
  fraud/scoring.py          Combined rules + ML decision service
  fraud/evaluation.py       Baseline vs ML vs combined comparison
  copilot/                  Grounded AI copilot (intent -> analytics -> LLM -> validation)
  program/                  Milestones, RAID, dependencies, gates, shadow mode
  product/                  Product KPIs + experiment design
  ui/theme.py               CSS design system
  ui/components.py          Metric cards, tables, chart helpers
  ui/pages/*.py             One module per navigation entry
tests/                      Pytest suite
docs/                       PRD, roadmap, program plan, architecture, model card...
```

Layering: `ui` -> service modules (`payments`, `fraud`, `compliance`, `copilot`,
`program`, `product`) -> `data` / `db` -> `config`. Never import `ui` from a
service module, and keep pricing/rules/feature code free of Streamlit imports.

## Leakage controls (do not break these)

Fields that are post-outcome or label-derived are blocked from the ML feature
set in `fraud/features.py::EXCLUDED_COLUMNS`:
`fraud_label`, `transaction_status`, `compliance_status`, `settlement_time`,
`risk_score`, `rules_*`, `_latent_fraud_prob`, `_in_episode`,
`recipient_amount_local`, and the realised cost columns derived after pricing.
Splits are **chronological** (train < validation < test by timestamp); the
decision threshold is tuned on validation and reported on test.

## Development commands

```bash
make install        # create .venv and install requirements-dev.txt
make data           # regenerate the synthetic dataset (seeded, reproducible)
make train          # train + evaluate FraudLens models, write model bundle
make test           # pytest
make run            # streamlit run app.py
make reset          # wipe generated data/models/SQLite and rebuild
```

Without make: `.venv/bin/python -m pytest -q`,
`.venv/bin/streamlit run app.py`, `.venv/bin/python scripts/generate_data.py`,
`.venv/bin/python scripts/train_model.py`.

## Dependency policy

`requirements.txt` (deployment) uses version RANGES and excludes SHAP so cloud builds resolve fast
on any recent Python. `requirements-dev.txt` (local + CI) pins exact versions and includes SHAP.
The suite must pass on BOTH stacks: run it against each before changing either file. UI code must
therefore stay compatible across Streamlit versions - for example, never pass `height=None` to
`st.dataframe`, since newer Streamlit rejects it.

## Conventions

- Python 3.11, 4-space indent, type hints on public functions, docstrings that
  state the assumption being modelled.
- Pure functions in service modules; Streamlit state only in `ui`.
- Money is USD float rounded at the presentation boundary, never mid-calculation.
- New rails/corridors go in `config.py` only - never hardcode a fee in the UI.
- Every analyst decision writes to `audit_log` via `storage.log_audit`.

## Testing requirements

Any change to pricing, rules, features, the model pipeline, the copilot
validator or a workflow state machine requires a test. The suite must cover:
data reproducibility, fee/FX/margin maths, payout and onboarding transitions,
rule evaluation, leakage controls, threshold tuning, copilot grounding
(hallucinated numbers, invalid transaction IDs, prompt injection, no-key mode)
and the program/release-readiness logic. Run `make test` before declaring work
done, and never state a test count that the suite did not actually produce.
