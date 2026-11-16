# PayBridge — Architecture

## 1. Overview

PayBridge is a single Streamlit process. There is no separate backend service: a second service
would add deployment complexity without adding capability at this scale, so the decision was to keep
one process and keep all business logic in importable, Streamlit-free modules that are unit tested
directly.

```
┌──────────────────────────────────────────────────────────────────────┐
│ app.py — Streamlit shell, navigation, bootstrap, demo reset          │
├──────────────────────────────────────────────────────────────────────┤
│ ui/           theme · components · filters · services (caching)      │
│ ui/pages/     one module per navigation entry (10)                   │
├──────────────────────────────────────────────────────────────────────┤
│ SERVICES (no Streamlit imports)                                      │
│  payments/    pricing engine · payout state machine                  │
│  compliance/  onboarding state machine · screening simulation        │
│  fraud/       rules · features · model · explain · scoring ·         │
│               evaluation · investigations                            │
│  copilot/     intents · analytics · evidence · validator · client    │
│  program/     definitions · program service · shadow mode            │
│  product/     metrics · experiments                                  │
│  analytics/   deterministic KPI layer (shared by UI and copilot)     │
├──────────────────────────────────────────────────────────────────────┤
│ DATA        data/generator · data/loader        db/storage · db/seeds│
├──────────────────────────────────────────────────────────────────────┤
│ config.py   corridors · rail economics · thresholds · paths          │
└──────────────────────────────────────────────────────────────────────┘
      Parquet/CSV (immutable synthetic dataset)   SQLite (user state)
```

Dependencies point one way: `ui` → services → `data`/`db` → `config`. No service module imports
`ui`, which is what makes the whole business layer testable without a browser.

## 2. Component responsibilities

| Component | Responsibility | Never does |
| --- | --- | --- |
| `config.py` | Single source of truth for corridors, rail economics, thresholds, paths | Compute anything |
| `data/generator.py` | Deterministic synthetic dataset from a fixed seed | Read user state |
| `data/loader.py` | Cached loading, first-run generation, derived columns | Mutate the dataset |
| `db/storage.py` | SQLite schema, generic CRUD, audit log | Hold business rules |
| `payments/pricing.py` | Pure quote maths, scalar and vectorised | Touch the database |
| `payments/payouts.py` | Payout state machine, lifecycle events | Price anything |
| `compliance/*` | Onboarding transitions, screening simulation | Claim to be real screening |
| `fraud/rules.py` | Deterministic, attributable rule scoring | See the label |
| `fraud/features.py` | Leakage-controlled feature matrix and chronological splits | Include post-outcome fields |
| `fraud/model.py` | Training, selection, threshold tuning, persistence | Tune on test data |
| `fraud/explain.py` | SHAP with a labelled deterministic fallback | Present a fallback as SHAP |
| `fraud/scoring.py` | Combined decision policy | Auto-approve high risk |
| `fraud/evaluation.py` | Approach comparison and matched operating points | Hardcode a result |
| `copilot/*` | Intent → analytics → evidence → LLM → validation | Compute a number in the LLM |
| `program/*` | Milestones, RAID, gates, shadow replay | Mark an unmeasured gate as passed |
| `analytics/kpis.py` | Every KPI in the product | Return a constant |

## 3. Data flow

**Startup.** `app.py` → `bootstrap()` → `ensure_dataset()` generates the Parquet files on first run
→ `storage.init_db()` creates the schema → `seeds.seed_all()` inserts program artefacts, experiment
designs and the initial investigation queue with `INSERT OR IGNORE`, so a restart never overwrites a
user's edits.

**Read path.** Page → `ui/services` (Streamlit cache) → `data/loader` (Parquet) → service module →
`analytics/kpis` → chart or table. The scored frame (rules + ML + combined) is computed once per
process and cached.

**Write path.** Page → service (`payouts`, `onboarding`, `investigations`, `program`) → `db/storage`
→ SQLite, plus an `audit_log` row written through `storage.log_audit` in the same call.

**Dataset vs user state.** The synthetic dataset is immutable: a compliance decision made in the app
is stored as an override row and overlaid at read time by `apply_overrides`. The dataset can be
regenerated at any time without destroying decision history semantics.

## 4. AI grounding design

```
question
  → detect_intent()            deterministic keyword/pattern routing, no model call
  → build_evidence()           Pandas/NumPy analytics; produces an EvidencePack
       facts (allowlisted numbers) · tables · citable transaction ids
       · untrusted notes (escaped and fenced) · deterministic_answer
  → [optional] Claude          system prompt forbids computing or inventing anything
  → validate_response()        numeric claims, transaction ids, decision claims
  → answer
```

- The deterministic answer is computed **every time**, so the product never depends on the LLM.
- The LLM sees only the evidence pack. Analyst notes enter inside `<untrusted_data>` after escaping
  and tag-stripping, so they cannot close the fence.
- Validation failure ⇒ the generated text is discarded and the deterministic answer is shown, with
  the rejection reason surfaced in the UI.
- Four outcome modes are reported honestly: `llm_verified`, `deterministic_after_rejection`,
  `deterministic_no_llm`, `deterministic`.

## 5. Model training pipeline

```
load_transactions()
  → chronological_split(0.60 / 0.20 / 0.20)     strictly by timestamp, never shuffled
  → build_feature_frame()                        20 features, leakage guard raises on violation
  → ColumnTransformer: StandardScaler + OneHotEncoder(handle_unknown="ignore")
  → fit three candidates (logistic regression, random forest, HistGradientBoosting)
  → tune_threshold() on VALIDATION only (F1; a capacity point is also solved for)
  → select by VALIDATION PR-AUC
  → report metrics on the untouched TEST split
  → joblib bundle: pipeline + threshold + full training report + feature names
```

Leakage control is enforced in code, not by convention: `EXCLUDED_COLUMNS` blocks the label,
post-outcome fields (status, settlement time, compliance status), the rules baseline score and
identifiers; `assert_no_leakage()` raises if any of them reach the matrix, and a test asserts the
feature set and the excluded set are disjoint.

## 6. Persistence and its limits

SQLite holds payouts, payout events, investigations, notes, compliance decisions, the audit log,
program artefacts, experiment designs and settings. WAL mode is enabled and connections are
short-lived.

**Limitation, stated plainly:** on Streamlit Community Cloud the container filesystem is ephemeral.
Writes survive within a running container and are lost when the app is rebuilt or recycled. This is
single-container storage with no multi-user durability guarantee, and the product makes no claim
otherwise. Durable multi-tenant storage is a Phase 2 item.

## 7. Performance

The dataset (~25k rows) is loaded once and cached; model scoring over the full set runs once per
process and is cached. Single-transaction scoring, including rules, model and policy, measures
p95 ≈ 18 ms in process on one container — an interactive-latency figure, not a load test.

## 8. Key architectural decisions

| Decision | Alternative | Why |
| --- | --- | --- |
| Single Streamlit process | Streamlit + FastAPI | A second service adds deployment surface without capability at this scale; `API_SPEC.md` documents the interface the services already implement in Python |
| Parquet dataset + SQLite state | Everything in SQLite | Keeps the generated dataset immutable and regenerable; keeps user state small and auditable |
| Rules score excluded from ML features | Feed rules into the model | Keeps the two layers independently comparable; a combined score that contained the baseline would make the comparison circular |
| Chronological splits | Random k-fold | Random folds leak the future into training for a time-ordered fraud problem |
| Validation-tuned threshold | Tune on test | Tuning on test invalidates the reported metrics |
| Deterministic analytics + LLM narration | LLM-computed answers | An LLM that computes figures cannot be audited; this design makes every number traceable |
| Rule override in the combined policy | Pure model decision | A deterministic red flag must never be silently suppressed by a confident model |
