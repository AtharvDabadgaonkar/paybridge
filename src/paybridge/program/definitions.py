"""Program delivery plan definitions (seed data for the Program Command Center).

This is a planning artefact for the PayBridge portfolio build: it models how a
TPM would sequence this programme. Milestone statuses describe the state of
THIS repository's delivery where that is verifiable, and are explicitly marked
as planned for work that is not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass

WORKSTREAMS: tuple[str, ...] = (
    "Product Discovery",
    "Payment Infrastructure",
    "Compliance Integration",
    "FraudLens Development",
    "Risk Evaluation",
    "System Testing",
    "Shadow-Mode Validation",
    "Pilot Launch",
    "General Availability Planning",
)

MILESTONE_STATUSES = ("not_started", "in_progress", "blocked", "at_risk", "complete")
RAID_TYPES = ("risk", "assumption", "issue", "dependency")
RAID_STATUSES = ("open", "mitigating", "monitoring", "closed")
DEPENDENCY_STATUSES = ("not_started", "in_progress", "satisfied", "blocked")
GATE_STATUSES = ("not_assessed", "pass", "fail", "waived")


@dataclass(frozen=True)
class MilestoneSeed:
    milestone_id: str
    workstream: str
    milestone: str
    owner_role: str
    start_date: str
    due_date: str
    status: str
    pct_complete: int
    notes: str


MILESTONES: tuple[MilestoneSeed, ...] = (
    MilestoneSeed("M-001", "Product Discovery", "Problem framing and persona definition", "Product Manager", "2026-01-06", "2026-01-24", "complete", 100, "Personas and JTBD captured in docs/PRD.md."),
    MilestoneSeed("M-002", "Product Discovery", "Corridor and rail opportunity sizing", "Product Manager", "2026-01-20", "2026-02-14", "complete", 100, "Corridor set and rail economics modelled in config.py."),
    MilestoneSeed("M-003", "Product Discovery", "MVP scope lock and RICE prioritisation", "Product Manager", "2026-02-10", "2026-02-28", "complete", 100, "RICE table in docs/PRD.md."),
    MilestoneSeed("M-010", "Payment Infrastructure", "Rail economics model and quote engine", "Engineering", "2026-02-17", "2026-03-20", "complete", 100, "payments/pricing.py with vectorised parity tests."),
    MilestoneSeed("M-011", "Payment Infrastructure", "Payout state machine and lifecycle audit", "Engineering", "2026-03-09", "2026-04-03", "complete", 100, "payments/payouts.py; transitions covered by tests."),
    MilestoneSeed("M-012", "Payment Infrastructure", "Executive payment analytics", "Engineering", "2026-03-30", "2026-04-24", "complete", 100, "analytics/kpis.py feeds the Executive Overview."),
    MilestoneSeed("M-013", "Payment Infrastructure", "Production provider integration (ACH/SWIFT/USDC partners)", "Engineering", "2026-09-07", "2026-12-18", "not_started", 0, "PLANNED - out of scope for this simulation; no provider connectivity exists."),
    MilestoneSeed("M-020", "Compliance Integration", "KYB onboarding workflow and status model", "Compliance Ops", "2026-04-06", "2026-05-01", "complete", 100, "compliance/onboarding.py state machine."),
    MilestoneSeed("M-021", "Compliance Integration", "Synthetic watchlist screening simulation", "Compliance Ops", "2026-04-20", "2026-05-15", "complete", 100, "compliance/screening.py against a fictional list."),
    MilestoneSeed("M-022", "Compliance Integration", "Audit trail and reviewer decision capture", "Compliance Ops", "2026-05-11", "2026-05-29", "complete", 100, "SQLite audit_log + compliance_decisions."),
    MilestoneSeed("M-023", "Compliance Integration", "Licensed vendor screening contract", "Compliance Ops", "2026-10-05", "2027-01-29", "not_started", 0, "PLANNED - real screening requires a licensed data vendor."),
    MilestoneSeed("M-030", "FraudLens Development", "Rule library and configurable weights", "Data Science", "2026-05-04", "2026-05-29", "complete", 100, "14 rules in fraud/rules.py, runtime-tunable."),
    MilestoneSeed("M-031", "FraudLens Development", "Leakage-controlled feature pipeline", "Data Science", "2026-05-18", "2026-06-12", "complete", 100, "fraud/features.py with an explicit exclusion list."),
    MilestoneSeed("M-032", "FraudLens Development", "Model training, selection and threshold tuning", "Data Science", "2026-06-08", "2026-07-03", "complete", 100, "Chronological splits; threshold tuned on validation."),
    MilestoneSeed("M-033", "FraudLens Development", "SHAP explanation service", "Data Science", "2026-06-29", "2026-07-24", "complete", 100, "fraud/explain.py with deterministic fallback."),
    MilestoneSeed("M-034", "FraudLens Development", "Grounded investigation copilot", "Data Science", "2026-07-20", "2026-08-14", "complete", 100, "copilot/ pipeline with claim validation."),
    MilestoneSeed("M-040", "Risk Evaluation", "Baseline vs ML vs combined comparison harness", "Data Science", "2026-07-06", "2026-07-31", "complete", 100, "fraud/evaluation.py at matched operating points."),
    MilestoneSeed("M-041", "Risk Evaluation", "Fraud loss and review-cost simulation", "Risk", "2026-07-27", "2026-08-21", "complete", 100, "Loss model documented in UNIT_ECONOMICS.md."),
    MilestoneSeed("M-042", "Risk Evaluation", "Corridor-level performance review", "Risk", "2026-08-17", "2026-09-11", "complete", 100, "Per-corridor metrics on the Model Evaluation page."),
    MilestoneSeed("M-043", "Risk Evaluation", "Independent model validation review", "Risk", "2026-11-02", "2026-12-11", "not_started", 0, "PLANNED - requires a second-line validation function."),
    MilestoneSeed("M-050", "System Testing", "Automated test suite and CI pipeline", "Engineering", "2026-08-10", "2026-09-04", "complete", 100, "Pytest suite + GitHub Actions workflow."),
    MilestoneSeed("M-051", "System Testing", "Workflow integration tests", "QA", "2026-08-24", "2026-09-18", "complete", 100, "End-to-end payout, investigation and onboarding tests."),
    MilestoneSeed("M-052", "System Testing", "Load and latency testing at pilot volume", "QA", "2026-10-12", "2026-11-20", "in_progress", 35, "Scoring latency measured in-process only; no load rig."),
    MilestoneSeed("M-060", "Shadow-Mode Validation", "Shadow scoring harness (no decision impact)", "Data Science", "2026-09-07", "2026-10-02", "complete", 100, "program/shadow_mode.py replays the dataset."),
    MilestoneSeed("M-061", "Shadow-Mode Validation", "Shadow-mode acceptance criteria and evidence pack", "TPM", "2026-09-28", "2026-10-23", "complete", 100, "Criteria evaluated from measured shadow output."),
    MilestoneSeed("M-062", "Shadow-Mode Validation", "Analyst workflow dry run with ops team", "Ops", "2026-10-19", "2026-11-27", "in_progress", 45, "Workflow implemented; no real analyst usability study run."),
    MilestoneSeed("M-070", "Pilot Launch", "Pilot cohort selection and success criteria", "Product Manager", "2026-11-02", "2026-11-27", "in_progress", 40, "Cohort criteria drafted in docs/PRODUCT_ROADMAP.md."),
    MilestoneSeed("M-071", "Pilot Launch", "Pilot runbook and rollback plan", "TPM", "2026-11-16", "2026-12-18", "not_started", 0, "PLANNED."),
    MilestoneSeed("M-072", "Pilot Launch", "Go / no-go review", "TPM", "2026-12-14", "2026-12-23", "not_started", 0, "PLANNED - gated on the release readiness checklist."),
    MilestoneSeed("M-080", "General Availability Planning", "Pricing and packaging decision", "Product Manager", "2027-01-04", "2027-01-29", "not_started", 0, "PLANNED."),
    MilestoneSeed("M-081", "General Availability Planning", "Support, SLA and incident model", "Ops", "2027-01-18", "2027-02-26", "not_started", 0, "PLANNED."),
    MilestoneSeed("M-082", "General Availability Planning", "GA launch readiness review", "TPM", "2027-02-22", "2027-03-19", "not_started", 0, "PLANNED."),
)


@dataclass(frozen=True)
class RaidSeed:
    item_id: str
    item_type: str
    title: str
    description: str
    workstream: str
    severity: str
    likelihood: str
    owner_role: str
    mitigation: str
    status: str


RAID_ITEMS: tuple[RaidSeed, ...] = (
    RaidSeed("R-001", "risk", "Synthetic fraud patterns may not generalise", "The fraud model is trained on simulated behaviour with a known generative process, so measured performance is an upper bound on real-world performance.", "Risk Evaluation", "high", "high", "Data Science", "Treat all metrics as synthetic-only; require shadow mode on real traffic before any decisioning.", "open"),
    RaidSeed("R-002", "risk", "False positives create operational load", "A low decision threshold raises manual review volume and analyst cost per payout.", "Risk Evaluation", "medium", "high", "Risk", "Tune the threshold on validation data at a fixed fraud-loss constraint and track review volume as a guardrail metric.", "mitigating"),
    RaidSeed("R-003", "risk", "Stablecoin payout rail carries regulatory uncertainty", "USDC settlement economics assume a licensed off-ramp partner in each destination market.", "Payment Infrastructure", "high", "medium", "Compliance Ops", "Model the rail as partner-dependent; do not commit corridor availability until partner licensing is confirmed.", "open"),
    RaidSeed("R-004", "risk", "Ephemeral storage on the demo host", "Streamlit Community Cloud containers do not retain the SQLite file across rebuilds, so demo decisions can disappear.", "System Testing", "low", "high", "Engineering", "Document the limitation, keep the dataset regenerable and expose a demo reset.", "monitoring"),
    RaidSeed("R-005", "risk", "LLM copilot could state an unsupported number", "A generative explanation layer can invent figures that contradict the computed evidence.", "FraudLens Development", "high", "medium", "Data Science", "Numbers come only from deterministic analytics; a validator rejects unsupported numeric claims and invalid transaction IDs.", "mitigating"),
    RaidSeed("A-001", "assumption", "Rail fee and FX assumptions approximate market levels", "Fee, FX markup and settlement assumptions are indicative, not a provider rate card.", "Payment Infrastructure", "medium", "high", "Product Manager", "Centralise every assumption in config.py and document it in UNIT_ECONOMICS.md for sensitivity testing.", "open"),
    RaidSeed("A-002", "assumption", "Fraud loss rate of 85% of settled fraudulent volume", "Used only for loss simulation; real recovery rates vary by rail and corridor.", "Risk Evaluation", "medium", "medium", "Risk", "Expose the loss rate as a tunable input on the evaluation page.", "open"),
    RaidSeed("A-003", "assumption", "One analyst-hour costs the equivalent of ~3.3 reviews", "Manual review cost of $18 per case is a modelling assumption for review-cost comparisons.", "Risk Evaluation", "low", "medium", "Ops", "Flag as an assumption wherever review cost is displayed.", "open"),
    RaidSeed("I-001", "issue", "No load testing rig at pilot volume", "Latency is measured in-process on a single container; concurrent-user behaviour is unverified.", "System Testing", "medium", "high", "QA", "Keep M-052 open and exclude latency claims from the release gate evidence until measured under load.", "open"),
    RaidSeed("I-002", "issue", "Onboarding duration target is unvalidated", "The sub-five-minute onboarding goal is a product target measured against simulated durations, not usability testing.", "Compliance Integration", "medium", "high", "Product Manager", "Label the target as proposed; schedule moderated usability testing before any claim.", "open"),
    RaidSeed("D-001", "dependency", "Local payout partner coverage per corridor", "Partner-assisted payout economics depend on a contracted partner in each destination market.", "Payment Infrastructure", "high", "medium", "Partnerships", "Track corridor-by-corridor partner status before enabling a corridor.", "open"),
    RaidSeed("D-002", "dependency", "Licensed screening data vendor", "Real KYB/AML screening cannot ship until a licensed vendor contract is in place.", "Compliance Integration", "high", "high", "Compliance Ops", "Blocking dependency on M-023; the simulation must stay labelled as a demonstration.", "open"),
)


@dataclass(frozen=True)
class DependencySeed:
    dependency_id: str
    description: str
    upstream: str
    downstream: str
    owner_role: str
    due_date: str
    status: str


DEPENDENCIES: tuple[DependencySeed, ...] = (
    DependencySeed("DEP-001", "Rail economics model must be final before executive analytics can report cost KPIs.", "Payment Infrastructure", "Product Discovery", "Engineering", "2026-03-20", "satisfied"),
    DependencySeed("DEP-002", "Leakage-controlled feature pipeline must exist before model training can start.", "FraudLens Development", "Risk Evaluation", "Data Science", "2026-06-12", "satisfied"),
    DependencySeed("DEP-003", "Model selection and threshold must be fixed before the evaluation harness can report a comparison.", "FraudLens Development", "Risk Evaluation", "Data Science", "2026-07-03", "satisfied"),
    DependencySeed("DEP-004", "Investigation workbench must persist decisions before shadow-mode audit completeness can be assessed.", "FraudLens Development", "Shadow-Mode Validation", "Engineering", "2026-10-02", "satisfied"),
    DependencySeed("DEP-005", "Shadow-mode evidence pack is required input to the pilot go/no-go review.", "Shadow-Mode Validation", "Pilot Launch", "TPM", "2026-10-23", "satisfied"),
    DependencySeed("DEP-006", "Load and latency testing must complete before the pilot runbook can set SLOs.", "System Testing", "Pilot Launch", "QA", "2026-11-20", "in_progress"),
    DependencySeed("DEP-007", "Licensed screening vendor contract blocks any real-data compliance launch.", "Compliance Integration", "General Availability Planning", "Compliance Ops", "2027-01-29", "blocked"),
    DependencySeed("DEP-008", "Payout provider integrations block GA pricing and packaging commitments.", "Payment Infrastructure", "General Availability Planning", "Engineering", "2026-12-18", "not_started"),
)


@dataclass(frozen=True)
class GateSeed:
    gate_id: str
    gate_area: str
    criterion: str
    target: str
    measurement_source: str


# Release gates. Targets are ACCEPTANCE THRESHOLDS, not measured results; the
# measured value is computed at assessment time from the named source.
RELEASE_GATES: tuple[GateSeed, ...] = (
    GateSeed("G-01", "Model performance", "Combined engine recall on the held-out test set", ">= 0.60", "fraud/evaluation.py (test split)"),
    GateSeed("G-02", "Model performance", "Combined engine precision on the held-out test set", ">= 0.15", "fraud/evaluation.py (test split)"),
    GateSeed("G-03", "Model performance", "False positive rate on the held-out test set", "<= 3.0%", "fraud/evaluation.py (test split)"),
    GateSeed("G-04", "Model performance", "False-positive reduction vs rules-only at matched recall", "> 0% (direction only)", "fraud/evaluation.py matched operating point"),
    GateSeed("G-05", "Operations", "Manual review volume within analyst capacity", "<= 4% of transactions", "fraud/evaluation.py review volume"),
    GateSeed("G-06", "Operations", "Scoring latency per transaction (in-process, single container)", "<= 50 ms p95", "program/shadow_mode.py measured latency"),
    GateSeed("G-07", "Operations", "High and critical risk payouts require a human decision", "100% of cases", "payments/payouts.py policy path"),
    GateSeed("G-08", "Compliance", "Every analyst decision produces an audit record", "100% of decisions", "db audit_log coverage check"),
    GateSeed("G-09", "Compliance", "Onboarding workflow rejects invalid status transitions", "100% of invalid transitions blocked", "compliance/onboarding.py tests"),
    GateSeed("G-10", "Engineering", "Automated test suite passes", "100% of tests pass", "pytest run"),
    GateSeed("G-11", "Engineering", "AI copilot operates without an API key", "Deterministic fallback available", "copilot fallback tests"),
    GateSeed("G-12", "Engineering", "Copilot rejects unsupported numeric claims", "100% of injected hallucinations blocked", "copilot validator tests"),
    GateSeed("G-13", "Launch", "Shadow-mode validation completed without decision impact", "0 payment decisions changed", "program/shadow_mode.py"),
    GateSeed("G-14", "Launch", "Open blocking issues resolved", "0 open blocking issues", "program RAID log"),
)


RESPONSIBILITY_MATRIX: tuple[dict[str, str], ...] = (
    {"activity": "Product scope and prioritisation", "Product Manager": "A/R", "TPM": "C", "Engineering": "C", "Data Science": "C", "Compliance Ops": "I", "Risk": "I"},
    {"activity": "Rail economics and pricing model", "Product Manager": "A", "TPM": "I", "Engineering": "R", "Data Science": "C", "Compliance Ops": "I", "Risk": "C"},
    {"activity": "Payout workflow implementation", "Product Manager": "C", "TPM": "A", "Engineering": "R", "Data Science": "I", "Compliance Ops": "C", "Risk": "I"},
    {"activity": "KYB / AML workflow design", "Product Manager": "C", "TPM": "C", "Engineering": "R", "Data Science": "I", "Compliance Ops": "A/R", "Risk": "C"},
    {"activity": "Fraud rule library", "Product Manager": "I", "TPM": "C", "Engineering": "C", "Data Science": "R", "Compliance Ops": "C", "Risk": "A"},
    {"activity": "Model training and selection", "Product Manager": "I", "TPM": "C", "Engineering": "C", "Data Science": "A/R", "Compliance Ops": "I", "Risk": "C"},
    {"activity": "Threshold and operating point decision", "Product Manager": "C", "TPM": "C", "Engineering": "I", "Data Science": "R", "Compliance Ops": "I", "Risk": "A"},
    {"activity": "Shadow-mode validation", "Product Manager": "C", "TPM": "A/R", "Engineering": "C", "Data Science": "R", "Compliance Ops": "C", "Risk": "C"},
    {"activity": "Release readiness and go/no-go", "Product Manager": "C", "TPM": "A/R", "Engineering": "C", "Data Science": "C", "Compliance Ops": "C", "Risk": "C"},
    {"activity": "Pilot launch and rollback", "Product Manager": "C", "TPM": "A/R", "Engineering": "R", "Data Science": "I", "Compliance Ops": "C", "Risk": "C"},
)

RACI_LEGEND = "A = Accountable, R = Responsible, C = Consulted, I = Informed"
