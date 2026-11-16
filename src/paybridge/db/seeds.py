"""Idempotent seeding of program artefacts, experiment designs and the
initial investigation queue.

Seeding uses INSERT OR IGNORE so a user's edits are never overwritten by a
later application start.
"""
from __future__ import annotations

from pathlib import Path

from paybridge.db import storage
from paybridge.program import definitions as pdefs
from paybridge.product.experiments import seed_designs

SEED_VERSION = "1"


def seed_program(db: Path | str | None = None) -> dict[str, int]:
    now = storage.now_iso()
    milestones = [
        {
            "milestone_id": m.milestone_id,
            "workstream": m.workstream,
            "milestone": m.milestone,
            "owner_role": m.owner_role,
            "start_date": m.start_date,
            "due_date": m.due_date,
            "status": m.status,
            "pct_complete": m.pct_complete,
            "notes": m.notes,
            "updated_at": now,
        }
        for m in pdefs.MILESTONES
    ]
    raid = [
        {
            "item_id": r.item_id,
            "item_type": r.item_type,
            "title": r.title,
            "description": r.description,
            "workstream": r.workstream,
            "severity": r.severity,
            "likelihood": r.likelihood,
            "owner_role": r.owner_role,
            "mitigation": r.mitigation,
            "status": r.status,
            "created_at": now,
            "updated_at": now,
        }
        for r in pdefs.RAID_ITEMS
    ]
    deps = [
        {
            "dependency_id": d.dependency_id,
            "description": d.description,
            "upstream": d.upstream,
            "downstream": d.downstream,
            "owner_role": d.owner_role,
            "due_date": d.due_date,
            "status": d.status,
            "updated_at": now,
        }
        for d in pdefs.DEPENDENCIES
    ]
    gates = [
        {
            "gate_id": g.gate_id,
            "gate_area": g.gate_area,
            "criterion": g.criterion,
            "target": g.target,
            "measurement_source": g.measurement_source,
            "status": "not_assessed",
            "evidence": "",
            "updated_at": now,
        }
        for g in pdefs.RELEASE_GATES
    ]
    counts = {
        "program_milestones": storage.insert_rows("program_milestones", milestones, db, "ignore"),
        "program_raid": storage.insert_rows("program_raid", raid, db, "ignore"),
        "program_dependencies": storage.insert_rows("program_dependencies", deps, db, "ignore"),
        "release_gates": storage.insert_rows("release_gates", gates, db, "ignore"),
    }
    return counts


def seed_experiments(db: Path | str | None = None) -> int:
    now = storage.now_iso()
    rows = []
    for design in seed_designs():
        rows.append(
            {
                "experiment_id": design.experiment_id,
                "name": design.name,
                "hypothesis": design.hypothesis,
                "primary_kpi": design.primary_kpi,
                "guardrails": "; ".join(design.guardrails),
                "baseline_rate": design.baseline_rate,
                "mde_relative": design.mde_relative,
                "alpha": design.alpha,
                "power": design.power,
                "sample_size_per_arm": design.sample_size_per_arm,
                "runtime_days": design.runtime_days,
                "rollout_criteria": design.rollout_criteria,
                "status": "designed",
                "created_at": now,
            }
        )
    return storage.insert_rows("experiments", rows, db, "ignore")


def seed_investigations(limit: int = 60, db: Path | str | None = None) -> int:
    """Open investigation cases for the highest-risk synthetic transactions."""
    from paybridge.data.loader import load_transactions

    existing = storage.query_df("SELECT COUNT(*) AS n FROM investigations", path=db)
    if not existing.empty and int(existing.iloc[0]["n"]) > 0:
        return 0

    txns = load_transactions()
    queue = txns[
        (txns["transaction_status"] == "under_review") | (txns["rules_band"].isin(["high", "critical"]))
    ].nlargest(limit, "rules_score")

    now = storage.now_iso()
    rows = []
    for i, row in enumerate(queue.itertuples(), start=1):
        rows.append(
            {
                "case_id": f"CASE-{i:04d}",
                "transaction_id": row.transaction_id,
                "business_id": row.business_id,
                "amount_usd": float(row.amount_usd),
                "risk_score": float(row.rules_score),
                "risk_band": str(row.rules_band),
                "ml_score": None,
                "rules_triggered": str(row.rules_triggered),
                "status": "open",
                "decision": "pending",
                "assigned_to": "",
                "created_at": now,
                "updated_at": now,
            }
        )
    return storage.insert_rows("investigations", rows, db, "ignore")


def seed_all(db: Path | str | None = None) -> dict[str, int]:
    storage.init_db(db)
    if storage.get_setting("seed_version", path=db) == SEED_VERSION:
        return {"skipped": 1}
    counts = seed_program(db)
    counts["experiments"] = seed_experiments(db)
    counts["investigations"] = seed_investigations(db=db)
    storage.set_setting("seed_version", SEED_VERSION, path=db)
    storage.set_setting("seeded_at", storage.now_iso(), path=db)
    return counts
