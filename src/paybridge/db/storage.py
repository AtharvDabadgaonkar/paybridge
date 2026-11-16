"""SQLite persistence layer.

Scope: everything a user CHANGES while using PayBridge - simulated payouts,
investigation decisions, compliance decisions, the audit trail, program
artefacts and experiment designs. The synthetic historical dataset itself
lives in Parquet/CSV files and is never mutated.

PERSISTENCE LIMITATION: on Streamlit Community Cloud the container filesystem
is ephemeral. Writes survive within a running container but are lost when the
app is rebuilt or the container is recycled. This is single-container storage
with no multi-user durability guarantee. See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paybridge import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS payouts (
    payout_id            TEXT PRIMARY KEY,
    reference            TEXT NOT NULL,
    business_id          TEXT NOT NULL,
    business_name        TEXT,
    corridor             TEXT NOT NULL,
    recipient_country    TEXT NOT NULL,
    currency             TEXT NOT NULL,
    amount_usd           REAL NOT NULL,
    payment_rail         TEXT NOT NULL,
    total_customer_cost  REAL NOT NULL,
    platform_direct_cost REAL NOT NULL,
    contribution_margin  REAL NOT NULL,
    fx_markup_cost       REAL NOT NULL,
    recipient_amount_local REAL NOT NULL,
    settlement_hours_p50 REAL NOT NULL,
    status               TEXT NOT NULL,
    risk_score           REAL,
    risk_band            TEXT,
    ml_score             REAL,
    rules_triggered      TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    quote_json           TEXT
);

CREATE TABLE IF NOT EXISTS payout_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    payout_id    TEXT NOT NULL,
    from_status  TEXT,
    to_status    TEXT NOT NULL,
    actor        TEXT NOT NULL,
    note         TEXT,
    occurred_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investigations (
    case_id        TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    business_id    TEXT NOT NULL,
    amount_usd     REAL,
    risk_score     REAL,
    risk_band      TEXT,
    ml_score       REAL,
    rules_triggered TEXT,
    status         TEXT NOT NULL,
    decision       TEXT NOT NULL DEFAULT 'pending',
    assigned_to    TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investigation_notes (
    note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    note       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS compliance_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT NOT NULL,
    business_id TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    reviewer    TEXT NOT NULL,
    note        TEXT,
    decided_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    actor       TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS program_milestones (
    milestone_id  TEXT PRIMARY KEY,
    workstream    TEXT NOT NULL,
    milestone     TEXT NOT NULL,
    owner_role    TEXT NOT NULL,
    start_date    TEXT,
    due_date      TEXT,
    status        TEXT NOT NULL,
    pct_complete  INTEGER NOT NULL DEFAULT 0,
    notes         TEXT,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS program_raid (
    item_id     TEXT PRIMARY KEY,
    item_type   TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT,
    workstream  TEXT,
    severity    TEXT,
    likelihood  TEXT,
    owner_role  TEXT,
    mitigation  TEXT,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS program_dependencies (
    dependency_id TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    upstream      TEXT NOT NULL,
    downstream    TEXT NOT NULL,
    owner_role    TEXT,
    due_date      TEXT,
    status        TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS release_gates (
    gate_id      TEXT PRIMARY KEY,
    gate_area    TEXT NOT NULL,
    criterion    TEXT NOT NULL,
    target       TEXT NOT NULL,
    measurement_source TEXT,
    status       TEXT NOT NULL,
    evidence     TEXT,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    hypothesis    TEXT NOT NULL,
    primary_kpi   TEXT NOT NULL,
    guardrails    TEXT,
    baseline_rate REAL,
    mde_relative  REAL,
    alpha         REAL,
    power         REAL,
    sample_size_per_arm INTEGER,
    runtime_days  INTEGER,
    rollout_criteria TEXT,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_payout_events_payout ON payout_events(payout_id);
CREATE INDEX IF NOT EXISTS idx_notes_case ON investigation_notes(case_id);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_inv_txn ON investigations(transaction_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return Path(config.DB_PATH)


@contextmanager
def connect(path: Path | str | None = None):
    """Yield a SQLite connection with foreign keys and row access by name."""
    target = Path(path) if path is not None else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path | str | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def reset_db(path: Path | str | None = None) -> None:
    """Drop every table and recreate the schema (demo reset)."""
    with connect(path) as conn:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def query_df(sql: str, params: Iterable[Any] = (), path: Path | str | None = None) -> pd.DataFrame:
    with connect(path) as conn:
        return pd.read_sql_query(sql, conn, params=tuple(params))


def execute(sql: str, params: Iterable[Any] = (), path: Path | str | None = None) -> int:
    with connect(path) as conn:
        cur = conn.execute(sql, tuple(params))
        return cur.rowcount


def insert_row(table: str, row: Mapping[str, Any], path: Path | str | None = None) -> None:
    cols = ",".join(row.keys())
    marks = ",".join("?" for _ in row)
    with connect(path) as conn:
        conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))


def log_audit(
    actor: str,
    entity_type: str,
    entity_id: str,
    action: str,
    detail: str | Mapping[str, Any] | None = None,
    path: Path | str | None = None,
) -> None:
    """Append an immutable-by-convention audit record."""
    payload = json.dumps(detail, default=str) if isinstance(detail, Mapping) else (detail or "")
    insert_row(
        "audit_log",
        {
            "occurred_at": now_iso(),
            "actor": actor,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "detail": payload,
        },
        path=path,
    )


def audit_trail(
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 200,
    path: Path | str | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM audit_log"
    params: list[Any] = []
    clauses = []
    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if entity_id:
        clauses.append("entity_id = ?")
        params.append(entity_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY audit_id DESC LIMIT ?"
    params.append(int(limit))
    return query_df(sql, params, path=path)


def get_setting(key: str, default: str | None = None, path: Path | str | None = None) -> str | None:
    df = query_df("SELECT value FROM app_settings WHERE key = ?", (key,), path=path)
    if df.empty:
        return default
    return str(df.iloc[0]["value"])


def set_setting(key: str, value: str, path: Path | str | None = None) -> None:
    insert_row("app_settings", {"key": key, "value": value, "updated_at": now_iso()}, path=path)
