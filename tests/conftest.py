"""Pytest configuration: shared fixtures and the QA results recorder."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paybridge import config  # noqa: E402

QA_RESULTS_FILE = config.DATA_DIR / "qa_results.json"


@pytest.fixture(scope="session")
def transactions():
    from paybridge.data.loader import load_transactions

    return load_transactions()


@pytest.fixture(scope="session")
def businesses():
    from paybridge.data.loader import load_businesses

    return load_businesses()


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """An isolated SQLite database for tests that write state."""
    from paybridge.db import storage

    path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    storage.init_db(path)
    return path


def pytest_sessionfinish(session, exitstatus):
    """Record the ACTUAL result of this run for the QA summary panel."""
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # pragma: no cover
        return
    stats = reporter.stats
    counts = {key: len(stats.get(key, [])) for key in ("passed", "failed", "error", "skipped", "xfailed")}
    payload = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["error"],
        "skipped": counts["skipped"],
        "total": counts["passed"] + counts["failed"] + counts["error"] + counts["skipped"],
        "exit_status": int(exitstatus),
    }
    try:
        QA_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        QA_RESULTS_FILE.write_text(json.dumps(payload, indent=2))
    except Exception:  # pragma: no cover
        pass
