"""Application bootstrap: dataset, database schema and seed data."""
from __future__ import annotations

from typing import Any

from paybridge.data.loader import clear_caches, ensure_dataset
from paybridge.db import seeds, storage


def bootstrap(force_data: bool = False) -> dict[str, Any]:
    """Make the app ready to serve. Safe to call repeatedly."""
    manifest = ensure_dataset(force=force_data)
    storage.init_db()
    seed_counts = seeds.seed_all()
    return {"manifest": manifest, "seeds": seed_counts}


def reset_demo(regenerate_data: bool = False) -> dict[str, Any]:
    """Demo reset: wipe user-generated state and reseed.

    With ``regenerate_data=True`` the synthetic dataset is rebuilt from the
    fixed seed as well, which restores the app to a pristine demo state.
    """
    storage.reset_db()
    if regenerate_data:
        ensure_dataset(force=True)
    clear_caches()
    result = bootstrap()
    storage.log_audit("system", "app", "demo", "reset", {"regenerate_data": regenerate_data})
    return result
