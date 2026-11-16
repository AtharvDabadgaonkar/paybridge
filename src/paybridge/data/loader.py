"""Dataset loading with caching and first-run generation."""
from __future__ import annotations

import functools
import json
from typing import Callable

import numpy as np
import pandas as pd

from paybridge import config


def _cached(func: Callable) -> Callable:
    """Use Streamlit's data cache when available, else an LRU cache."""
    try:  # pragma: no cover - exercised only under Streamlit
        import streamlit as st
        from streamlit.runtime import exists as runtime_exists

        if runtime_exists():
            return st.cache_data(show_spinner=False)(func)
    except Exception:
        pass
    return functools.lru_cache(maxsize=8)(func)


def dataset_exists() -> bool:
    return all(
        p.exists()
        for p in (
            config.TRANSACTIONS_FILE,
            config.BUSINESSES_FILE,
            config.COMPLIANCE_FILE,
            config.EVENTS_FILE,
            config.WATCHLIST_FILE,
        )
    )


def ensure_dataset(force: bool = False) -> dict:
    """Generate and persist the synthetic dataset if it is missing."""
    if dataset_exists() and not force:
        return load_manifest()
    from paybridge.data.generator import generate_dataset, write_dataset

    dataset = generate_dataset()
    manifest = write_dataset(dataset)
    clear_caches()
    return manifest


def clear_caches() -> None:
    for fn in (
        load_transactions,
        load_businesses,
        load_compliance_cases,
        load_product_events,
        load_watchlist,
        load_manifest,
    ):
        for attr in ("clear", "cache_clear"):
            if hasattr(fn, attr):
                getattr(fn, attr)()


def _enrich_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
    df["date"] = df["transaction_timestamp"].dt.date
    df["week"] = df["transaction_timestamp"].dt.to_period("W").dt.start_time
    df["month"] = df["transaction_timestamp"].dt.to_period("M").astype(str)
    df["is_successful"] = (df["transaction_status"] == "completed").astype(int)
    df["is_under_review"] = (df["transaction_status"] == "under_review").astype(int)
    df["cost_pct_of_amount"] = np.where(
        df["amount_usd"] > 0, df["total_cost_usd"] / df["amount_usd"] * 100.0, np.nan
    )
    df["settlement_days"] = df["settlement_time"] / 24.0
    return df


@_cached
def load_transactions() -> pd.DataFrame:
    ensure_dataset()
    return _enrich_transactions(pd.read_parquet(config.TRANSACTIONS_FILE))


@_cached
def load_businesses() -> pd.DataFrame:
    ensure_dataset()
    return pd.read_parquet(config.BUSINESSES_FILE)


@_cached
def load_compliance_cases() -> pd.DataFrame:
    ensure_dataset()
    df = pd.read_parquet(config.COMPLIANCE_FILE)
    for col in ("submitted_at", "decided_at"):
        df[col] = pd.to_datetime(df[col])
    return df


@_cached
def load_product_events() -> pd.DataFrame:
    ensure_dataset()
    df = pd.read_parquet(config.EVENTS_FILE)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["date"] = df["event_timestamp"].dt.date
    df["week"] = df["event_timestamp"].dt.to_period("W").dt.start_time
    return df


@_cached
def load_watchlist() -> pd.DataFrame:
    ensure_dataset()
    return pd.read_parquet(config.WATCHLIST_FILE)


@_cached
def load_manifest() -> dict:
    if not config.DATASET_MANIFEST.exists():
        return {}
    return json.loads(config.DATASET_MANIFEST.read_text())


def business_options() -> list[tuple[str, str]]:
    df = load_businesses()
    return [(str(r.business_id), str(r.legal_name)) for r in df.itertuples()]
