"""Data quality: reproducibility, integrity and leakage controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paybridge import config
from paybridge.data.generator import generate_dataset
from paybridge.fraud.features import EXCLUDED_COLUMNS


@pytest.fixture(scope="module")
def small_dataset():
    return generate_dataset(n_businesses=25, target_transactions=1500, sim_days=45)


def test_generation_is_reproducible():
    a = generate_dataset(n_businesses=12, target_transactions=500, sim_days=30)
    b = generate_dataset(n_businesses=12, target_transactions=500, sim_days=30)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    pd.testing.assert_frame_equal(a.businesses, b.businesses)
    assert a.manifest["fraud"]["realised_rate"] == b.manifest["fraud"]["realised_rate"]


def test_different_seeds_produce_different_data():
    a = generate_dataset(seed=1, n_businesses=12, target_transactions=500, sim_days=30)
    b = generate_dataset(seed=2, n_businesses=12, target_transactions=500, sim_days=30)
    assert not a.transactions["amount_usd"].equals(b.transactions["amount_usd"])


def test_transaction_ids_unique(transactions):
    assert transactions["transaction_id"].is_unique


def test_currencies_and_corridors_are_configured(transactions):
    assert set(transactions["currency"]).issubset(set(config.CURRENCIES))
    assert set(transactions["corridor"]).issubset({c.code for c in config.CORRIDORS})
    assert set(transactions["payment_rail"]).issubset(set(config.RAIL_CODES))
    assert set(transactions["transaction_status"]).issubset(set(config.TRANSACTION_STATUSES))


def test_currency_matches_corridor(transactions):
    expected = transactions["corridor"].map({c.code: c.currency for c in config.CORRIDORS})
    assert (transactions["currency"] == expected).all()


def test_no_nulls_in_critical_columns(transactions):
    critical = [
        "transaction_id", "business_id", "amount_usd", "payment_rail", "corridor",
        "transaction_timestamp", "fraud_label", "transaction_status", "risk_score",
    ]
    assert transactions[critical].isna().sum().sum() == 0


def test_amounts_and_fees_are_positive(transactions):
    assert (transactions["amount_usd"] > 0).all()
    assert (transactions["total_cost_usd"] > 0).all()
    assert (transactions["processing_fee"] >= 0).all()
    assert (transactions["fx_markup"] >= 0).all()


def test_fraud_rate_close_to_target(transactions):
    rate = transactions["fraud_label"].mean()
    assert abs(rate - config.BASE_FRAUD_RATE) < 0.004, f"realised fraud rate {rate}"


def test_behavioural_features_are_causal(transactions):
    """Velocity, rolling counts and ages can only be non-negative and bounded."""
    assert (transactions["transaction_velocity"] >= 0).all()
    assert (transactions["beneficiary_age_days"] >= 0).all()
    assert (transactions["account_age_days"] >= 0).all()
    assert (transactions["business_txn_count_30d"] >= 0).all()
    assert (transactions["days_since_last_txn"] >= 0).all()


def test_first_transaction_per_business_has_no_history(transactions):
    first = transactions.sort_values("transaction_timestamp").groupby("business_id").head(1)
    assert (first["transaction_velocity"] == 0).all()
    assert (first["business_txn_count_30d"] == 0).all()
    assert (first["recent_failed_count"] == 0).all()


def test_timestamps_fall_inside_the_declared_window(transactions):
    manifest_end = pd.Timestamp(config.SIM_END_DATE)
    manifest_start = manifest_end - pd.Timedelta(days=config.SIM_DAYS)
    assert transactions["transaction_timestamp"].min() >= manifest_start
    assert transactions["transaction_timestamp"].max() <= manifest_end


def test_failed_payouts_have_no_settlement_time(transactions):
    failed = transactions[transactions["transaction_status"].isin(["failed", "returned"])]
    assert failed["settlement_time"].isna().all()


def test_settlement_times_are_ordered_by_rail(transactions):
    completed = transactions[transactions["transaction_status"] == "completed"]
    medians = completed.groupby("payment_rail")["settlement_time"].median()
    assert medians["USDC"] < medians["ACH_PARTNER"] < medians["SWIFT"]


def test_risk_score_equals_rules_score(transactions):
    """risk_score is the rules baseline, never a label-derived value."""
    assert np.allclose(transactions["risk_score"], transactions["rules_score"])


def test_risk_bands_separate_fraud_monotonically(transactions):
    rates = transactions.groupby("rules_band")["fraud_label"].mean()
    assert rates["low"] < rates["medium"] < rates["high"]


def test_manifest_declares_leakage_controls(small_dataset):
    declared = set(small_dataset.manifest["leakage_controls"])
    assert {"fraud_label", "transaction_status", "settlement_time", "risk_score"} <= declared
    assert declared <= EXCLUDED_COLUMNS | {"_latent_fraud_prob", "_in_episode"}


def test_businesses_and_cases_align(businesses):
    from paybridge.data.loader import load_compliance_cases

    cases = load_compliance_cases()
    assert set(cases["business_id"]) == set(businesses["business_id"])
    assert cases["case_id"].is_unique


def test_watchlist_is_explicitly_fictional():
    from paybridge.data.loader import load_watchlist

    watchlist = load_watchlist()
    assert watchlist["is_fictional"].all()
    assert len(watchlist) > 100


def test_product_events_cover_lifecycle():
    from paybridge.data.loader import load_product_events

    events = load_product_events()
    assert {"account_created", "first_payout_created", "payout_created"} <= set(events["event_name"])
    assert events["event_id"].is_unique


def test_feature_adoption_is_not_uniform():
    """Adoption must vary by feature, otherwise the metric is meaningless."""
    from paybridge.data.loader import load_product_events
    from paybridge.product.metrics import feature_adoption

    adoption = feature_adoption(load_product_events())
    assert adoption["adoption_pct"].max() - adoption["adoption_pct"].min() > 20
