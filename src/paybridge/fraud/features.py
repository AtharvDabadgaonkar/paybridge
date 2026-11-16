"""FraudLens feature pipeline with explicit leakage controls.

A feature is admissible only if its value is knowable AT THE MOMENT the
payment is initiated. Everything produced after the payment is attempted -
the status, the realised settlement time, the compliance outcome, the fraud
label itself - is blocked by :data:`EXCLUDED_COLUMNS` and the pipeline raises
if one of them reaches the model matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that must never be used as model inputs.
EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        # label and label-derived
        "fraud_label",
        "_latent_fraud_prob",
        "_in_episode",
        # post-outcome
        "transaction_status",
        "compliance_status",
        "settlement_time",
        "settlement_days",
        "is_successful",
        "is_under_review",
        # rules baseline (kept out so rules and ML stay independently comparable)
        "risk_score",
        "rules_score",
        "rules_band",
        "rules_flagged",
        "rules_triggered",
        "rules_n_triggered",
        # identifiers / leakage-prone keys
        "transaction_id",
        "business_id",
        "beneficiary_id",
        "transaction_timestamp",
        "date",
        "week",
        "month",
        # pricing outputs that are deterministic functions of rail/corridor/amount
        "processing_fee",
        "fx_markup",
        "total_cost_usd",
        "platform_direct_cost",
        "contribution_margin",
        "recipient_amount_local",
        "settlement_hours_p50",
        "cost_pct_of_amount",
    }
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "amount_usd",
    "log_amount",
    "account_age_days",
    "beneficiary_age_days",
    "transaction_velocity",
    "days_since_last_txn",
    "business_txn_count_30d",
    "business_avg_amount_30d",
    "amount_vs_business_mean_ratio",
    "recent_failed_count",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_round_amount",
    "account_details_changed_days",
    "corridor_risk",
    "is_new_beneficiary",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("payment_rail", "corridor", "currency")

ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "fraud_label"


class LeakageError(RuntimeError):
    """Raised when an excluded column reaches the model feature matrix."""


def assert_no_leakage(columns: list[str] | pd.Index) -> None:
    offenders = sorted(set(columns) & EXCLUDED_COLUMNS)
    if offenders:
        raise LeakageError(f"Excluded columns present in the feature matrix: {offenders}")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that only use payment-time information."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount_usd"].clip(lower=0))
    for col in ("beneficiary_age_days", "account_details_changed_days", "days_since_last_txn"):
        if col in out.columns:
            out[col] = out[col].fillna(999.0)
    if "is_new_beneficiary" not in out.columns:
        out["is_new_beneficiary"] = (out["beneficiary_age_days"] <= 7).astype(int)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input frame (features only, leakage-checked)."""
    enriched = add_derived_features(df)
    missing = [c for c in ALL_FEATURES if c not in enriched.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    features = enriched[list(ALL_FEATURES)].copy()
    assert_no_leakage(features.columns)
    return features


def build_preprocessor() -> ColumnTransformer:
    """Numeric scaling + one-hot categoricals, shared by every candidate model."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def expanded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Feature names after preprocessing (used for SHAP and importances)."""
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover - defensive
        return list(ALL_FEATURES)


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp

    def summary(self) -> dict:
        return {
            "train_rows": int(len(self.train)),
            "validation_rows": int(len(self.validation)),
            "test_rows": int(len(self.test)),
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "train_fraud_rate": float(self.train[TARGET].mean()) if len(self.train) else 0.0,
            "validation_fraud_rate": float(self.validation[TARGET].mean()) if len(self.validation) else 0.0,
            "test_fraud_rate": float(self.test[TARGET].mean()) if len(self.test) else 0.0,
        }


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.60, validation_frac: float = 0.20
) -> ChronologicalSplit:
    """Split strictly by time so no future information trains the model."""
    if not 0 < train_frac < 1 or not 0 < validation_frac < 1:
        raise ValueError("Fractions must be between 0 and 1")
    if train_frac + validation_frac >= 1.0:
        raise ValueError("train_frac + validation_frac must leave room for a test split")

    ordered = df.sort_values("transaction_timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + validation_frac))

    train = ordered.iloc[:train_end_idx]
    validation = ordered.iloc[train_end_idx:val_end_idx]
    test = ordered.iloc[val_end_idx:]

    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=pd.Timestamp(train["transaction_timestamp"].max()),
        validation_end=pd.Timestamp(validation["transaction_timestamp"].max()),
    )


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
