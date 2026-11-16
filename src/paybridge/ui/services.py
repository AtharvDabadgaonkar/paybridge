"""Cached UI-side access to expensive services (model, scored dataset)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from paybridge.data.loader import load_transactions
from paybridge.fraud.model import ModelBundle
from paybridge.fraud.scoring import load_bundle_safe, score_frame


@st.cache_resource(show_spinner="Loading the FraudLens model...")
def model_bundle() -> ModelBundle | None:
    return load_bundle_safe()


@st.cache_data(show_spinner="Scoring transactions with FraudLens...")
def scored_transactions() -> pd.DataFrame:
    """The full synthetic dataset with rules, ML and combined scores attached."""
    bundle = model_bundle()
    return score_frame(load_transactions(), bundle)


@st.cache_data(show_spinner=False)
def background_sample(n: int = 300) -> pd.DataFrame:
    df = load_transactions()
    return df.sample(min(n, len(df)), random_state=11)


def model_status_note() -> str:
    bundle = model_bundle()
    if bundle is None:
        return "No trained model is loaded — FraudLens is running on the rules layer only."
    report = bundle.report
    return (
        f"Model `{bundle.model_name}` selected on {report.selection_criterion}; "
        f"decision threshold {bundle.threshold:.3f} tuned on the validation split."
    )
