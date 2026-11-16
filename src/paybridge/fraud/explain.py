"""FraudLens Layer C - explainability.

Primary path: SHAP values computed from the trained model. Fallback path: a
deterministic, clearly-labelled attribution based on how far each feature sits
from the training population together with the model's own importance
weights. The method actually used is always reported so a rule-based or
statistical explanation is never presented as a SHAP explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from paybridge.fraud import features as F
from paybridge.fraud.model import ModelBundle

try:  # pragma: no cover - import guard
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False

_EXPLAINER_CACHE: dict[str, Any] = {}

METHOD_SHAP_TREE = "shap_tree"
METHOD_SHAP_LINEAR = "shap_linear"
METHOD_FALLBACK = "deterministic_fallback"


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    contribution: float
    direction: str          # "increases risk" | "reduces risk"

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": self.value,
            "contribution": self.contribution,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class ExplanationResult:
    method: str
    base_value: float
    contributions: tuple[FeatureContribution, ...]
    shap_available: bool
    note: str = ""

    @property
    def is_shap(self) -> bool:
        return self.method in {METHOD_SHAP_TREE, METHOD_SHAP_LINEAR}

    def top(self, k: int = 8) -> tuple[FeatureContribution, ...]:
        return tuple(sorted(self.contributions, key=lambda c: abs(c.contribution), reverse=True)[:k])

    def as_frame(self, k: int = 8) -> pd.DataFrame:
        return pd.DataFrame([c.as_dict() for c in self.top(k)])


def _transformed(bundle: ModelBundle, df: pd.DataFrame) -> np.ndarray:
    X = F.build_feature_frame(df)
    return np.asarray(bundle.pipeline.named_steps["prep"].transform(X))


def _background(bundle: ModelBundle, sample: pd.DataFrame | None, n: int = 200) -> np.ndarray | None:
    if sample is None or sample.empty:
        return None
    take = sample.sample(min(n, len(sample)), random_state=0)
    return _transformed(bundle, take)


def get_explainer(bundle: ModelBundle, background: pd.DataFrame | None = None) -> tuple[Any, str]:
    """Build (and cache) a SHAP explainer for the bundle's estimator."""
    if not SHAP_AVAILABLE:
        return None, METHOD_FALLBACK

    key = f"{bundle.model_name}:{id(bundle.pipeline)}"
    if key in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[key]

    estimator = bundle.pipeline.named_steps["clf"]
    try:
        if bundle.model_name in {"random_forest", "hist_gradient_boosting"}:
            explainer = shap.TreeExplainer(estimator)
            method = METHOD_SHAP_TREE
        else:
            bg = _background(bundle, background)
            if bg is None:
                return None, METHOD_FALLBACK
            explainer = shap.LinearExplainer(estimator, bg)
            method = METHOD_SHAP_LINEAR
    except Exception:
        return None, METHOD_FALLBACK

    _EXPLAINER_CACHE[key] = (explainer, method)
    return explainer, method


def _positive_class_values(raw: Any, n_features: int) -> np.ndarray:
    """Normalise SHAP output shapes across versions to a 1-D positive-class vector."""
    values = raw.values if hasattr(raw, "values") else raw
    if isinstance(values, list):               # [class0, class1]
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:                       # (rows, features, classes)
        values = values[0, :, -1]
    elif values.ndim == 2:
        values = values[0] if values.shape[0] == 1 else values[:, -1]
    values = np.ravel(values)
    if values.size != n_features and values.size % n_features == 0:
        values = values[:n_features]
    return values


def _base_value(explainer: Any) -> float:
    expected = getattr(explainer, "expected_value", 0.0)
    if isinstance(expected, (list, tuple, np.ndarray)):
        arr = np.ravel(np.asarray(expected))
        return float(arr[-1])
    return float(expected)


def _fallback_explanation(
    bundle: ModelBundle, row: pd.DataFrame, background: pd.DataFrame | None, top_k: int
) -> ExplanationResult:
    """Deterministic attribution: |z-score| x model importance, signed.

    This is NOT a SHAP value and is labelled as such everywhere it is shown.
    """
    x = _transformed(bundle, row)[0]
    names = bundle.feature_names
    estimator = bundle.pipeline.named_steps["clf"]

    if hasattr(estimator, "feature_importances_"):
        weights = np.asarray(estimator.feature_importances_, dtype=float)
        signed = None
    elif hasattr(estimator, "coef_"):
        coef = np.ravel(np.asarray(estimator.coef_, dtype=float))
        weights = np.abs(coef)
        signed = coef
    else:  # pragma: no cover - defensive
        weights = np.ones(len(x))
        signed = None

    bg = _background(bundle, background, 500)
    if bg is not None and len(bg):
        centre = bg.mean(axis=0)
        spread = bg.std(axis=0)
    else:
        centre = np.zeros_like(x)
        spread = np.ones_like(x)
    spread = np.where(spread <= 1e-9, 1.0, spread)
    z = (x - centre) / spread

    contribution = weights[: len(z)] * z if signed is None else signed[: len(z)] * (x - centre)

    contributions = [
        FeatureContribution(
            feature=names[i] if i < len(names) else f"f{i}",
            value=float(x[i]),
            contribution=float(contribution[i]),
            direction="increases risk" if contribution[i] >= 0 else "reduces risk",
        )
        for i in range(len(contribution))
    ]
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    return ExplanationResult(
        method=METHOD_FALLBACK,
        base_value=0.0,
        contributions=tuple(contributions[: max(top_k, 12)]),
        shap_available=SHAP_AVAILABLE,
        note=(
            "SHAP was unavailable for this model, so contributions are a deterministic "
            "statistical attribution (feature deviation from the training population x model "
            "importance). Treat the ordering as indicative, not as exact SHAP values."
        ),
    )


def explain_transaction(
    bundle: ModelBundle,
    row: pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_k: int = 8,
) -> ExplanationResult:
    """Explain one transaction. ``row`` must be a single-row DataFrame."""
    if len(row) != 1:
        raise ValueError("explain_transaction expects exactly one row")

    explainer, method = get_explainer(bundle, background)
    if explainer is None:
        return _fallback_explanation(bundle, row, background, top_k)

    try:
        x = _transformed(bundle, row)
        raw = explainer.shap_values(x) if hasattr(explainer, "shap_values") else explainer(x)
        values = _positive_class_values(raw, x.shape[1])
        if values.size != x.shape[1]:
            raise ValueError("Unexpected SHAP output shape")
        names = bundle.feature_names
        contributions = [
            FeatureContribution(
                feature=names[i] if i < len(names) else f"f{i}",
                value=float(x[0, i]),
                contribution=float(values[i]),
                direction="increases risk" if values[i] >= 0 else "reduces risk",
            )
            for i in range(len(values))
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return ExplanationResult(
            method=method,
            base_value=_base_value(explainer),
            contributions=tuple(contributions[: max(top_k, 12)]),
            shap_available=True,
        )
    except Exception:
        return _fallback_explanation(bundle, row, background, top_k)


def global_importance(bundle: ModelBundle, sample: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    """Mean absolute SHAP value per feature over a sample (model-level view)."""
    if sample.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_contribution", "method"])

    take = sample.sample(min(max_rows, len(sample)), random_state=7)
    explainer, method = get_explainer(bundle, sample)
    names = bundle.feature_names

    if explainer is not None:
        try:
            x = _transformed(bundle, take)
            raw = explainer.shap_values(x) if hasattr(explainer, "shap_values") else explainer(x)
            values = raw.values if hasattr(raw, "values") else raw
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            mean_abs = np.abs(values).mean(axis=0)
            if mean_abs.size == len(names):
                return (
                    pd.DataFrame(
                        {"feature": names, "mean_abs_contribution": mean_abs, "method": method}
                    )
                    .sort_values("mean_abs_contribution", ascending=False)
                    .reset_index(drop=True)
                )
        except Exception:
            pass

    estimator = bundle.pipeline.named_steps["clf"]
    if hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        imp = np.abs(np.ravel(np.asarray(estimator.coef_, dtype=float)))
    else:  # pragma: no cover
        imp = np.ones(len(names))
    return (
        pd.DataFrame(
            {"feature": names[: len(imp)], "mean_abs_contribution": imp, "method": METHOD_FALLBACK}
        )
        .sort_values("mean_abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
