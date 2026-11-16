"""FraudLens Layer B - supervised model training, selection and thresholding.

Protocol
--------
1. Chronological train / validation / test split (no shuffling).
2. Three candidate models on an identical preprocessing pipeline.
3. Model selection on VALIDATION PR-AUC (the dataset is heavily imbalanced, so
   PR-AUC is the discriminating metric; ROC-AUC is reported for reference).
4. Decision threshold tuned on VALIDATION only, by maximising F1 and also
   solving for a fixed review-capacity operating point.
5. Final metrics reported on the untouched TEST split.

Nothing here is hardcoded: every number in the UI comes from this run.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from paybridge import config
from paybridge.fraud import features as F

RANDOM_STATE = 42
MODEL_FORMAT_VERSION = 3


def candidate_models() -> dict[str, Any]:
    """Candidate estimators. Class weighting handles the 1.5% positive rate."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Full metric set at one operating point."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positives = tp + fn
    negatives = tn + fp
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if positives else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if positives and negatives else 0.0,
        "false_positive_rate": float(fp / negatives) if negatives else 0.0,
        "false_negative_rate": float(fn / positives) if positives else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "alert_rate": float((tp + fp) / len(y_true)) if len(y_true) else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "n": int(len(y_true)),
        "positives": int(positives),
    }


def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1", grid: int = 400
) -> tuple[float, dict[str, float]]:
    """Pick a decision threshold on validation data.

    objective 'f1'      - maximise F1
    objective 'recall@review' - highest precision subject to alert rate <= 4%
    """
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.50, 0.9999, grid)))
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]
    if candidates.size == 0:
        return 0.5, classification_metrics(y_true, y_prob, 0.5)

    best_t, best_score, best_metrics = 0.5, -1.0, None
    for t in candidates:
        m = classification_metrics(y_true, y_prob, float(t))
        if objective == "f1":
            score = m["f1"]
        elif objective == "recall@review":
            score = m["recall"] if m["alert_rate"] <= 0.04 else -1.0
        else:
            raise ValueError(f"Unknown objective: {objective}")
        if score > best_score:
            best_t, best_score, best_metrics = float(t), score, m
    if best_metrics is None:
        best_metrics = classification_metrics(y_true, y_prob, best_t)
    return best_t, best_metrics


@dataclass
class ModelResult:
    name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    threshold: float
    fit_seconds: float
    validation_pr_auc: float
    selected: bool = False


@dataclass
class TrainingReport:
    selected_model: str
    selection_criterion: str
    threshold: float
    threshold_objective: str
    split_summary: dict
    results: list[dict] = field(default_factory=list)
    feature_count: int = 0
    features: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    trained_at: str = ""
    dataset_rows: int = 0
    capacity_threshold: float | None = None
    capacity_metrics: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelBundle:
    pipeline: Any
    threshold: float
    model_name: str
    report: TrainingReport
    feature_names: list[str]
    format_version: int = MODEL_FORMAT_VERSION

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = F.build_feature_frame(df)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(df) >= t).astype(int)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_models(
    df: pd.DataFrame,
    threshold_objective: str = "f1",
    models: dict[str, Any] | None = None,
) -> tuple[ModelBundle, TrainingReport]:
    """Train every candidate, select on validation PR-AUC, report on test."""
    split = F.chronological_split(df)
    X_train = F.build_feature_frame(split.train)
    X_val = F.build_feature_frame(split.validation)
    X_test = F.build_feature_frame(split.test)
    y_train = split.train[F.TARGET].to_numpy()
    y_val = split.validation[F.TARGET].to_numpy()
    y_test = split.test[F.TARGET].to_numpy()

    results: list[ModelResult] = []
    fitted: dict[str, Any] = {}

    for name, estimator in (models or candidate_models()).items():
        pipeline = F.make_pipeline(estimator)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - started

        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold, val_metrics = tune_threshold(y_val, val_prob, threshold_objective)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test, test_prob, threshold)

        fitted[name] = pipeline
        results.append(
            ModelResult(
                name=name,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                threshold=threshold,
                fit_seconds=round(elapsed, 3),
                validation_pr_auc=val_metrics["pr_auc"],
            )
        )

    best = max(results, key=lambda r: r.validation_pr_auc)
    best.selected = True

    # Secondary operating point: highest recall within a 4% review budget.
    val_prob_best = fitted[best.name].predict_proba(X_val)[:, 1]
    capacity_threshold, _ = tune_threshold(y_val, val_prob_best, "recall@review")
    test_prob_best = fitted[best.name].predict_proba(X_test)[:, 1]
    capacity_metrics = classification_metrics(y_test, test_prob_best, capacity_threshold)

    preprocessor = fitted[best.name].named_steps["prep"]
    feature_names = F.expanded_feature_names(preprocessor)

    report = TrainingReport(
        selected_model=best.name,
        selection_criterion="highest PR-AUC on the chronological validation split",
        threshold=best.threshold,
        threshold_objective=threshold_objective,
        split_summary=split.summary(),
        results=[
            {
                "name": r.name,
                "selected": r.selected,
                "threshold": r.threshold,
                "fit_seconds": r.fit_seconds,
                "validation": r.validation_metrics,
                "test": r.test_metrics,
            }
            for r in results
        ],
        feature_count=len(feature_names),
        features=list(F.ALL_FEATURES),
        excluded_columns=sorted(F.EXCLUDED_COLUMNS),
        trained_at=pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        dataset_rows=int(len(df)),
        capacity_threshold=capacity_threshold,
        capacity_metrics=capacity_metrics,
    )

    bundle = ModelBundle(
        pipeline=fitted[best.name],
        threshold=best.threshold,
        model_name=best.name,
        report=report,
        feature_names=feature_names,
    )
    return bundle, report


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_bundle(bundle: ModelBundle, path: Path | None = None) -> Path:
    target = Path(path or config.MODEL_BUNDLE)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": bundle.pipeline,
            "threshold": bundle.threshold,
            "model_name": bundle.model_name,
            "report": bundle.report.as_dict(),
            "feature_names": bundle.feature_names,
            "format_version": MODEL_FORMAT_VERSION,
        },
        target,
    )
    Path(config.MODEL_METRICS).write_text(json.dumps(bundle.report.as_dict(), indent=2, default=str))
    return target


def load_bundle(path: Path | None = None) -> ModelBundle | None:
    target = Path(path or config.MODEL_BUNDLE)
    if not target.exists():
        return None
    try:
        payload = joblib.load(target)
        if int(payload.get("format_version", 0)) != MODEL_FORMAT_VERSION:
            return None
        report = TrainingReport(**payload["report"])
        return ModelBundle(
            pipeline=payload["pipeline"],
            threshold=float(payload["threshold"]),
            model_name=str(payload["model_name"]),
            report=report,
            feature_names=list(payload["feature_names"]),
        )
    except Exception:
        return None


def ensure_model(df: pd.DataFrame | None = None, retrain: bool = False) -> ModelBundle:
    """Load the trained bundle, training it on first use if necessary."""
    if not retrain:
        bundle = load_bundle()
        if bundle is not None:
            return bundle
    if df is None:
        from paybridge.data.loader import load_transactions

        df = load_transactions()
    bundle, _ = train_models(df)
    save_bundle(bundle)
    return bundle
