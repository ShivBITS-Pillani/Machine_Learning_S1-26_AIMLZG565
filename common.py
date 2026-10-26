"""Shared data schema and evaluation logic for the UCI online shoppers dataset.

Study notes:
- Keep feature order and preprocessing consistent during train and evaluation.
- The trained Pipeline saves imputation/encoding/scaling and the estimator together.
- ROC AUC needs a continuous score, not a predicted class label.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "online_shoppers_intention.csv"
TEST_FILE = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
METRICS_FILE = ROOT / "model_comparison.csv"
TARGET = "Revenue"

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def validate_test_frame(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate a labelled test dataset and normalize column types.

    The target must be present: upload of feature-only rows is not sufficient for
    calculating evaluation metrics, even though it would support predictions.
    """
    if raw_frame.empty:
        raise ValueError("The CSV is empty.")
    missing = [col for col in FEATURES + [TARGET] if col not in raw_frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    frame = raw_frame[FEATURES + [TARGET]].copy()
    labels = frame[TARGET].astype(str).str.strip().str.lower()
    y = labels.map({"true": 1, "false": 0, "1": 1, "0": 0})
    if y.isna().any():
        invalid = sorted(labels[y.isna()].unique())
        raise ValueError(f"Unrecognized Revenue target values: {invalid}")

    X = frame[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return X, y.to_numpy(dtype=int)


def score_model(model, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Compute all six assignment metrics on the same labelled dataset."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        positive_scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        positive_scores = model.decision_function(X)
    else:
        raise ValueError("Classifier has no score for ROC AUC calculation")

    # AUC is undefined when the uploaded test data contains only one class.
    auc = roc_auc_score(y, positive_scores) if len(np.unique(y)) == 2 else np.nan
    return {
        "Accuracy": float(accuracy_score(y, y_pred)),
        "AUC": float(auc),
        "Precision": float(precision_score(y, y_pred, zero_division=0)),
        "Recall": float(recall_score(y, y_pred, zero_division=0)),
        "F1": float(f1_score(y, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, y_pred)),
    }
