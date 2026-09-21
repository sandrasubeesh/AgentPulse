"""Random Forest Model Training for AgentPulse Supervised ML Pipeline (Step 6).

Loads preprocessed dataset, splits into 80% train / 20% test sets, trains a
reproducible RandomForestClassifier, evaluates metrics, and saves the trained
model artifact for future runtime behavioral inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from behavior.feature_engineering import LABEL_MAPPING, prepare_behavioral_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_behavior_model.joblib"


def train_random_forest_model(
    csv_path: Path | str | None = None,
    model_output_path: Path | str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 100,
) -> dict[str, Any]:
    """Train and evaluate a Random Forest classifier on the behavioral dataset.

    Returns dict containing model instance, evaluation metrics, train/test counts,
    feature list, and artifact path.
    """
    out_model_path = Path(model_output_path) if model_output_path is not None else DEFAULT_MODEL_PATH

    # 1. Load feature matrix X, label vector y, and feature names using feature engineering
    X, y, feature_names = prepare_behavioral_dataset(csv_path=csv_path)

    # 2. Perform stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 3. Train RandomForestClassifier on training set only
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    # 4. Predict and evaluate on test set
    y_pred = rf_model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

    # 5. Save model artifact with feature metadata
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": rf_model,
        "feature_names": feature_names,
        "label_mapping": LABEL_MAPPING,
        "metrics": metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_estimators": n_estimators,
        "random_state": random_state,
        "train_class_distribution": {
            "0_NORMAL": int(np.sum(y_train == 0)),
            "1_SUSPICIOUS": int(np.sum(y_train == 1)),
        },
        "test_class_distribution": {
            "0_NORMAL": int(np.sum(y_test == 0)),
            "1_SUSPICIOUS": int(np.sum(y_test == 1)),
        },
    }

    joblib.dump(artifact, out_model_path)

    return {
        "model": rf_model,
        "artifact": artifact,
        "model_path": str(out_model_path),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_names),
        "n_estimators": n_estimators,
        "metrics": metrics,
        "feature_names": feature_names,
        "train_class_distribution": artifact["train_class_distribution"],
        "test_class_distribution": artifact["test_class_distribution"],
    }


def load_random_forest_model(
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load saved Random Forest model artifact."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(f"Random Forest model artifact not found at: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Loaded file is not a valid Random Forest model artifact")

    return artifact
