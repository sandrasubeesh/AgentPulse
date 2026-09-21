"""Feature Engineering module for AgentPulse Supervised ML Pipeline (Step 5).

Loads behavioral dataset CSV, separates features and labels, maps categorical
labels to binary integers ('NORMAL' -> 0, 'SUSPICIOUS' -> 1), performs missing
and infinite value validation, and prevents label leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "behavioral_dataset.csv"

FEATURE_NAMES = [
    "total_events",
    "tool_call_count",
    "tool_result_count",
    "tool_error_count",
    "unique_tool_count",
    "file_read_count",
    "search_call_count",
    "email_call_count",
    "security_detection_count",
    "prompt_event_count",
    "session_duration_seconds",
]

LABEL_MAPPING = {
    "NORMAL": 0,
    "SUSPICIOUS": 1,
}


def validate_dataset(
    df: pd.DataFrame,
    expected_features: list[str] | None = None,
) -> None:
    """Validate DataFrame columns, data integrity, NaN/Inf values, and labels."""
    features = expected_features or FEATURE_NAMES

    # 1. Check all expected feature columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in dataset: {missing_cols}")

    # 2. Check target label column exists
    if "label" not in df.columns:
        raise ValueError("Target 'label' column is missing from dataset")

    # 3. Check feature matrix X for missing values (NaN / Null)
    X_df = df[features]
    if X_df.isnull().any().any():
        null_counts = X_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Dataset contains missing (NaN) values in feature columns: {cols_with_nulls}")

    # 4. Check feature matrix X for infinite values (+Inf / -Inf)
    numeric_values = X_df.to_numpy(dtype=np.float64)
    if np.isinf(numeric_values).any():
        raise ValueError("Dataset contains infinite (Inf / -Inf) values in feature columns")

    # 5. Check labels contain only valid categorical values ('NORMAL' / 'SUSPICIOUS') or (0 / 1)
    unique_labels = set(df["label"].unique())
    valid_str_labels = set(LABEL_MAPPING.keys())
    valid_int_labels = set(LABEL_MAPPING.values())

    if not (unique_labels.issubset(valid_str_labels) or unique_labels.issubset(valid_int_labels)):
        raise ValueError(f"Dataset contains invalid label values: {unique_labels}. Expected subset of {valid_str_labels}")


def prepare_behavioral_dataset(
    csv_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load, validate, and prepare the behavioral dataset for Supervised ML.

    Returns:
      - X: 2D numpy array of shape (N, 11) containing float64 feature vectors
      - y: 1D numpy array of shape (N,) containing int64 binary labels (0=NORMAL, 1=SUSPICIOUS)
      - feature_names: list of 11 feature column names in exact order
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_DATASET_CSV

    if not path.exists():
        raise FileNotFoundError(f"Behavioral dataset CSV file not found at: {path}")

    df = pd.read_csv(path)

    # Perform thorough validation before splitting
    validate_dataset(df, expected_features=FEATURE_NAMES)

    # Separate feature matrix X (numerical columns only)
    X_df = df[FEATURE_NAMES].copy()
    X = X_df.to_numpy(dtype=np.float64)

    # Extract target label y and map to binary integers
    y_raw = df["label"]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.to_numpy(dtype=np.int64)
    else:
        y = y_raw.map(LABEL_MAPPING).to_numpy(dtype=np.int64)

    # Extra assertion checks post-conversion
    if np.isnan(X).any():
        raise ValueError("Converted feature matrix X contains NaN values")
    if np.isinf(X).any():
        raise ValueError("Converted feature matrix X contains Inf values")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Converted label vector y contains invalid values: {np.unique(y)}")

    return X, y, list(FEATURE_NAMES)
