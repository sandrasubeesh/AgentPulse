"""Tests for AgentPulse behavior/feature_engineering.py module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavior.feature_engineering import (
    FEATURE_NAMES,
    LABEL_MAPPING,
    prepare_behavioral_dataset,
    validate_dataset,
)


def test_prepare_behavioral_dataset_shape_and_types():
    """Test loading existing CSV dataset, verifying X shape (200, 11) and y shape (200,)."""
    X, y, feature_names = prepare_behavioral_dataset()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(feature_names, list)

    assert X.shape == (200, 11)
    assert y.shape == (200,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64


def test_feature_selection_and_order():
    """Verify that feature names and matrix columns match FEATURE_NAMES in exact order."""
    X, y, feature_names = prepare_behavioral_dataset()

    assert feature_names == FEATURE_NAMES
    assert len(feature_names) == 11
    assert feature_names[0] == "total_events"
    assert feature_names[-1] == "session_duration_seconds"


def test_label_encoding_mapping():
    """Verify that labels are converted correctly: NORMAL -> 0, SUSPICIOUS -> 1."""
    X, y, feature_names = prepare_behavioral_dataset()

    unique_labels = set(np.unique(y))
    assert unique_labels == {0, 1}

    # Count NORMAL (0) and SUSPICIOUS (1)
    normal_count = int(np.sum(y == 0))
    suspicious_count = int(np.sum(y == 1))

    assert normal_count == 100
    assert suspicious_count == 100



def test_no_label_leakage():
    """Verify that label column is stripped from X and feature_names."""
    X, y, feature_names = prepare_behavioral_dataset()

    assert "label" not in feature_names
    assert X.shape[1] == 11
    # Check that y is 1D array of integers, separate from X
    assert y.ndim == 1


def test_missing_value_validation():
    """Verify that validate_dataset raises ValueError when NaN values exist in features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "nan_dataset.csv"

        df_data = {col: [1.0, 2.0, 3.0] for col in FEATURE_NAMES}
        df_data["total_events"] = [1.0, np.nan, 3.0]
        df_data["label"] = ["NORMAL", "NORMAL", "SUSPICIOUS"]

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="NaN"):
            prepare_behavioral_dataset(csv_path=csv_path)


def test_infinite_value_validation():
    """Verify that validate_dataset raises ValueError when Inf values exist in features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "inf_dataset.csv"

        df_data = {col: [1.0, 2.0, 3.0] for col in FEATURE_NAMES}
        df_data["session_duration_seconds"] = [1.0, np.inf, 3.0]
        df_data["label"] = ["NORMAL", "SUSPICIOUS", "NORMAL"]

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="infinite"):
            prepare_behavioral_dataset(csv_path=csv_path)


def test_invalid_label_validation():
    """Verify that validate_dataset raises ValueError when unknown labels are present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "bad_label_dataset.csv"

        df_data = {col: [1.0, 2.0, 3.0] for col in FEATURE_NAMES}
        df_data["label"] = ["NORMAL", "UNKNOWN_ATTACK", "SUSPICIOUS"]

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="invalid label values"):
            prepare_behavioral_dataset(csv_path=csv_path)
