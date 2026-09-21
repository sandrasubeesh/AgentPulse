"""Tests for AgentPulse ml/random_forest_model.py Supervised Classifier."""

import tempfile
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier

from behavior.feature_engineering import FEATURE_NAMES
from ml.random_forest_model import (
    DEFAULT_MODEL_PATH,
    load_random_forest_model,
    train_random_forest_model,
)


def test_train_random_forest_model_creation_and_split():
    """Verify model training, split sizes (160/40), feature count (11), and evaluation metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_rf_model.joblib"
        res = train_random_forest_model(model_output_path=model_path)

        assert model_path.exists()
        assert res["train_samples"] == 160
        assert res["test_samples"] == 40
        assert res["feature_count"] == 11
        assert res["n_estimators"] == 100
        assert res["feature_names"] == FEATURE_NAMES

        # Verify model instance
        rf = res["model"]
        assert isinstance(rf, RandomForestClassifier)
        assert rf.n_estimators == 100
        assert rf.class_weight == "balanced"
        assert rf.n_features_in_ == 11

        # Verify metrics structure
        metrics = res["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics


def test_model_file_creation_and_loading():
    """Verify model artifact serialization to joblib file and artifact contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "saved_model.joblib"
        train_random_forest_model(model_output_path=model_path)

        assert model_path.exists()

        artifact = load_random_forest_model(model_path=model_path)
        assert "model" in artifact
        assert isinstance(artifact["model"], RandomForestClassifier)
        assert artifact["feature_names"] == FEATURE_NAMES
        assert artifact["train_size"] == 160
        assert artifact["test_size"] == 40



def test_model_prediction():
    """Verify that loaded model can predict on test inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_pred.joblib"
        res = train_random_forest_model(model_output_path=model_path)

        rf = res["model"]
        sample_normal = [[4.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0]]
        sample_suspicious = [[7.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 3.5]]

        pred_normal = rf.predict(sample_normal)
        pred_suspicious = rf.predict(sample_suspicious)

        assert pred_normal[0] == 0
        assert pred_suspicious[0] == 1


def test_reproducibility_with_random_state():
    """Verify that model training is fully reproducible when random_state=42 is specified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m1_path = Path(tmpdir) / "m1.joblib"
        m2_path = Path(tmpdir) / "m2.joblib"

        res1 = train_random_forest_model(model_output_path=m1_path, random_state=42)
        res2 = train_random_forest_model(model_output_path=m2_path, random_state=42)

        assert res1["metrics"] == res2["metrics"]
        assert res1["train_class_distribution"] == res2["train_class_distribution"]
        assert res1["test_class_distribution"] == res2["test_class_distribution"]
