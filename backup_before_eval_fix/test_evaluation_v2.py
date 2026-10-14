"""Tests for AgentPulse ml/evaluate_model_v2.py Improved Evaluation Methodology."""

import tempfile
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.evaluate_model_v2 import (
    BEHAVIORAL_9_FEATURES,
    evaluate_and_train_v2_model,
    extract_scenario_groups_from_logs,
    load_random_forest_model_v2,
)


def test_scenario_group_extraction():
    """Verify scenario group extraction maps 200 sessions into 9 scenario families."""
    groups = extract_scenario_groups_from_logs()

    assert len(groups) == 200
    unique_groups = set(groups)

    # Verify key scenario groups exist
    assert "prompt_injection" in unique_groups
    assert "tool_misuse" in unique_groups
    assert "behavioral_anomaly" in unique_groups
    assert "benign_file_read" in unique_groups
    assert "benign_search" in unique_groups
    assert "benign_email" in unique_groups


def test_v2_feature_selection_excludes_detector_flags():
    """Verify that BEHAVIORAL_9_FEATURES strictly excludes detector flags."""
    assert len(BEHAVIORAL_9_FEATURES) == 9
    assert "security_detection_count" not in BEHAVIORAL_9_FEATURES
    assert "prompt_event_count" not in BEHAVIORAL_9_FEATURES
    assert "total_events" in BEHAVIORAL_9_FEATURES
    assert "session_duration_seconds" in BEHAVIORAL_9_FEATURES


def test_zero_group_overlap_in_split():
    """Verify that GroupShuffleSplit produces zero train/test scenario group overlap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "v2_test.joblib"
        res = evaluate_and_train_v2_model(model_output_path=model_path)

        groups = res["groups"]
        assert groups["group_overlap_count"] == 0
        assert set(groups["train_groups"]).intersection(set(groups["test_groups"])) == set()


def test_v2_model_training_and_serialization():
    """Verify v2 Random Forest training (200 trees) and artifact creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "v2_saved.joblib"
        res = evaluate_and_train_v2_model(model_output_path=model_path)

        assert model_path.exists()
        assert res["model_path"] == str(model_path)

        rf = res["model"]
        assert isinstance(rf, RandomForestClassifier)
        assert rf.n_estimators == 200
        assert rf.n_features_in_ == 9

        artifact = load_random_forest_model_v2(model_path=model_path)
        assert "model" in artifact
        assert artifact["feature_names"] == BEHAVIORAL_9_FEATURES
        assert artifact["group_overlap_count"] == 0


def test_v2_cross_validation_metrics():
    """Verify 5-Fold Stratified Group Cross-Validation results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "v2_cv.joblib"
        res = evaluate_and_train_v2_model(model_output_path=model_path)

        cv = res["cv_metrics"]
        assert len(cv["fold_accuracies"]) == 5
        assert 0.0 <= cv["mean_accuracy"] <= 1.0
        assert 0.0 <= cv["std_accuracy"] <= 1.0


def test_v2_reproducibility():
    """Verify deterministic evaluation outputs with random_state=42."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m1 = Path(tmpdir) / "m1.joblib"
        m2 = Path(tmpdir) / "m2.joblib"

        res1 = evaluate_and_train_v2_model(model_output_path=m1, random_state=42)
        res2 = evaluate_and_train_v2_model(model_output_path=m2, random_state=42)

        assert res1["test_metrics"]["accuracy"] == res2["test_metrics"]["accuracy"]
        assert res1["groups"] == res2["groups"]
        assert res1["cv_metrics"] == res2["cv_metrics"]
