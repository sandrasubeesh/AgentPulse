"""Tests for AgentPulse ml/evaluate_model_v2.py Evaluation Methodology."""

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


def test_unavailable_logs_returns_empty_groups_without_fabrication(tmp_path: Path):
    """Verify missing runtime logs return an empty list and never fabricate scenario groups."""
    missing_log = tmp_path / "nonexistent_runtime_events.jsonl"
    groups = extract_scenario_groups_from_logs(log_path=missing_log)
    assert groups == []


def test_v2_feature_selection_excludes_detector_flags():
    """Verify that BEHAVIORAL_9_FEATURES strictly excludes detector flags."""
    assert len(BEHAVIORAL_9_FEATURES) == 9
    assert "security_detection_count" not in BEHAVIORAL_9_FEATURES
    assert "prompt_event_count" not in BEHAVIORAL_9_FEATURES
    assert "total_events" in BEHAVIORAL_9_FEATURES
    assert "session_duration_seconds" in BEHAVIORAL_9_FEATURES


def test_stratified_evaluation_when_groups_unavailable(tmp_path: Path):
    """Verify StratifiedShuffleSplit and StratifiedKFold are used when scenario groups are unavailable."""
    model_path = tmp_path / "v2_stratified.joblib"
    missing_log = tmp_path / "missing_events.jsonl"

    res = evaluate_and_train_v2_model(
        log_path=missing_log,
        model_output_path=model_path,
    )

    assert res["scenario_group_evaluation_available"] is False
    assert res["evaluation_method"] == "stratified_row_level"
    assert "NOT scenario-group validated" in res["evaluation_limitation"]
    assert res["sample_counts"] == {
        "total_samples": 200,
        "train_samples": 160,
        "test_samples": 40,
    }

    groups = res["groups"]
    assert groups["scenario_group_evaluation_available"] is False
    assert groups["train_groups"] == []
    assert groups["test_groups"] == []
    assert groups["group_overlap_count"] is None


def test_v2_model_training_and_serialization(tmp_path: Path):
    """Verify v2 Random Forest training (200 trees), 9 features, and artifact metadata persistence."""
    model_path = tmp_path / "v2_saved.joblib"
    missing_log = tmp_path / "missing_events.jsonl"

    res = evaluate_and_train_v2_model(
        log_path=missing_log,
        model_output_path=model_path,
    )

    assert model_path.exists()
    assert res["model_path"] == str(model_path)

    rf = res["model"]
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 200
    assert rf.n_features_in_ == 9

    artifact = load_random_forest_model_v2(model_path=model_path)
    assert "model" in artifact
    assert artifact["feature_names"] == BEHAVIORAL_9_FEATURES
    assert artifact["evaluation_method"] == "stratified_row_level"
    assert artifact["scenario_group_evaluation_available"] is False
    assert "NOT scenario-group validated" in artifact["evaluation_limitation"]
    assert artifact["group_overlap_count"] is None


def test_v2_cross_validation_and_holdout_metrics_in_valid_ranges(tmp_path: Path):
    """Verify 5-Fold StratifiedKFold CV and holdout metrics are in valid [0.0, 1.0] ranges."""
    model_path = tmp_path / "v2_cv.joblib"
    res = evaluate_and_train_v2_model(
        log_path=tmp_path / "missing.jsonl",
        model_output_path=model_path,
    )

    tm = res["test_metrics"]
    for key in ("accuracy", "precision", "recall", "f1_score"):
        assert 0.0 <= tm[key] <= 1.0

    cv = res["cv_metrics"]
    assert cv["cv_method"] == "StratifiedKFold"
    assert cv["n_splits"] == 5
    assert len(cv["fold_accuracies"]) == 5
    for fold_acc in cv["fold_accuracies"]:
        assert 0.0 <= fold_acc <= 1.0
    assert 0.0 <= cv["mean_accuracy"] <= 1.0
    assert 0.0 <= cv["std_accuracy"] <= 1.0
    assert 0.0 <= cv["mean_precision"] <= 1.0
    assert 0.0 <= cv["mean_recall"] <= 1.0
    assert 0.0 <= cv["mean_f1"] <= 1.0


def test_v2_reproducibility(tmp_path: Path):
    """Verify deterministic evaluation outputs with fixed random_state=42."""
    m1 = tmp_path / "m1.joblib"
    m2 = tmp_path / "m2.joblib"
    missing_log = tmp_path / "missing.jsonl"

    res1 = evaluate_and_train_v2_model(log_path=missing_log, model_output_path=m1, random_state=42)
    res2 = evaluate_and_train_v2_model(log_path=missing_log, model_output_path=m2, random_state=42)

    assert res1["test_metrics"]["accuracy"] == res2["test_metrics"]["accuracy"]
    assert res1["test_metrics"]["confusion_matrix"] == res2["test_metrics"]["confusion_matrix"]
    assert res1["cv_metrics"] == res2["cv_metrics"]
    assert res1["feature_importances"] == res2["feature_importances"]


def test_explicit_valid_scenario_groups_supported_when_provided(tmp_path: Path):
    """Verify group-aware evaluation is enabled ONLY when valid scenario groups are explicitly provided."""
    import pandas as pd

    csv_path = Path(__file__).resolve().parent.parent / "data" / "behavioral_dataset.csv"
    df = pd.read_csv(csv_path)

    # Construct explicit valid scenario group labels aligned with each row's class label
    explicit_groups = []
    normal_idx = 0
    susp_idx = 0
    for label in df["label"]:
        if label == "NORMAL":
            explicit_groups.append(f"benign_family_{normal_idx % 5}")
            normal_idx += 1
        else:
            explicit_groups.append(f"attack_family_{susp_idx % 5}")
            susp_idx += 1

    model_path = tmp_path / "v2_grouped.joblib"
    res = evaluate_and_train_v2_model(
        model_output_path=model_path,
        scenario_groups=explicit_groups,
    )

    assert res["scenario_group_evaluation_available"] is True
    assert res["evaluation_method"] == "group_aware_stratified"
    assert res["groups"]["group_overlap_count"] == 0
    assert set(res["groups"]["train_groups"]).intersection(set(res["groups"]["test_groups"])) == set()

    # Invalid length must raise ValueError rather than inventing fake groups
    with pytest.raises(ValueError, match="does not match dataset row count"):
        evaluate_and_train_v2_model(
            model_output_path=model_path,
            scenario_groups=["only_one_group"],
        )

