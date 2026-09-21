"""Tests for AgentPulse ml/evaluate_model.py module."""

import pytest

from ml.evaluate_model import evaluate_random_forest


def test_evaluate_random_forest_metrics_and_structure():
    """Verify test metrics, confusion matrix, classification report, and sample counts."""
    res = evaluate_random_forest()

    assert "test_metrics" in res
    tm = res["test_metrics"]

    assert tm["accuracy"] == 1.0
    assert tm["precision"] == 1.0
    assert tm["recall"] == 1.0
    assert tm["f1_score"] == 1.0
    assert tm["confusion_matrix"] == [[20, 0], [0, 20]]

    # Class-wise metrics
    cwm = tm["class_wise_metrics"]
    assert cwm["NORMAL"]["precision"] == 1.0
    assert cwm["NORMAL"]["support"] == 20
    assert cwm["SUSPICIOUS"]["precision"] == 1.0
    assert cwm["SUSPICIOUS"]["support"] == 20

    # Sample counts
    sc = res["sample_counts"]
    assert sc["total_samples"] == 200
    assert sc["train_samples"] == 160
    assert sc["test_samples"] == 40


def test_feature_importance_ranking():
    """Verify feature importances for all 11 features sorted in descending order."""
    res = evaluate_random_forest()

    assert "feature_importance_ranking" in res
    ranking = res["feature_importance_ranking"]

    assert len(ranking) == 11
    for item in ranking:
        assert "feature" in item
        assert "importance" in item

    # Verify descending sort order
    importances = [item["importance"] for item in ranking]
    assert importances == sorted(importances, reverse=True)

    # Top feature should be security_detection_count
    assert ranking[0]["feature"] == "security_detection_count"
    assert ranking[0]["importance"] > 0.40


def test_cross_validation_metrics():
    """Verify 5-fold Stratified Cross-Validation scores and statistics."""
    res = evaluate_random_forest()

    assert "cross_validation" in res
    cv = res["cross_validation"]

    assert len(cv["fold_scores"]) == 5
    assert cv["mean_cv_accuracy"] >= 0.95
    assert cv["std_cv_accuracy"] <= 0.05
    for fold_score in cv["fold_scores"]:
        assert 0.0 <= fold_score <= 1.0


def test_ablation_evaluation():
    """Verify feature ablation study removing security_detection_count & prompt_event_count."""
    res = evaluate_random_forest()

    assert "ablation_results" in res
    abl = res["ablation_results"]

    assert set(abl["excluded_features"]) == {"prompt_event_count", "security_detection_count"}
    assert abl["remaining_features_count"] == 9
    assert abl["accuracy"] == 1.0
    assert abl["precision"] == 1.0
    assert abl["recall"] == 1.0
    assert abl["f1_score"] == 1.0


def test_evaluation_reproducibility():
    """Verify deterministic evaluation outputs when random_state=42 is specified."""
    res1 = evaluate_random_forest(random_state=42)
    res2 = evaluate_random_forest(random_state=42)

    assert res1["test_metrics"]["accuracy"] == res2["test_metrics"]["accuracy"]
    assert res1["cross_validation"] == res2["cross_validation"]
    assert res1["ablation_results"] == res2["ablation_results"]
    assert res1["feature_importance_ranking"] == res2["feature_importance_ranking"]
