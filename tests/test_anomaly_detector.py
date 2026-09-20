"""Tests for Stage 4 BehaviorAnomalyDetector."""

from __future__ import annotations

import pytest

from behavior.anomaly_detector import BehaviorAnomalyDetector, get_baseline_training_data
from behavior.feature_extractor import extract_features


def test_insufficient_training_data_raises_error() -> None:
    detector = BehaviorAnomalyDetector()
    with pytest.raises(ValueError, match="Insufficient training data"):
        detector.train([[1.0] * 11])


def test_model_trains_and_predicts_normal_vs_anomalous() -> None:
    detector = BehaviorAnomalyDetector(contamination=0.1, random_state=42)
    baseline_data = get_baseline_training_data()

    detector.train(baseline_data)
    assert detector.is_trained

    # Normal session feature vector
    normal_events = [
        {"event_type": "USER_REQUEST", "input": "What is Python?"},
        {"event_type": "SECURITY_ANALYSIS", "detected": False},
        {"event_type": "TOOL_CALL", "tool": "search_knowledge"},
        {"event_type": "TOOL_RESULT", "tool": "search_knowledge"},
        {"event_type": "AGENT_RESPONSE", "input": "Python is..."},
    ]
    normal_vec = extract_features(normal_events)
    normal_result = detector.predict(normal_vec)

    assert normal_result["detection_type"] == "BEHAVIORAL_ANOMALY"
    assert normal_result["detected"] is False
    assert "anomaly_score" in normal_result
    assert normal_result["severity"] == "LOW"

    # Anomalous session feature vector (unusually high tool calls & tool errors)
    anomalous_events = [
        {"event_type": "USER_REQUEST", "input": "Heavy scan"},
    ]
    for i in range(15):
        anomalous_events.append({"event_type": "TOOL_CALL", "tool": f"tool_{i % 3}"})
        anomalous_events.append({"event_type": "TOOL_ERROR", "tool": f"tool_{i % 3}"})
    anomalous_events.append({"event_type": "AGENT_RESPONSE", "input": "Done"})

    anomalous_vec = extract_features(anomalous_events)
    anomalous_result = detector.predict(anomalous_vec)

    assert anomalous_result["detected"] is True
    assert anomalous_result["severity"] in ("MEDIUM", "HIGH")
    assert "reason" in anomalous_result
    assert len(anomalous_result["reason"]) > 0
