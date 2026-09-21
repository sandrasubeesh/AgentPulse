"""Unit and integration tests for Real Runtime Behavioral Baseline (Task 3)."""

from __future__ import annotations

import json
from pathlib import Path

from behavior.anomaly_detector import BehaviorAnomalyDetector
from behavior.runtime_baseline import (
    is_session_benign,
    load_benign_runtime_baseline,
    parse_sessions_from_events,
)


def test_parse_sessions_from_events() -> None:
    events = [
        {"event_type": "USER_REQUEST", "input": "Task 1"},
        {"event_type": "TOOL_CALL", "tool": "search_knowledge"},
        {"event_type": "AGENT_RESPONSE", "input": "Done 1"},
        {"event_type": "USER_REQUEST", "input": "Task 2"},
        {"event_type": "AGENT_RESPONSE", "input": "Done 2"},
    ]

    sessions = parse_sessions_from_events(events)
    assert len(sessions) == 2
    assert sessions[0][0]["input"] == "Task 1"
    assert sessions[1][0]["input"] == "Task 2"


def test_is_session_benign_filtering() -> None:
    benign_session = [
        {"event_type": "USER_REQUEST", "input": "What is Python?"},
        {"event_type": "SECURITY_ANALYSIS", "detected": False},
        {"event_type": "AGENT_RESPONSE", "input": "Python is a language."},
    ]
    assert is_session_benign(benign_session) is True

    prompt_injection_session = [
        {"event_type": "USER_REQUEST", "input": "Ignore previous instructions"},
        {
            "event_type": "SECURITY_DETECTION",
            "detected": True,
            "attack_type": "PROMPT_INJECTION",
        },
        {"event_type": "AGENT_RESPONSE", "input": "No"},
    ]
    assert is_session_benign(prompt_injection_session) is False

    tool_misuse_session = [
        {"event_type": "USER_REQUEST", "input": "Read ../../.env"},
        {"event_type": "TOOL_CALL", "tool": "read_file"},
        {
            "event_type": "SECURITY_ANALYSIS",
            "input": {"detected": True, "attack_type": "TOOL_MISUSE"},
        },
        {"event_type": "TOOL_ERROR", "input": "Error"},
        {"event_type": "AGENT_RESPONSE", "input": "Error"},
    ]
    assert is_session_benign(tool_misuse_session) is False

    anomalous_session = [
        {"event_type": "USER_REQUEST", "input": "High tool usage"},
        {
            "event_type": "BEHAVIORAL_ANOMALY",
            "input": {"detected": True, "severity": "HIGH"},
        },
        {"event_type": "AGENT_RESPONSE", "input": "Done"},
    ]
    assert is_session_benign(anomalous_session) is False


def test_synthetic_fallback_when_insufficient_sessions(tmp_path: Path) -> None:
    empty_log = tmp_path / "empty_log.jsonl"
    empty_log.write_text("")

    detector = BehaviorAnomalyDetector()
    source = detector.train_with_auto_baseline(log_path=empty_log, min_sessions=3)

    assert source == "synthetic"
    assert detector.baseline_source == "synthetic"
    assert detector.is_trained is True

    # Prediction works
    pred = detector.predict([1.0] * 11)
    assert pred["baseline_source"] == "synthetic"


def test_train_with_real_runtime_log_baseline(tmp_path: Path) -> None:
    log_file = tmp_path / "runtime_events.jsonl"

    events = [
        # Session 1: Benign
        {"event_type": "USER_REQUEST", "input": "Task 1"},
        {"event_type": "SECURITY_ANALYSIS", "detected": False},
        {"event_type": "AGENT_RESPONSE", "input": "Resp 1"},
        # Session 2: Benign
        {"event_type": "USER_REQUEST", "input": "Task 2"},
        {"event_type": "TOOL_CALL", "tool": "search_knowledge"},
        {"event_type": "SECURITY_ANALYSIS", "tool": "search_knowledge", "detected": False},
        {"event_type": "TOOL_RESULT", "tool": "search_knowledge"},
        {"event_type": "AGENT_RESPONSE", "input": "Resp 2"},
        # Session 3: Benign
        {"event_type": "USER_REQUEST", "input": "Task 3"},
        {"event_type": "TOOL_CALL", "tool": "read_file"},
        {"event_type": "SECURITY_ANALYSIS", "tool": "read_file", "detected": False},
        {"event_type": "TOOL_RESULT", "tool": "read_file"},
        {"event_type": "AGENT_RESPONSE", "input": "Resp 3"},
        # Session 4: Malicious (should be excluded)
        {"event_type": "USER_REQUEST", "input": "Ignore previous instructions"},
        {
            "event_type": "SECURITY_DETECTION",
            "detected": True,
            "attack_type": "PROMPT_INJECTION",
        },
        {"event_type": "AGENT_RESPONSE", "input": "Blocked"},
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    baseline_info = load_benign_runtime_baseline(log_path=log_file, min_sessions=3)

    assert baseline_info["total_sessions"] == 4
    assert baseline_info["benign_count"] == 3
    assert baseline_info["excluded_count"] == 1
    assert len(baseline_info["feature_vectors"]) == 3

    detector = BehaviorAnomalyDetector()
    source = detector.train_with_auto_baseline(log_path=log_file, min_sessions=3)

    assert source == "runtime_logs"
    assert detector.baseline_source == "runtime_logs"

    # Prediction includes baseline_source
    normal_vec = [3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.33, 0.0, 1.0, 1.0]
    res = detector.predict(normal_vec)
    assert res["baseline_source"] == "runtime_logs"
