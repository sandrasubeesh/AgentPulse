"""Tests for Stage 4 Feature Extractor."""

from __future__ import annotations

from behavior.feature_extractor import FEATURE_NAMES, extract_features


def test_empty_session_handled_safely() -> None:
    vec = extract_features([])
    assert len(vec) == len(FEATURE_NAMES)
    assert vec == [0.0] * len(FEATURE_NAMES)

    vec_none = extract_features(None)
    assert len(vec_none) == len(FEATURE_NAMES)
    assert vec_none == [0.0] * len(FEATURE_NAMES)


def test_feature_vector_structure_and_counts() -> None:
    events = [
        {"event_type": "USER_REQUEST", "input": "Read notes.txt"},
        {"event_type": "SECURITY_ANALYSIS", "detected": False},
        {"event_type": "TOOL_CALL", "tool": "read_file", "input": {"filename": "notes.txt"}},
        {"event_type": "SECURITY_ANALYSIS", "tool": "read_file", "detected": False},
        {"event_type": "TOOL_RESULT", "tool": "read_file", "input": "Content"},
        {"event_type": "AGENT_RESPONSE", "input": "Here is the content"},
    ]

    vec = extract_features(events)
    assert len(vec) == 11
    # total_events = 6
    assert vec[0] == 6.0
    # total_tool_calls = 1
    assert vec[1] == 1.0
    # total_tool_errors = 0
    assert vec[2] == 0.0
    # file_read_calls = 1
    assert vec[3] == 1.0
    # search_calls = 0
    assert vec[4] == 0.0
    # email_calls = 0
    assert vec[5] == 0.0
    # unique_tools = 1
    assert vec[6] == 1.0
    # tool_call_ratio = 1/6
    assert round(vec[7], 2) == round(1.0 / 6.0, 2)
    # security_detection_count = 0
    assert vec[8] == 0.0
    # prompt_event_count = 1
    assert vec[9] == 1.0
    # agent_response_count = 1
    assert vec[10] == 1.0
