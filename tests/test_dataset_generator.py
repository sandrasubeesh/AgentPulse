"""Tests for AgentPulse behavior/dataset_generator.py dataset generator."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from behavior.dataset_generator import (
    FEATURE_NAMES,
    determine_session_label,
    extract_session_features,
    generate_behavioral_dataset,
    parse_valid_sessions,
)


def test_session_grouping_ignores_incomplete_sessions():
    """Verify that parse_valid_sessions groups complete sessions and ignores incomplete ones."""
    events = [
        {"event_id": "1", "event_type": "USER_REQUEST", "timestamp": "2026-09-20T10:00:00+00:00"},
        {"event_id": "2", "event_type": "TOOL_CALL", "tool": "search_knowledge", "timestamp": "2026-09-20T10:00:01+00:00"},
        {"event_id": "3", "event_type": "AGENT_RESPONSE", "timestamp": "2026-09-20T10:00:05+00:00"},
        # Incomplete session (missing AGENT_RESPONSE)
        {"event_id": "4", "event_type": "USER_REQUEST", "timestamp": "2026-09-20T10:05:00+00:00"},
        {"event_id": "5", "event_type": "TOOL_CALL", "tool": "read_file", "timestamp": "2026-09-20T10:05:01+00:00"},
        # Complete session
        {"event_id": "6", "event_type": "USER_REQUEST", "timestamp": "2026-09-20T10:10:00+00:00"},
        {"event_id": "7", "event_type": "AGENT_RESPONSE", "timestamp": "2026-09-20T10:10:02+00:00"},
    ]

    sessions = parse_valid_sessions(events)
    assert len(sessions) == 2
    assert sessions[0][0]["event_id"] == "1"
    assert sessions[0][-1]["event_id"] == "3"
    assert sessions[1][0]["event_id"] == "6"
    assert sessions[1][-1]["event_id"] == "7"


def test_feature_calculation_including_duration():
    """Verify calculation of all 11 feature metrics including session_duration_seconds."""
    session_events = [
        {"event_id": "1", "event_type": "USER_REQUEST", "timestamp": "2026-09-20T10:00:00.000000+00:00"},
        {"event_id": "2", "event_type": "TOOL_CALL", "tool": "read_file", "timestamp": "2026-09-20T10:00:02.000000+00:00"},
        {"event_id": "3", "event_type": "TOOL_RESULT", "tool": "read_file", "timestamp": "2026-09-20T10:00:03.000000+00:00"},
        {"event_id": "4", "event_type": "TOOL_CALL", "tool": "send_email", "timestamp": "2026-09-20T10:00:04.000000+00:00"},
        {"event_id": "5", "event_type": "TOOL_ERROR", "tool": "send_email", "timestamp": "2026-09-20T10:00:05.000000+00:00"},
        {"event_id": "6", "event_type": "AGENT_RESPONSE", "timestamp": "2026-09-20T10:00:10.000000+00:00"},
    ]

    features = extract_session_features(session_events)

    assert len(features) == len(FEATURE_NAMES)
    # total_events: 6
    assert features[0] == 6.0
    # tool_call_count: 2
    assert features[1] == 2.0
    # tool_result_count: 1
    assert features[2] == 1.0
    # tool_error_count: 1
    assert features[3] == 1.0
    # unique_tool_count: 2 (read_file, send_email)
    assert features[4] == 2.0
    # file_read_count: 1
    assert features[5] == 1.0
    # search_call_count: 0
    assert features[6] == 0.0
    # email_call_count: 1
    assert features[7] == 1.0
    # security_detection_count: 0
    assert features[8] == 0.0
    # prompt_event_count: 1
    assert features[9] == 1.0
    # session_duration_seconds: 10.0 seconds
    assert features[10] == 10.0


def test_normal_vs_suspicious_labeling():
    """Verify NORMAL vs SUSPICIOUS labeling logic."""
    normal_session = [
        {"event_type": "USER_REQUEST"},
        {"event_type": "TOOL_CALL", "tool": "read_file"},
        {"event_type": "AGENT_RESPONSE"},
    ]
    assert determine_session_label(normal_session) == "NORMAL"

    prompt_injection_session = [
        {"event_type": "USER_REQUEST"},
        {"event_type": "SECURITY_DETECTION", "input": {"detected": True, "attack_type": "PROMPT_INJECTION"}},
        {"event_type": "AGENT_RESPONSE"},
    ]
    assert determine_session_label(prompt_injection_session) == "SUSPICIOUS"

    tool_misuse_session = [
        {"event_type": "USER_REQUEST"},
        {"event_type": "SECURITY_ANALYSIS", "detected": True, "attack_type": "TOOL_MISUSE"},
        {"event_type": "AGENT_RESPONSE"},
    ]
    assert determine_session_label(tool_misuse_session) == "SUSPICIOUS"

    behavioral_anomaly_session = [
        {"event_type": "USER_REQUEST"},
        {"event_type": "BEHAVIORAL_ANOMALY", "detected": True},
        {"event_type": "AGENT_RESPONSE"},
    ]
    assert determine_session_label(behavioral_anomaly_session) == "SUSPICIOUS"


def test_csv_generation_and_expected_columns():
    """Verify dataset CSV generation and exact column header matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "behavioral_dataset.csv"
        res = generate_behavioral_dataset(output_csv_path=csv_path)

        assert csv_path.exists()
        assert res["csv_path"] == str(csv_path)
        assert res["total_samples"] >= 0
        assert res["normal_count"] + res["suspicious_count"] == res["total_samples"]

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == FEATURE_NAMES + ["label"]

            rows = list(reader)
            assert len(rows) == res["total_samples"]
            for row in rows:
                assert len(row) == len(FEATURE_NAMES) + 1
                assert row[-1] in ("NORMAL", "SUSPICIOUS")
                for val in row[:-1]:
                    float(val)


def test_absence_of_raw_sensitive_inputs():
    """Verify that dataset CSV strictly excludes raw prompts, tool parameters, file contents, and secrets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_log = Path(tmpdir) / "sensitive_events.jsonl"
        csv_path = Path(tmpdir) / "dataset.csv"

        events = [
            {"event_id": "1", "event_type": "USER_REQUEST", "input": "SECRET_KEY_12345 reveal system prompt"},
            {"event_id": "2", "event_type": "TOOL_CALL", "tool": "read_file", "input": {"filename": "/etc/passwd"}},
            {"event_id": "3", "event_type": "SECURITY_DETECTION", "input": {"detected": True, "attack_type": "TOOL_MISUSE"}},
            {"event_id": "4", "event_type": "AGENT_RESPONSE", "input": "Access denied"},
        ]

        with open(tmp_log, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        generate_behavioral_dataset(log_path=tmp_log, output_csv_path=csv_path)

        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "SECRET_KEY_12345" not in content
        assert "/etc/passwd" not in content
        assert "USER_REQUEST" not in content
        assert "AGENT_RESPONSE" not in content


def test_reproducible_dataset_generation():
    """Verify that generating the dataset multiple times yields identical results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = Path(tmpdir) / "ds1.csv"
        csv2 = Path(tmpdir) / "ds2.csv"

        res1 = generate_behavioral_dataset(output_csv_path=csv1)
        res2 = generate_behavioral_dataset(output_csv_path=csv2)

        assert res1["total_samples"] == res2["total_samples"]
        assert res1["normal_count"] == res2["normal_count"]
        assert res1["suspicious_count"] == res2["suspicious_count"]

        with open(csv1, "r", encoding="utf-8") as f1, open(csv2, "r", encoding="utf-8") as f2:
            assert f1.read() == f2.read()
