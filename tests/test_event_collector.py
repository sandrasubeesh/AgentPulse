"""Tests for Stage 2 Runtime Event Collector."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from agent.tools import read_file, simulate_send_email
from monitoring.event_collector import EventCollector, redact_sensitive_data


def test_collector_creates_event(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    collector = EventCollector(agent_id="agent_001", log_file=log_file)

    event = collector.record_event("USER_REQUEST", input="Test request")

    assert len(collector.events) == 1
    assert event["event_type"] == "USER_REQUEST"
    assert event["input"] == "Test request"


def test_event_has_uuid(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    collector = EventCollector(log_file=log_file)

    event = collector.record_event("USER_REQUEST", input="Test request")

    assert "event_id" in event
    # Verify valid UUID format
    parsed_uuid = uuid.UUID(event["event_id"])
    assert str(parsed_uuid) == event["event_id"]


def test_event_has_timestamp(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    collector = EventCollector(log_file=log_file)

    event = collector.record_event("USER_REQUEST", input="Test request")

    assert "timestamp" in event
    assert isinstance(event["timestamp"], str)
    assert len(event["timestamp"]) > 0


def test_agent_id_recorded(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    collector = EventCollector(agent_id="agent_custom_007", log_file=log_file)

    event = collector.record_event("USER_REQUEST", input="Test request")

    assert event["agent_id"] == "agent_custom_007"


def test_event_type_recorded(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    collector = EventCollector(log_file=log_file)

    event1 = collector.record_event("USER_REQUEST", input="Hi")
    event2 = collector.record_event("AGENT_RESPONSE", input="Hello")

    assert event1["event_type"] == "USER_REQUEST"
    assert event2["event_type"] == "AGENT_RESPONSE"


def test_log_file_created(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    collector = EventCollector(log_file=log_file)

    assert not log_file.exists()

    collector.record_event("USER_REQUEST", input="Hello world")

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "USER_REQUEST"
    assert parsed["input"] == "Hello world"


def test_read_file_produces_events(monkeypatch, tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    test_collector = EventCollector(log_file=log_file)

    import agent.tools as tools_module

    monkeypatch.setattr(tools_module, "get_event_collector", lambda: test_collector)

    result = read_file.invoke({"filename": "notes.txt"})

    assert "AgentPulse" in result
    events = test_collector.events
    assert len(events) >= 2

    call_event = next(e for e in events if e["event_type"] == "TOOL_CALL")
    result_event = next(e for e in events if e["event_type"] == "TOOL_RESULT")

    assert call_event["tool"] == "read_file"
    assert call_event["input"] == {"filename": "notes.txt"}
    assert result_event["tool"] == "read_file"
    assert "AgentPulse" in result_event["input"]


def test_send_email_produces_events(monkeypatch, tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    test_collector = EventCollector(log_file=log_file)

    import agent.tools as tools_module

    monkeypatch.setattr(tools_module, "get_event_collector", lambda: test_collector)

    result = simulate_send_email("Rahul", "hello")

    assert "SIMULATED EMAIL" in result


def test_tool_error_produces_tool_error_event(monkeypatch, tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    test_collector = EventCollector(log_file=log_file)

    import agent.tools as tools_module

    monkeypatch.setattr(tools_module, "get_event_collector", lambda: test_collector)

    result = read_file.invoke({"filename": "../../.env"})

    assert result.startswith("Error:")
    events = test_collector.events

    call_event = next(e for e in events if e["event_type"] == "TOOL_CALL")
    error_event = next(e for e in events if e["event_type"] == "TOOL_ERROR")

    assert call_event["tool"] == "read_file"
    assert error_event["tool"] == "read_file"
    assert "Error:" in error_event["input"]


def test_redact_sensitive_data() -> None:
    secret_key = "sk-proj-1234567890abcdef1234567890"
    bearer_token = "Bearer secret_jwt_token_here"

    redacted_key = redact_sensitive_data(f"My key is {secret_key}")
    assert secret_key not in redacted_key
    assert "[REDACTED]" in redacted_key

    redacted_bearer = redact_sensitive_data(f"Authorization: {bearer_token}")
    assert "secret_jwt_token_here" not in redacted_bearer
    assert "[REDACTED]" in redacted_bearer
