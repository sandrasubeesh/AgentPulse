"""Integration tests for Stage 4 Behavioral Anomaly Detection."""

from __future__ import annotations

from pathlib import Path

from agent.agent import run_task
from monitoring.event_collector import EventCollector


def test_behavioral_anomaly_logged_on_task_execution(monkeypatch, tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "runtime_events.jsonl"
    test_collector = EventCollector(log_file=log_file)

    import agent.agent as agent_module

    monkeypatch.setattr(agent_module, "get_event_collector", lambda: test_collector)

    class DummyGraph:
        def invoke(self, inputs):
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="Dummy answer")]}

    run_task("What is LangGraph?", agent=DummyGraph(), collector=test_collector)

    events = test_collector.events
    behavior_event = next((e for e in events if e.get("event_type") == "BEHAVIORAL_ANOMALY"), None)

    assert behavior_event is not None
    assert behavior_event["detection_type"] == "BEHAVIORAL_ANOMALY"
    assert behavior_event["detected"] is False
    assert behavior_event["severity"] == "LOW"


def test_behavioral_contrast_normal_vs_anomalous_session(tmp_path: Path) -> None:
    from behavior.anomaly_detector import get_anomaly_detector
    from behavior.feature_extractor import extract_features

    detector = get_anomaly_detector()

    # Normal session
    normal_session = [
        {"event_type": "USER_REQUEST", "input": "Read notes.txt"},
        {"event_type": "SECURITY_ANALYSIS", "detected": False},
        {"event_type": "TOOL_CALL", "tool": "read_file", "input": {"filename": "notes.txt"}},
        {"event_type": "SECURITY_ANALYSIS", "tool": "read_file", "detected": False},
        {"event_type": "TOOL_RESULT", "tool": "read_file", "input": "AgentPulse notes"},
        {"event_type": "AGENT_RESPONSE", "input": "Here are the notes."},
    ]
    res_normal = detector.predict(extract_features(normal_session))
    assert res_normal["detected"] is False
    assert res_normal["severity"] == "LOW"

    # Anomalous session (excessive tool calls, errors, multiple tools)
    anomalous_session = [
        {"event_type": "USER_REQUEST", "input": "Exfiltrate files"},
    ]
    for i in range(12):
        anomalous_session.append({"event_type": "TOOL_CALL", "tool": f"tool_{i % 4}"})
        anomalous_session.append({"event_type": "TOOL_ERROR", "tool": f"tool_{i % 4}"})
    anomalous_session.append({"event_type": "SECURITY_DETECTION", "detected": True})
    anomalous_session.append({"event_type": "AGENT_RESPONSE", "input": "Finished"})

    res_anomalous = detector.predict(extract_features(anomalous_session))
    assert res_anomalous["detected"] is True
    assert res_anomalous["severity"] in ("MEDIUM", "HIGH")
