"""Feature Extraction for AgentPulse Behavioral Anomaly Detection.

Converts a session/window of runtime events into a numerical feature vector.
"""

from __future__ import annotations

from typing import Any

FEATURE_NAMES = [
    "total_events",
    "total_tool_calls",
    "total_tool_errors",
    "file_read_calls",
    "search_calls",
    "email_calls",
    "unique_tools",
    "tool_call_ratio",
    "security_detection_count",
    "prompt_event_count",
    "agent_response_count",
]


def extract_features(events: list[dict[str, Any]] | None) -> list[float]:
    """Extract a numerical feature vector from a list of runtime events.

    Returns a 11-dimensional float vector. Handles empty or None event lists safely.
    """
    if not events:
        return [0.0] * len(FEATURE_NAMES)

    total_events = float(len(events))
    total_tool_calls = 0.0
    total_tool_errors = 0.0
    file_read_calls = 0.0
    search_calls = 0.0
    email_calls = 0.0
    unique_tools_set: set[str] = set()
    security_detection_count = 0.0
    prompt_event_count = 0.0
    agent_response_count = 0.0

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")
        tool = event.get("tool")

        if event_type == "USER_REQUEST":
            prompt_event_count += 1.0
        elif event_type == "AGENT_RESPONSE":
            agent_response_count += 1.0
        elif event_type == "TOOL_CALL":
            total_tool_calls += 1.0
            if tool:
                unique_tools_set.add(str(tool))
                if tool == "read_file":
                    file_read_calls += 1.0
                elif tool == "search_knowledge":
                    search_calls += 1.0
                elif tool == "send_email":
                    email_calls += 1.0
        elif event_type == "TOOL_ERROR":
            total_tool_errors += 1.0
        elif event_type in ("SECURITY_DETECTION", "SECURITY_ANALYSIS"):
            if event.get("detected") is True or (
                isinstance(event.get("input"), dict) and event["input"].get("detected") is True
            ):
                security_detection_count += 1.0

    unique_tools = float(len(unique_tools_set))
    tool_call_ratio = total_tool_calls / max(total_events, 1.0)

    return [
        total_events,
        total_tool_calls,
        total_tool_errors,
        file_read_calls,
        search_calls,
        email_calls,
        unique_tools,
        tool_call_ratio,
        security_detection_count,
        prompt_event_count,
        agent_response_count,
    ]
