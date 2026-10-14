"""Feature schemas and extraction for the isolated AgentPulse credibility experiments.

Separates detector-leaking features from behavioral features and explicitly computes
`non_security_total_events` to eliminate indirect leakage through `total_events`
(which previously included SECURITY_ANALYSIS, SECURITY_DETECTION, and BEHAVIORAL_ANOMALY events).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from behavior.dataset_generator import determine_session_label, extract_session_features

SECURITY_EVENT_TYPES = frozenset({
    "SECURITY_DETECTION",
    "SECURITY_ANALYSIS",
    "BEHAVIORAL_ANOMALY",
})

# 1. Full 11-feature schema (includes direct detector leakage via security_detection_count)
FEATURES_11_WITH_DETECTOR = [
    "total_events",
    "tool_call_count",
    "tool_result_count",
    "tool_error_count",
    "unique_tool_count",
    "file_read_count",
    "search_call_count",
    "email_call_count",
    "security_detection_count",
    "prompt_event_count",
    "session_duration_seconds",
]

# 2. v2 9-feature schema (excludes explicit detector flags, but retains raw total_events
#    which indirectly counts SECURITY_* events)
FEATURES_9_RAW_TOTAL_EVENTS = [
    "total_events",
    "tool_call_count",
    "tool_result_count",
    "tool_error_count",
    "unique_tool_count",
    "file_read_count",
    "search_call_count",
    "email_call_count",
    "session_duration_seconds",
]

# 3. De-leaked 9-feature schema (replaces raw total_events with non_security_total_events
#    so security-detector events are completely excluded from event counts)
FEATURES_9_DELEAKED_BEHAVIORAL = [
    "non_security_total_events",
    "tool_call_count",
    "tool_result_count",
    "tool_error_count",
    "unique_tool_count",
    "file_read_count",
    "search_call_count",
    "email_call_count",
    "session_duration_seconds",
]

# 4. Pure tool/duration 8-feature schema (excludes all event-count aggregates and detector flags)
FEATURES_8_PURE_TOOL_TIMING = [
    "tool_call_count",
    "tool_result_count",
    "tool_error_count",
    "unique_tool_count",
    "file_read_count",
    "search_call_count",
    "email_call_count",
    "session_duration_seconds",
]

METADATA_COLUMNS = [
    "session_id",
    "dataset_source",
    "scenario_family",
]

ALL_CSV_FEATURE_COLUMNS = [
    "total_events",
    "non_security_total_events",
    "security_event_overhead",
    "tool_call_count",
    "tool_result_count",
    "tool_error_count",
    "unique_tool_count",
    "file_read_count",
    "search_call_count",
    "email_call_count",
    "security_detection_count",
    "prompt_event_count",
    "session_duration_seconds",
]

EXPERIMENTAL_CSV_HEADERS = METADATA_COLUMNS + ALL_CSV_FEATURE_COLUMNS + ["label"]

LABEL_MAPPING = {"NORMAL": 0, "SUSPICIOUS": 1}


def extract_experimental_session_record(
    session_events: list[dict[str, Any]],
    session_id: str,
    scenario_family: str,
    dataset_source: str = "synthetic",
    explicit_label: str | None = None,
) -> dict[str, Any]:
    """Extract both legacy 11 features and de-leaked behavioral features from a session.

    Computes `non_security_total_events` by counting only operational agent events
    (excluding SECURITY_DETECTION, SECURITY_ANALYSIS, and BEHAVIORAL_ANOMALY).
    """
    base_vec = extract_session_features(session_events)
    (
        total_events,
        tool_call_count,
        tool_result_count,
        tool_error_count,
        unique_tool_count,
        file_read_count,
        search_call_count,
        email_call_count,
        security_detection_count,
        prompt_event_count,
        session_duration_seconds,
    ) = base_vec

    non_security_events = [
        ev
        for ev in session_events
        if isinstance(ev, dict) and ev.get("event_type") not in SECURITY_EVENT_TYPES
    ]
    non_security_total_events = float(len(non_security_events))
    security_event_overhead = round(total_events - non_security_total_events, 4)

    # Recalculate duration safely if needed
    if session_events:
        start_ts_str = session_events[0].get("timestamp")
        end_ts_str = session_events[-1].get("timestamp")
        if start_ts_str and end_ts_str:
            try:
                start_dt = datetime.fromisoformat(str(start_ts_str))
                end_dt = datetime.fromisoformat(str(end_ts_str))
                session_duration_seconds = round(
                    max(0.0, float((end_dt - start_dt).total_seconds())), 4
                )
            except (ValueError, TypeError):
                pass

    label = explicit_label if explicit_label is not None else determine_session_label(session_events)
    if label not in LABEL_MAPPING:
        raise ValueError(f"Invalid session label '{label}'. Expected one of {list(LABEL_MAPPING.keys())}")

    return {
        "session_id": str(session_id),
        "dataset_source": str(dataset_source),
        "scenario_family": str(scenario_family),
        "total_events": float(total_events),
        "non_security_total_events": float(non_security_total_events),
        "security_event_overhead": float(security_event_overhead),
        "tool_call_count": float(tool_call_count),
        "tool_result_count": float(tool_result_count),
        "tool_error_count": float(tool_error_count),
        "unique_tool_count": float(unique_tool_count),
        "file_read_count": float(file_read_count),
        "search_call_count": float(search_call_count),
        "email_call_count": float(email_call_count),
        "security_detection_count": float(security_detection_count),
        "prompt_event_count": float(prompt_event_count),
        "session_duration_seconds": float(session_duration_seconds),
        "label": str(label),
    }
