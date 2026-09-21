"""Dataset generator for AgentPulse Supervised ML Pipeline.

Parses runtime event sessions from JSONL log files, extracts numerical session
features, calculates session durations, assigns uppercase 'NORMAL' or 'SUSPICIOUS'
labels, and exports a clean behavioral CSV dataset excluding secrets and raw inputs.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "runtime_events.jsonl"
DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "behavioral_dataset.csv"

FEATURE_NAMES = [
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


def parse_valid_sessions(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group runtime events into valid complete sessions.

    A valid session starts with USER_REQUEST and ends with AGENT_RESPONSE.
    Incomplete or invalid sessions are ignored.
    """
    valid_sessions: list[list[dict[str, Any]]] = []
    current_session: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")

        if event_type == "USER_REQUEST":
            # If there was a previous session that was not completed, discard it
            current_session = [event]
        elif current_session:
            current_session.append(event)
            if event_type == "AGENT_RESPONSE":
                valid_sessions.append(current_session)
                current_session = []

    return valid_sessions


def extract_session_features(session_events: list[dict[str, Any]]) -> list[float]:
    """Extract 11 numerical behavioral features from a valid runtime session."""
    if not session_events:
        return [0.0] * len(FEATURE_NAMES)

    total_events = float(len(session_events))
    tool_call_count = 0.0
    tool_result_count = 0.0
    tool_error_count = 0.0
    unique_tools: set[str] = set()
    file_read_count = 0.0
    search_call_count = 0.0
    email_call_count = 0.0
    security_detection_count = 0.0
    prompt_event_count = 0.0

    for event in session_events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")
        tool = event.get("tool")

        if event_type == "USER_REQUEST":
            prompt_event_count += 1.0
        elif event_type == "TOOL_CALL":
            tool_call_count += 1.0
            if tool:
                unique_tools.add(str(tool))
                if tool == "read_file":
                    file_read_count += 1.0
                elif tool == "search_knowledge":
                    search_call_count += 1.0
                elif tool == "send_email":
                    email_call_count += 1.0
        elif event_type == "TOOL_RESULT":
            tool_result_count += 1.0
        elif event_type == "TOOL_ERROR":
            tool_error_count += 1.0
        elif event_type in ("SECURITY_DETECTION", "SECURITY_ANALYSIS", "BEHAVIORAL_ANOMALY"):
            if event.get("detected") is True or (
                isinstance(event.get("input"), dict) and event["input"].get("detected") is True
            ):
                security_detection_count += 1.0

    unique_tool_count = float(len(unique_tools))

    # Calculate session duration in seconds
    session_duration_seconds = 0.0
    start_ts_str = session_events[0].get("timestamp")
    end_ts_str = session_events[-1].get("timestamp")
    if start_ts_str and end_ts_str:
        try:
            start_dt = datetime.fromisoformat(str(start_ts_str))
            end_dt = datetime.fromisoformat(str(end_ts_str))
            duration = (end_dt - start_dt).total_seconds()
            session_duration_seconds = round(max(0.0, float(duration)), 4)
        except (ValueError, TypeError):
            session_duration_seconds = 0.0

    return [
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
    ]


def determine_session_label(session_events: list[dict[str, Any]]) -> str:
    """Determine session label: 'NORMAL' or 'SUSPICIOUS'.

    Returns 'SUSPICIOUS' if any confirmed prompt injection, tool misuse, or
    behavioral anomaly is present in the session; otherwise returns 'NORMAL'.
    """
    for event in session_events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")

        if event_type == "SECURITY_DETECTION":
            return "SUSPICIOUS"

        if event_type in ("SECURITY_ANALYSIS", "BEHAVIORAL_ANOMALY"):
            if event.get("detected") is True:
                return "SUSPICIOUS"
            inp = event.get("input")
            if isinstance(inp, dict) and inp.get("detected") is True:
                return "SUSPICIOUS"

    return "NORMAL"


def generate_behavioral_dataset(
    log_path: Path | str | None = None,
    output_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """Read runtime log, parse valid sessions, extract features & labels, export CSV.

    Returns dict containing:
      - 'csv_path': path to generated CSV file
      - 'total_samples': count of valid session samples
      - 'normal_count': count of NORMAL samples
      - 'suspicious_count': count of SUSPICIOUS samples
      - 'features': list of feature column names
      - 'class_distribution': dict mapping label -> count
    """
    log_file = Path(log_path) if log_path is not None else DEFAULT_LOG_FILE
    out_csv = Path(output_csv_path) if output_csv_path is not None else DEFAULT_DATASET_CSV

    raw_events: list[dict[str, Any]] = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            raw_events.append(json.loads(line_str))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            pass

    valid_sessions = parse_valid_sessions(raw_events)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = list(FEATURE_NAMES) + ["label"]
    rows: list[list[Any]] = []
    normal_count = 0
    suspicious_count = 0

    for session in valid_sessions:
        feature_vector = extract_session_features(session)
        label = determine_session_label(session)

        if label == "NORMAL":
            normal_count += 1
        else:
            suspicious_count += 1

        row = feature_vector + [label]
        rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return {
        "csv_path": str(out_csv),
        "total_samples": len(rows),
        "normal_count": normal_count,
        "suspicious_count": suspicious_count,
        "features": list(FEATURE_NAMES),
        "class_distribution": {
            "NORMAL": normal_count,
            "SUSPICIOUS": suspicious_count,
        },
    }
