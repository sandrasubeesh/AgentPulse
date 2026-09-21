"""Runtime Behavioral Baseline loader for AgentPulse Stage 4 IsolationForest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from behavior.feature_extractor import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "runtime_events.jsonl"


def parse_sessions_from_events(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group a flat list of runtime events into discrete session lists based on USER_REQUEST boundary."""
    sessions: list[list[dict[str, Any]]] = []
    current_session: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")
        if event_type == "USER_REQUEST":
            if current_session:
                sessions.append(current_session)
                current_session = []

        if current_session or event_type == "USER_REQUEST":
            current_session.append(event)

        if event_type == "AGENT_RESPONSE":
            if current_session:
                sessions.append(current_session)
                current_session = []

    if current_session:
        sessions.append(current_session)

    return sessions


def is_session_benign(session_events: list[dict[str, Any]]) -> bool:
    """Check if a session contains only benign normal activity (no security detections or anomalies)."""
    if not session_events:
        return False

    for event in session_events:
        event_type = event.get("event_type")

        # Exclude sessions with explicit SECURITY_DETECTION
        if event_type == "SECURITY_DETECTION":
            return False

        # Exclude sessions with detected SECURITY_ANALYSIS
        if event_type == "SECURITY_ANALYSIS":
            if event.get("detected") is True:
                return False
            inp = event.get("input")
            if isinstance(inp, dict) and inp.get("detected") is True:
                return False

        # Exclude sessions with detected BEHAVIORAL_ANOMALY
        if event_type == "BEHAVIORAL_ANOMALY":
            if event.get("detected") is True:
                return False
            inp = event.get("input")
            if isinstance(inp, dict) and inp.get("detected") is True:
                return False

    return True


def load_benign_runtime_baseline(
    log_path: Path | str | None = None,
    min_sessions: int = 3,
) -> dict[str, Any]:
    """Load benign runtime sessions from log file and extract feature vectors.

    Returns dict containing:
      - 'feature_vectors': list of float vectors for benign sessions
      - 'total_sessions': count of all parsed sessions
      - 'benign_count': count of benign sessions
      - 'excluded_count': count of excluded malicious/anomalous sessions
      - 'benign_sessions': list of benign event lists
      - 'excluded_sessions': list of excluded event lists
    """
    path = Path(log_path) if log_path is not None else DEFAULT_LOG_FILE

    if not path.exists():
        return {
            "feature_vectors": [],
            "total_sessions": 0,
            "benign_count": 0,
            "excluded_count": 0,
            "benign_sessions": [],
            "excluded_sessions": [],
        }

    raw_events: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        raw_events.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return {
            "feature_vectors": [],
            "total_sessions": 0,
            "benign_count": 0,
            "excluded_count": 0,
            "benign_sessions": [],
            "excluded_sessions": [],
        }

    all_sessions = parse_sessions_from_events(raw_events)
    benign_sessions: list[list[dict[str, Any]]] = []
    excluded_sessions: list[list[dict[str, Any]]] = []

    for s in all_sessions:
        if is_session_benign(s):
            benign_sessions.append(s)
        else:
            excluded_sessions.append(s)

    feature_vectors = [extract_features(s) for s in benign_sessions]

    return {
        "feature_vectors": feature_vectors,
        "total_sessions": len(all_sessions),
        "benign_count": len(benign_sessions),
        "excluded_count": len(excluded_sessions),
        "benign_sessions": benign_sessions,
        "excluded_sessions": excluded_sessions,
    }
