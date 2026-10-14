"""Service layer for AgentPulse API.

Provides functions for reading JSONL event logs, grouping session timelines,
aggregating SOC metrics, and checking agent runtime health.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import urllib.request
import urllib.error

from api.schemas import (
    MetricsResponse,
    SecurityAssessmentSummary,
    SessionDetailResponse,
    SessionSummary,
    SystemStatusResponse,
)
from monitoring.event_collector import DEFAULT_LOG_FILE, PROJECT_ROOT

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def read_all_events(log_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read all runtime events from the JSONL log file safely."""
    path = Path(log_path) if log_path is not None else DEFAULT_LOG_FILE
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    events.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    return events


def group_sessions_from_events(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split an ordered list of runtime events into discrete session blocks.

    Each session begins with a USER_REQUEST event and continues until the next
    USER_REQUEST event or the end of the log.
    """
    sessions: list[list[dict[str, Any]]] = []
    current_session: list[dict[str, Any]] = []

    for event in events:
        if event.get("event_type") == "USER_REQUEST":
            if current_session:
                sessions.append(current_session)
            current_session = [event]
        else:
            if current_session:
                current_session.append(event)
            else:
                # Event occurred before any USER_REQUEST; group into an initial block
                current_session = [event]

    if current_session:
        sessions.append(current_session)

    return sessions


def extract_session_security_summary(session_events: list[dict[str, Any]]) -> SecurityAssessmentSummary:
    """Extract the correlated security and risk findings from a session block."""
    risk_score = 0
    risk_level = "LOW"
    response_action = "ALLOW"
    signals_used: list[str] = []
    reason = "Normal activity."
    score_breakdown: dict[str, int] = {}

    rule_detections = 0
    anomaly_detected = False
    anomaly_score = 0.0
    ml_prediction = "NORMAL"
    ml_confidence = 0.0

    for ev in session_events:
        etype = ev.get("event_type")

        # Check rule-based detections
        if etype == "SECURITY_DETECTION":
            rule_detections += 1
        elif etype == "SECURITY_ANALYSIS" and (ev.get("detected") is True or ev.get("status") == "detected"):
            # Count if not already counted as a detection event
            pass

        # Check Isolation Forest anomaly
        if etype == "BEHAVIORAL_ANOMALY":
            anomaly_detected = bool(ev.get("detected") is True or (isinstance(ev.get("input"), dict) and ev["input"].get("detected") is True))
            score = ev.get("anomaly_score")
            if score is None and isinstance(ev.get("input"), dict):
                score = ev["input"].get("anomaly_score", 0.0)
            anomaly_score = float(score) if score is not None else 0.0

        # Check Random Forest prediction
        if etype == "ML_BEHAVIOR_PREDICTION":
            pred = ev.get("predicted_label") or (ev.get("input", {}).get("predicted_label") if isinstance(ev.get("input"), dict) else None)
            if pred:
                ml_prediction = str(pred)
            conf = ev.get("confidence") or (ev.get("input", {}).get("confidence") if isinstance(ev.get("input"), dict) else None)
            if conf is not None:
                ml_confidence = float(conf)

        # Check Risk Assessment event
        if etype == "RISK_ASSESSMENT":
            inp = ev.get("input") if isinstance(ev.get("input"), dict) else {}
            risk_score = int(ev.get("risk_score", inp.get("risk_score", 0)) or 0)
            risk_level = str(ev.get("risk_level", inp.get("risk_level", "LOW")))
            response_action = str(ev.get("response_action", inp.get("response_action", "ALLOW"))).upper()
            signals_used = list(ev.get("signals_used", inp.get("signals_used", [])))
            reason = str(ev.get("reason", inp.get("reason", reason)))
            sb = ev.get("score_breakdown", inp.get("score_breakdown", {}))
            if isinstance(sb, dict):
                score_breakdown = {k: int(v) for k, v in sb.items()}

    return SecurityAssessmentSummary(
        risk_score=risk_score,
        risk_level=risk_level,
        response_action=response_action,
        rule_based_detections=rule_detections,
        behavioral_anomaly_detected=anomaly_detected,
        anomaly_score=round(anomaly_score, 4),
        ml_prediction=ml_prediction,
        ml_confidence=round(ml_confidence, 4),
        signals_used=signals_used,
        reason=reason,
        score_breakdown=score_breakdown,
    )


def summarize_session(session_events: list[dict[str, Any]]) -> SessionSummary:
    """Create a SessionSummary from an ordered block of session events."""
    first_ev = session_events[0] if session_events else {}
    session_id = first_ev.get("event_id", "session_unknown")
    timestamp = first_ev.get("timestamp", "")
    prompt = ""
    response = ""
    correlated = False

    for ev in session_events:
        etype = ev.get("event_type")
        if etype == "USER_REQUEST":
            prompt = str(ev.get("input", ""))
            session_id = ev.get("event_id", session_id)
            timestamp = ev.get("timestamp", timestamp)
        elif etype == "AGENT_RESPONSE":
            response = str(ev.get("input", ""))
        elif etype == "SECURITY_CORRELATION":
            correlated = bool(ev.get("correlated") or (isinstance(ev.get("input"), dict) and ev["input"].get("correlated")))

    sec = extract_session_security_summary(session_events)

    return SessionSummary(
        session_id=session_id,
        timestamp=timestamp,
        prompt=prompt,
        response=response,
        event_count=len(session_events),
        risk_score=sec.risk_score,
        risk_level=sec.risk_level,
        response_action=sec.response_action,
        correlated=correlated,
        ml_prediction=sec.ml_prediction,
        anomaly_score=sec.anomaly_score,
        rule_detections=sec.rule_based_detections,
    )


def get_all_sessions(log_path: Path | str | None = None) -> list[SessionSummary]:
    """Retrieve all sessions from the log, newest first."""
    events = read_all_events(log_path)
    session_blocks = group_sessions_from_events(events)
    summaries = [summarize_session(block) for block in session_blocks]
    # Reverse so newest sessions appear first
    summaries.reverse()
    return summaries


def get_session_detail(session_id: str, log_path: Path | str | None = None) -> SessionDetailResponse | None:
    """Retrieve the full timeline and summary of a specific session."""
    events = read_all_events(log_path)
    session_blocks = group_sessions_from_events(events)

    for block in session_blocks:
        summary = summarize_session(block)
        if summary.session_id == session_id:
            return SessionDetailResponse(summary=summary, events=block)

    return None


def calculate_metrics(log_path: Path | str | None = None) -> MetricsResponse:
    """Compute aggregated cybersecurity metrics across all logged runtime sessions."""
    events = read_all_events(log_path)
    session_blocks = group_sessions_from_events(events)

    total_sessions = len(session_blocks)
    total_events = len(events)
    rule_detections_count = 0
    anomaly_count = 0
    suspicious_predictions_count = 0
    risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    action_distribution = {"ALLOW": 0, "ALERT": 0, "HUMAN_REVIEW": 0}
    event_type_distribution: dict[str, int] = {}
    total_risk_score = 0

    for ev in events:
        etype = ev.get("event_type", "UNKNOWN")
        event_type_distribution[etype] = event_type_distribution.get(etype, 0) + 1
        if etype == "SECURITY_DETECTION":
            rule_detections_count += 1

    for block in session_blocks:
        sec = extract_session_security_summary(block)
        total_risk_score += sec.risk_score
        risk_level = sec.risk_level.upper()
        if risk_level in risk_distribution:
            risk_distribution[risk_level] += 1
        else:
            risk_distribution[risk_level] = 1

        action = sec.response_action.upper()
        if action in action_distribution:
            action_distribution[action] += 1
        else:
            action_distribution[action] = 1

        if sec.behavioral_anomaly_detected:
            anomaly_count += 1
        if sec.ml_prediction == "SUSPICIOUS":
            suspicious_predictions_count += 1

    avg_risk = round(total_risk_score / max(1, total_sessions), 2) if total_sessions > 0 else 0.0

    return MetricsResponse(
        total_sessions=total_sessions,
        total_events=total_events,
        rule_detections_count=rule_detections_count,
        anomaly_count=anomaly_count,
        suspicious_predictions_count=suspicious_predictions_count,
        risk_distribution=risk_distribution,
        action_distribution=action_distribution,
        average_risk_score=avg_risk,
        event_type_distribution=event_type_distribution,
    )


def check_system_health(log_path: Path | str | None = None) -> SystemStatusResponse:
    """Verify runtime services, models, and log files."""
    log_file = Path(log_path) if log_path is not None else DEFAULT_LOG_FILE
    events = read_all_events(log_file)

    # Check Ollama connection
    ollama_ok = False
    try:
        url = f"{DEFAULT_OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "AgentPulse/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                ollama_ok = True
    except (urllib.error.URLError, TimeoutError, OSError):
        ollama_ok = False

    # Check models
    rf_model_path = PROJECT_ROOT / "models" / "random_forest_behavior_model.joblib"
    rf_loaded = rf_model_path.exists()

    status = "online" if ollama_ok and rf_loaded else ("degraded" if rf_loaded else "offline")

    return SystemStatusResponse(
        status=status,
        ollama_connected=ollama_ok,
        ollama_model=DEFAULT_OLLAMA_MODEL,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
        models_loaded={
            "random_forest_classifier": rf_loaded,
            "isolation_forest_baseline": True,
            "rule_prompt_detector": True,
            "rule_tool_misuse_detector": True,
        },
        log_file_path=str(log_file),
        log_file_events_count=len(events),
        agent_id="agent_001",
    )
