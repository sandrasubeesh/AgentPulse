"""Pydantic schemas for the AgentPulse REST API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for submitting a task or query to the AI agent."""

    message: str = Field(..., min_length=1, max_length=10000, description="The user prompt or task.")


class SecurityAssessmentSummary(BaseModel):
    """Summary of security evaluations for a completed session."""

    risk_score: int = Field(default=0, ge=0, le=100, description="Composite risk score from 0 to 100.")
    risk_level: str = Field(default="LOW", description="LOW, MEDIUM, or HIGH.")
    response_action: str = Field(default="ALLOW", description="ALLOW, ALERT, or HUMAN_REVIEW.")
    rule_based_detections: int = Field(default=0, ge=0, description="Count of rule-based attack detections.")
    behavioral_anomaly_detected: bool = Field(default=False, description="Isolation Forest anomaly flag.")
    anomaly_score: float = Field(default=0.0, description="Isolation Forest anomaly score.")
    ml_prediction: str = Field(default="NORMAL", description="Random Forest prediction label (NORMAL/SUSPICIOUS).")
    ml_confidence: float = Field(default=0.0, description="Random Forest prediction confidence (0.0 to 1.0).")
    signals_used: list[str] = Field(default_factory=list, description="List of detected signal labels.")
    reason: str = Field(default="", description="Detailed reason for the risk score.")
    score_breakdown: dict[str, int] = Field(default_factory=dict, description="Point contribution breakdown.")


class ChatResponse(BaseModel):
    """Response returned after running a task through the agent and security pipeline."""

    response: str = Field(..., description="The agent's natural-language answer.")
    session_id: str = Field(..., description="Unique UUID for this session (from USER_REQUEST).")
    timestamp: str = Field(..., description="UTC ISO timestamp of the session.")
    security: SecurityAssessmentSummary = Field(..., description="Correlated security and risk assessment.")
    event_count: int = Field(default=0, description="Number of runtime events recorded in this session.")
    events: list[dict[str, Any]] = Field(default_factory=list, description="Chronological events of this session.")


class SessionSummary(BaseModel):
    """High-level summary of a historical or active session."""

    session_id: str
    timestamp: str
    prompt: str
    response: str
    event_count: int
    risk_score: int
    risk_level: str
    response_action: str
    correlated: bool
    ml_prediction: str
    anomaly_score: float
    rule_detections: int


class SessionDetailResponse(BaseModel):
    """Detailed view of a session including full event timeline."""

    summary: SessionSummary
    events: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    """Aggregated security metrics for dashboard visualization."""

    total_sessions: int
    total_events: int
    rule_detections_count: int
    anomaly_count: int
    suspicious_predictions_count: int
    risk_distribution: dict[str, int]  # e.g., {"LOW": 5, "MEDIUM": 1, "HIGH": 0}
    action_distribution: dict[str, int]  # e.g., {"ALLOW": 5, "ALERT": 1, "HUMAN_REVIEW": 0}
    average_risk_score: float
    event_type_distribution: dict[str, int]


class SystemStatusResponse(BaseModel):
    """System health and integration status."""

    status: str
    ollama_connected: bool
    ollama_model: str
    ollama_base_url: str
    models_loaded: dict[str, bool]
    log_file_path: str
    log_file_events_count: int
    agent_id: str
