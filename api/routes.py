"""FastAPI router endpoints for AgentPulse."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Query

from agent.agent import run_task
from api.schemas import (
    ChatRequest,
    ChatResponse,
    MetricsResponse,
    SessionDetailResponse,
    SessionSummary,
    SystemStatusResponse,
)
from api.service import (
    calculate_metrics,
    check_system_health,
    extract_session_security_summary,
    get_all_sessions,
    get_session_detail,
    group_sessions_from_events,
    read_all_events,
)
from monitoring.event_collector import get_event_collector

router = APIRouter(prefix="/api", tags=["AgentPulse"])


@router.post("/chat", response_model=ChatResponse)
def submit_chat_task(req: ChatRequest) -> ChatResponse:
    """Submit a task/prompt to the real AgentPulse AI agent and security pipeline."""
    task_text = req.message.strip()
    if not task_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    ec = get_event_collector()
    start_idx = len(ec.events)

    try:
        reply = run_task(task_text, collector=ec)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    session_events = list(ec.events[start_idx:])
    if not session_events:
        # Fallback to reading the latest session from disk if in-memory list was modified
        disk_events = read_all_events()
        blocks = group_sessions_from_events(disk_events)
        session_events = blocks[-1] if blocks else []

    first_event = session_events[0] if session_events else {}
    session_id = first_event.get("event_id", "session_unknown")
    timestamp = first_event.get("timestamp", "")

    security_summary = extract_session_security_summary(session_events)

    return ChatResponse(
        response=reply,
        session_id=session_id,
        timestamp=timestamp,
        security=security_summary,
        event_count=len(session_events),
        events=session_events,
    )


@router.get("/events", response_model=list[dict[str, Any]])
def list_runtime_events(
    limit: int = Query(default=100, ge=1, le=1000, description="Max events to return"),
    event_type: str | None = Query(default=None, description="Filter by event_type"),
    session_id: str | None = Query(default=None, description="Filter by session_id"),
) -> list[dict[str, Any]]:
    """Retrieve runtime events from logs/runtime_events.jsonl (newest first)."""
    events = read_all_events()

    if session_id:
        blocks = group_sessions_from_events(events)
        matching_block = next((b for b in blocks if b and b[0].get("event_id") == session_id), None)
        events = matching_block if matching_block else []

    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]

    # Return newest first up to limit
    events_reversed = list(reversed(events))
    return events_reversed[:limit]


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    """Retrieve high-level summaries of all logged sessions (newest first)."""
    return get_all_sessions()


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str) -> SessionDetailResponse:
    """Retrieve full event timeline and security details for a single session."""
    detail = get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return detail


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Retrieve aggregated SOC metrics for the dashboard."""
    return calculate_metrics()


@router.get("/status", response_model=SystemStatusResponse)
def get_status() -> SystemStatusResponse:
    """Retrieve system, Ollama, model, and event log status."""
    return check_system_health()
