"""Runtime Event Collector for AgentPulse.

Records AI agent activities (USER_REQUEST, AGENT_RESPONSE, TOOL_CALL, TOOL_RESULT, TOOL_ERROR)
in memory and appends them to logs/runtime_events.jsonl with UTC ISO timestamps, UUIDs,
and sensitive credential redaction.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "runtime_events.jsonl"

REDACTION_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{10,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|password|secret|token|credential)s?\s*[:=]\s*['\"]?[^\s'\";]+['\"]?", re.IGNORECASE),
]


def redact_sensitive_data(val: Any) -> Any:
    """Recursively redact sensitive tokens or credentials from logged values."""
    if isinstance(val, str):
        redacted = val
        for pattern in REDACTION_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    if isinstance(val, dict):
        return {k: redact_sensitive_data(v) for k, v in val.items()}
    if isinstance(val, list):
        return [redact_sensitive_data(item) for item in val]
    return val


class EventCollector:
    """Collects and logs AgentPulse runtime events."""

    def __init__(self, agent_id: str = "agent_001", log_file: Path | str | None = None) -> None:
        self.agent_id = agent_id
        if log_file is None:
            self.log_file = DEFAULT_LOG_FILE
        else:
            self.log_file = Path(log_file)

        self.events: list[dict[str, Any]] = []

    def _ensure_log_dir(self) -> None:
        """Create the parent logs directory if it does not exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def record_event(
        self,
        event_type: str,
        tool: str | None = None,
        input: Any = None,
        status: str = "success",
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Record a single runtime event in memory and write to JSONL log file."""
        self._ensure_log_dir()

        event: dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "event_type": event_type,
            "tool": tool,
            "input": redact_sensitive_data(input),
            "status": redact_sensitive_data(status),
        }
        for k, v in extra_fields.items():
            if k not in event:
                event[k] = redact_sensitive_data(v)


        # Append to in-memory event store
        self.events.append(event)

        # Write to JSONL file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as exc:
            print(f"Warning: Failed to write runtime event to {self.log_file}: {exc}")

        return event

    def clear(self) -> None:
        """Clear in-memory events list."""
        self.events.clear()


# Global default collector instance
_GLOBAL_COLLECTOR: EventCollector | None = None


def get_event_collector() -> EventCollector:
    """Return the singleton global EventCollector instance."""
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is None:
        _GLOBAL_COLLECTOR = EventCollector()
    return _GLOBAL_COLLECTOR
