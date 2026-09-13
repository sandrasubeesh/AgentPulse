"""Controlled tools for the stage-1 baseline agent.

These tools are intentionally independent of any security-monitoring layer.
A later AgentPulse Runtime Event Collector can wrap tool calls without
changing the core logic here.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from monitoring.event_collector import get_event_collector
from security.tool_misuse_detector import ToolMisuseDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


KNOWLEDGE_BASE: dict[str, str] = {
    "python": (
        "Python is a high-level programming language often used for "
        "automation, data work, and AI applications. This baseline agent "
        "is written in Python 3.11+."
    ),
    "ai agents": (
        "An AI agent is a system that can receive a goal, reason about it, "
        "optionally use tools, observe results, and produce a response. "
        "This project uses a ReAct-style tool-using agent."
    ),
    "langgraph": (
        "LangGraph is a library for building stateful, graph-based LLM "
        "workflows. This baseline agent uses a LangGraph ReAct loop: the "
        "model node decides whether to call tools, a tools node executes "
        "them, and control returns to the model until a final answer is ready."
    ),
    "cybersecurity": (
        "Cybersecurity is the practice of protecting systems, data, and "
        "users from attacks and misuse. AgentPulse will later study runtime "
        "security and behavioral trust for AI agents. This stage does not "
        "implement those defenses yet."
    ),
    "agentpulse": (
        "AgentPulse is a research framework for digital-twin-driven runtime "
        "security and behavioral trust of AI agents. This repository currently "
        "contains only the baseline agent that AgentPulse will later monitor."
    ),
}


class FileAccessError(ValueError):
    """Raised when a file request is invalid or outside the data directory."""


def _validate_filename(filename: str) -> str:
    """Allow only a simple filename inside the project data directory."""
    if filename is None or not str(filename).strip():
        raise FileAccessError("Filename cannot be empty.")

    cleaned = str(filename).strip()

    # Reject absolute paths, including POSIX and Windows drive paths.
    if Path(cleaned).is_absolute() or cleaned.startswith(("/", "\\")):
        raise FileAccessError("Absolute paths are not allowed. Provide a filename only.")
    if len(cleaned) >= 2 and cleaned[1] == ":":
        raise FileAccessError("Absolute paths are not allowed. Provide a filename only.")

    if ".." in cleaned:
        raise FileAccessError(
            "Path traversal is not allowed. Access is restricted to the data directory."
        )

    # Allow only a filename, not a nested path.
    path_obj = Path(cleaned)
    if path_obj.name != cleaned or "/" in cleaned or "\\" in cleaned:
        raise FileAccessError("Only a filename is allowed, not a directory path.")

    return cleaned


def resolve_data_file(filename: str) -> Path:
    """Return a path that is guaranteed to stay inside DATA_DIR."""
    safe_name = _validate_filename(filename)
    data_root = DATA_DIR.resolve()
    candidate = (data_root / safe_name).resolve()

    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise FileAccessError(
            "Access denied. Files can only be read from the project data directory."
        ) from exc

    return candidate


def read_file_content(filename: str) -> str:
    """Read a text file from the local data directory, or return a clear error."""
    try:
        path = resolve_data_file(filename)
    except FileAccessError as exc:
        return f"Error: {exc}"

    if not path.exists():
        return (
            f"Error: File '{filename}' was not found in the data directory. "
            "No other locations were searched."
        )
    if not path.is_file():
        return f"Error: '{filename}' is not a readable file."

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error: Could not read '{filename}': {exc}"


def search_knowledge_base(query: str) -> str:
    """Search the small local knowledge base for matching topics."""
    if query is None or not str(query).strip():
        return "Error: Search query cannot be empty."

    needle = str(query).strip().lower()
    key_matches: list[str] = []
    value_matches: list[str] = []

    for topic, text in KNOWLEDGE_BASE.items():
        entry = f"{topic}: {text}"
        if needle in topic.lower():
            key_matches.append(entry)
        elif needle in text.lower():
            value_matches.append(entry)

    matches = key_matches or value_matches

    if not matches:
        available = ", ".join(sorted(KNOWLEDGE_BASE))
        return (
            f"No knowledge-base entries matched '{query}'. "
            f"Available topics: {available}."
        )

    return "\n\n".join(matches)


def simulate_send_email(recipient: str, message: str) -> str:
    """Simulate sending an email. This never contacts a mail server."""
    if recipient is None or not str(recipient).strip():
        return "Error: Recipient cannot be empty. No real email was sent."
    if message is None or not str(message).strip():
        return "Error: Message cannot be empty. No real email was sent."

    return (
        "SIMULATED EMAIL\n"
        f"Recipient: {str(recipient).strip()}\n"
        f"Message: {str(message).strip()}\n"
        "Status: simulated_success\n"
        "Note: No real email was sent. This tool is a simulation only."
    )


def _log_tool_call(tool_name: str, args: dict[str, object], func) -> str:
    collector = get_event_collector()
    collector.record_event(
        event_type="TOOL_CALL",
        tool=tool_name,
        input=args,
        status="initiated",
    )

    tool_detector = ToolMisuseDetector()
    t_res = tool_detector.detect(tool_name, args)
    collector.record_event(
        event_type="SECURITY_ANALYSIS",
        tool=tool_name,
        input=t_res.to_dict(),
        status="detected" if t_res.detected else "clear",
        detected=t_res.detected,
        attack_type=t_res.attack_type,
        severity=t_res.severity,
        confidence=t_res.confidence,
        reason=t_res.reason,
    )
    if t_res.detected:
        collector.record_event(
            event_type="SECURITY_DETECTION",
            tool=tool_name,
            input=t_res.to_dict(),
            status="detected",
            attack_type=t_res.attack_type,
            severity=t_res.severity,
            confidence=t_res.confidence,
            reason=t_res.reason,
        )


    try:
        res = func()
    except Exception as exc:
        err_msg = f"Error: {exc}"
        collector.record_event(
            event_type="TOOL_ERROR",
            tool=tool_name,
            input=err_msg,
            status="error",
        )
        raise exc

    if isinstance(res, str) and res.startswith("Error:"):
        collector.record_event(
            event_type="TOOL_ERROR",
            tool=tool_name,
            input=res,
            status="error",
        )
    else:
        collector.record_event(
            event_type="TOOL_RESULT",
            tool=tool_name,
            input=res,
            status="success",
        )
    return res


@tool
def read_file(filename: str) -> str:
    """Read a text file from the project's local data directory.

    Use this when the user asks to read, open, or show a file.
    Pass only a filename such as notes.txt. Absolute paths and
    parent-directory paths such as ../ are rejected.
    """
    return _log_tool_call("read_file", {"filename": filename}, lambda: read_file_content(filename))


@tool
def search_knowledge(query: str) -> str:
    """Search the local knowledge base about Python, AI agents, LangGraph, cybersecurity, and AgentPulse.

    Use this when the user asks a factual question about those topics.
    """
    return _log_tool_call("search_knowledge", {"query": query}, lambda: search_knowledge_base(query))


@tool
def send_email(recipient: str, message: str) -> str:
    """Simulate sending an email. This does not send a real email.

    Use this when the user asks to email someone. Pass the recipient name
    or address and the message text. The result is always a simulation.
    """
    return _log_tool_call(
        "send_email",
        {"recipient": recipient, "message": message},
        lambda: simulate_send_email(recipient, message),
    )


def get_tools() -> list[BaseTool]:
    """Return the LangChain tools used by the baseline agent."""
    return [read_file, search_knowledge, send_email]

