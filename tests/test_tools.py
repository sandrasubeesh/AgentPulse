"""Unit tests for the baseline tools. These tests do not call the LLM."""

from __future__ import annotations

from agent.tools import read_file_content, search_knowledge_base, simulate_send_email


def test_read_allowed_file() -> None:
    content = read_file_content("notes.txt")
    assert "AgentPulse" in content
    assert not content.startswith("Error:")


def test_prevent_path_traversal() -> None:
    result = read_file_content("../../.env")
    assert result.startswith("Error:")
    assert "traversal" in result.lower() or "not allowed" in result.lower()
    assert "OPENAI_API_KEY" not in result


def test_reject_absolute_path() -> None:
    result = read_file_content("/etc/passwd")
    assert result.startswith("Error:")
    assert "absolute" in result.lower() or "filename" in result.lower()


def test_missing_file_returns_error() -> None:
    result = read_file_content("does_not_exist.txt")
    assert result.startswith("Error:")
    assert "not found" in result.lower()


def test_search_knowledge() -> None:
    result = search_knowledge_base("LangGraph")
    assert "LangGraph" in result or "langgraph" in result.lower()
    assert "ReAct" in result or "graph" in result.lower()
    assert not result.startswith("Error:")


def test_search_knowledge_empty_query() -> None:
    result = search_knowledge_base("   ")
    assert result.startswith("Error:")


def test_simulated_email() -> None:
    result = simulate_send_email("Rahul", "The project meeting is tomorrow.")
    assert "SIMULATED EMAIL" in result
    assert "Rahul" in result
    assert "The project meeting is tomorrow." in result
    assert "simulated_success" in result
    assert "No real email was sent" in result


def test_simulated_email_rejects_empty_fields() -> None:
    missing_recipient = simulate_send_email(" ", "hello")
    missing_message = simulate_send_email("Rahul", " ")
    assert missing_recipient.startswith("Error:")
    assert missing_message.startswith("Error:")
    assert "No real email was sent" in missing_recipient
    assert "No real email was sent" in missing_message
