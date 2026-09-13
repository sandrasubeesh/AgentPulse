"""LangGraph ReAct-style baseline agent.

The graph has two nodes:
- agent: the LLM decides whether to answer or call a tool
- tools: selected tools are executed, then control returns to the agent

A later Runtime Event Collector can wrap these nodes without rewriting
the tool implementations.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent.tools import get_tools
from monitoring.event_collector import EventCollector, get_event_collector
from security.prompt_detector import PromptInjectionDetector



load_dotenv()

DEFAULT_OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are the AgentPulse baseline assistant, a helpful tool-using AI agent.

You have three tools:
1. search_knowledge(query) — local facts about Python, AI agents, LangGraph, cybersecurity, and AgentPulse.
2. read_file(filename) — read a text file from the project data directory. Use a filename only, such as notes.txt.
3. send_email(recipient, message) — SIMULATE an email. It never sends a real email.

When a tool can help, call it. After you receive the tool result, give a clear natural-language answer.
If the user asks a general question that does not need a tool, answer directly.
If a tool returns an error, explain the error clearly and do not invent file contents.
Never claim that a real email was sent. The email tool is always a simulation.
"""


def _build_llm() -> ChatOllama:
    """Create the local Ollama chat model used by the CLI."""
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    kwargs: dict[str, object] = {
        "model": model_name,
        "temperature": 0,
    }
    base_url = os.getenv("OLLAMA_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)


def _friendly_llm_error(exc: Exception) -> str:
    """Turn Ollama connection/model failures into a clear user message."""
    text = str(exc).lower()
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    connection_problem = (
        isinstance(exc, (ConnectionError, OSError, TimeoutError))
        or "connection" in text
        or "connect" in text
        or "refused" in text
        or "timed out" in text
        or "unreachable" in text
    )
    if connection_problem:
        return (
            "Error: Ollama does not appear to be running. "
            "Start it with `ollama serve`, then run `ollama pull llama3.2`."
        )

    model_missing = (
        "not found" in text
        or "404" in text
        or "pull" in text
        or "does not exist" in text
        or "no such model" in text
    )
    if model_missing:
        return (
            f"Error: The Ollama model '{model_name}' is not available. "
            f"Run `ollama pull {model_name}` and try again."
        )

    return f"Error: The language model request failed ({exc}). Please try again."


def create_agent(llm: BaseChatModel | None = None):
    """Compile a LangGraph ReAct agent that can call the baseline tools.

    Pass an LLM only in tests. The CLI uses local Ollama (llama3.2 by default).
    """
    tools = get_tools()
    model = llm if llm is not None else _build_llm()
    try:
        llm_with_tools = model.bind_tools(tools)
    except NotImplementedError:
        # Test doubles such as FakeMessagesListChatModel do not bind tools.
        llm_with_tools = model

    def agent_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _route_after_agent(state: MessagesState) -> Literal["tools"] | Literal["__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


@lru_cache(maxsize=1)
def _cached_agent():
    return create_agent()


def run_task(task: str, agent=None, collector: EventCollector | None = None) -> str:
    """Run one user task and return the agent's final text reply."""
    ec = collector if collector is not None else get_event_collector()

    if task is None or not str(task).strip():
        err_msg = "Error: Please enter a non-empty task."
        ec.record_event(event_type="USER_REQUEST", input=task, status="rejected")
        ec.record_event(event_type="AGENT_RESPONSE", input=err_msg, status="error")
        return err_msg

    task_str = str(task).strip()
    ec.record_event(event_type="USER_REQUEST", input=task_str, status="success")

    prompt_detector = PromptInjectionDetector()
    p_res = prompt_detector.detect(task_str)
    ec.record_event(
        event_type="SECURITY_ANALYSIS",
        tool=None,
        input=p_res.to_dict(),
        status="detected" if p_res.detected else "clear",
        detected=p_res.detected,
        attack_type=p_res.attack_type,
        severity=p_res.severity,
        confidence=p_res.confidence,
        reason=p_res.reason,
    )
    if p_res.detected:
        ec.record_event(
            event_type="SECURITY_DETECTION",
            tool=None,
            input=p_res.to_dict(),
            status="detected",
            attack_type=p_res.attack_type,
            severity=p_res.severity,
            confidence=p_res.confidence,
            reason=p_res.reason,
        )



    try:
        graph = agent if agent is not None else _cached_agent()
        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=task_str),
                ]
            }
        )
    except RuntimeError as exc:
        err_msg = f"Error: {exc}"
        ec.record_event(event_type="AGENT_RESPONSE", input=err_msg, status="error")
        return err_msg
    except Exception as exc:  # noqa: BLE001 - keep the CLI from crashing on LLM failures
        err_msg = _friendly_llm_error(exc)
        ec.record_event(event_type="AGENT_RESPONSE", input=err_msg, status="error")
        return err_msg

    messages = result.get("messages", [])
    if not messages:
        err_msg = "Error: The agent returned no messages."
        ec.record_event(event_type="AGENT_RESPONSE", input=err_msg, status="error")
        return err_msg

    final = messages[-1]
    content = getattr(final, "content", None)
    if isinstance(content, str) and content.strip():
        ans = content
        ec.record_event(event_type="AGENT_RESPONSE", input=ans, status="success")
        return ans

    err_msg = "Error: The agent did not produce a text response."
    ec.record_event(event_type="AGENT_RESPONSE", input=err_msg, status="error")
    return err_msg

