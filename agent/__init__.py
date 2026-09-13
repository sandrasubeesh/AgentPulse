"""Baseline AgentPulse agent package.

This package currently contains only the stage-1 tool-using agent.
Security monitoring will wrap this agent in a later stage.
"""

from agent.agent import create_agent, run_task

__all__ = ["create_agent", "run_task"]
