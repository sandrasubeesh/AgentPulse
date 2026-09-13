"""Command-line interface for the AgentPulse baseline agent."""

from __future__ import annotations

from agent.agent import run_task

BANNER = """
========================================
        AgentPulse Basic Agent
========================================
""".strip()


def main() -> None:
    print(BANNER)
    print()
    print("Enter a task, or type exit to quit.")
    print()

    while True:
        try:
            user_input = input("Enter your task:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            print("Please enter a task, or type exit to quit.\n")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print()
        print(run_task(user_input))
        print()


if __name__ == "__main__":
    main()
