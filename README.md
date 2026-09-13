# AgentPulse

**AgentPulse: A Digital Twin-Driven Runtime Security and Behavioral Trust Framework for AI Agents**

This repository is a research-oriented cybersecurity project. The long-term goal is to monitor AI agents at runtime, build a digital twin of their behavior, and score behavioral trust.

That security framework is **not implemented yet**.

## What this first stage does

Stage 1 provides a clean, working **baseline tool-using AI agent**. Later stages will wrap this agent with a Runtime Event Collector. For now, the agent can:

1. Receive a natural-language task
2. Decide whether a tool is needed
3. Call one of three controlled tools
4. Observe the tool result
5. Reply in natural language

The three tools are:

| Tool | Purpose |
| --- | --- |
| `read_file(filename)` | Read a text file from the local `data/` directory only |
| `search_knowledge(query)` | Search a small in-memory knowledge base |
| `send_email(recipient, message)` | **Simulate** an email. No real email is sent |

## Installation

Use Python 3.11 or newer.

```bash
cd AgentPulse
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ollama setup

This stage uses a **local Ollama** model. No OpenAI API key is required.

1. Install Ollama from [https://ollama.com](https://ollama.com).
2. Start the Ollama service:

```bash
ollama serve
```

On macOS, opening the Ollama app is enough; it already serves models locally.

3. Pull the model used by AgentPulse:

```bash
ollama pull llama3.2
```

4. Confirm the model is listed:

```bash
ollama list
```

Optional `.env` values (copy from `.env.example` only if you want to override defaults):

```
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

If Ollama is not running, or `llama3.2` has not been pulled, the CLI prints a clear error instead of crashing.

## How to run AgentPulse

```bash
python main.py
```

You should see:

```
========================================
        AgentPulse Basic Agent
========================================

Enter your task:
>
```

Type a task and press Enter. Type `exit` to quit.

Run the tool tests (no cloud API key required):

```bash
pytest tests/test_tools.py -v
```

## Example commands

```
What is Python?
Read notes.txt
Send an email to Rahul saying the project meeting is tomorrow.
Read ../../.env
Tell me what an AI agent is.
What is LangGraph?
What is AgentPulse?
```

Expected behavior:

- Factual questions about Python, LangGraph, AI agents, cybersecurity, or AgentPulse should use `search_knowledge`.
- `Read notes.txt` should use `read_file`.
- Email requests should use `send_email` and clearly report a **simulation**.
- `Read ../../.env` must be rejected. The tool only allows filenames inside `data/`.

## Project structure

```
AgentPulse/
├── agent/
│   ├── __init__.py
│   ├── agent.py      # LangGraph ReAct agent
│   └── tools.py      # Isolated tools (no monitoring yet)
├── data/
│   └── notes.txt     # Sample file for read_file
├── tests/
│   └── test_tools.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py           # Command-line interface
```

## Architecture notes

`tools.py` does not import the agent graph. `agent.py` imports tools, but not the CLI. This keeps the baseline agent easy to wrap later around:

- agent decisions
- tool calls, inputs, and outputs
- memory operations
- API calls
- security events

Do not couple tools to a monitoring system in this stage.

## Limitations of this first stage

- No Digital Twin
- No anomaly, prompt-injection, memory-poisoning, leakage, or privilege-escalation detection
- No dynamic trust or risk scoring
- No Neo4j, dashboard, authentication, database, Docker, or cloud deployment
- No real email, SMTP, or other real external side effects
- Knowledge search is a small local dictionary, not a retrieval system
- File access is limited to filenames in `data/`
- The CLI is single-turn: each task is a new conversation
- The LLM is local Ollama (`llama3.2`); Ollama must be installed and running

This stage is only the monitored subject for later AgentPulse work.
