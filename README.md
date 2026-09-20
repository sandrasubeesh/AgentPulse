# AgentPulse

**AgentPulse: A Digital Twin-Driven Runtime Security and Behavioral Trust Framework for AI Agents**

This repository is a research-oriented cybersecurity framework. It monitors AI agents at runtime, collects runtime events, performs rule-based attack detection, and analyzes agent behavior using machine learning.

---

## Architecture & Implementation Stages

### Stage 1 — Baseline LangGraph AI Agent
- **LLM**: Local Ollama (`llama3.2`). No cloud API keys required.
- **Agent Architecture**: LangGraph ReAct loop.
- **Controlled Tools**:
  - `read_file(filename)`: Read text files from `data/` only (with path-traversal protection).
  - `search_knowledge(query)`: Search local knowledge base.
  - `send_email(recipient, message)`: Simulated email tool.

### Stage 2 — Runtime Event Collector
- **Event Logger**: Records runtime events to `logs/runtime_events.jsonl` with UTC ISO timestamps and unique UUIDs.
- **Event Types**: `USER_REQUEST`, `AGENT_RESPONSE`, `TOOL_CALL`, `TOOL_RESULT`, `TOOL_ERROR`.
- **Credential Redaction**: Automatic sanitization of API keys, bearer tokens, and secrets.

### Stage 3 — Attack Detection Layer
- **Rule-Based Security Detection**: Analyzes prompts and tool calls for known attack patterns.
- **Detectors**:
  - `PromptInjectionDetector`: Detects instruction overrides, system prompt extractions, and jailbreaks.
  - `ToolMisuseDetector`: Detects path traversal, absolute path access, and sensitive file access.
- **Event Types**: `SECURITY_ANALYSIS` (logged for all checks) and `SECURITY_DETECTION` (logged for positive detections).

### Stage 4 — Behavioral Anomaly Detection
- **Machine Learning Detector**: Uses Scikit-learn `IsolationForest` to identify unusual or suspicious agent behavior patterns that do not match predefined signatures.
- **Rule-Based Detection**: Detects known attacks using predefined patterns and security policies.
- **Behavioral Anomaly Detection**: Uses Isolation Forest trained on a normal agent interaction baseline to detect unusual behavior (e.g. excessive tool calls, high tool error rates, unusual action sequences).
- **Event Type**: `BEHAVIORAL_ANOMALY` recorded with `detection_type="BEHAVIORAL_ANOMALY"`, `detected`, `anomaly_score`, `model`, `severity`, and `reason`.

---

## Installation

Requirements: Python 3.11+ and Ollama.

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

---

## Ollama Setup

1. Install Ollama from [https://ollama.com](https://ollama.com).
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the required model:
   ```bash
   ollama pull llama3.2
   ```

---

## Running AgentPulse

Start the CLI:

```bash
python main.py
```

Run tests:

```bash
pytest -v
```

---

## Project Structure

```
AgentPulse/
├── agent/
│   ├── __init__.py
│   ├── agent.py               # LangGraph ReAct agent & pipeline integration
│   └── tools.py               # Controlled tools with event & security hooks
├── behavior/
│   ├── __init__.py
│   ├── feature_extractor.py   # Converts session events to numerical feature vectors
│   └── anomaly_detector.py    # IsolationForest ML anomaly detector
├── monitoring/
│   ├── __init__.py
│   └── event_collector.py     # Runtime Event Collector & JSONL logging
├── security/
│   ├── __init__.py
│   ├── types.py               # Security detection result types
│   ├── prompt_detector.py     # Prompt Injection Detector
│   └── tool_misuse_detector.py# Tool Misuse Detector
├── data/
│   └── notes.txt              # Sample file for read_file
├── logs/
│   └── runtime_events.jsonl   # Local runtime log (gitignored)
├── tests/
│   ├── test_tools.py
│   ├── test_event_collector.py
│   ├── test_prompt_detector.py
│   ├── test_tool_misuse_detector.py
│   ├── test_security_integration.py
│   ├── test_feature_extractor.py
│   ├── test_anomaly_detector.py
│   └── test_behavior_integration.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                    # Command-line interface
```
