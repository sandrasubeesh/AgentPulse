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

### Stage 4 — Behavioral Anomaly Detection & Supervised ML Pipeline
- **Machine Learning Detector**: Uses Scikit-learn `IsolationForest` to identify unusual or suspicious agent behavior patterns that do not match predefined signatures.
- **Rule-Based Detection**: Detects known attacks using predefined patterns and security policies.
- **Behavioral Anomaly Detection**: Uses Isolation Forest trained on a normal agent interaction baseline to detect unusual behavior (e.g. excessive tool calls, high tool error rates, unusual action sequences).
- **Behavioral Baseline**:
  - Uses unsupervised `IsolationForest` to learn normal agent behavior.
  - Automatically parses benign runtime sessions from `logs/runtime_events.jsonl` to establish the baseline (`baseline_source: "runtime_logs"`).
  - Suspicious or malicious sessions (containing prompt injection, tool misuse, or detected anomalies) are strictly excluded from baseline training data.
  - Synthetic sessions remain available as a fallback when insufficient log data exists (`baseline_source: "synthetic"`).
- **Runtime Behavioral Dataset Generation (`behavior/dataset_generator.py`)**:
  - Generates a controlled runtime behavioral dataset CSV saved at `data/behavioral_dataset.csv`.
  - **Session-Based Feature Extraction**: Reads `logs/runtime_events.jsonl` and parses discrete valid sessions bounded between `USER_REQUEST` and `AGENT_RESPONSE`. For each valid session, extracts 11 numerical features: `total_events`, `tool_call_count`, `tool_result_count`, `tool_error_count`, `unique_tool_count`, `file_read_count`, `search_call_count`, `email_call_count`, `security_detection_count`, `prompt_event_count`, and `session_duration_seconds`.
  - **NORMAL / SUSPICIOUS Labels**: Reuses Stage 3 security detections and Stage 4 behavioral anomaly results to label sessions as `NORMAL` (valid session with zero confirmed detections/anomalies) or `SUSPICIOUS` (session containing prompt injection, tool misuse, or behavioral anomaly).
  - **Controlled Dataset**: The generated CSV contains purely numerical behavioral metrics and categorical labels (`NORMAL` vs `SUSPICIOUS`), ensuring that no raw user prompts, tool parameters, file contents, passwords, API keys, or secrets are exposed.
- **Feature Engineering (`behavior/feature_engineering.py`)**:
  - **Feature Selection & Ordering**: Loads `data/behavioral_dataset.csv` and constructs matrix `X` with exact 11-column feature ordering (`total_events` through `session_duration_seconds`).
  - **Label Encoding**: Converts target strings to numeric binary integers (`NORMAL` $\to$ `0`, `SUSPICIOUS` $\to$ `1`).
  - **Label Leakage Prevention**: Target `label` column is strictly isolated from feature matrix `X` and feature names.
  - **Scaling Rationale**: Feature normalization/standardization is intentionally omitted because tree-based models (Random Forest) use monotonic axis-aligned splits and are scale-invariant.
  - **Random Forest Classifier (`ml/random_forest_model.py`)**:
  - **Supervised Classification**: Trains a `RandomForestClassifier` (`n_estimators=100`, `random_state=42`, `class_weight="balanced"`) to classify runtime agent behavior into `NORMAL` (0) vs `SUSPICIOUS` (1).
  - **Stratified Train/Test Split**: Splits the expanded 200-session dataset into 80% training (`160` samples: 80 NORMAL, 80 SUSPICIOUS) and 20% testing (`40` samples: 20 NORMAL, 20 SUSPICIOUS) with `random_state=42` and stratification.
  - **Model Persistence**: Serializes the trained classifier instance, exact feature names, label mapping, and evaluation metadata to `models/random_forest_behavior_model.joblib`.
- **Improved ML Evaluation Methodology (`ml/evaluate_model_v2.py`)**:
  - **Ground-Truth Scenario Grouping**: Maps sessions into 9 deterministic scenario families (`benign_qa`, `benign_file_read`, `benign_search`, `benign_email`, `benign_multitool`, `benign_tool_error`, `prompt_injection`, `tool_misuse`, `behavioral_anomaly`).
  - **Pure Behavioral Features (9 Features)**: Strictly excludes detector-derived flags (`security_detection_count` and `prompt_event_count`) from the feature matrix $X$ to eliminate shortcut leakage.
  - **Group-Aware Train/Test Split**: Uses `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` ensuring **zero group overlap** so that the test set evaluates unseen attack and scenario families (`benign_email`, `prompt_injection`).
  - **5-Fold Stratified Group Cross-Validation**: Applies `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` across scenario groups (`70.02%` mean CV accuracy $\pm 23.50\%$).
  - **Model Persistence (v2 Model)**: Serializes the improved classifier instance to `models/random_forest_behavior_model_v2.joblib` without altering the baseline v1 model.
  - **Controlled Research Disclaimer**: Controlled experimental datasets measure pipeline validity under specific scenario splits, not general real-world benchmark performance.
- **Event Type**: `BEHAVIORAL_ANOMALY` recorded with `detection_type="BEHAVIORAL_ANOMALY"`, `detected`, `anomaly_score`, `model`, `severity`, `reason`, and `baseline_source`.

### Stage 5 — ML-Based Runtime Detection (`ml/runtime_predictor.py`)
- **Offline Model Training**: The Random Forest classifier is trained offline on the controlled behavioral dataset (`data/behavioral_dataset.csv`) and saved as a joblib model artifact (`models/random_forest_behavior_model.joblib`).
- **Runtime Model Loading**: During agent execution, `RuntimePredictor` loads the saved model artifact along with its feature names, ordering, and label mapping.
- **Session-Level Behavioral Feature Extraction**: At session completion (bounded between `USER_REQUEST` and `AGENT_RESPONSE`), runtime events are converted into a numerical feature vector dynamically aligned with the model's exact training feature order.
- **Supervised Classification & Confidence**: The classifier predicts `NORMAL` (0) or `SUSPICIOUS` (1) and outputs a confidence score ($[0.0, 1.0]$) derived from Random Forest class prediction probabilities.
- **Independent Event Logging**: Logs a structured `ML_BEHAVIOR_PREDICTION` event containing `prediction`, `predicted_label`, `confidence`, `model`, and non-sensitive `feature_summary`. Rule-based detectors (`PromptInjectionDetector`, `ToolMisuseDetector`) and unsupervised anomaly detectors (`IsolationForest`) operate independently.
- **Non-Blocking Architecture**: The ML predictor operates in **detect/report mode only** and does NOT automatically block actions. Any model loading or feature extraction failure is caught safely to prevent agent execution crashes.
- **Controlled Dataset Disclaimer**: The Random Forest model was trained and evaluated on a controlled runtime dataset. Metrics reflect performance under specified controlled scenarios and should not be interpreted as guaranteed real-world detection accuracy.

### Stage 6 — Security Correlation (`security/correlation_engine.py`)
- **Multi-Signal Evidence Aggregation**: Security Correlation combines evidence from rule-based attack detection, Isolation Forest behavioral anomaly detection, and Random Forest runtime classification. Rather than relying on a single detector, AgentPulse preserves the provenance of each signal and produces a correlated security finding.
- **Signal Provenance Preservation**: Captures metadata for every active detection source (`RULE_BASED`, `ISOLATION_FOREST`, `RANDOM_FOREST`), tracking attack types, anomaly scores, ML confidence probabilities, and severity levels.
- **Correlation Severity Matrix**:
  - `LOW`: Zero active security signals (`correlated = False`).
  - `MEDIUM`: Exactly one active security signal (`correlated = False`).
  - `HIGH`: Two or more independent detector signals agree (`correlated = True`).
- **Structured Event Logging**: Records `SECURITY_CORRELATION` runtime events with full signal lists, confidence metrics, correlation status, and natural language explanations.
- **Detection-Only Mode**: Operates strictly in detection/correlation mode without automated blocking or score flattening.

### Stage 7 — Basic Risk Scoring and Response Engine (`security/risk_engine.py`)
- **Explainable Risk Aggregation**: Risk scoring is an explainable aggregation layer over independent runtime security signals. It does not represent a mathematically calibrated probability of attack. The score is used to prioritize response decisions and preserve detector provenance.
- **Deterministic Scoring Formula** (0–100, additive, capped):
  - Rule-based detection confirmed: **+30**
  - Isolation Forest behavioral anomaly detected: **+25**
  - Random Forest prediction is SUSPICIOUS: **+25**
  - All three independent detector sources agree: **+10** (triple-correlation bonus)
- **Risk Level Thresholds**:
  - `LOW` (0–29) → `ALLOW` — no significant signals
  - `MEDIUM` (30–59) → `ALERT` — single or partial evidence
  - `HIGH` (60–100) → `HUMAN_REVIEW` — multi-source agreement
- **Signal Provenance in Every Result**: Each `RISK_ASSESSMENT` event records `signals_used` (which detectors contributed), `score_breakdown` (per-detector point allocation), `ml_confidence`, and a natural-language `reason`.
- **Detector Roles**:
  - Rule-based detection identifies known attack patterns (Prompt Injection, Tool Misuse).
  - Isolation Forest detects unusual behavioral patterns without requiring attack labels.
  - Random Forest performs supervised behavioral classification.
  - Security Correlation combines the independent signals.
  - Risk Engine converts the correlated evidence into an interpretable response level.
- **Non-Blocking**: Response actions are `ALLOW`, `ALERT`, and `HUMAN_REVIEW`. The engine does NOT automatically terminate or block the agent.
- **Structured Event Logging**: Records `RISK_ASSESSMENT` events to `logs/runtime_events.jsonl` without exposing raw user prompts, secrets, passwords, or API keys.

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
│   ├── anomaly_detector.py    # IsolationForest ML anomaly detector
│   ├── runtime_baseline.py    # Parses & filters benign runtime sessions for baseline
│   ├── dataset_builder.py     # Legacy dataset builder
│   ├── dataset_generator.py   # Exports controlled runtime behavioral dataset CSV
│   ├── expand_dataset.py      # Generates expanded 200-session dataset
│   └── feature_engineering.py # Loads, validates, & prepares X, y matrices for Supervised ML
├── ml/
│   ├── __init__.py
│   ├── random_forest_model.py # Supervised RandomForestClassifier training & serialization
│   ├── evaluate_model.py      # Holdout evaluation, 5-fold CV, feature importance, & ablation
│   ├── evaluate_model_v2.py   # Group-aware evaluation methodology, 9 behavioral features, & v2 model
│   └── runtime_predictor.py   # ML runtime inference module & session feature alignment
├── monitoring/
│   ├── __init__.py
│   └── event_collector.py     # Runtime Event Collector & JSONL logging
├── security/
│   ├── __init__.py
│   ├── types.py               # Security detection result types
│   ├── prompt_detector.py     # Prompt Injection Detector
│   ├── tool_misuse_detector.py# Tool Misuse Detector
│   ├── correlation_engine.py  # Multi-signal security correlation engine & provenance tracking
│   └── risk_engine.py         # Explainable risk scoring and response action engine
├── data/
│   ├── notes.txt              # Sample file for read_file
│   └── behavioral_dataset.csv # Controlled behavioral dataset for ML pipeline (200 sessions)
├── models/
│   ├── random_forest_behavior_model.joblib    # Saved Random Forest v1 model artifact (gitignored)
│   └── random_forest_behavior_model_v2.joblib # Saved Random Forest v2 model artifact (gitignored)
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
│   ├── test_behavior_integration.py
│   ├── test_runtime_baseline.py
│   ├── test_dataset_builder.py
│   ├── test_dataset_generator.py
│   ├── test_feature_engineering.py
│   ├── test_random_forest.py
│   ├── test_evaluate_model.py
│   ├── test_evaluation_v2.py
│   ├── test_runtime_predictor.py
│   ├── test_correlation_engine.py
│   └── test_risk_engine.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                    # Command-line interface
```
