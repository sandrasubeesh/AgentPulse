"""Behavioral Anomaly Detector for AgentPulse Stage 4 using IsolationForest."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import IsolationForest

from behavior.feature_extractor import FEATURE_NAMES, extract_features


def get_baseline_training_data() -> list[list[float]]:
    """Generate normal/benign agent interaction sessions for baseline model training."""
    normal_sessions: list[list[dict[str, Any]]] = []

    # 1. Direct Q&A sessions (no tool calls)
    for _ in range(10):
        normal_sessions.append([
            {"event_type": "USER_REQUEST", "input": "What is Python?"},
            {"event_type": "SECURITY_ANALYSIS", "detected": False},
            {"event_type": "AGENT_RESPONSE", "input": "Python is a programming language."},
        ])

    # 2. Normal Knowledge Search (1 tool call)
    for _ in range(15):
        normal_sessions.append([
            {"event_type": "USER_REQUEST", "input": "What is LangGraph?"},
            {"event_type": "SECURITY_ANALYSIS", "detected": False},
            {"event_type": "TOOL_CALL", "tool": "search_knowledge", "input": {"query": "LangGraph"}},
            {"event_type": "SECURITY_ANALYSIS", "tool": "search_knowledge", "detected": False},
            {"event_type": "TOOL_RESULT", "tool": "search_knowledge", "input": "LangGraph is..."},
            {"event_type": "AGENT_RESPONSE", "input": "LangGraph is a library..."},
        ])

    # 3. Normal Read File (1 tool call)
    for _ in range(15):
        normal_sessions.append([
            {"event_type": "USER_REQUEST", "input": "Read notes.txt"},
            {"event_type": "SECURITY_ANALYSIS", "detected": False},
            {"event_type": "TOOL_CALL", "tool": "read_file", "input": {"filename": "notes.txt"}},
            {"event_type": "SECURITY_ANALYSIS", "tool": "read_file", "detected": False},
            {"event_type": "TOOL_RESULT", "tool": "read_file", "input": "AgentPulse notes..."},
            {"event_type": "AGENT_RESPONSE", "input": "Here are the notes..."},
        ])

    # 4. Normal Email Simulation (1 tool call)
    for _ in range(10):
        normal_sessions.append([
            {"event_type": "USER_REQUEST", "input": "Send an email to Rahul"},
            {"event_type": "SECURITY_ANALYSIS", "detected": False},
            {"event_type": "TOOL_CALL", "tool": "send_email", "input": {"recipient": "Rahul", "message": "Hi"}},
            {"event_type": "SECURITY_ANALYSIS", "tool": "send_email", "detected": False},
            {"event_type": "TOOL_RESULT", "tool": "send_email", "input": "SIMULATED EMAIL..."},
            {"event_type": "AGENT_RESPONSE", "input": "I sent a simulated email to Rahul."},
        ])

    # 5. Search + Read multi-tool normal task (2 tool calls)
    for _ in range(10):
        normal_sessions.append([
            {"event_type": "USER_REQUEST", "input": "Search and read"},
            {"event_type": "SECURITY_ANALYSIS", "detected": False},
            {"event_type": "TOOL_CALL", "tool": "search_knowledge", "input": {"query": "AgentPulse"}},
            {"event_type": "SECURITY_ANALYSIS", "tool": "search_knowledge", "detected": False},
            {"event_type": "TOOL_RESULT", "tool": "search_knowledge", "input": "AgentPulse info"},
            {"event_type": "TOOL_CALL", "tool": "read_file", "input": {"filename": "notes.txt"}},
            {"event_type": "SECURITY_ANALYSIS", "tool": "read_file", "detected": False},
            {"event_type": "TOOL_RESULT", "tool": "read_file", "input": "Notes content"},
            {"event_type": "AGENT_RESPONSE", "input": "Found info and read notes."},
        ])

    return [extract_features(s) for s in normal_sessions]


class BehaviorAnomalyDetector:
    """IsolationForest-based behavioral anomaly detector for AI agent event sessions."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.is_trained = False

    def train(self, normal_feature_vectors: list[list[float]]) -> None:
        """Fit the IsolationForest model on benign session feature vectors."""
        if not normal_feature_vectors or len(normal_feature_vectors) < 2:
            raise ValueError("Insufficient training data. At least 2 feature vectors are required.")

        self.model.fit(normal_feature_vectors)
        self.is_trained = True

    def score(self, feature_vector: list[float]) -> float:
        """Return the raw IsolationForest decision score for a feature vector."""
        if not self.is_trained:
            raise RuntimeError("BehaviorAnomalyDetector must be trained before scoring.")

        raw_score = float(self.model.decision_function([feature_vector])[0])
        return raw_score

    def predict(self, feature_vector: list[float]) -> dict[str, Any]:
        """Predict whether a session feature vector represents normal or anomalous behavior."""
        if not self.is_trained:
            # Fallback if not trained
            return {
                "detection_type": "BEHAVIORAL_ANOMALY",
                "detected": False,
                "anomaly_score": 0.0,
                "model": "IsolationForest",
                "severity": "LOW",
                "reason": "Detector is not trained; default low risk.",
            }

        score_val = self.score(feature_vector)
        pred = int(self.model.predict([feature_vector])[0])  # 1 for inlier/normal, -1 for outlier/anomaly

        has_behavioral_red_flags = (
            feature_vector[1] > 4.0 or  # total_tool_calls > 4
            feature_vector[2] > 1.0 or  # total_tool_errors > 1
            feature_vector[8] > 0.0     # security_detection_count > 0
        )
        is_anomaly = bool(pred == -1 or score_val < 0.0 or has_behavioral_red_flags)

        # Interpret specific feature drivers for reason description
        reasons: list[str] = []
        if feature_vector[1] > 4.0:
            reasons.append("unusually high number of tool calls")
        if feature_vector[2] > 1.0:
            reasons.append("elevated tool-call errors")
        if feature_vector[8] > 0.0:
            reasons.append("security detection events present")

        if is_anomaly:
            severity = "HIGH" if (feature_vector[2] > 1.0 or feature_vector[8] > 0.0) else "MEDIUM"
            reason_text = (
                f"Anomalous agent behavior detected: {', '.join(reasons)}."
                if reasons
                else "Unusual agent behavior pattern detected compared to normal baseline."
            )
        else:
            severity = "LOW"
            reason_text = "Behavior is consistent with the learned baseline."

        return {
            "detection_type": "BEHAVIORAL_ANOMALY",
            "detected": is_anomaly,
            "anomaly_score": float(round(score_val, 4)),
            "model": "IsolationForest",
            "severity": severity,
            "reason": reason_text,
        }




# Singleton default trained detector
_DEFAULT_DETECTOR: BehaviorAnomalyDetector | None = None


def get_anomaly_detector() -> BehaviorAnomalyDetector:
    """Return a pre-trained singleton BehaviorAnomalyDetector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        detector = BehaviorAnomalyDetector()
        baseline_data = get_baseline_training_data()
        detector.train(baseline_data)
        _DEFAULT_DETECTOR = detector
    return _DEFAULT_DETECTOR
