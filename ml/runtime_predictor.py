"""ML Runtime Predictor for AgentPulse (Step 8).

Loads saved Random Forest model artifact, extracts session-level behavioral features
matching the trained model's feature order, computes predicted class labels and
probabilities (confidence), and generates structured ML prediction results safely.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_behavior_model.joblib"

logger = logging.getLogger("agentpulse.ml.runtime_predictor")


def extract_raw_session_feature_dict(session_events: list[dict[str, Any]]) -> dict[str, float]:
    """Extract a comprehensive dictionary of raw numerical behavioral features from a session.

    Handles empty or None event lists safely and calculates duration from timestamps.
    """
    if not session_events:
        return {
            "total_events": 0.0,
            "tool_call_count": 0.0,
            "tool_result_count": 0.0,
            "tool_error_count": 0.0,
            "unique_tool_count": 0.0,
            "file_read_count": 0.0,
            "search_call_count": 0.0,
            "email_call_count": 0.0,
            "security_detection_count": 0.0,
            "prompt_event_count": 0.0,
            "session_duration_seconds": 0.0,
        }

    total_events = float(len(session_events))
    tool_call_count = 0.0
    tool_result_count = 0.0
    tool_error_count = 0.0
    unique_tools: set[str] = set()
    file_read_count = 0.0
    search_call_count = 0.0
    email_call_count = 0.0
    security_detection_count = 0.0
    prompt_event_count = 0.0

    for event in session_events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")
        tool = event.get("tool")

        if event_type == "USER_REQUEST":
            prompt_event_count += 1.0
        elif event_type == "TOOL_CALL":
            tool_call_count += 1.0
            if tool:
                unique_tools.add(str(tool))
                if tool == "read_file":
                    file_read_count += 1.0
                elif tool == "search_knowledge":
                    search_call_count += 1.0
                elif tool == "send_email":
                    email_call_count += 1.0
        elif event_type == "TOOL_RESULT":
            tool_result_count += 1.0
        elif event_type == "TOOL_ERROR":
            tool_error_count += 1.0
        elif event_type in ("SECURITY_DETECTION", "SECURITY_ANALYSIS", "BEHAVIORAL_ANOMALY"):
            if event.get("detected") is True or (
                isinstance(event.get("input"), dict) and event["input"].get("detected") is True
            ):
                security_detection_count += 1.0

    unique_tool_count = float(len(unique_tools))

    # Calculate session duration in seconds
    session_duration_seconds = 0.0
    start_ts_str = session_events[0].get("timestamp")
    end_ts_str = session_events[-1].get("timestamp")
    if start_ts_str and end_ts_str:
        try:
            start_dt = datetime.fromisoformat(str(start_ts_str))
            end_dt = datetime.fromisoformat(str(end_ts_str))
            duration = (end_dt - start_dt).total_seconds()
            session_duration_seconds = round(max(0.0, float(duration)), 4)
        except (ValueError, TypeError):
            session_duration_seconds = 0.0

    return {
        "total_events": total_events,
        "tool_call_count": tool_call_count,
        "tool_result_count": tool_result_count,
        "tool_error_count": tool_error_count,
        "unique_tool_count": unique_tool_count,
        "file_read_count": file_read_count,
        "search_call_count": search_call_count,
        "email_call_count": email_call_count,
        "security_detection_count": security_detection_count,
        "prompt_event_count": prompt_event_count,
        "session_duration_seconds": session_duration_seconds,
    }


class RuntimePredictor:
    """Session-level ML Behavioral Predictor using a trained Random Forest artifact."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.model: Any = None
        self.feature_names: list[str] = []
        self.label_mapping: dict[str, int] = {"NORMAL": 0, "SUSPICIOUS": 1}
        self.inv_label_mapping: dict[int, str] = {0: "NORMAL", 1: "SUSPICIOUS"}
        self.is_loaded: bool = False
        self.load_error: str | None = None

        self._load_artifact()

    def _load_artifact(self) -> None:
        """Load joblib model artifact and extract classifier + feature ordering."""
        if not self.model_path.exists():
            self.is_loaded = False
            self.load_error = f"Model artifact file not found at: {self.model_path}"
            logger.warning(self.load_error)
            return

        try:
            artifact = joblib.load(self.model_path)
            if not isinstance(artifact, dict) or "model" not in artifact:
                self.is_loaded = False
                self.load_error = f"Invalid model artifact structure in {self.model_path}"
                logger.error(self.load_error)
                return

            self.model = artifact["model"]
            self.feature_names = list(artifact.get("feature_names", []))
            self.label_mapping = artifact.get("label_mapping", {"NORMAL": 0, "SUSPICIOUS": 1})
            self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}
            self.is_loaded = True
            self.load_error = None
        except Exception as exc:  # noqa: BLE001
            self.is_loaded = False
            self.load_error = f"Failed to load Random Forest model artifact: {exc}"
            logger.error(self.load_error)

    def predict_session(self, session_events: list[dict[str, Any]]) -> dict[str, Any]:
        """Perform ML prediction on a completed session event list.

        Returns structured prediction result dict. Never raises exceptions; returns error
        status dict if model loading or inference fails.
        """
        if not self.is_loaded or self.model is None:
            return {
                "prediction": 0,
                "predicted_label": "NORMAL",
                "confidence": 0.0,
                "model": "RandomForestClassifier",
                "features_used": {},
                "status": "error",
                "error": self.load_error or "Model not loaded",
            }

        try:
            # 1. Extract raw feature dictionary
            raw_features = extract_raw_session_feature_dict(session_events)

            # 2. Align features with exact model artifact ordering
            feature_vector: list[float] = []
            features_used: dict[str, float] = {}

            for fname in self.feature_names:
                val = float(raw_features.get(fname, 0.0))
                feature_vector.append(val)
                features_used[fname] = val

            # 3. Reshape for sklearn input (1, n_features)
            X = np.array([feature_vector], dtype=np.float64)

            # 4. Predict class label and class probability confidence
            pred_class = int(self.model.predict(X)[0])
            pred_probas = self.model.predict_proba(X)[0]

            # Confidence = probability corresponding to predicted class
            confidence = float(pred_probas[pred_class])
            predicted_label = self.inv_label_mapping.get(pred_class, "NORMAL" if pred_class == 0 else "SUSPICIOUS")

            return {
                "prediction": pred_class,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 4),
                "model": type(self.model).__name__,
                "features_used": features_used,
                "feature_vector": feature_vector,
                "feature_names": list(self.feature_names),
                "status": "detected" if pred_class == 1 else "clear",
            }
        except Exception as exc:  # noqa: BLE001 - keep agent from crashing on prediction failures
            err_msg = f"Runtime ML prediction error: {exc}"
            logger.error(err_msg)
            return {
                "prediction": 0,
                "predicted_label": "NORMAL",
                "confidence": 0.0,
                "model": type(self.model).__name__ if self.model else "RandomForestClassifier",
                "features_used": {},
                "status": "error",
                "error": err_msg,
            }


_GLOBAL_PREDICTOR: RuntimePredictor | None = None


def get_runtime_predictor(model_path: Path | str | None = None) -> RuntimePredictor:
    """Return a singleton or newly initialized RuntimePredictor instance."""
    global _GLOBAL_PREDICTOR
    if model_path is not None:
        return RuntimePredictor(model_path=model_path)

    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = RuntimePredictor()
    return _GLOBAL_PREDICTOR


def predict_runtime_session(
    session_events: list[dict[str, Any]],
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Helper function to predict session behavior using default or specified model."""
    predictor = get_runtime_predictor(model_path=model_path)
    return predictor.predict_session(session_events)
