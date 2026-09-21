"""Security Correlation Engine for AgentPulse (Step 9).

Aggregates evidence from rule-based attack detectors, Isolation Forest anomaly detection,
and Random Forest ML classification to produce a multi-signal correlated security finding
while strictly preserving signal provenance.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agentpulse.security.correlation_engine")


def extract_session_signals(session_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract individual security evidence signals from a session while preserving provenance.

    Captures:
    - RULE_BASED: Prompt Injection, Tool Misuse, etc.
    - ISOLATION_FOREST: Behavioral Anomaly Detection
    - RANDOM_FOREST: Supervised Behavior Classification
    """
    signals: list[dict[str, Any]] = []
    seen_rule_types: set[str] = set()

    for event in session_events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")

        # 1. Rule-Based Security Signals
        if event_type in ("SECURITY_DETECTION", "SECURITY_ANALYSIS"):
            detected = event.get("detected") is True or event.get("status") == "detected"
            inp = event.get("input")
            if isinstance(inp, dict) and not detected:
                detected = inp.get("detected") is True

            if detected:
                attack_type = (
                    event.get("attack_type")
                    or (inp.get("attack_type") if isinstance(inp, dict) else None)
                    or "SECURITY_DETECTION"
                )

                # Deduplicate repeated logs for the same rule attack within one session
                if attack_type not in seen_rule_types:
                    seen_rule_types.add(attack_type)
                    severity = event.get("severity") or (inp.get("severity") if isinstance(inp, dict) else "MEDIUM")
                    signals.append(
                        {
                            "source": "RULE_BASED",
                            "type": attack_type,
                            "detected": True,
                            "severity": severity,
                            "reason": event.get("reason") or (inp.get("reason") if isinstance(inp, dict) else f"Rule-based {attack_type} detected"),
                        }
                    )

        # 2. Isolation Forest Behavioral Anomaly Signal
        elif event_type == "BEHAVIORAL_ANOMALY":
            detected = event.get("detected") is True or (
                isinstance(event.get("input"), dict) and event["input"].get("detected") is True
            )
            score = event.get("anomaly_score")
            if score is None and isinstance(event.get("input"), dict):
                score = event["input"].get("anomaly_score", 0.0)

            signals.append(
                {
                    "source": "ISOLATION_FOREST",
                    "type": "BEHAVIORAL_ANOMALY",
                    "detected": bool(detected),
                    "anomaly_score": round(float(score), 4) if score is not None else 0.0,
                    "severity": event.get("severity", "LOW" if not detected else "MEDIUM"),
                }
            )

        # 3. Random Forest ML Prediction Signal
        elif event_type == "ML_BEHAVIOR_PREDICTION":
            inp = event.get("input")
            pred_label = event.get("predicted_label")
            if not pred_label and isinstance(inp, dict):
                pred_label = inp.get("predicted_label", "NORMAL")
            elif not pred_label:
                pred_label = "NORMAL"

            confidence = event.get("confidence")
            if confidence is None and isinstance(inp, dict):
                confidence = inp.get("confidence", 0.0)
            elif confidence is None:
                confidence = 0.0

            prediction_code = event.get("prediction")
            if prediction_code is None and isinstance(inp, dict):
                prediction_code = inp.get("prediction", 0)

            signals.append(
                {
                    "source": "RANDOM_FOREST",
                    "type": "BEHAVIOR_CLASSIFICATION",
                    "prediction": str(pred_label),
                    "prediction_code": int(prediction_code or 0),
                    "confidence": round(float(confidence), 4),
                    "detected": pred_label == "SUSPICIOUS" or prediction_code == 1,
                }
            )

    return signals


class SecurityCorrelationEngine:
    """Combines multi-detector evidence for a runtime session and computes correlation result."""

    def correlate_session(
        self,
        session_events: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process session events and produce a structured SECURITY_CORRELATION result.

        Never raises exceptions; returns safe fallback result if correlation fails.
        """
        try:
            signals = extract_session_signals(session_events)

            # Extract signal counts by source
            rb_signals = [s for s in signals if s["source"] == "RULE_BASED" and s.get("detected") is True]
            if_signals = [s for s in signals if s["source"] == "ISOLATION_FOREST" and s.get("detected") is True]
            rf_signals = [s for s in signals if s["source"] == "RANDOM_FOREST" and s.get("detected") is True]

            rule_based_count = len(rb_signals)
            behavioral_anomaly_detected = len(if_signals) > 0

            # Latest RF ML Prediction signal
            rf_latest = next((s for s in reversed(signals) if s["source"] == "RANDOM_FOREST"), None)
            if rf_latest:
                ml_prediction = rf_latest.get("prediction", "NORMAL")
                ml_confidence = float(rf_latest.get("confidence", 0.0))
            else:
                ml_prediction = "NORMAL"
                ml_confidence = 0.0

            # Count independent active detector sources
            active_sources: list[str] = []
            if rule_based_count > 0:
                active_sources.append("RULE_BASED")
            if behavioral_anomaly_detected:
                active_sources.append("ISOLATION_FOREST")
            if ml_prediction == "SUSPICIOUS":
                active_sources.append("RANDOM_FOREST")

            independent_signal_count = len(active_sources)

            # Determine Correlation & Severity Logic
            if independent_signal_count == 0:
                correlated = False
                severity = "LOW"
                reason = "No significant security signals were correlated."
            elif independent_signal_count == 1:
                correlated = False
                severity = "MEDIUM"
                src = active_sources[0]
                if src == "RULE_BASED":
                    rb_type = rb_signals[0].get("type", "SECURITY_DETECTION")
                    reason = f"Single security signal detected: Rule-based {rb_type}."
                elif src == "ISOLATION_FOREST":
                    reason = "Single security signal detected: Isolation Forest behavioral anomaly."
                else:
                    reason = "Single security signal detected: Random Forest suspicious classification."
            else:  # independent_signal_count >= 2
                correlated = True
                severity = "HIGH"
                sources_str = ", ".join(active_sources)
                reason = f"Multiple independent security signals indicate suspicious runtime behavior ({sources_str})."

            # Determine session_id
            sid = session_id
            if not sid and session_events:
                sid = session_events[0].get("event_id") or session_events[0].get("timestamp")
            if not sid:
                sid = "session_unknown"

            return {
                "detection_type": "SECURITY_CORRELATION",
                "session_id": str(sid),
                "rule_based_detections": rule_based_count,
                "behavioral_anomaly_detected": behavioral_anomaly_detected,
                "ml_prediction": ml_prediction,
                "ml_confidence": round(ml_confidence, 4),
                "correlated": correlated,
                "severity": severity,
                "reason": reason,
                "signals": signals,
            }
        except Exception as exc:  # noqa: BLE001
            err_msg = f"Security correlation error: {exc}"
            logger.error(err_msg)
            return {
                "detection_type": "SECURITY_CORRELATION",
                "session_id": str(session_id or "session_unknown"),
                "rule_based_detections": 0,
                "behavioral_anomaly_detected": False,
                "ml_prediction": "NORMAL",
                "ml_confidence": 0.0,
                "correlated": False,
                "severity": "LOW",
                "reason": f"Correlation process encountered error: {exc}",
                "signals": [],
                "error": err_msg,
            }


_GLOBAL_CORRELATION_ENGINE: SecurityCorrelationEngine | None = None


def get_correlation_engine() -> SecurityCorrelationEngine:
    """Return singleton SecurityCorrelationEngine instance."""
    global _GLOBAL_CORRELATION_ENGINE
    if _GLOBAL_CORRELATION_ENGINE is None:
        _GLOBAL_CORRELATION_ENGINE = SecurityCorrelationEngine()
    return _GLOBAL_CORRELATION_ENGINE


def correlate_runtime_session(
    session_events: list[dict[str, Any]],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Helper function to correlate security signals for a runtime session."""
    engine = get_correlation_engine()
    return engine.correlate_session(session_events, session_id=session_id)
