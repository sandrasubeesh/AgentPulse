"""Risk Scoring and Response Engine for AgentPulse (Step 10).

Converts correlated security signals from SecurityCorrelationEngine into an
explicit, deterministic risk score (0–100), risk level (LOW/MEDIUM/HIGH), and
a research-safe response action (ALLOW/ALERT/HUMAN_REVIEW).

Scoring formula (deterministic, documented):
  +30  — at least one rule-based detection confirmed
  +25  — Isolation Forest behavioral anomaly detected
  +25  — Random Forest prediction is SUSPICIOUS
  +10  — three independent detector sources all indicate suspicious activity
   →0  — no signals at all

Cap: final score is clamped to [0, 100].

Risk level thresholds:
   0–29  → LOW    → ALLOW
  30–59  → MEDIUM → ALERT
  60–100 → HIGH   → HUMAN_REVIEW

IMPORTANT: The risk score is an interpretable aggregation of independent signals,
NOT a mathematically calibrated probability of attack.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agentpulse.security.risk_engine")

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
SCORE_RULE_BASED = 30        # One or more rule-based detections confirmed
SCORE_ISOLATION_FOREST = 25  # Isolation Forest anomaly detected
SCORE_RANDOM_FOREST = 25     # Random Forest predicts SUSPICIOUS
SCORE_TRIPLE_CORRELATION = 10  # All three independent sources agree

# Risk level thresholds (inclusive lower bound)
THRESHOLD_HIGH = 60
THRESHOLD_MEDIUM = 30


def _determine_risk_level(score: int) -> str:
    if score >= THRESHOLD_HIGH:
        return "HIGH"
    if score >= THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _determine_response_action(risk_level: str) -> str:
    if risk_level == "HIGH":
        return "HUMAN_REVIEW"
    if risk_level == "MEDIUM":
        return "ALERT"
    return "ALLOW"


class RiskEngine:
    """Converts correlated security evidence into a risk score and response action.

    Accepts a SECURITY_CORRELATION result dict (from SecurityCorrelationEngine)
    and produces a RISK_ASSESSMENT result dict.
    """

    def score(self, correlation_result: dict[str, Any]) -> dict[str, Any]:
        """Calculate risk score from a SECURITY_CORRELATION result.

        Args:
            correlation_result: dict produced by SecurityCorrelationEngine.correlate_session()

        Returns:
            Structured RISK_ASSESSMENT dict.  Never raises.
        """
        try:
            return self._compute_risk(correlation_result)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"Risk scoring error: {exc}"
            logger.error(err_msg)
            return {
                "detection_type": "RISK_ASSESSMENT",
                "risk_score": 0,
                "risk_level": "LOW",
                "response_action": "ALLOW",
                "signals_used": [],
                "reason": f"Risk scoring encountered an error: {exc}",
                "error": err_msg,
            }

    def _compute_risk(self, corr: dict[str, Any]) -> dict[str, Any]:
        raw_score = 0
        signals_used: list[str] = []
        contributing_sources: list[str] = []

        # --- Rule-based contribution ---
        rb_count: int = corr.get("rule_based_detections", 0) or 0
        if rb_count > 0:
            raw_score += SCORE_RULE_BASED
            signals_used.append("RULE_BASED")
            contributing_sources.append("RULE_BASED")

        # --- Isolation Forest contribution ---
        if_detected: bool = corr.get("behavioral_anomaly_detected", False)
        if if_detected:
            raw_score += SCORE_ISOLATION_FOREST
            signals_used.append("ISOLATION_FOREST")
            contributing_sources.append("ISOLATION_FOREST")

        # --- Random Forest contribution ---
        ml_pred: str = corr.get("ml_prediction", "NORMAL") or "NORMAL"
        ml_conf: float = float(corr.get("ml_confidence", 0.0) or 0.0)
        if ml_pred == "SUSPICIOUS":
            raw_score += SCORE_RANDOM_FOREST
            signals_used.append("RANDOM_FOREST")
            contributing_sources.append("RANDOM_FOREST")

        # --- Triple-source correlation bonus ---
        if len(contributing_sources) >= 3:
            raw_score += SCORE_TRIPLE_CORRELATION

        # --- Cap at 100 ---
        final_score = min(100, max(0, raw_score))

        risk_level = _determine_risk_level(final_score)
        response_action = _determine_response_action(risk_level)

        # Build reason string
        if not signals_used:
            reason = "No significant security signals were detected."
        elif len(signals_used) == 1:
            reason = f"Single security signal detected ({signals_used[0]}). Score: {final_score}."
        else:
            reason = (
                "Multiple independent security detectors identified suspicious runtime behavior "
                f"({', '.join(signals_used)}). Score: {final_score}."
            )

        # Extract session_id from correlation result for traceability
        session_id = corr.get("session_id", "session_unknown")

        return {
            "detection_type": "RISK_ASSESSMENT",
            "session_id": str(session_id),
            "risk_score": final_score,
            "risk_level": risk_level,
            "response_action": response_action,
            "ml_prediction": ml_pred,
            "ml_confidence": round(ml_conf, 4),
            "signals_used": signals_used,
            "reason": reason,
            # Scoring breakdown for auditability
            "score_breakdown": {
                "rule_based": SCORE_RULE_BASED if "RULE_BASED" in contributing_sources else 0,
                "isolation_forest": SCORE_ISOLATION_FOREST if "ISOLATION_FOREST" in contributing_sources else 0,
                "random_forest": SCORE_RANDOM_FOREST if "RANDOM_FOREST" in contributing_sources else 0,
                "triple_correlation_bonus": SCORE_TRIPLE_CORRELATION if len(contributing_sources) >= 3 else 0,
            },
        }

    def score_from_events(
        self,
        session_events: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Convenience method: correlate events and then compute risk score."""
        from security.correlation_engine import SecurityCorrelationEngine

        engine = SecurityCorrelationEngine()
        corr = engine.correlate_session(session_events, session_id=session_id)
        return self.score(corr)


# ---------------------------------------------------------------------------
# Module-level singleton + helpers
# ---------------------------------------------------------------------------

_GLOBAL_RISK_ENGINE: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    """Return the singleton RiskEngine instance."""
    global _GLOBAL_RISK_ENGINE
    if _GLOBAL_RISK_ENGINE is None:
        _GLOBAL_RISK_ENGINE = RiskEngine()
    return _GLOBAL_RISK_ENGINE


def compute_risk_score(correlation_result: dict[str, Any]) -> dict[str, Any]:
    """Helper: compute a RISK_ASSESSMENT dict from a correlation result."""
    return get_risk_engine().score(correlation_result)
