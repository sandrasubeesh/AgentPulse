"""Unit and integration tests for Risk Scoring and Response Engine (Step 10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.agent import run_task
from monitoring.event_collector import EventCollector
from security.risk_engine import (
    RiskEngine,
    compute_risk_score,
    get_risk_engine,
)


# ---------------------------------------------------------------------------
# Helpers: build correlation dicts without going through the full pipeline
# ---------------------------------------------------------------------------

def _corr(
    *,
    rb: int = 0,
    anomaly: bool = False,
    ml: str = "NORMAL",
    ml_conf: float = 0.50,
    correlated: bool = False,
) -> dict[str, Any]:
    """Build a minimal SECURITY_CORRELATION dict for risk-engine unit tests."""
    return {
        "detection_type": "SECURITY_CORRELATION",
        "session_id": "sess_test",
        "rule_based_detections": rb,
        "behavioral_anomaly_detected": anomaly,
        "ml_prediction": ml,
        "ml_confidence": ml_conf,
        "correlated": correlated,
        "severity": "LOW",
        "reason": "",
        "signals": [],
    }


# ---------------------------------------------------------------------------
# 1. Completely normal session
# ---------------------------------------------------------------------------

def test_normal_session_low_allow() -> None:
    """Test 1: Normal session → LOW, ALLOW, score = 0."""
    engine = RiskEngine()
    res = engine.score(_corr())

    assert res["detection_type"] == "RISK_ASSESSMENT"
    assert res["risk_score"] == 0
    assert res["risk_level"] == "LOW"
    assert res["response_action"] == "ALLOW"
    assert res["signals_used"] == []


# ---------------------------------------------------------------------------
# 2. Only rule-based detection
# ---------------------------------------------------------------------------

def test_only_rule_based() -> None:
    """Test 2: One rule-based detection only → score = 30, MEDIUM, ALERT."""
    engine = RiskEngine()
    res = engine.score(_corr(rb=1))

    assert res["risk_score"] == 30
    assert res["risk_level"] == "MEDIUM"
    assert res["response_action"] == "ALERT"
    assert "RULE_BASED" in res["signals_used"]


# ---------------------------------------------------------------------------
# 3. Only Isolation Forest anomaly
# ---------------------------------------------------------------------------

def test_only_isolation_forest() -> None:
    """Test 3: IF anomaly only → score = 25, LOW, ALLOW."""
    engine = RiskEngine()
    res = engine.score(_corr(anomaly=True))

    assert res["risk_score"] == 25
    assert res["risk_level"] == "LOW"
    assert res["response_action"] == "ALLOW"
    assert "ISOLATION_FOREST" in res["signals_used"]


# ---------------------------------------------------------------------------
# 4. Only Random Forest suspicious
# ---------------------------------------------------------------------------

def test_only_random_forest() -> None:
    """Test 4: RF suspicious only → score = 25, LOW, ALLOW."""
    engine = RiskEngine()
    res = engine.score(_corr(ml="SUSPICIOUS", ml_conf=0.87))

    assert res["risk_score"] == 25
    assert res["risk_level"] == "LOW"
    assert res["response_action"] == "ALLOW"
    assert "RANDOM_FOREST" in res["signals_used"]


# ---------------------------------------------------------------------------
# 5. Rule-based + Isolation Forest
# ---------------------------------------------------------------------------

def test_rule_and_isolation_forest() -> None:
    """Test 5: Rule + IF → score = 55, MEDIUM, ALERT."""
    engine = RiskEngine()
    res = engine.score(_corr(rb=1, anomaly=True, correlated=True))

    assert res["risk_score"] == 55
    assert res["risk_level"] == "MEDIUM"
    assert res["response_action"] == "ALERT"
    assert "RULE_BASED" in res["signals_used"]
    assert "ISOLATION_FOREST" in res["signals_used"]


# ---------------------------------------------------------------------------
# 6. Rule-based + Isolation Forest + Random Forest
# ---------------------------------------------------------------------------

def test_triple_signal_high_human_review() -> None:
    """Test 6: Rule + IF + RF → score = 90, HIGH, HUMAN_REVIEW."""
    engine = RiskEngine()
    res = engine.score(_corr(rb=1, anomaly=True, ml="SUSPICIOUS", ml_conf=0.97, correlated=True))

    assert res["risk_score"] == 90
    assert res["risk_level"] == "HIGH"
    assert res["response_action"] == "HUMAN_REVIEW"
    assert set(res["signals_used"]) == {"RULE_BASED", "ISOLATION_FOREST", "RANDOM_FOREST"}


# ---------------------------------------------------------------------------
# 7. Score never below 0
# ---------------------------------------------------------------------------

def test_score_never_below_zero() -> None:
    """Test 7: Risk score is always >= 0."""
    engine = RiskEngine()
    res = engine.score(_corr())
    assert res["risk_score"] >= 0


# ---------------------------------------------------------------------------
# 8. Score never exceeds 100
# ---------------------------------------------------------------------------

def test_score_never_above_100() -> None:
    """Test 8: Risk score is always <= 100, even with multiple rule detections."""
    engine = RiskEngine()
    # Artificially bump rb count to many — should still cap at 100
    corr = _corr(rb=10, anomaly=True, ml="SUSPICIOUS", ml_conf=1.0, correlated=True)
    res = engine.score(corr)
    assert res["risk_score"] <= 100


# ---------------------------------------------------------------------------
# 9. Signal provenance preserved
# ---------------------------------------------------------------------------

def test_signal_provenance_preserved() -> None:
    """Test 9: signals_used lists every contributing source in order."""
    engine = RiskEngine()
    res = engine.score(_corr(rb=1, anomaly=True, ml="SUSPICIOUS", ml_conf=0.92, correlated=True))

    assert "RULE_BASED" in res["signals_used"]
    assert "ISOLATION_FOREST" in res["signals_used"]
    assert "RANDOM_FOREST" in res["signals_used"]

    # score_breakdown records per-detector contributions
    breakdown = res["score_breakdown"]
    assert breakdown["rule_based"] == 30
    assert breakdown["isolation_forest"] == 25
    assert breakdown["random_forest"] == 25
    assert breakdown["triple_correlation_bonus"] == 10


# ---------------------------------------------------------------------------
# 10. Risk assessment written to runtime_events.jsonl
# ---------------------------------------------------------------------------

def test_risk_assessment_written_to_event_log(tmp_path: Path) -> None:
    """Test 10: RISK_ASSESSMENT event is recorded in the EventCollector and JSONL log."""
    log_file = tmp_path / "risk_test_events.jsonl"
    ec = EventCollector(log_file=log_file)

    class FakeAgent:
        def invoke(self, state: Any) -> dict[str, Any]:
            class Msg:
                content = "All done."
            return {"messages": [Msg()]}

    reply = run_task("What is AI safety?", agent=FakeAgent(), collector=ec)
    assert reply == "All done."

    event_types = [e["event_type"] for e in ec.events]
    assert "RISK_ASSESSMENT" in event_types

    risk_events = [e for e in ec.events if e["event_type"] == "RISK_ASSESSMENT"]
    assert len(risk_events) == 1
    re = risk_events[0]

    assert "risk_score" in re
    assert "risk_level" in re
    assert "response_action" in re
    assert "signals_used" in re
    assert "reason" in re

    assert isinstance(re["risk_score"], int)
    assert re["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert re["response_action"] in ("ALLOW", "ALERT", "HUMAN_REVIEW")
    assert 0 <= re["risk_score"] <= 100

    # Verify it is also written to the JSONL log file
    assert log_file.exists()
    lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    ra_lines = [l for l in lines if l.get("event_type") == "RISK_ASSESSMENT"]
    assert len(ra_lines) >= 1


# ---------------------------------------------------------------------------
# 11. No raw sensitive data in RISK_ASSESSMENT events
# ---------------------------------------------------------------------------

def test_no_sensitive_data_in_risk_event(tmp_path: Path) -> None:
    """Test 11: RISK_ASSESSMENT events must not contain raw secrets or prompts."""
    log_file = tmp_path / "sensitive_risk.jsonl"
    ec = EventCollector(log_file=log_file)

    class FakeAgent:
        def invoke(self, state: Any) -> dict[str, Any]:
            class Msg:
                content = "Done."
            return {"messages": [Msg()]}

    secret = "password=SuperSecret123 api_key=sk-proj-abc123"
    run_task(secret, agent=FakeAgent(), collector=ec)

    risk_event = next(e for e in ec.events if e["event_type"] == "RISK_ASSESSMENT")
    risk_json = json.dumps(risk_event)
    assert "SuperSecret123" not in risk_json
    assert "sk-proj-abc123" not in risk_json
    assert secret not in risk_json


# ---------------------------------------------------------------------------
# 12. compute_risk_score helper
# ---------------------------------------------------------------------------

def test_compute_risk_score_helper() -> None:
    """Test 12: module-level helper function returns the same result as engine.score()."""
    corr = _corr(rb=1, anomaly=False, ml="NORMAL")
    res = compute_risk_score(corr)
    assert res["risk_score"] == 30
    assert res["risk_level"] == "MEDIUM"


# ---------------------------------------------------------------------------
# 13. ML confidence flows through to RISK_ASSESSMENT
# ---------------------------------------------------------------------------

def test_ml_confidence_in_risk_result() -> None:
    """Test 13: ml_confidence from correlation result is preserved in risk assessment."""
    corr = _corr(ml="SUSPICIOUS", ml_conf=0.8833)
    res = compute_risk_score(corr)
    assert res["ml_confidence"] == 0.8833
