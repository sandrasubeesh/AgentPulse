"""Security types and dataclasses for AgentPulse Attack Detection Layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectionResult:
    """Standard security detection result returned by all detectors."""

    detected: bool
    attack_type: str | None
    severity: str
    confidence: float
    reason: str
    rule_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a serializable dictionary."""
        d: dict[str, Any] = {
            "detected": self.detected,
            "attack_type": self.attack_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "reason": self.reason,
        }
        if self.rule_name is not None:
            d["rule_name"] = self.rule_name
        return d
