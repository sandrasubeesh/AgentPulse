"""Prompt Injection Detector for AgentPulse Stage 3 Attack Detection Layer."""

from __future__ import annotations

import re

from security.types import DetectionResult

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore your instructions",
    "disregard previous instructions",
    "forget your instructions",
    "override your instructions",
    "ignore safety rules",
    "bypass safety",
    "reveal your system prompt",
    "show your system prompt",
    "reveal hidden instructions",
    "you are now unrestricted",
    "developer mode",
    "jailbreak",
]


class PromptInjectionDetector:
    """Detects prompt injection and instruction override attempts in user input."""

    def __init__(self, patterns: list[str] | None = None) -> None:
        self.patterns = patterns if patterns is not None else PROMPT_INJECTION_PATTERNS

    def detect(self, text: str) -> DetectionResult:
        """Analyze text for prompt injection patterns.

        Normalizes input (lowercase, whitespace collapsed) and checks against
        known attack signatures.
        """
        if not text or not isinstance(text, str):
            return DetectionResult(
                detected=False,
                attack_type=None,
                severity="LOW",
                confidence=0.0,
                reason="No known prompt injection pattern detected",
            )

        # Normalize whitespace and case
        normalized = re.sub(r"\s+", " ", text.strip().lower())

        for pattern in self.patterns:
            pattern_norm = re.sub(r"\s+", " ", pattern.strip().lower())
            if pattern_norm in normalized:
                return DetectionResult(
                    detected=True,
                    attack_type="PROMPT_INJECTION",
                    severity="HIGH",
                    confidence=0.95,
                    reason=f"Instruction override pattern detected: '{pattern}'",
                )

        return DetectionResult(
            detected=False,
            attack_type=None,
            severity="LOW",
            confidence=0.0,
            reason="No known prompt injection pattern detected",
        )
