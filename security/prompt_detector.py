"""Prompt Injection Detector for AgentPulse Stage 3 Attack Detection Layer."""

from __future__ import annotations

import re
from typing import NamedTuple

from security.types import DetectionResult

# Legacy / base literal patterns preserved for backward compatibility
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


class InjectionRule(NamedTuple):
    name: str
    pattern: re.Pattern[str]
    description: str


# Comprehensive contextual rules covering prompt injection attack vectors
STRUCTURED_RULES: list[InjectionRule] = [
    # 1. Instruction disregard / override / conflict negation
    InjectionRule(
        name="INSTRUCTION_DISREGARD",
        pattern=re.compile(
            r"\b(?:ignore|disregard|forget|override|drop|negate|skip)\s+"
            r"(?:all\s+)?(?:any\s+)?(?:your\s+)?(?:the\s+)?"
            r"(?:prior|previous|existing|above|system|developer|safety|conflicting|other)?\s*"
            r"(?:instructions?|rules?|directives?|guidelines?|constraints?|prompts?|restrictions?)\b",
            re.IGNORECASE,
        ),
        description="Attempt to disregard or override existing/prior instructions or conflicting rules",
    ),
    InjectionRule(
        name="INSTRUCTION_DISREGARD",
        pattern=re.compile(
            r"\bfollow\s+only\s+(?:the\s+)?instructions?\s+in\s+this\s+(?:message|prompt|conversation|task)\b",
            re.IGNORECASE,
        ),
        description="Attempt to isolate instructions to current message exclusively",
    ),
    InjectionRule(
        name="INSTRUCTION_DISREGARD",
        pattern=re.compile(
            r"\bdo\s+not\s+follow\s+(?:any\s+)?(?:previous|prior|system|developer|other)\s+instructions?\b",
            re.IGNORECASE,
        ),
        description="Attempt to block execution of previous or system instructions",
    ),
    InjectionRule(
        name="INSTRUCTION_DISREGARD",
        pattern=re.compile(
            r"\bstop\s+following\s+(?:your\s+)?(?:previous|prior|system|safety)\s+rules?\b",
            re.IGNORECASE,
        ),
        description="Attempt to halt adherence to safety rules",
    ),
    # 2. Security restriction circumvention
    InjectionRule(
        name="RESTRICTION_CIRCUMVENTION",
        pattern=re.compile(
            r"\bpretend\s+(?:that\s+)?(?:you\s+have\s+)?no\s+(?:security|safety|ethical)?\s*restrictions?\b",
            re.IGNORECASE,
        ),
        description="Roleplay attempt pretending security restrictions do not exist",
    ),
    InjectionRule(
        name="RESTRICTION_CIRCUMVENTION",
        pattern=re.compile(
            r"\b(?:security|safety|ethical)\s+restrictions?\s+(?:do\s+not|no\s+longer|dont)\s+apply\b",
            re.IGNORECASE,
        ),
        description="Claim that security restrictions no longer apply",
    ),
    InjectionRule(
        name="RESTRICTION_CIRCUMVENTION",
        pattern=re.compile(
            r"\b(?:bypass|ignore|circumvent|disable)\s+(?:all\s+)?(?:safety|security|guardrails?|filters?|rules?|protections?|policies)\b",
            re.IGNORECASE,
        ),
        description="Direct instruction to bypass safety or security guardrails",
    ),
    InjectionRule(
        name="RESTRICTION_CIRCUMVENTION",
        pattern=re.compile(
            r"\byou\s+are\s+(?:now\s+)?(?:unrestricted|freed|liberated|uncensored|unfiltered)\b",
            re.IGNORECASE,
        ),
        description="Assertion that agent is now unrestricted or unfiltered",
    ),
    InjectionRule(
        name="RESTRICTION_CIRCUMVENTION",
        pattern=re.compile(
            r"\b(?:without|free\s+of)\s+(?:any\s+)?(?:safety|security)\s+restrictions?\b",
            re.IGNORECASE,
        ),
        description="Request to operate without safety or security restrictions",
    ),
    # 3. System prompt and internal instructions leakage
    InjectionRule(
        name="SYSTEM_PROMPT_LEAKAGE",
        pattern=re.compile(
            r"\b(?:reveal|show|print|display|leak|output|repeat|share)\s+"
            r"(?:your\s+)?(?:hidden\s+|initial\s+|internal\s+|secret\s+)?(?:system\s+)?"
            r"(?:prompt|instructions?|directives?)\b",
            re.IGNORECASE,
        ),
        description="Request to leak or reveal internal system prompt or instructions",
    ),
    InjectionRule(
        name="SYSTEM_PROMPT_LEAKAGE",
        pattern=re.compile(
            r"\bgive\s+me\s+(?:your\s+)?(?:hidden\s+|initial\s+|internal\s+|secret\s+)(?:system\s+)?(?:prompt|instructions?)\b",
            re.IGNORECASE,
        ),
        description="Request to extract hidden or secret system prompt",
    ),
    InjectionRule(
        name="SYSTEM_PROMPT_LEAKAGE",
        pattern=re.compile(
            r"\bwhat\s+(?:is|are)\s+your\s+(?:system\s+prompt|internal\s+instructions|hidden\s+(?:prompt|instructions))\b",
            re.IGNORECASE,
        ),
        description="Direct query for system prompt or hidden instructions",
    ),
    # 4. Authority impersonation and fake system updates
    InjectionRule(
        name="AUTHORITY_IMPERSONATION",
        pattern=re.compile(
            r"(?:^|[\r\n])\s*\[?\s*(?:system|admin|root|developer|superuser|operator)\s+(?:update|override|command|message|alert|mode)\s*\]?[:\-]",
            re.IGNORECASE,
        ),
        description="Fake system or admin update header impersonation",
    ),
    InjectionRule(
        name="AUTHORITY_IMPERSONATION",
        pattern=re.compile(
            r"\b(?:admin|root|developer|operator)\s+override\b",
            re.IGNORECASE,
        ),
        description="Authority override keyword attempt",
    ),
    InjectionRule(
        name="AUTHORITY_IMPERSONATION",
        pattern=re.compile(
            r"\benable\s+(?:developer|debug|god|root|jailbreak|unrestricted)\s+mode\b",
            re.IGNORECASE,
        ),
        description="Developer or unrestricted mode activation attempt",
    ),
    InjectionRule(
        name="AUTHORITY_IMPERSONATION",
        pattern=re.compile(
            r"\bjailbreak\b",
            re.IGNORECASE,
        ),
        description="Jailbreak keyword attempt",
    ),
]


class PromptInjectionDetector:
    """Detects prompt injection and instruction override attempts in user input."""

    def __init__(self, patterns: list[str] | None = None) -> None:
        self.custom_patterns = patterns
        self.rules = STRUCTURED_RULES

    def detect(self, text: str) -> DetectionResult:
        """Analyze text for prompt injection patterns.

        Normalizes input (lowercase, whitespace collapsed) and checks against
        contextual attack signatures and custom patterns.
        """
        if not text or not isinstance(text, str):
            return DetectionResult(
                detected=False,
                attack_type=None,
                severity="LOW",
                confidence=0.0,
                reason="No known prompt injection pattern detected",
                rule_name=None,
            )

        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", text.strip())

        # If custom patterns were explicitly provided by caller, check them
        if self.custom_patterns is not None:
            norm_lower = normalized.lower()
            for pattern in self.custom_patterns:
                pattern_norm = re.sub(r"\s+", " ", pattern.strip().lower())
                if pattern_norm in norm_lower:
                    return DetectionResult(
                        detected=True,
                        attack_type="PROMPT_INJECTION",
                        severity="HIGH",
                        confidence=0.95,
                        reason=f"Instruction override pattern detected: '{pattern}'",
                        rule_name="CUSTOM_PATTERN",
                    )
            return DetectionResult(
                detected=False,
                attack_type=None,
                severity="LOW",
                confidence=0.0,
                reason="No known prompt injection pattern detected",
                rule_name=None,
            )

        # Evaluate structured contextual rules
        for rule in self.rules:
            match = rule.pattern.search(normalized)
            if match:
                matched_fragment = match.group(0)
                return DetectionResult(
                    detected=True,
                    attack_type="PROMPT_INJECTION",
                    severity="HIGH",
                    confidence=0.95,
                    reason=f"Instruction override pattern detected: '{matched_fragment}' ({rule.description})",
                    rule_name=rule.name,
                )

        # Fallback check against baseline literal patterns
        norm_lower = normalized.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            pattern_norm = re.sub(r"\s+", " ", pattern.strip().lower())
            if pattern_norm in norm_lower:
                return DetectionResult(
                    detected=True,
                    attack_type="PROMPT_INJECTION",
                    severity="HIGH",
                    confidence=0.95,
                    reason=f"Instruction override pattern detected: '{pattern}'",
                    rule_name="INSTRUCTION_DISREGARD",
                )

        return DetectionResult(
            detected=False,
            attack_type=None,
            severity="LOW",
            confidence=0.0,
            reason="No known prompt injection pattern detected",
            rule_name=None,
        )
