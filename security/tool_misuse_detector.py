"""Tool Misuse Detector for AgentPulse Stage 3 Attack Detection Layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from security.types import DetectionResult

SENSITIVE_PATTERNS = [
    ".env",
    "/etc/passwd",
    "id_rsa",
    ".ssh",
    ".aws",
    "shadow",
]


class ToolMisuseDetector:
    """Detects unauthorized tool usage, path traversal, and sensitive file access attempts."""

    def detect(self, tool_name: str, args: Any) -> DetectionResult:
        """Analyze tool call name and arguments for security violations."""
        if not tool_name or not isinstance(tool_name, str):
            return DetectionResult(
                detected=False,
                attack_type=None,
                severity="LOW",
                confidence=0.0,
                reason="No tool misuse detected",
            )

        if tool_name == "read_file":
            filename = ""
            if isinstance(args, dict):
                filename = str(args.get("filename", "")).strip()
            elif isinstance(args, str):
                filename = args.strip()

            if not filename:
                return DetectionResult(
                    detected=False,
                    attack_type=None,
                    severity="LOW",
                    confidence=0.0,
                    reason="No tool misuse detected",
                )

            # Check 1: Path Traversal
            if ".." in filename:
                return DetectionResult(
                    detected=True,
                    attack_type="TOOL_MISUSE",
                    severity="HIGH",
                    confidence=0.98,
                    reason="Path traversal attempt detected",
                )

            # Check 2: Absolute Path Access
            p = Path(filename)
            if p.is_absolute() or filename.startswith(("/", "\\", "~")) or (len(filename) >= 2 and filename[1] == ":"):
                return DetectionResult(
                    detected=True,
                    attack_type="TOOL_MISUSE",
                    severity="HIGH",
                    confidence=0.98,
                    reason="Absolute path access attempt detected",
                )

            # Check 3: Sensitive File Access
            fn_lower = filename.lower()
            for sensitive in SENSITIVE_PATTERNS:
                if sensitive in fn_lower:
                    return DetectionResult(
                        detected=True,
                        attack_type="TOOL_MISUSE",
                        severity="HIGH",
                        confidence=0.95,
                        reason=f"Access to sensitive file pattern detected: '{sensitive}'",
                    )

        return DetectionResult(
            detected=False,
            attack_type=None,
            severity="LOW",
            confidence=0.0,
            reason="No tool misuse detected",
        )
