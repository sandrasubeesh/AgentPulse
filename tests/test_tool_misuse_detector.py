"""Tests for ToolMisuseDetector."""

from __future__ import annotations

from security.tool_misuse_detector import ToolMisuseDetector


def test_notes_txt_allowed() -> None:
    detector = ToolMisuseDetector()
    result = detector.detect("read_file", {"filename": "notes.txt"})

    assert not result.detected
    assert result.attack_type is None
    assert result.severity == "LOW"


def test_dot_dot_env_detected() -> None:
    detector = ToolMisuseDetector()
    result = detector.detect("read_file", {"filename": "../../.env"})

    assert result.detected
    assert result.attack_type == "TOOL_MISUSE"
    assert result.severity == "HIGH"
    assert "traversal" in result.reason.lower()


def test_single_dot_dot_env_detected() -> None:
    detector = ToolMisuseDetector()
    result = detector.detect("read_file", {"filename": "../.env"})

    assert result.detected
    assert result.attack_type == "TOOL_MISUSE"


def test_etc_passwd_detected() -> None:
    detector = ToolMisuseDetector()
    result = detector.detect("read_file", {"filename": "/etc/passwd"})

    assert result.detected
    assert result.attack_type == "TOOL_MISUSE"
    assert "absolute" in result.reason.lower() or "sensitive" in result.reason.lower()


def test_ssh_key_detected() -> None:
    detector = ToolMisuseDetector()
    result = detector.detect("read_file", {"filename": "~/.ssh/id_rsa"})

    assert result.detected
    assert result.attack_type == "TOOL_MISUSE"


def test_normal_filenames_not_detected() -> None:
    detector = ToolMisuseDetector()
    res1 = detector.detect("read_file", {"filename": "data.csv"})
    res2 = detector.detect("read_file", "sample_report.txt")

    assert not res1.detected
    assert not res2.detected


def test_result_contains_all_required_fields() -> None:
    detector = ToolMisuseDetector()
    result = detector.detect("read_file", {"filename": "../../.env"})
    res_dict = result.to_dict()

    assert "detected" in res_dict
    assert "attack_type" in res_dict
    assert "severity" in res_dict
    assert "confidence" in res_dict
    assert "reason" in res_dict
