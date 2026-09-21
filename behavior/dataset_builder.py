"""Dataset builder for AgentPulse Supervised ML Pipeline (25% Milestone).

Parses runtime event sessions, extracts numerical features, assigns binary
behavioral labels ('Normal' vs 'Suspicious'), and exports a clean dataset CSV
excluding raw text inputs or secrets.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from behavior.feature_extractor import FEATURE_NAMES, extract_features
from behavior.runtime_baseline import is_session_benign, parse_sessions_from_events

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "runtime_events.jsonl"
DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "behavioral_dataset.csv"


def build_runtime_behavioral_dataset(
    log_path: Path | str | None = None,
    output_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a controlled runtime behavioral dataset CSV from runtime event logs.

    Extracts numerical features and labels for benign ('Normal') and security-test
    ('Suspicious') sessions, excluding raw text inputs or secrets.

    Returns dict containing:
      - 'csv_path': path to generated CSV file
      - 'total_samples': total dataset count
      - 'features': feature column names
      - 'class_distribution': dict mapping label -> count
    """
    log_file = Path(log_path) if log_path is not None else DEFAULT_LOG_FILE
    out_csv = Path(output_csv_path) if output_csv_path is not None else DEFAULT_DATASET_CSV

    raw_events: list[dict[str, Any]] = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            raw_events.append(json.loads(line_str))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            pass

    sessions = parse_sessions_from_events(raw_events)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = list(FEATURE_NAMES) + ["label"]
    rows: list[list[Any]] = []
    normal_count = 0
    suspicious_count = 0

    for session in sessions:
        feature_vector = extract_features(session)
        is_benign = is_session_benign(session)
        label = "Normal" if is_benign else "Suspicious"

        if is_benign:
            normal_count += 1
        else:
            suspicious_count += 1

        row = feature_vector + [label]
        rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return {
        "csv_path": str(out_csv),
        "total_samples": len(rows),
        "features": list(FEATURE_NAMES),
        "class_distribution": {
            "Normal": normal_count,
            "Suspicious": suspicious_count,
        },
    }
