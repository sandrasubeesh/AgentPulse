"""Behavior package for AgentPulse Behavioral Anomaly Detection."""

from behavior.anomaly_detector import BehaviorAnomalyDetector, get_anomaly_detector
from behavior.dataset_builder import build_runtime_behavioral_dataset
from behavior.dataset_generator import (
    DEFAULT_DATASET_CSV,
    determine_session_label,
    extract_session_features,
    generate_behavioral_dataset,
    parse_valid_sessions,
)
from behavior.feature_engineering import LABEL_MAPPING, prepare_behavioral_dataset, validate_dataset
from behavior.feature_extractor import FEATURE_NAMES, extract_features
from behavior.runtime_baseline import (
    is_session_benign,
    load_benign_runtime_baseline,
    parse_sessions_from_events,
)

__all__ = [
    "DEFAULT_DATASET_CSV",
    "FEATURE_NAMES",
    "LABEL_MAPPING",
    "BehaviorAnomalyDetector",
    "build_runtime_behavioral_dataset",
    "determine_session_label",
    "extract_features",
    "extract_session_features",
    "generate_behavioral_dataset",
    "get_anomaly_detector",
    "is_session_benign",
    "load_benign_runtime_baseline",
    "parse_sessions_from_events",
    "parse_valid_sessions",
    "prepare_behavioral_dataset",
    "validate_dataset",
]




