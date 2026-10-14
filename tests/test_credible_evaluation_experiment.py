"""Isolated unit and integrity tests for the AgentPulse experimental credibility pipeline.

Verifies:
1. Existing files (`data/behavioral_dataset.csv`, `logs/runtime_events.jsonl`,
   `models/random_forest_behavior_model.joblib`, `models/random_forest_behavior_model_v2.joblib`)
   are completely untouched (byte-for-byte SHA-256 verification).
2. Exact 152-session synthetic benchmark has explicit `session_id`, `scenario_family`,
   and `label` columns, with 74 NORMAL and 78 SUSPICIOUS sessions across 9 families.
3. Enriched 240-session synthetic benchmark has 120 NORMAL and 120 SUSPICIOUS sessions
   across 12 scenario families, including benign tool errors and hard negatives.
4. Indirect leakage through `total_events` is explicitly quantified and removed by
   `non_security_total_events`.
5. Scenario-family-grouped evaluation enforces zero group overlap between train and test
   sets and checks class coverage before running grouped cross-validation.
6. Full deterministic reproducibility with fixed seeds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from experiments.credible_evaluation import (
    FEATURE_SETS,
    audit_existing_200_csv_readonly,
    evaluate_benchmark_across_feature_sets,
    run_complete_experimental_pipeline,
)
from experiments.feature_ablation_definitions import (
    EXPERIMENTAL_CSV_HEADERS,
    FEATURES_8_PURE_TOOL_TIMING,
    FEATURES_9_DELEAKED_BEHAVIORAL,
    FEATURES_9_RAW_TOTAL_EVENTS,
    FEATURES_11_WITH_DETECTOR,
)
from experiments.synthetic_benchmark_generator import (
    generate_enriched_synthetic_benchmark,
    generate_exact_152_tagged_benchmark,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_project_files_strictly_preserved(tmp_path: Path) -> None:
    """Verify that running the experimental generators and evaluator never modifies existing project files."""
    protected_files = [
        PROJECT_ROOT / "data" / "behavioral_dataset.csv",
        PROJECT_ROOT / "logs" / "runtime_events.jsonl",
        PROJECT_ROOT / "models" / "random_forest_behavior_model.joblib",
        PROJECT_ROOT / "models" / "random_forest_behavior_model_v2.joblib",
        PROJECT_ROOT / "ml" / "evaluate_model_v2.py",
        PROJECT_ROOT / "ml" / "runtime_predictor.py",
        PROJECT_ROOT / "behavior" / "dataset_generator.py",
        PROJECT_ROOT / "behavior" / "expand_dataset.py",
    ]
    hashes_before = {p: _sha256(p) for p in protected_files if p.exists()}

    # Run complete experimental pipeline in tmp_path
    data_dir = tmp_path / "exp_data"
    rep_dir = tmp_path / "exp_reports"
    run_complete_experimental_pipeline(data_output_dir=data_dir, report_output_dir=rep_dir)

    hashes_after = {p: _sha256(p) for p in protected_files if p.exists()}
    assert hashes_before == hashes_after


def test_readonly_audit_of_existing_200_csv() -> None:
    """Verify the read-only diagnostic audit of the existing 200-row dataset."""
    audit = audit_existing_200_csv_readonly()

    assert audit["total_rows"] == 200
    assert audit["class_distribution"] == {"NORMAL": 100, "SUSPICIOUS": 100}
    assert audit["has_scenario_group_column"] is False
    assert audit["security_detection_rule_accuracy"] == 1.0

    stumps = audit["single_feature_5fold_stump"]
    assert stumps["security_detection_count"]["mean_cv_accuracy"] == 1.0
    assert stumps["tool_result_count"]["mean_cv_accuracy"] == 0.865


def test_exact_152_benchmark_generation_and_integrity(tmp_path: Path) -> None:
    """Verify the tagged 152-session benchmark matches expand_dataset.py counts and schema."""
    res = generate_exact_152_tagged_benchmark(output_dir=tmp_path, seed=42)

    assert res["is_synthetic"] is True
    assert res["total_samples"] == 152
    assert res["normal_count"] == 74
    assert res["suspicious_count"] == 78

    df = pd.read_csv(res["csv_path"])
    assert list(df.columns) == EXPERIMENTAL_CSV_HEADERS
    assert len(df) == 152
    assert df["session_id"].nunique() == 152
    assert not df.isna().any().any()

    # Check indirect leakage column: in exact_152, NORMAL sessions have 0 security events,
    # whereas SUSPICIOUS sessions have 1 or 2 security events counted inside raw total_events!
    normal_df = df[df["label"] == "NORMAL"]
    susp_df = df[df["label"] == "SUSPICIOUS"]

    assert (normal_df["security_event_overhead"] == 0.0).all()
    assert (susp_df["security_event_overhead"] >= 1.0).all()
    assert (df["non_security_total_events"] == df["total_events"] - df["security_event_overhead"]).all()

    # Specifically verify 0-tool prompt injection vs 0-tool benign Q&A:
    # Raw total_events is 4.0 vs 2.0, but non_security_total_events is 2.0 for both!
    qa_df = df[df["scenario_family"] == "benign_qa"]
    pi_df = df[df["scenario_family"] == "suspicious_prompt_injection"]
    assert (qa_df["total_events"] == 2.0).all()
    assert (pi_df["total_events"] == 4.0).all()
    assert (qa_df["non_security_total_events"] == 2.0).all()
    assert (pi_df["non_security_total_events"] == 2.0).all()


def test_enriched_240_benchmark_diversity_and_hard_negatives(tmp_path: Path) -> None:
    """Verify the enriched 240-session benchmark contains 12 balanced scenario families and hard negatives."""
    res = generate_enriched_synthetic_benchmark(output_dir=tmp_path, seed=2026)

    assert res["is_synthetic"] is True
    assert res["total_samples"] == 240
    assert res["normal_count"] == 120
    assert res["suspicious_count"] == 120
    assert len(res["scenario_family_counts"]) == 12
    for count in res["scenario_family_counts"].values():
        assert count == 20

    df = pd.read_csv(res["csv_path"])
    assert list(df.columns) == EXPERIMENTAL_CSV_HEADERS
    assert len(df) == 240
    assert df["session_id"].nunique() == 240

    # Verify benign hard negatives with tool errors exist
    normal_with_errors = df[(df["label"] == "NORMAL") & (df["tool_error_count"] > 0)]
    assert len(normal_with_errors) >= 40

    # Verify suspicious sessions with successful tool results exist (breaking the tool_result_count == 0 shortcut)
    susp_with_results = df[(df["label"] == "SUSPICIOUS") & (df["tool_result_count"] > 0)]
    assert len(susp_with_results) >= 30


def test_group_separation_and_coverage_verification(tmp_path: Path) -> None:
    """Verify zero scenario-family overlap in grouped splits and honest fold-coverage reporting."""
    res_152 = generate_exact_152_tagged_benchmark(output_dir=tmp_path, seed=42)
    df_152 = pd.read_csv(res_152["csv_path"])
    eval_152 = evaluate_benchmark_across_feature_sets(df_152, benchmark_name="synthetic_exact_152")

    for fset_name, fset_data in eval_152["feature_set_evaluations"].items():
        grp = fset_data["scenario_family_grouped_evaluation"]
        h_info = grp["holdout_split_info"]
        assert h_info["valid_class_coverage"] is True
        assert h_info["group_overlap_count"] == 0
        assert set(h_info["train_groups"]).isdisjoint(set(h_info["test_groups"]))

        # Because exact_152 has only 3 SUSPICIOUS scenario families, 5-fold group CV cannot
        # place SUSPICIOUS sessions in all 5 folds; verify the pipeline detects this and uses 3 folds
        cv = grp["cv_metrics"]
        assert cv["requested_splits"] == 5
        assert cv["requested_splits_valid_coverage"] is False
        assert cv["actual_splits_used"] == 3
        assert cv["all_folds_have_both_classes"] is True
        for fold in cv["folds"]:
            assert fold["group_overlap_count"] == 0
            assert fold["both_classes_present"] is True

    # In enriched_240, there are 6 NORMAL and 6 SUSPICIOUS families, so 5-fold StratifiedGroupKFold
    # has full 2-class coverage across all 5 folds
    res_240 = generate_enriched_synthetic_benchmark(output_dir=tmp_path, seed=2026)
    df_240 = pd.read_csv(res_240["csv_path"])
    eval_240 = evaluate_benchmark_across_feature_sets(df_240, benchmark_name="synthetic_enriched_240")

    for fset_name, fset_data in eval_240["feature_set_evaluations"].items():
        grp = fset_data["scenario_family_grouped_evaluation"]
        cv = grp["cv_metrics"]
        assert cv["requested_splits"] == 5
        assert cv["requested_splits_valid_coverage"] is True
        assert cv["actual_splits_used"] == 5
        assert cv["all_folds_have_both_classes"] is True
        for fold in cv["folds"]:
            assert fold["group_overlap_count"] == 0
            assert fold["both_classes_present"] is True


def test_reproducibility_across_runs(tmp_path: Path) -> None:
    """Verify deterministic reproducibility of both dataset generation and evaluation metrics."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    rep1 = run_complete_experimental_pipeline(data_output_dir=dir1 / "d", report_output_dir=dir1 / "r", random_state=42)
    rep2 = run_complete_experimental_pipeline(data_output_dir=dir2 / "d", report_output_dir=dir2 / "r", random_state=42)

    assert rep1["existing_200_csv_readonly_audit"] == rep2["existing_200_csv_readonly_audit"]
    assert (
        rep1["benchmarks"]["synthetic_exact_152"]["evaluation"]
        == rep2["benchmarks"]["synthetic_exact_152"]["evaluation"]
    )
    assert (
        rep1["benchmarks"]["synthetic_enriched_240"]["evaluation"]
        == rep2["benchmarks"]["synthetic_enriched_240"]["evaluation"]
    )
