"""Isolated Credible Evaluation Pipeline for AgentPulse Behavioral ML.

Evaluates:
1. Read-Only Audit of existing `data/behavioral_dataset.csv` (200 rows):
   - Single-feature baselines (`security_detection_count`, `tool_result_count`, `total_events`)
   - Discrete template duplicate analysis
   - Verification that `data/behavioral_dataset.csv` lacks `session_id`/`scenario_family`
     provenance for grouped evaluation without heuristic fabrication.
2. Benchmark A (`synthetic_exact_152`):
   - Evaluates the exact 152 synthetic sessions from `behavior/expand_dataset.py`
     with verified ground-truth `scenario_family` IDs across 4 feature sets:
       a) 11-feature (with `security_detection_count` detector leakage)
       b) 9-feature raw (`total_events` includes `SECURITY_*` event overhead)
       c) 9-feature de-leaked (`non_security_total_events` excludes `SECURITY_*` overhead)
       d) 8-feature pure tool/duration (`FEATURES_8_PURE_TOOL_TIMING`)
   - Compares row-level (`StratifiedShuffleSplit` / `StratifiedKFold`) vs
     scenario-family-grouped (`GroupShuffleSplit` / `StratifiedGroupKFold`) splits.
   - Explicitly checks class coverage per fold/split and reports when coarse group count
     (e.g. only 3 suspicious families in `exact_152`) cannot support 5-fold group CV
     with both classes in every validation fold, using 3-fold `StratifiedGroupKFold`
     where every fold is guaranteed to contain both classes.
3. Benchmark B (`synthetic_enriched_240`):
   - Evaluates the 240-session enriched synthetic benchmark (12 scenario families:
     6 NORMAL, 6 SUSPICIOUS) across the same 4 feature sets and row-level vs
     5-fold `StratifiedGroupKFold` / `GroupShuffleSplit` splits.

All outputs are saved strictly under `models/experiments/` (or a caller-supplied directory).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeClassifier

from experiments.feature_ablation_definitions import (
    FEATURES_8_PURE_TOOL_TIMING,
    FEATURES_9_DELEAKED_BEHAVIORAL,
    FEATURES_9_RAW_TOTAL_EVENTS,
    FEATURES_11_WITH_DETECTOR,
    LABEL_MAPPING,
)
from experiments.synthetic_benchmark_generator import (
    DEFAULT_EXPERIMENTS_DATA_DIR,
    generate_enriched_synthetic_benchmark,
    generate_exact_152_tagged_benchmark,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXISTING_CSV_PATH = PROJECT_ROOT / "data" / "behavioral_dataset.csv"
DEFAULT_EXPERIMENTS_OUTPUT_DIR = PROJECT_ROOT / "models" / "experiments"

FEATURE_SETS: dict[str, list[str]] = {
    "11_with_detector_leakage": FEATURES_11_WITH_DETECTOR,
    "9_raw_total_events_indirect_leakage": FEATURES_9_RAW_TOTAL_EVENTS,
    "9_deleaked_behavioral": FEATURES_9_DELEAKED_BEHAVIORAL,
    "8_pure_tool_timing": FEATURES_8_PURE_TOOL_TIMING,
}


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, macro-F1, confusion matrix, and class distributions."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_bin = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    clf_rep = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "SUSPICIOUS"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1_bin, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "class_distribution": {
            "NORMAL": int(np.sum(y_true == 0)),
            "SUSPICIOUS": int(np.sum(y_true == 1)),
        },
        "class_wise_metrics": {
            "NORMAL": {
                "precision": round(float(clf_rep["NORMAL"]["precision"]), 4),
                "recall": round(float(clf_rep["NORMAL"]["recall"]), 4),
                "f1_score": round(float(clf_rep["NORMAL"]["f1-score"]), 4),
                "support": int(clf_rep["NORMAL"]["support"]),
            },
            "SUSPICIOUS": {
                "precision": round(float(clf_rep["SUSPICIOUS"]["precision"]), 4),
                "recall": round(float(clf_rep["SUSPICIOUS"]["recall"]), 4),
                "f1_score": round(float(clf_rep["SUSPICIOUS"]["f1-score"]), 4),
                "support": int(clf_rep["SUSPICIOUS"]["support"]),
            },
        },
    }


def audit_existing_200_csv_readonly(csv_path: Path | str | None = None) -> dict[str, Any]:
    """Perform a strictly read-only diagnostic audit of the existing 200-row dataset."""
    path = Path(csv_path) if csv_path is not None else EXISTING_CSV_PATH
    df = pd.read_csv(path)
    y = df["label"].map(LABEL_MAPPING).to_numpy(dtype=np.int64)

    # 1. Single-feature 5-fold StratifiedKFold decision stump evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    single_feature_cv: dict[str, dict[str, float]] = {}

    for col in FEATURES_11_WITH_DETECTOR:
        X_col = df[[col]].to_numpy(dtype=np.float64)
        fold_accs, fold_macro_f1s = [], []
        for tr_idx, val_idx in skf.split(X_col, y):
            stump = DecisionTreeClassifier(max_depth=1, random_state=42, class_weight="balanced")
            stump.fit(X_col[tr_idx], y[tr_idx])
            preds = stump.predict(X_col[val_idx])
            fold_accs.append(float(accuracy_score(y[val_idx], preds)))
            fold_macro_f1s.append(float(f1_score(y[val_idx], preds, average="macro", zero_division=0)))
        single_feature_cv[col] = {
            "mean_cv_accuracy": round(float(np.mean(fold_accs)), 4),
            "std_cv_accuracy": round(float(np.std(fold_accs)), 4),
            "mean_cv_macro_f1": round(float(np.mean(fold_macro_f1s)), 4),
        }

    # 2. Check deterministic rule: security_detection_count > 0
    sec_rule_preds = (df["security_detection_count"].to_numpy() > 0).astype(np.int64)
    sec_rule_acc = round(float(accuracy_score(y, sec_rule_preds)), 4)

    # 3. Count unique discrete feature combinations (excluding duration and detector flags)
    discrete_behavioral_cols = [
        c for c in FEATURES_9_RAW_TOTAL_EVENTS if c != "session_duration_seconds"
    ]
    unique_discrete_patterns = int(len(df.drop_duplicates(subset=discrete_behavioral_cols)))

    return {
        "dataset_path": str(path),
        "total_rows": len(df),
        "class_distribution": {
            "NORMAL": int(np.sum(y == 0)),
            "SUSPICIOUS": int(np.sum(y == 1)),
        },
        "has_scenario_group_column": "scenario_group" in df.columns or "scenario_family" in df.columns,
        "security_detection_rule_accuracy": sec_rule_acc,
        "unique_discrete_behavioral_patterns": unique_discrete_patterns,
        "single_feature_5fold_stump": single_feature_cv,
    }


def _find_valid_group_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform GroupShuffleSplit and verify both classes are represented in train and test sets."""
    # Try deterministic sequence of seeds starting from random_state to ensure non-empty class coverage
    for offset in range(25):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + offset)
        train_idx, test_idx = next(gss.split(X, y, groups))
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_tr)) == 2 and len(set(y_te)) == 2:
            tr_groups = sorted(list(set(groups[train_idx])))
            te_groups = sorted(list(set(groups[test_idx])))
            overlap = sorted(list(set(tr_groups).intersection(set(te_groups))))
            return {
                "valid_class_coverage": True,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_groups": tr_groups,
                "test_groups": te_groups,
                "group_overlap": overlap,
                "group_overlap_count": len(overlap),
                "seed_used": random_state + offset,
            }

    return {
        "valid_class_coverage": False,
        "reason": "GroupShuffleSplit could not place both NORMAL and SUSPICIOUS classes in both train and test splits.",
    }


def _evaluate_group_cv_with_coverage_check(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run StratifiedGroupKFold with strict verification that every fold contains both classes.

    Never fabricates groups. If `requested_splits` exceeds the minimum number of scenario
    families in either class (for example, 5-fold CV when SUSPICIOUS has only 3 families),
    explicitly reports the class-coverage failure and evaluates at the maximum supported
    fold count (`max_supported_splits`).
    """
    normal_groups = sorted(list(set(groups[y == 0])))
    suspicious_groups = sorted(list(set(groups[y == 1])))
    min_class_groups = min(len(normal_groups), len(suspicious_groups))

    if min_class_groups < 2:
        return {
            "supported": False,
            "requested_splits": requested_splits,
            "reason": (
                f"Insufficient scenario families per class for grouped CV "
                f"(NORMAL families={len(normal_groups)}, SUSPICIOUS families={len(suspicious_groups)})."
            ),
        }

    # Check whether requested_splits can cover both classes in every validation fold
    requested_coverage_ok = requested_splits <= min_class_groups
    actual_splits = min(requested_splits, min_class_groups)

    sgkf = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    fold_details: list[dict[str, Any]] = []
    oof_preds = np.zeros_like(y)
    oof_covered = np.zeros_like(y, dtype=bool)

    all_folds_have_both_classes = True

    for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        tr_groups = sorted(list(set(groups[tr_idx])))
        val_groups = sorted(list(set(groups[val_idx])))
        overlap = set(tr_groups).intersection(set(val_groups))
        if overlap:
            raise AssertionError(f"Group leakage detected in fold {fold_idx}: {overlap}")

        y_tr, y_val = y[tr_idx], y[val_idx]
        val_classes = set(np.unique(y_val))
        tr_classes = set(np.unique(y_tr))
        has_both = (val_classes == {0, 1}) and (tr_classes == {0, 1})
        if not has_both:
            all_folds_have_both_classes = False

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        clf.fit(X[tr_idx], y_tr)
        preds = clf.predict(X[val_idx])
        oof_preds[val_idx] = preds
        oof_covered[val_idx] = True

        fold_metrics = _compute_binary_metrics(y_val, preds)
        fold_details.append({
            "fold": fold_idx + 1,
            "train_groups": tr_groups,
            "val_groups": val_groups,
            "group_overlap_count": len(overlap),
            "both_classes_present": has_both,
            "metrics": fold_metrics,
        })

    fold_accs = [f["metrics"]["accuracy"] for f in fold_details]
    fold_precs = [f["metrics"]["precision"] for f in fold_details]
    fold_recs = [f["metrics"]["recall"] for f in fold_details]
    fold_f1s = [f["metrics"]["f1_score"] for f in fold_details]
    fold_macro_f1s = [f["metrics"]["macro_f1"] for f in fold_details]

    pooled_metrics = _compute_binary_metrics(y, oof_preds)

    coverage_note = (
        f"All {actual_splits} grouped folds contain both NORMAL and SUSPICIOUS classes with 0 group overlap."
        if requested_coverage_ok and all_folds_have_both_classes
        else (
            f"Requested {requested_splits}-fold StratifiedGroupKFold cannot place SUSPICIOUS sessions in all "
            f"{requested_splits} validation folds because dataset has {len(normal_groups)} NORMAL families and "
            f"{len(suspicious_groups)} SUSPICIOUS families. Automatically used {actual_splits}-fold "
            f"StratifiedGroupKFold (max folds with full two-class coverage)."
        )
    )

    return {
        "supported": True,
        "cv_method": "StratifiedGroupKFold",
        "requested_splits": requested_splits,
        "requested_splits_valid_coverage": requested_coverage_ok,
        "actual_splits_used": actual_splits,
        "all_folds_have_both_classes": all_folds_have_both_classes,
        "normal_family_count": len(normal_groups),
        "suspicious_family_count": len(suspicious_groups),
        "coverage_note": coverage_note,
        "fold_accuracies": fold_accs,
        "mean_accuracy": round(float(np.mean(fold_accs)), 4),
        "std_accuracy": round(float(np.std(fold_accs)), 4),
        "mean_precision": round(float(np.mean(fold_precs)), 4),
        "mean_recall": round(float(np.mean(fold_recs)), 4),
        "mean_f1": round(float(np.mean(fold_f1s)), 4),
        "mean_macro_f1": round(float(np.mean(fold_macro_f1s)), 4),
        "pooled_oof_metrics": pooled_metrics,
        "folds": fold_details,
    }


def evaluate_benchmark_across_feature_sets(
    df: pd.DataFrame,
    benchmark_name: str,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate a tagged synthetic benchmark comparing row-level vs scenario-family-grouped splits."""
    if "scenario_family" not in df.columns or "session_id" not in df.columns:
        raise ValueError(f"Benchmark '{benchmark_name}' must contain 'session_id' and 'scenario_family' columns.")

    y = df["label"].map(LABEL_MAPPING).to_numpy(dtype=np.int64)
    groups = df["scenario_family"].astype(str).to_numpy()

    # Quantify indirect leakage of total_events vs non_security_total_events
    sec_overhead = (df["total_events"] - df["non_security_total_events"]).to_numpy()
    normal_overhead_mean = round(float(np.mean(sec_overhead[y == 0])), 4)
    susp_overhead_mean = round(float(np.mean(sec_overhead[y == 1])), 4)

    feature_set_comparisons: dict[str, Any] = {}

    for fset_name, fset_cols in FEATURE_SETS.items():
        X = df[fset_cols].to_numpy(dtype=np.float64)

        # -----------------------------------------------------------------
        # A. Row-Level Evaluation (StratifiedShuffleSplit + 5-fold StratifiedKFold)
        # -----------------------------------------------------------------
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
        tr_idx_row, te_idx_row = next(sss.split(X, y))

        rf_row = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        rf_row.fit(X[tr_idx_row], y[tr_idx_row])
        preds_row_holdout = rf_row.predict(X[te_idx_row])
        row_holdout_metrics = _compute_binary_metrics(y[te_idx_row], preds_row_holdout)

        # Check how many scenario families leaked across train and test in the row-level split
        row_tr_fams = set(groups[tr_idx_row])
        row_te_fams = set(groups[te_idx_row])
        row_shared_fams = sorted(list(row_tr_fams.intersection(row_te_fams)))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        row_cv_accs, row_cv_precs, row_cv_recs, row_cv_f1s, row_cv_macro_f1s = [], [], [], [], []
        row_oof_preds = np.zeros_like(y)

        for tr_i, val_i in skf.split(X, y):
            clf_fold = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight="balanced",
            )
            clf_fold.fit(X[tr_i], y[tr_i])
            p_val = clf_fold.predict(X[val_i])
            row_oof_preds[val_i] = p_val
            m_fold = _compute_binary_metrics(y[val_i], p_val)
            row_cv_accs.append(m_fold["accuracy"])
            row_cv_precs.append(m_fold["precision"])
            row_cv_recs.append(m_fold["recall"])
            row_cv_f1s.append(m_fold["f1_score"])
            row_cv_macro_f1s.append(m_fold["macro_f1"])

        row_cv_summary = {
            "cv_method": "StratifiedKFold",
            "n_splits": 5,
            "fold_accuracies": row_cv_accs,
            "mean_accuracy": round(float(np.mean(row_cv_accs)), 4),
            "std_accuracy": round(float(np.std(row_cv_accs)), 4),
            "mean_precision": round(float(np.mean(row_cv_precs)), 4),
            "mean_recall": round(float(np.mean(row_cv_recs)), 4),
            "mean_f1": round(float(np.mean(row_cv_f1s)), 4),
            "mean_macro_f1": round(float(np.mean(row_cv_macro_f1s)), 4),
            "pooled_oof_metrics": _compute_binary_metrics(y, row_oof_preds),
        }

        # -----------------------------------------------------------------
        # B. Scenario-Family-Grouped Evaluation (GroupShuffleSplit + StratifiedGroupKFold)
        # -----------------------------------------------------------------
        grp_holdout_split = _find_valid_group_holdout_split(
            X, y, groups, test_size=0.25, random_state=random_state
        )
        if grp_holdout_split["valid_class_coverage"]:
            g_tr_idx = grp_holdout_split["train_idx"]
            g_te_idx = grp_holdout_split["test_idx"]
            rf_grp = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight="balanced",
            )
            rf_grp.fit(X[g_tr_idx], y[g_tr_idx])
            preds_grp_holdout = rf_grp.predict(X[g_te_idx])
            grp_holdout_metrics = _compute_binary_metrics(y[g_te_idx], preds_grp_holdout)
            importances = [
                {"feature": fset_cols[i], "importance": round(float(rf_grp.feature_importances_[i]), 4)}
                for i in np.argsort(rf_grp.feature_importances_)[::-1]
            ]
        else:
            grp_holdout_metrics = None
            importances = []

        grp_cv_summary = _evaluate_group_cv_with_coverage_check(
            X=X,
            y=y,
            groups=groups,
            requested_splits=5,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        # Compute delta (Grouped minus Row-Level)
        acc_delta_cv = round(grp_cv_summary["mean_accuracy"] - row_cv_summary["mean_accuracy"], 4)
        macro_f1_delta_cv = round(grp_cv_summary["mean_macro_f1"] - row_cv_summary["mean_macro_f1"], 4)

        feature_set_comparisons[fset_name] = {
            "features": list(fset_cols),
            "feature_count": len(fset_cols),
            "row_level_evaluation": {
                "holdout_metrics": row_holdout_metrics,
                "shared_scenario_families_in_holdout": row_shared_fams,
                "shared_family_count": len(row_shared_fams),
                "cv_metrics": row_cv_summary,
            },
            "scenario_family_grouped_evaluation": {
                "holdout_split_info": {
                    "valid_class_coverage": grp_holdout_split["valid_class_coverage"],
                    "train_groups": grp_holdout_split.get("train_groups", []),
                    "test_groups": grp_holdout_split.get("test_groups", []),
                    "group_overlap_count": grp_holdout_split.get("group_overlap_count"),
                },
                "holdout_metrics": grp_holdout_metrics,
                "cv_metrics": grp_cv_summary,
                "feature_importances": importances,
            },
            "row_vs_grouped_delta": {
                "cv_mean_accuracy_diff_grouped_minus_row": acc_delta_cv,
                "cv_mean_macro_f1_diff_grouped_minus_row": macro_f1_delta_cv,
            },
        }

    return {
        "benchmark_name": benchmark_name,
        "is_synthetic_data": True,
        "disclaimer": (
            "All sessions in this benchmark are synthetically generated for controlled "
            "evaluation methodology analysis and do NOT represent real-world production security effectiveness."
        ),
        "total_samples": len(df),
        "class_distribution": {
            "NORMAL": int(np.sum(y == 0)),
            "SUSPICIOUS": int(np.sum(y == 1)),
        },
        "scenario_family_distribution": (
            df.groupby(["scenario_family", "label"]).size().unstack(fill_value=0).to_dict(orient="index")
        ),
        "indirect_total_events_leakage_analysis": {
            "mean_security_event_overhead_normal": normal_overhead_mean,
            "mean_security_event_overhead_suspicious": susp_overhead_mean,
            "explanation": (
                "In raw 'total_events', SECURITY_ANALYSIS, SECURITY_DETECTION, and BEHAVIORAL_ANOMALY "
                "events are counted in the session length. 'non_security_total_events' strips these "
                "detector-generated events so session length reflects only operational user/tool/agent events."
            ),
        },
        "feature_set_evaluations": feature_set_comparisons,
    }


def run_complete_experimental_pipeline(
    data_output_dir: Path | str | None = None,
    report_output_dir: Path | str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Generate both isolated synthetic benchmarks, evaluate all feature sets, and save reports."""
    data_dir = Path(data_output_dir) if data_output_dir is not None else DEFAULT_EXPERIMENTS_DATA_DIR
    rep_dir = Path(report_output_dir) if report_output_dir is not None else DEFAULT_EXPERIMENTS_OUTPUT_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read-only audit of existing 200-row CSV
    existing_audit = audit_existing_200_csv_readonly()

    # 2. Generate and evaluate Benchmark A (Exact 152 synthetic sessions)
    bench_152_gen = generate_exact_152_tagged_benchmark(output_dir=data_dir, seed=random_state)
    df_152 = pd.read_csv(bench_152_gen["csv_path"])
    eval_152 = evaluate_benchmark_across_feature_sets(
        df=df_152,
        benchmark_name="synthetic_exact_152",
        n_estimators=100,
        random_state=random_state,
    )

    # 3. Generate and evaluate Benchmark B (Enriched 240 synthetic sessions)
    bench_240_gen = generate_enriched_synthetic_benchmark(output_dir=data_dir, seed=2026)
    df_240 = pd.read_csv(bench_240_gen["csv_path"])
    eval_240 = evaluate_benchmark_across_feature_sets(
        df=df_240,
        benchmark_name="synthetic_enriched_240",
        n_estimators=100,
        random_state=random_state,
    )

    full_report = {
        "pipeline_type": "isolated_experimental_credibility_evaluation",
        "synthetic_data_notice": (
            "All generated datasets and evaluations in this report use synthetic agent session "
            "simulations. Results measure evaluation methodology sensitivity (label leakage, "
            "template cloning, and row-level vs scenario-grouped splitting) and must NOT be "
            "interpreted as real-world attack detection effectiveness."
        ),
        "existing_200_csv_readonly_audit": existing_audit,
        "benchmarks": {
            "synthetic_exact_152": {
                "csv_path": bench_152_gen["csv_path"],
                "jsonl_path": bench_152_gen["jsonl_path"],
                "evaluation": eval_152,
            },
            "synthetic_enriched_240": {
                "csv_path": bench_240_gen["csv_path"],
                "jsonl_path": bench_240_gen["jsonl_path"],
                "evaluation": eval_240,
            },
        },
    }

    report_json_path = rep_dir / "credibility_evaluation_report.json"
    report_md_path = rep_dir / "credibility_evaluation_summary.md"

    with open(report_json_path, "w", encoding="utf-8") as f_json:
        json.dump(full_report, f_json, indent=2)

    md_text = _format_markdown_summary(full_report)
    with open(report_md_path, "w", encoding="utf-8") as f_md:
        f_md.write(md_text)

    full_report["report_json_path"] = str(report_json_path)
    full_report["report_md_path"] = str(report_md_path)
    return full_report


def _format_markdown_summary(report: dict[str, Any]) -> str:
    """Format a clear, honest Markdown summary of the experimental results."""
    lines: list[str] = [
        "# AgentPulse Behavioral ML Credibility Evaluation Report (Synthetic Data)",
        "",
        f"> **Note**: {report['synthetic_data_notice']}",
        "",
        "## 1. Read-Only Audit of Existing `data/behavioral_dataset.csv` (200 Rows)",
    ]
    audit = report["existing_200_csv_readonly_audit"]
    lines.extend([
        f"- **Total Rows**: {audit['total_rows']} (`NORMAL`: {audit['class_distribution']['NORMAL']}, `SUSPICIOUS`: {audit['class_distribution']['SUSPICIOUS']})",
        f"- **Has Scenario-Family Column**: `{audit['has_scenario_group_column']}`",
        f"- **Unique Discrete 8-Feature Patterns**: {audit['unique_discrete_behavioral_patterns']} across 200 rows",
        f"- **Rule `security_detection_count > 0` Accuracy**: {audit['security_detection_rule_accuracy'] * 100:.2f}%",
        f"- **Single-Feature 5-Fold Decision Stump (`security_detection_count`)**: {audit['single_feature_5fold_stump']['security_detection_count']['mean_cv_accuracy'] * 100:.2f}%",
        f"- **Single-Feature 5-Fold Decision Stump (`tool_result_count`)**: {audit['single_feature_5fold_stump']['tool_result_count']['mean_cv_accuracy'] * 100:.2f}%",
        f"- **Single-Feature 5-Fold Decision Stump (`total_events`)**: {audit['single_feature_5fold_stump']['total_events']['mean_cv_accuracy'] * 100:.2f}%",
        "",
    ])

    for b_key, b_title in [
        ("synthetic_exact_152", "2. Benchmark A: Exact 152 Synthetic Sessions (`expand_dataset.py` Templates)"),
        ("synthetic_enriched_240", "3. Benchmark B: Enriched 240 Synthetic Sessions (12 Scenario Families with Hard Negatives)"),
    ]:
        b_eval = report["benchmarks"][b_key]["evaluation"]
        ind = b_eval["indirect_total_events_leakage_analysis"]
        lines.extend([
            f"## {b_title}",
            f"- **Samples**: {b_eval['total_samples']} (`NORMAL`: {b_eval['class_distribution']['NORMAL']}, `SUSPICIOUS`: {b_eval['class_distribution']['SUSPICIOUS']})",
            f"- **Mean Security Event Overhead (`total_events - non_security_total_events`)**: NORMAL = `{ind['mean_security_event_overhead_normal']}`, SUSPICIOUS = `{ind['mean_security_event_overhead_suspicious']}`",
            "",
            "| Feature Set | Split Mode | Holdout Acc | Holdout Macro-F1 | CV Method (Folds) | CV Mean Acc (±Std) | CV Mean Macro-F1 | Pooled OOF Confusion Matrix `[[TN, FP], [FN, TP]]` |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for fset_name, fset_res in b_eval["feature_set_evaluations"].items():
            row_ev = fset_res["row_level_evaluation"]
            grp_ev = fset_res["scenario_family_grouped_evaluation"]

            r_hm = row_ev["holdout_metrics"]
            r_cv = row_ev["cv_metrics"]
            r_cm = r_cv["pooled_oof_metrics"]["confusion_matrix"]

            lines.append(
                f"| `{fset_name}` | **Row-Level** (Leaks templates) | {r_hm['accuracy']:.4f} | {r_hm['macro_f1']:.4f} | "
                f"`{r_cv['cv_method']}` ({r_cv['n_splits']}) | {r_cv['mean_accuracy']:.4f} (±{r_cv['std_accuracy']:.4f}) | "
                f"{r_cv['mean_macro_f1']:.4f} | `{r_cm}` |"
            )

            g_hm = grp_ev["holdout_metrics"]
            g_cv = grp_ev["cv_metrics"]
            g_cm = g_cv["pooled_oof_metrics"]["confusion_matrix"]
            g_hm_acc = f"{g_hm['accuracy']:.4f}" if g_hm else "N/A"
            g_hm_mf1 = f"{g_hm['macro_f1']:.4f}" if g_hm else "N/A"

            lines.append(
                f"| `{fset_name}` | **Scenario-Grouped** (0 overlap) | {g_hm_acc} | {g_hm_mf1} | "
                f"`{g_cv['cv_method']}` ({g_cv['actual_splits_used']}) | **{g_cv['mean_accuracy']:.4f}** (±{g_cv['std_accuracy']:.4f}) | "
                f"**{g_cv['mean_macro_f1']:.4f}** | `{g_cm}` |"
            )

        # Add coverage note from the first feature set's grouped CV
        first_grp_cv = next(iter(b_eval["feature_set_evaluations"].values()))["scenario_family_grouped_evaluation"]["cv_metrics"]
        lines.extend([
            "",
            f"- **Group CV Class-Coverage Check**: {first_grp_cv['coverage_note']}",
            "",
        ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    res = run_complete_experimental_pipeline()
    print(Path(res["report_md_path"]).read_text(encoding="utf-8"))
