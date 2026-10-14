"""Improved ML Evaluation Methodology Module for AgentPulse (v2).

Evaluates the 9 pure behavioral features (excluding detector flags) using
StratifiedShuffleSplit for holdout evaluation and 5-fold StratifiedKFold
cross-validation when scenario-group IDs are unavailable, or group-aware
splitting (GroupShuffleSplit / StratifiedGroupKFold) ONLY when genuine,
validated scenario-group IDs are explicitly available. Serializes the trained
v2 model artifact and evaluation provenance to
models/random_forest_behavior_model_v2.joblib.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
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

from behavior.runtime_baseline import parse_sessions_from_events

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_V2_PATH = PROJECT_ROOT / "models" / "random_forest_behavior_model_v2.joblib"
DEFAULT_BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_behavior_model.joblib"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "runtime_events.jsonl"

BEHAVIORAL_9_FEATURES = [
    "total_events",
    "tool_call_count",
    "tool_result_count",
    "tool_error_count",
    "unique_tool_count",
    "file_read_count",
    "search_call_count",
    "email_call_count",
    "session_duration_seconds",
]

LABEL_MAPPING = {"NORMAL": 0, "SUSPICIOUS": 1}


def extract_scenario_groups_from_logs(log_path: Path | str | None = None) -> list[str]:
    """Parse runtime event log sessions and map each to a scenario group if logs exist.

    Returns an empty list when the log file is unavailable. Never fabricates groups.
    """
    path = Path(log_path) if log_path is not None else DEFAULT_LOG_FILE
    if not path.exists():
        return []

    raw_events: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        raw_events.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []

    sessions = parse_sessions_from_events(raw_events)
    groups: list[str] = []

    for session in sessions:
        has_prompt_inj = False
        has_tool_misuse = False
        has_anom = False

        for ev in session:
            etype = ev.get("event_type")
            inp = ev.get("input")
            inp_str = str(inp).lower()

            if etype in ("SECURITY_DETECTION", "SECURITY_ANALYSIS"):
                atk = ev.get("attack_type") or (inp.get("attack_type") if isinstance(inp, dict) else "")
                if atk == "PROMPT_INJECTION" or "prompt injection" in inp_str or "ignore previous" in inp_str:
                    has_prompt_inj = True
                elif atk == "TOOL_MISUSE" or "path traversal" in inp_str or ".." in inp_str or "/etc/" in inp_str:
                    has_tool_misuse = True
            elif etype == "BEHAVIORAL_ANOMALY" and (
                ev.get("detected") is True or (isinstance(inp, dict) and inp.get("detected") is True)
            ):
                has_anom = True

            if etype == "USER_REQUEST":
                if any(
                    term in inp_str
                    for term in ["ignore previous", "system prompt", "developer mode", "override", "disregard", "bypass"]
                ):
                    has_prompt_inj = True
                elif any(
                    term in inp_str
                    for term in ["../", "..\\", "/etc/passwd", "id_rsa", "/etc/shadow", "credentials"]
                ):
                    has_tool_misuse = True
                elif "anomalous automated task" in inp_str:
                    has_anom = True

        if has_prompt_inj:
            groups.append("prompt_injection")
            continue
        if has_tool_misuse:
            groups.append("tool_misuse")
            continue
        if has_anom:
            groups.append("behavioral_anomaly")
            continue

        tools = [ev.get("tool") for ev in session if ev.get("event_type") == "TOOL_CALL"]
        errors = [ev for ev in session if ev.get("event_type") == "TOOL_ERROR"]

        if errors:
            groups.append("benign_tool_error")
        elif len(tools) > 1:
            groups.append("benign_multitool")
        elif len(tools) == 1:
            t = tools[0]
            if t == "read_file":
                groups.append("benign_file_read")
            elif t == "search_knowledge":
                groups.append("benign_search")
            elif t == "send_email":
                groups.append("benign_email")
            else:
                groups.append("benign_qa")
        else:
            groups.append("benign_qa")

    return groups


def _resolve_valid_scenario_groups(
    df: pd.DataFrame,
    y: np.ndarray,
    n_samples: int,
    explicit_groups: Sequence[str] | np.ndarray | None = None,
    log_path: Path | str | None = None,
) -> tuple[np.ndarray | None, str]:
    """Validate whether genuine scenario-group IDs are available.

    Never invents fake groups or infers scenario families from row positions.
    Returns (groups_array_or_None, reason_message).
    """
    candidates: list[str] | None = None
    source_desc = ""

    if explicit_groups is not None:
        if len(explicit_groups) != n_samples:
            raise ValueError(
                f"Explicit scenario_groups length ({len(explicit_groups)}) does not match dataset row count ({n_samples})."
            )
        candidates = [str(g).strip() if g is not None else "" for g in explicit_groups]
        source_desc = "explicit scenario_groups argument"
    elif "scenario_group" in df.columns:
        raw_col = df["scenario_group"]
        if raw_col.isna().any():
            return (
                None,
                "Column 'scenario_group' contains missing values; falling back to stratified row-level evaluation.",
            )
        candidates = [str(g).strip() for g in raw_col.tolist()]
        source_desc = "CSV 'scenario_group' column"
    elif log_path is not None:
        log_groups = extract_scenario_groups_from_logs(log_path=log_path)
        if len(log_groups) == n_samples:
            candidates = [str(g).strip() for g in log_groups]
            source_desc = f"runtime log file ({log_path})"
        else:
            return (
                None,
                f"Runtime log group count ({len(log_groups)}) does not match dataset row count ({n_samples}); "
                "falling back to stratified row-level evaluation (NOT scenario-group validated).",
            )
    else:
        log_groups = extract_scenario_groups_from_logs(log_path=DEFAULT_LOG_FILE)
        if len(log_groups) == n_samples:
            candidates = [str(g).strip() for g in log_groups]
            source_desc = f"default runtime log file ({DEFAULT_LOG_FILE})"
        else:
            return (
                None,
                "data/behavioral_dataset.csv contains no session_id or scenario_group column and "
                "logs/runtime_events.jsonl is unavailable; evaluation uses stratified row-level splitting "
                "(StratifiedShuffleSplit and StratifiedKFold) and is NOT scenario-group validated.",
            )

    if not candidates or any(not g for g in candidates):
        return (
            None,
            "Scenario group identifiers contain empty values; falling back to stratified row-level evaluation.",
        )

    groups_arr = np.array(candidates, dtype=object)
    unique_groups = set(groups_arr)
    if len(unique_groups) < 2:
        return (
            None,
            "At least 2 distinct scenario groups are required for group-aware splitting; "
            "falling back to stratified row-level evaluation.",
        )

    # Ensure each class has at least 2 groups so group-aware splits can include both classes
    for cls_val in (0, 1):
        cls_groups = set(groups_arr[y == cls_val])
        if len(cls_groups) < 2:
            return (
                None,
                f"Class {cls_val} has fewer than 2 distinct scenario groups ({len(cls_groups)}); "
                "falling back to stratified row-level evaluation.",
            )

    return groups_arr, f"Verified scenario-group evaluation enabled via {source_desc}."


def evaluate_and_train_v2_model(
    csv_path: Path | str | None = None,
    log_path: Path | str | None = None,
    model_output_path: Path | str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 200,
    scenario_groups: Sequence[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Train and evaluate the 9-feature v2 Random Forest classifier.

    Uses StratifiedShuffleSplit (holdout) and 5-fold StratifiedKFold (CV) when
    genuine scenario-group IDs are unavailable, or GroupShuffleSplit and
    StratifiedGroupKFold when verified scenario-group IDs are explicitly provided.
    """
    out_model_path = Path(model_output_path) if model_output_path is not None else DEFAULT_MODEL_V2_PATH

    # 1. Load dataset & extract 9 behavioral features
    dataset_path = Path(csv_path) if csv_path is not None else PROJECT_ROOT / "data" / "behavioral_dataset.csv"
    df = pd.read_csv(dataset_path)
    X = df[BEHAVIORAL_9_FEATURES].to_numpy(dtype=np.float64)
    y = df["label"].map(LABEL_MAPPING).to_numpy(dtype=np.int64)

    # 2. Check whether genuine scenario groups exist (never fabricate groups)
    valid_groups, limitation_or_reason = _resolve_valid_scenario_groups(
        df=df,
        y=y,
        n_samples=len(X),
        explicit_groups=scenario_groups,
        log_path=log_path,
    )

    scenario_group_evaluation_available = valid_groups is not None

    # 3. Perform Holdout Split
    if scenario_group_evaluation_available and valid_groups is not None:
        evaluation_method = "group_aware_stratified"
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, valid_groups))
        train_groups: list[str] = sorted(list(set(valid_groups[train_idx])))
        test_groups: list[str] = sorted(list(set(valid_groups[test_idx])))
        group_overlap: list[str] | None = sorted(list(set(train_groups).intersection(set(test_groups))))
        group_overlap_count: int | None = len(group_overlap)
    else:
        evaluation_method = "stratified_row_level"
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(sss.split(X, y))
        train_groups = []
        test_groups = []
        group_overlap = None
        group_overlap_count = None

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 4. Train Random Forest v2
    rf_v2 = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    rf_v2.fit(X_train, y_train)

    # 5. Evaluate on held-out test set
    y_pred = rf_v2.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=["NORMAL", "SUSPICIOUS"],
        output_dict=True,
        zero_division=0,
    )

    test_metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "class_wise_metrics": {
            "NORMAL": {
                "precision": round(float(clf_report["NORMAL"]["precision"]), 4),
                "recall": round(float(clf_report["NORMAL"]["recall"]), 4),
                "f1-score": round(float(clf_report["NORMAL"]["f1-score"]), 4),
                "support": int(clf_report["NORMAL"]["support"]),
            },
            "SUSPICIOUS": {
                "precision": round(float(clf_report["SUSPICIOUS"]["precision"]), 4),
                "recall": round(float(clf_report["SUSPICIOUS"]["recall"]), 4),
                "f1-score": round(float(clf_report["SUSPICIOUS"]["f1-score"]), 4),
                "support": int(clf_report["SUSPICIOUS"]["support"]),
            },
        },
    }

    # 6. Feature Importances (9 behavioral features)
    importances = rf_v2.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importances = [
        {
            "feature": BEHAVIORAL_9_FEATURES[i],
            "importance": round(float(importances[i]), 4),
        }
        for i in sorted_idx
    ]

    # 7. 5-Fold Cross-Validation
    cv_accs, cv_precs, cv_recs, cv_f1s = [], [], [], []
    if scenario_group_evaluation_available and valid_groups is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        split_iter = cv_splitter.split(X, y, valid_groups)
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        split_iter = cv_splitter.split(X, y)

    for _, (tr_i, val_i) in enumerate(split_iter):
        X_tr, X_val = X[tr_i], X[val_i]
        y_tr, y_val = y[tr_i], y[val_i]

        fold_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        fold_clf.fit(X_tr, y_tr)
        preds = fold_clf.predict(X_val)

        cv_accs.append(round(float(accuracy_score(y_val, preds)), 4))
        cv_precs.append(round(float(precision_score(y_val, preds, zero_division=0)), 4))
        cv_recs.append(round(float(recall_score(y_val, preds, zero_division=0)), 4))
        cv_f1s.append(round(float(f1_score(y_val, preds, zero_division=0)), 4))

    cv_metrics = {
        "cv_method": "StratifiedGroupKFold" if scenario_group_evaluation_available else "StratifiedKFold",
        "n_splits": 5,
        "fold_accuracies": cv_accs,
        "mean_accuracy": round(float(np.mean(cv_accs)), 4),
        "std_accuracy": round(float(np.std(cv_accs)), 4),
        "mean_precision": round(float(np.mean(cv_precs)), 4),
        "mean_recall": round(float(np.mean(cv_recs)), 4),
        "mean_f1": round(float(np.mean(cv_f1s)), 4),
    }

    groups_info = {
        "scenario_group_evaluation_available": scenario_group_evaluation_available,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "group_overlap": group_overlap,
        "group_overlap_count": group_overlap_count,
        "reason": limitation_or_reason,
    }

    # 8. Save Model Artifact v2
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_v2 = {
        "model": rf_v2,
        "feature_names": list(BEHAVIORAL_9_FEATURES),
        "label_mapping": dict(LABEL_MAPPING),
        "evaluation_method": evaluation_method,
        "scenario_group_evaluation_available": scenario_group_evaluation_available,
        "evaluation_limitation": limitation_or_reason,
        "limitation_reason": limitation_or_reason,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_groups": train_groups,
        "test_groups": test_groups,
        "group_overlap_count": group_overlap_count,
        "n_estimators": n_estimators,
        "random_state": random_state,
    }

    joblib.dump(artifact_v2, out_model_path)

    return {
        "model_path": str(out_model_path),
        "model": rf_v2,
        "evaluation_method": evaluation_method,
        "scenario_group_evaluation_available": scenario_group_evaluation_available,
        "evaluation_limitation": limitation_or_reason,
        "limitation_reason": limitation_or_reason,
        "sample_counts": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "groups": groups_info,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "feature_importances": feature_importances,
    }


def load_random_forest_model_v2(
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load saved Random Forest v2 model artifact."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_V2_PATH

    if not path.exists():
        raise FileNotFoundError(f"Random Forest v2 model artifact not found at: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Loaded file is not a valid Random Forest v2 model artifact")

    return artifact


def _load_verified_baseline_metrics(baseline_path: Path | None = None) -> dict[str, Any] | None:
    """Load verified baseline v1 metrics from artifact if available on disk."""
    path = baseline_path if baseline_path is not None else DEFAULT_BASELINE_MODEL_PATH
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
        if isinstance(artifact, dict) and "metrics" in artifact:
            return {
                "metrics": artifact["metrics"],
                "feature_count": len(artifact.get("feature_names", [])),
            }
    except Exception:
        return None
    return None


def run_v2_evaluation_report() -> None:
    """Execute v2 model training/evaluation and display methodological report."""
    res = evaluate_and_train_v2_model()

    sc = res["sample_counts"]
    grp = res["groups"]
    tm = res["test_metrics"]
    cv = res["cv_metrics"]
    is_grouped = res["scenario_group_evaluation_available"]

    print("==========================================================================")
    print("   AGENTPULSE ML EVALUATION METHODOLOGY REPORT (v2 9-FEATURE MODEL)       ")
    print("==========================================================================")
    print(f"Dataset Size            : {sc['total_samples']} Total Sessions")
    print(f"Features (9 Behavioral) : {BEHAVIORAL_9_FEATURES}")
    print(f"Evaluation Method       : {res['evaluation_method']}")
    print(f"Scenario-Group Validated: {is_grouped}")
    print(f"Methodological Note     : {res['evaluation_limitation']}")
    print("--------------------------------------------------------------------------")
    if is_grouped:
        print("1. GROUP-AWARE TRAIN/TEST SPLIT (GroupShuffleSplit):")
        print(f"   - Train / Test Samples     : {sc['train_samples']} Train / {sc['test_samples']} Test")
        print(f"   - Training Scenario Groups : {grp['train_groups']}")
        print(f"   - Testing Scenario Groups  : {grp['test_groups']}")
        print(f"   - Group Overlap Count      : {grp['group_overlap_count']}")
        print("--------------------------------------------------------------------------")
        print("2. HOLDOUT TEST SET METRICS (Group-Separated Split):")
    else:
        print("1. STRATIFIED ROW-LEVEL TRAIN/TEST SPLIT (StratifiedShuffleSplit):")
        print(f"   - Train / Test Samples     : {sc['train_samples']} Train / {sc['test_samples']} Test")
        print("   - Validation Scope         : Stratified row-level evaluation (NOT scenario-group validated)")
        print("--------------------------------------------------------------------------")
        print("2. HOLDOUT TEST SET METRICS (Stratified Row-Level Split):")

    print(f"   - Accuracy  : {tm['accuracy']:.4f}")
    print(f"   - Precision : {tm['precision']:.4f}")
    print(f"   - Recall    : {tm['recall']:.4f}")
    print(f"   - F1-Score  : {tm['f1_score']:.4f}")
    print(f"   - Confusion Matrix:\n     {np.array(tm['confusion_matrix'])}")
    print("--------------------------------------------------------------------------")
    print("3. CLASS-WISE METRICS:")
    print(
        f"   - NORMAL     : Prec={tm['class_wise_metrics']['NORMAL']['precision']:.2f}, "
        f"Rec={tm['class_wise_metrics']['NORMAL']['recall']:.2f}, "
        f"F1={tm['class_wise_metrics']['NORMAL']['f1-score']:.2f} "
        f"(Support={tm['class_wise_metrics']['NORMAL']['support']})"
    )
    print(
        f"   - SUSPICIOUS : Prec={tm['class_wise_metrics']['SUSPICIOUS']['precision']:.2f}, "
        f"Rec={tm['class_wise_metrics']['SUSPICIOUS']['recall']:.2f}, "
        f"F1={tm['class_wise_metrics']['SUSPICIOUS']['f1-score']:.2f} "
        f"(Support={tm['class_wise_metrics']['SUSPICIOUS']['support']})"
    )
    print("--------------------------------------------------------------------------")
    print("4. FEATURE IMPORTANCE RANKING (9 Pure Behavioral Features):")
    for item in res["feature_importances"]:
        print(f"   - {item['feature']:<25}: {item['importance']:.4f}")
    print("--------------------------------------------------------------------------")
    if is_grouped:
        print("5. 5-FOLD STRATIFIED GROUP CROSS-VALIDATION (StratifiedGroupKFold):")
    else:
        print("5. 5-FOLD STRATIFIED CROSS-VALIDATION (StratifiedKFold, Row-Level):")
    print(f"   - Fold Accuracies    : {cv['fold_accuracies']}")
    print(f"   - Mean CV Accuracy   : {cv['mean_accuracy']:.4f} (+/- {cv['std_accuracy']:.4f})")
    print(f"   - Mean CV Precision  : {cv['mean_precision']:.4f}")
    print(f"   - Mean CV Recall     : {cv['mean_recall']:.4f}")
    print(f"   - Mean CV F1-Score   : {cv['mean_f1']:.4f}")

    baseline = _load_verified_baseline_metrics()
    if baseline is not None:
        bm = baseline["metrics"]
        print("--------------------------------------------------------------------------")
        print("6. VERIFIED ARTIFACT COMPARISON (v1 11-Feature vs v2 9-Feature Holdout):")
        print(f"   - v1 Artifact ({baseline['feature_count']} features) Holdout Accuracy : {float(bm.get('accuracy', 0.0)):.4f}")
        print(f"   - v2 Artifact (9 behavioral features) Holdout Accuracy: {tm['accuracy']:.4f}")
    print("==========================================================================")


if __name__ == "__main__":
    run_v2_evaluation_report()

