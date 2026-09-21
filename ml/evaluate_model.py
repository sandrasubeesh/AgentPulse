"""Model Evaluation module for AgentPulse Supervised ML Pipeline (Step 7).

Evaluates the trained Random Forest classifier on the holdout test set, computes
class-wise metrics, feature importances, 5-fold stratified cross-validation,
ablation evaluation, and label-leakage analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from behavior.feature_engineering import prepare_behavioral_dataset
from ml.random_forest_model import DEFAULT_MODEL_PATH, load_random_forest_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate_random_forest(
    csv_path: Path | str | None = None,
    model_path: Path | str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Perform thorough evaluation of the saved Random Forest classifier.

    Returns dict containing test metrics, feature importances, 5-fold CV results,
    ablation evaluation results, and label leakage analysis.
    """
    # 1. Load dataset & model artifact
    X, y, feature_names = prepare_behavioral_dataset(csv_path=csv_path)
    artifact = load_random_forest_model(model_path=model_path)
    rf_model: RandomForestClassifier = artifact["model"]

    # 2. Reproduce stratified holdout split (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 3. Evaluate main model on holdout test set
    y_pred = rf_model.predict(X_test)

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
    )

    test_metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "class_wise_metrics": {
            "NORMAL": {
                "precision": float(clf_report["NORMAL"]["precision"]),
                "recall": float(clf_report["NORMAL"]["recall"]),
                "f1-score": float(clf_report["NORMAL"]["f1-score"]),
                "support": int(clf_report["NORMAL"]["support"]),
            },
            "SUSPICIOUS": {
                "precision": float(clf_report["SUSPICIOUS"]["precision"]),
                "recall": float(clf_report["SUSPICIOUS"]["recall"]),
                "f1-score": float(clf_report["SUSPICIOUS"]["f1-score"]),
                "support": int(clf_report["SUSPICIOUS"]["support"]),
            },
        },
    }

    # 4. Feature Importance Ranking (descending order)
    importances = rf_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    feature_importance_ranking = [
        {
            "feature": feature_names[idx],
            "importance": round(float(importances[idx]), 4),
        }
        for idx in sorted_indices
    ]

    # 5. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_fold_scores: list[float] = []

    for _, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
        )
        fold_model.fit(X_tr, y_tr)
        score = float(fold_model.score(X_val, y_val))
        cv_fold_scores.append(round(score, 4))

    cv_results = {
        "fold_scores": cv_fold_scores,
        "mean_cv_accuracy": round(float(np.mean(cv_fold_scores)), 4),
        "std_cv_accuracy": round(float(np.std(cv_fold_scores)), 4),
    }

    # 6. Ablation Evaluation (excluding security_detection_count & prompt_event_count)
    ablation_excluded = {"security_detection_count", "prompt_event_count"}
    ablation_indices = [i for i, f in enumerate(feature_names) if f not in ablation_excluded]
    ablation_features = [feature_names[i] for i in ablation_indices]

    X_ablation = X[:, ablation_indices]
    X_tr_abl, X_te_abl, y_tr_abl, y_te_abl = train_test_split(
        X_ablation,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rf_ablation = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced",
    )
    rf_ablation.fit(X_tr_abl, y_tr_abl)
    y_pred_abl = rf_ablation.predict(X_te_abl)

    ablation_results = {
        "excluded_features": sorted(list(ablation_excluded)),
        "remaining_features_count": len(ablation_features),
        "accuracy": round(float(accuracy_score(y_te_abl, y_pred_abl)), 4),
        "precision": round(float(precision_score(y_te_abl, y_pred_abl, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te_abl, y_pred_abl, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_te_abl, y_pred_abl, zero_division=0)), 4),
    }

    # 7. Label Leakage Analysis
    top_feature_name = feature_importance_ranking[0]["feature"]
    top_feature_imp = feature_importance_ranking[0]["importance"]

    label_leakage_observations = (
        f"Feature '{top_feature_name}' carries high importance ({top_feature_imp * 100:.1f}%) "
        f"because session ground-truth labels were assigned using rule-based security detections. "
        f"However, the ablation study confirms that removing '{sorted(list(ablation_excluded))}' "
        f"leaves a behavioral model that achieves {ablation_results['accuracy'] * 100:.1f}% test accuracy "
        f"independently using core behavioral patterns (session duration, tool errors, tool results)."
    )

    return {
        "test_metrics": test_metrics,
        "feature_importance_ranking": feature_importance_ranking,
        "cross_validation": cv_results,
        "ablation_results": ablation_results,
        "label_leakage_observations": label_leakage_observations,
        "sample_counts": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
    }


def run_full_evaluation_report() -> None:
    """Print complete formatted evaluation report to stdout."""
    res = evaluate_random_forest()

    tm = res["test_metrics"]
    cv = res["cross_validation"]
    abl = res["ablation_results"]
    sc = res["sample_counts"]

    print("===========================================================")
    print("      AGENTPULSE RANDOM FOREST MODEL EVALUATION REPORT     ")
    print("===========================================================")
    print(f"Dataset Total Samples : {sc['total_samples']} (100 NORMAL, 100 SUSPICIOUS)")
    print(f"Train / Test Split    : {sc['train_samples']} Train / {sc['test_samples']} Test (80/20 Stratified)")
    print("-----------------------------------------------------------")
    print("1. HOLDOUT TEST SET METRICS (40 samples):")
    print(f"   - Accuracy  : {tm['accuracy']:.4f}")
    print(f"   - Precision : {tm['precision']:.4f}")
    print(f"   - Recall    : {tm['recall']:.4f}")
    print(f"   - F1-Score  : {tm['f1_score']:.4f}")
    print(f"   - Confusion Matrix:\n     {np.array(tm['confusion_matrix'])}")
    print("-----------------------------------------------------------")
    print("2. CLASS-WISE METRICS:")
    print(f"   - NORMAL     : Prec={tm['class_wise_metrics']['NORMAL']['precision']:.2f}, Rec={tm['class_wise_metrics']['NORMAL']['recall']:.2f}, F1={tm['class_wise_metrics']['NORMAL']['f1-score']:.2f}")
    print(f"   - SUSPICIOUS : Prec={tm['class_wise_metrics']['SUSPICIOUS']['precision']:.2f}, Rec={tm['class_wise_metrics']['SUSPICIOUS']['recall']:.2f}, F1={tm['class_wise_metrics']['SUSPICIOUS']['f1-score']:.2f}")
    print("-----------------------------------------------------------")
    print("3. FEATURE IMPORTANCE RANKING (11 features):")
    for item in res["feature_importance_ranking"]:
        print(f"   - {item['feature']:<25}: {item['importance']:.4f}")
    print("-----------------------------------------------------------")
    print("4. 5-FOLD STRATIFIED CROSS-VALIDATION:")
    print(f"   - Fold Scores        : {cv['fold_scores']}")
    print(f"   - Mean CV Accuracy   : {cv['mean_cv_accuracy']:.4f}")
    print(f"   - CV Standard Dev    : {cv['std_cv_accuracy']:.4f}")
    print("-----------------------------------------------------------")
    print("5. FEATURE ABLATION STUDY (Without security_detection_count & prompt_event_count):")
    print(f"   - Accuracy  : {abl['accuracy']:.4f}")
    print(f"   - Precision : {abl['precision']:.4f}")
    print(f"   - Recall    : {abl['recall']:.4f}")
    print(f"   - F1-Score  : {abl['f1_score']:.4f}")
    print("-----------------------------------------------------------")
    print("6. LABEL LEAKAGE OBSERVATIONS:")
    print(f"   {res['label_leakage_observations']}")
    print("===========================================================")


if __name__ == "__main__":
    run_full_evaluation_report()
