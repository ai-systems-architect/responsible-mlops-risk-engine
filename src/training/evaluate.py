"""
evaluate.py — Model Evaluation and Fairness Audit
---------------------------------------------------
Loads the trained XGBoost model and runs two evaluations:

1. Overall metrics — AUC-ROC, F1, precision, recall on the test set

2. Fairness audit — per-group metrics across demographic groups
   Race, sex, and nativity were physically separated from model
   features during preprocessing and never used as model inputs.
   They are rejoined here for post-prediction fairness analysis only.

   For each demographic group the following are computed:
       - Positive prediction rate (PPR)
       - Precision, recall, F1
       - AUC-ROC where sample size permits

   A fairness gate checks whether any group's positive prediction
   rate deviates from the overall rate by more than FAIRNESS_THRESHOLD.
   Models that fail this gate do not proceed to the deployment step.

NIST AI RMF alignment:
    MEASURE 2.5 — bias and fairness evaluated across affected groups
    MANAGE 2.2  — findings documented for risk response decisions
"""

import pandas as pd
import numpy as np
import joblib
import logging
from glob import glob
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config import (
    FAIRNESS_THRESHOLD,
    PROCESSED_DATA_DIR,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"

# Census race codes → readable labels for audit report
# Reference: https://api.census.gov/data/2023/acs/acs1/pums/variables/RAC1P.json
RACE_LABELS = {
    "1": "White alone",
    "2": "Black or African American alone",
    "3": "American Indian alone",
    "4": "Alaska Native alone",
    "6": "Asian alone",
    "7": "Native Hawaiian / Pacific Islander alone",
    "8": "Some other race alone",
    "9": "Two or more races",
}

SEX_LABELS = {
    "1": "Male",
    "2": "Female",
}

NATIVITY_LABELS = {
    "1": "Native born",
    "2": "Foreign born",
}


def load_artifacts(
    data_dir: str = PROCESSED_DATA_DIR,
    models_dir: str = MODELS_DIR,
) -> tuple:
    """
    Load the test set, sensitive features, and trained model.

    Returns:
        X_test, y_test, sensitive_df, model
    """
    def latest(pattern):
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern}")
        return files[-1]

    X_test = pd.read_parquet(latest(f"{data_dir}/X_test_*.parquet"))
    y_test = pd.read_parquet(latest(f"{data_dir}/y_test_*.parquet")).squeeze()
    sensitive_df = pd.read_parquet(latest(f"{data_dir}/sensitive_*.parquet"))
    model = joblib.load(latest(f"{models_dir}/xgboost_*.joblib"))

    logger.info(f"Loaded X_test: {X_test.shape}")
    logger.info(f"Loaded sensitive features: {sensitive_df.shape}")
    logger.info(f"Loaded model: {latest(f'{models_dir}/xgboost_*.joblib')}")

    return X_test, y_test, sensitive_df, model


def compute_overall_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """
    Compute overall binary classification metrics on the full test set.
    """
    metrics = {
        "auc_roc":   round(roc_auc_score(y_test, y_pred_proba), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }
    return metrics


def compute_group_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    group_name: str,
) -> dict:
    """
    Compute metrics for a single demographic group.

    AUC requires both classes to be present — skipped for very small
    groups where one class may be absent.

    Returns dict with PPR, precision, recall, F1, AUC, and sample size.
    """
    n = len(y_true)
    if n == 0:
        return None

    ppr = round(y_pred.mean(), 4)
    prec = round(precision_score(y_true, y_pred, zero_division=0), 4)
    rec = round(recall_score(y_true, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)

    # AUC requires both classes present in the group
    try:
        auc = round(roc_auc_score(y_true, y_pred_proba), 4)
    except ValueError:
        auc = None

    return {
        "group": group_name,
        "n": n,
        "ppr": ppr,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc,
    }


def run_fairness_audit(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_df: pd.DataFrame,
    overall_ppr: float,
) -> tuple:
    """
    Compute per-group metrics across all demographic dimensions.

    Compares each group's positive prediction rate (PPR) against the
    overall PPR. Groups deviating by more than FAIRNESS_THRESHOLD
    are flagged in the audit report.

    Args:
        y_test:        True labels aligned with X_test index
        y_pred:        Binary predictions
        y_pred_proba:  Predicted probabilities
        sensitive_df:  Demographic features aligned with X_test index
        overall_ppr:   PPR across the full test set

    Returns:
        results_df:    Per-group metrics dataframe
        passed:        True if all groups within FAIRNESS_THRESHOLD
    """
    results = []

    group_configs = [
        ("race", RACE_LABELS),
        ("sex", SEX_LABELS),
        ("nativity", NATIVITY_LABELS),
    ]

    for feature, labels in group_configs:
        if feature not in sensitive_df.columns:
            continue

        for code, label in labels.items():
            mask = sensitive_df[feature].astype(str) == code
            if mask.sum() < 30:
                # Groups smaller than 30 records excluded —
                # metrics unreliable at very small sample sizes
                continue

            group_metrics = compute_group_metrics(
                y_test[mask],
                y_pred[mask],
                y_pred_proba[mask],
                group_name=f"{feature}: {label}",
            )
            if group_metrics:
                group_metrics["ppr_delta"] = round(
                    abs(group_metrics["ppr"] - overall_ppr), 4
                )
                group_metrics["flag"] = (
                    group_metrics["ppr_delta"] > FAIRNESS_THRESHOLD
                )
                results.append(group_metrics)

    results_df = pd.DataFrame(results)
    flagged = results_df[results_df["flag"]]
    passed = len(flagged) == 0

    return results_df, passed


def run_evaluation(
    data_dir: str = PROCESSED_DATA_DIR,
    models_dir: str = MODELS_DIR,
) -> tuple:
    """
    Run full evaluation — overall metrics and fairness audit.

    Steps:
        1. Load test set, sensitive features, and trained model
        2. Generate predictions on test set
        3. Compute overall metrics
        4. Run fairness audit across demographic groups
        5. Print full report
        6. Return pass/fail status for CI/CD gate

    Returns:
        metrics:  Overall metrics dict
        passed:   True if fairness gate passed
    """
    logger.info("=" * 55)
    logger.info("Evaluation — Metrics and Fairness Audit")
    logger.info("=" * 55)

    # Load artifacts
    X_test, y_test, sensitive_df, model = load_artifacts(data_dir, models_dir)

    # Drop person_weight — not used as a model input
    X_test = X_test.drop(columns=["person_weight"], errors="ignore")

    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Align sensitive_df index with test set
    sensitive_df = sensitive_df.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # --- Overall metrics ---
    metrics = compute_overall_metrics(y_test, y_pred, y_pred_proba)
    overall_ppr = round(y_pred.mean(), 4)

    print("\n--- Overall Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  overall_ppr: {overall_ppr}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # --- Fairness audit ---
    results_df, passed = run_fairness_audit(
        y_test, y_pred, y_pred_proba,
        sensitive_df, overall_ppr,
    )

    print("\n--- Fairness Audit — Per-Group Metrics ---")
    print(f"Overall PPR: {overall_ppr} | Threshold: ±{FAIRNESS_THRESHOLD}")
    print()
    print(results_df[[
        "group", "n", "ppr", "ppr_delta", "precision",
        "recall", "f1", "auc_roc", "flag"
    ]].to_string(index=False))

    print("\n--- Fairness Gate ---")
    if passed:
        print(f"  PASSED — all groups within ±{FAIRNESS_THRESHOLD} PPR threshold")
    else:
        flagged = results_df[results_df["flag"]]
        print(f"  FAILED — {len(flagged)} group(s) exceed ±{FAIRNESS_THRESHOLD} threshold:")
        for _, row in flagged.iterrows():
            print(f"    {row['group']} — PPR: {row['ppr']} | Delta: {row['ppr_delta']}")

    logger.info("=" * 55)
    logger.info(f"Evaluation complete — fairness gate: {'PASSED' if passed else 'FAILED'}")
    logger.info(f"AUC-ROC: {metrics['auc_roc']} | F1: {metrics['f1']}")
    logger.info("=" * 55)

    return metrics, passed


if __name__ == "__main__":
    metrics, passed = run_evaluation()

    if not passed:
        # Non-zero exit code signals CI/CD pipeline to block deployment
        exit(1)
