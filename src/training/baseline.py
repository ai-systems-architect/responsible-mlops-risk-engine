"""
baseline.py — Logistic Regression Baseline
--------------------------------------------
First stage in the three-model progression: Logistic Regression → Ridge → XGBoost.

Purpose:
    Establishes a fully interpretable performance baseline.
    Each subsequent model is only justified if it demonstrates
    meaningful improvement over this one — documented in
    docs/decision_log.md with the specific metric delta.

    If this model performs well enough for the use case,
    the simpler model is preferred. Complexity is earned, not assumed.

Why Logistic Regression first:
    - Coefficients are directly interpretable by non-technical reviewers
    - No hyperparameter tuning required — clean reference point
    - Confirms the full pipeline works end to end before adding complexity
    - Standard practice in regulated and federal environments

person_weight is dropped here — it is a Census survey sampling artifact,
not a personal characteristic. It is passed as sample_weight in
XGBoost training (train.py) instead.

MLflow logging is added after all three models are trained and
metrics are reviewed manually. No value in logging a single run
before a comparison exists.
"""

import pandas as pd
import logging
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config import (
    RANDOM_STATE,
    PROCESSED_DATA_DIR,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Dropped before training — Census sampling artifact, not a personal feature
# Passed as sample_weight in XGBoost training (train.py)
WEIGHT_COL = "person_weight"


def load_processed(data_dir: str = PROCESSED_DATA_DIR) -> tuple:
    """
    Load the most recently saved train/test splits from data/processed/.

    Returns:
        X_train, X_test, y_train, y_test
    """
    def latest(pattern):
        files = sorted(glob(f"{data_dir}/{pattern}"))
        if not files:
            raise FileNotFoundError(
                f"No files matching {pattern} in {data_dir}. "
                "Run src/data/preprocess.py first."
            )
        return files[-1]

    X_train = pd.read_parquet(latest("X_train_*.parquet"))
    X_test = pd.read_parquet(latest("X_test_*.parquet"))
    y_train = pd.read_parquet(latest("y_train_*.parquet")).squeeze()
    y_test = pd.read_parquet(latest("y_test_*.parquet")).squeeze()

    logger.info(f"Loaded X_train: {X_train.shape} | X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def prepare_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Drop person_weight before training.

    person_weight is retained through preprocessing for use
    by XGBoost as sample_weight. It is not a meaningful input
    for linear models.
    """
    X_train = X_train.drop(columns=[WEIGHT_COL], errors="ignore")
    X_test = X_test.drop(columns=[WEIGHT_COL], errors="ignore")

    logger.info(f"Training features ({len(X_train.columns)}): {list(X_train.columns)}")

    return X_train, X_test


def compute_metrics(y_true, y_pred_proba) -> dict:
    """
    Compute standard binary classification metrics.

    AUC-ROC is the primary metric — threshold-independent and
    appropriate when class imbalance is present.
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return {
        "auc_roc":   round(roc_auc_score(y_true, y_pred_proba), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }


def run_baseline(data_dir: str = PROCESSED_DATA_DIR) -> dict:
    """
    Train Logistic Regression baseline and print results.

    Steps:
        1. Load processed train/test splits
        2. Drop person_weight
        3. Train LogisticRegression
        4. Evaluate on held-out test set
        5. Print metrics and coefficient summary

    MLflow logging is deferred until all three models are trained
    and metrics are reviewed side by side.

    Returns:
        metrics dict — compared against Ridge and XGBoost results
    """
    logger.info("=" * 55)
    logger.info("Baseline — Logistic Regression")
    logger.info("=" * 55)

    # Load data
    X_train, X_test, y_train, y_test = load_processed(data_dir)

    # Drop sampling weight
    X_train, X_test = prepare_features(X_train, X_test)

    # Train
    # max_iter=1000 ensures convergence on this dataset size
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    logger.info("Model trained")

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred_proba)

    # Coefficient summary
    # Positive = feature increases probability of high income
    # Negative = feature decreases probability of high income
    coef_df = pd.DataFrame({
        "feature":     X_train.columns,
        "coefficient": model.coef_[0].round(4),
    }).sort_values("coefficient", ascending=False)

    print("\n--- Logistic Regression — Coefficient Summary ---")
    print(coef_df.to_string(index=False))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    logger.info("=" * 55)
    logger.info("Baseline complete")
    logger.info(f"AUC-ROC: {metrics['auc_roc']} | F1: {metrics['f1']}")
    logger.info("=" * 55)

    return metrics


if __name__ == "__main__":
    metrics = run_baseline()
