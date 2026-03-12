"""
ridge.py — Ridge Logistic Regression (L2 Regularization)
----------------------------------------------------------
Second stage in the three-model progression: Logistic Regression → Ridge → XGBoost.

Purpose:
    Determines whether L2 regularization improves on the baseline.
    Ridge penalizes large coefficients, which reduces overfitting when
    features are correlated or coefficients are inflated.

    Results are compared directly against baseline.py metrics.
    Improvement is documented in docs/decision_log.md with the
    specific delta. If improvement is marginal, that finding is
    documented and XGBoost is evaluated as the next step.

Why Ridge over plain Logistic Regression:
    - Education, occupation, and age are likely correlated
    - L2 penalty shrinks correlated coefficients toward each other
      rather than arbitrarily amplifying one and suppressing another
    - Produces more stable, generalizable coefficients
    - Still fully interpretable — coefficients remain meaningful

Regularization strength C:
    - C controls the inverse of regularization strength
    - Low C = strong regularization (coefficients shrunk aggressively)
    - High C = weak regularization (approaches plain logistic regression)
    - Optimal C is found via cross-validation on the training set

person_weight is dropped here for the same reason as baseline.py —
it is a Census sampling artifact used only in XGBoost as sample_weight.
"""

import pandas as pd
import logging
from glob import glob
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config import (
    TARGET,
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
WEIGHT_COL = "person_weight"

# Candidate regularization strengths evaluated during cross-validation
# Range spans weak (100) to strong (0.001) regularization
# CV selects the value that maximizes AUC on held-out folds
C_VALUES = [100, 10, 1.0, 0.1, 0.01, 0.001]


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

    person_weight is retained through preprocessing for use by
    XGBoost as sample_weight. It is not a meaningful input for
    linear models.
    """
    X_train = X_train.drop(columns=[WEIGHT_COL], errors="ignore")
    X_test = X_test.drop(columns=[WEIGHT_COL], errors="ignore")

    logger.info(f"Training features ({len(X_train.columns)}): {list(X_train.columns)}")

    return X_train, X_test


def compute_metrics(y_true, y_pred_proba) -> dict:
    """
    Compute standard binary classification metrics.

    AUC-ROC is the primary metric for comparison against baseline.py.
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return {
        "auc_roc":   round(roc_auc_score(y_true, y_pred_proba), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }


def run_ridge(data_dir: str = PROCESSED_DATA_DIR) -> dict:
    """
    Train Ridge Logistic Regression and compare results against baseline.

    Uses LogisticRegressionCV — fits one model per C value using
    5-fold stratified cross-validation on the training set, then
    retrains on the full training set using the best C found.

    Steps:
        1. Load processed train/test splits
        2. Drop person_weight
        3. Train LogisticRegressionCV with L2 penalty
        4. Report best regularization strength selected by CV
        5. Evaluate on held-out test set
        6. Print coefficient comparison against baseline

    Returns:
        metrics dict — compared against baseline and XGBoost results
    """
    logger.info("=" * 55)
    logger.info("Ridge — Logistic Regression with L2 Regularization")
    logger.info("=" * 55)

    # Load data
    X_train, X_test, y_train, y_test = load_processed(data_dir)

    # Drop sampling weight
    X_train, X_test = prepare_features(X_train, X_test)

    # Train with cross-validated regularization strength
    # cv=5 — 5-fold stratified cross-validation on training set
    # scoring="roc_auc" — selects C that maximizes AUC, consistent
    #   with the primary metric used to compare all three models
    # penalty="l2" — Ridge regularization
    model = LogisticRegressionCV(
        Cs=C_VALUES,
        penalty="l2",
        cv=5,
        scoring="roc_auc",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    best_C = model.C_[0]
    logger.info(f"Best regularization strength (C): {best_C}")
    logger.info(
        f"Interpretation: "
        f"{'Strong regularization applied' if best_C < 0.1 else 'Weak regularization — similar to baseline'}"
    )

    # Evaluate on held-out test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred_proba)

    # Coefficient summary
    coef_df = pd.DataFrame({
        "feature":     X_train.columns,
        "coefficient": model.coef_[0].round(4),
    }).sort_values("coefficient", ascending=False)

    print("\n--- Ridge — Coefficient Summary ---")
    print(coef_df.to_string(index=False))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n  Best C (regularization strength): {best_C}")
    print(
        "  Note: Compare AUC-ROC and F1 against baseline results.\n"
        "  Document delta and justification in docs/decision_log.md."
    )

    logger.info("=" * 55)
    logger.info("Ridge complete")
    logger.info(f"AUC-ROC: {metrics['auc_roc']} | F1: {metrics['f1']} | Best C: {best_C}")
    logger.info("=" * 55)

    return metrics


if __name__ == "__main__":
    metrics = run_ridge()