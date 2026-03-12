"""
xgboost.py — XGBoost with Optuna Hyperparameter Tuning
--------------------------------------------------------
Third stage in the three-model progression: Logistic Regression → Ridge → XGBoost.

Motivation for this stage:
    Logistic Regression and Ridge both reached AUC 0.9108 — the linear
    model ceiling. Near-zero coefficients on occupation (-0.005) and
    class_of_worker (-0.051) in the baseline suggest these features carry
    non-linear signal that linear models cannot capture. XGBoost finds
    feature interactions and threshold effects — for example, occupation
    combined with education level, or hours_per_week above a certain
    threshold interacting with class_of_worker.

    Improvement over AUC 0.9108 and F1 0.6508 is required to justify
    the added complexity. Results documented in docs/decision_log.md DL-007.

person_weight handling:
    Passed as sample_weight to XGBoost — not used as a model input feature.
    Census sampling weights make the model representative of the full U.S.
    population rather than just the survey sample. This is the statistically
    correct use of PWGTP as documented in docs/decision_log.md DL-004.

Optuna:
    Finds the best combination of XGBoost hyperparameters by running
    multiple trials, each evaluated via 5-fold cross-validated AUC on
    the training set. The best parameters are then used to train a final
    model evaluated on the held-out test set.

    Trial count is controlled by config.py:
        OPTUNA_TRIALS_LOCAL = 30  — thorough local runs
        OPTUNA_TRIALS_CI    = 5   — fast CI/CD runs
"""

import pandas as pd
import numpy as np
import logging
import optuna
from glob import glob
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config import (
    RANDOM_STATE,
    OPTUNA_TRIALS_LOCAL,
    PROCESSED_DATA_DIR,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Optuna logs one line per trial by default — reduced to WARNING to keep
# output readable. Final best parameters are logged explicitly below.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Extracted from X_train before training — used as sample_weight, not a feature
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


def prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Extract person_weight from X_train and X_test before training.

    person_weight is passed as sample_weight to XGBoost — not used as
    a model input. pop() removes the column and returns it as a series
    in one operation.

    Returns:
        X_train, X_test with person_weight removed
        train_weights, test_weights as separate series
    """
    train_weights = X_train.pop(WEIGHT_COL)
    test_weights = X_test.pop(WEIGHT_COL)

    logger.info(f"Training features ({len(X_train.columns)}): {list(X_train.columns)}")
    logger.info("person_weight extracted — passed as sample_weight to XGBoost")

    return X_train, X_test, train_weights, test_weights


def compute_metrics(y_true, y_pred_proba) -> dict:
    """
    Compute standard binary classification metrics.

    AUC-ROC is the primary metric for comparison against baseline
    and Ridge results.
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return {
        "auc_roc":   round(roc_auc_score(y_true, y_pred_proba), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }


def build_objective(X_train, y_train, train_weights):
    """
    Build the Optuna objective function for XGBoost hyperparameter tuning.

    Each trial samples a different combination of hyperparameters and
    evaluates them using 5-fold stratified cross-validation on the
    training set. Optuna minimizes the negative AUC (equivalent to
    maximizing AUC).

    Hyperparameters tuned:
        n_estimators    — number of boosting rounds
        max_depth       — maximum depth of each tree
        learning_rate   — step size shrinkage to prevent overfitting
        subsample       — fraction of training samples per tree
        colsample_bytree — fraction of features per tree
        min_child_weight — minimum sum of weights in a leaf node
        gamma           — minimum loss reduction to make a split
        scale_pos_weight — compensates for class imbalance
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            # scale_pos_weight corrects for class imbalance
            # ratio of negative to positive class count
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "random_state": RANDOM_STATE,
            "eval_metric": "auc",
        }

        model = XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        # Manual CV loop — cross_val_score does not support sample_weight
        # via fit_params in this version of scikit-learn
        scores = []
        X_arr = X_train.values
        y_arr = y_train.values
        w_arr = train_weights.values

        for train_idx, val_idx in cv.split(X_arr, y_arr):
            model.fit(
                X_arr[train_idx], y_arr[train_idx],
                sample_weight=w_arr[train_idx],
            )
            val_proba = model.predict_proba(X_arr[val_idx])[:, 1]
            scores.append(roc_auc_score(y_arr[val_idx], val_proba))

        return np.mean(scores)

    return objective


def run_xgboost(
    data_dir: str = PROCESSED_DATA_DIR,
    n_trials: int = OPTUNA_TRIALS_LOCAL,
) -> dict:
    """
    Train XGBoost with Optuna hyperparameter tuning and evaluate results.

    Steps:
        1. Load processed train/test splits
        2. Extract person_weight as sample_weight
        3. Run Optuna to find best hyperparameters via cross-validation
        4. Retrain final model on full training set with best parameters
        5. Evaluate on held-out test set
        6. Print full results for comparison against baseline and Ridge

    Args:
        data_dir: Directory containing processed parquet files
        n_trials: Number of Optuna trials — controls tuning thoroughness

    Returns:
        metrics dict — compared against baseline (0.9108) and Ridge (0.9108)
    """
    logger.info("=" * 55)
    logger.info("XGBoost — Optuna Hyperparameter Tuning")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 55)

    # Load data
    X_train, X_test, y_train, y_test = load_processed(data_dir)

    # Extract person_weight — used as sample_weight, not a feature
    X_train, X_test, train_weights, test_weights = prepare_features(X_train, X_test)

    # --- Optuna tuning ---
    logger.info(f"Running {n_trials} Optuna trials — this may take a few minutes")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        build_objective(X_train, y_train, train_weights),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_auc = round(study.best_value, 4)

    logger.info(f"Best cross-validated AUC: {best_cv_auc}")
    logger.info(f"Best parameters: {best_params}")

    # --- Final model ---
    # Retrain on full training set using best parameters found by Optuna
    # eval_metric is fixed, not tuned
    final_params = {
        **best_params,
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
    }

    model = XGBClassifier(**final_params)
    model.fit(X_train, y_train, sample_weight=train_weights)
    logger.info("Final model trained on full training set")

    # --- Evaluate on held-out test set ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred_proba)

    # --- Feature importance ---
    importance_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": model.feature_importances_.round(4),
    }).sort_values("importance", ascending=False)

    # --- Results ---
    print("\n--- XGBoost — Feature Importance ---")
    print(importance_df.to_string(index=False))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n  Best CV AUC (Optuna): {best_cv_auc}")
    print(f"  Best parameters: {best_params}")

    print("\n--- Model Progression Summary ---")
    print("  Logistic Regression — AUC: 0.9108 | F1: 0.6508")
    print("  Ridge               — AUC: 0.9108 | F1: 0.6507")
    print(f"  XGBoost             — AUC: {metrics['auc_roc']} | F1: {metrics['f1']}")

    logger.info("=" * 55)
    logger.info("XGBoost complete")
    logger.info(f"AUC-ROC: {metrics['auc_roc']} | F1: {metrics['f1']}")
    logger.info("=" * 55)

    return metrics, model, best_params


if __name__ == "__main__":
    metrics, model, best_params = run_xgboost()
