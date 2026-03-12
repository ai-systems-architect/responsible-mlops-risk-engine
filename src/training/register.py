"""
register.py — MLflow Experiment Logging and Model Registry
-----------------------------------------------------------
Logs the trained XGBoost model to MLflow with the full artifact bundle
required for reproducible deployment:

    - All hyperparameters
    - Performance metrics — AUC, F1, precision, recall
    - Fairness audit results — per-group PPR and AUC
    - Feature importance scores
    - Model artifact with input/output signature
    - Preprocessing artifacts — encoders and scaler
    - Model registered in MLflow Model Registry as Staging

MLflow stores all artifacts locally in mlruns/ (gitignored).
The UI is available at http://localhost:5000 when mlflow ui is running.

The model signature enforces that inference inputs match training inputs
exactly — correct feature names, types, and order. SageMaker validates
against this signature on every prediction request.

NIST AI RMF alignment:
    GOVERN 1.4 — model versioning and lineage documented in registry
    MEASURE 2.5 — fairness metrics logged alongside performance metrics
    MANAGE 1.3 — staging gate enforced before production promotion
"""

import logging
import mlflow
import mlflow.xgboost
import pandas as pd
from glob import glob
from mlflow.models.signature import infer_signature

from config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
)
from src.training.evaluate import (
    load_artifacts,
    compute_overall_metrics,
    run_fairness_audit,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"

# Census race/sex/nativity labels — matches evaluate.py
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
SEX_LABELS = {"1": "Male", "2": "Female"}
NATIVITY_LABELS = {"1": "Native born", "2": "Foreign born"}


def get_best_xgboost_params(model) -> dict:
    """
    Extract hyperparameters from the trained XGBoost model object.

    Returns a flat dict suitable for mlflow.log_params().
    """
    params = model.get_params()
    keys = [
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree", "min_child_weight",
        "gamma", "scale_pos_weight",
    ]
    return {k: round(float(params[k]), 6) if isinstance(params[k], float)
            else params[k] for k in keys if k in params}


def get_feature_importance(model, feature_names: list) -> dict:
    """
    Return feature importance scores as a flat dict for MLflow logging.

    Keys are prefixed with "importance/" for clean grouping in the UI.
    """
    scores = model.feature_importances_
    return {
        f"importance/{name}": round(float(score), 4)
        for name, score in zip(feature_names, scores)
    }


def flatten_fairness_for_mlflow(results_df: pd.DataFrame) -> dict:
    """
    Flatten per-group fairness metrics into a dict for MLflow logging.

    Keys use the pattern: fairness/<group_short>/<metric>
    Group names are shortened to avoid MLflow key length limits.
    """
    metrics = {}
    for _, row in results_df.iterrows():
        # Shorten group name for MLflow key — replace spaces and colons
        short = (
            row["group"]
            .replace("race: ", "")
            .replace("sex: ", "")
            .replace("nativity: ", "")
            .replace(" ", "_")
            .replace("/", "_")
            .lower()
        )
        prefix = f"fairness/{short}"
        metrics[f"{prefix}/ppr"] = row["ppr"]
        metrics[f"{prefix}/ppr_delta"] = row["ppr_delta"]
        metrics[f"{prefix}/precision"] = row["precision"]
        metrics[f"{prefix}/recall"] = row["recall"]
        metrics[f"{prefix}/f1"] = row["f1"]
        if row["auc_roc"] is not None:
            metrics[f"{prefix}/auc_roc"] = row["auc_roc"]
        metrics[f"{prefix}/n"] = int(row["n"])
    return metrics


def register_model(
    data_dir: str = PROCESSED_DATA_DIR,
    models_dir: str = MODELS_DIR,
) -> str:
    """
    Log XGBoost model and all associated metrics to MLflow, then
    register it in the Model Registry at Staging.

    Steps:
        1. Load test set, sensitive features, and trained model
        2. Generate predictions
        3. Compute overall metrics and fairness audit
        4. Start MLflow run — log params, metrics, artifacts
        5. Log model with input/output signature
        6. Register model version in MLflow Model Registry
        7. Transition to Staging

    Returns:
        run_id — MLflow run identifier for this registration
    """
    logger.info("=" * 55)
    logger.info("MLflow — Model Registration")
    logger.info("=" * 55)

    # --- MLflow setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- Load artifacts ---
    X_test, y_test, sensitive_df, model = load_artifacts(data_dir, models_dir)

    # Locate encoder and scaler artifacts
    encoder_path = sorted(glob(f"{data_dir}/encoders_*.joblib"))[-1]
    scaler_path = sorted(glob(f"{data_dir}/scaler_*.joblib"))[-1]

    # Drop person_weight — not a model input
    X_test = X_test.drop(columns=["person_weight"], errors="ignore")

    # --- Predictions ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Align indices
    sensitive_df = sensitive_df.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # --- Metrics ---
    metrics = compute_overall_metrics(y_test, y_pred, y_pred_proba)
    overall_ppr = round(float(y_pred.mean()), 4)

    fairness_df, fairness_passed = run_fairness_audit(
        y_test, y_pred, y_pred_proba,
        sensitive_df, overall_ppr,
    )

    # --- Hyperparameters ---
    params = get_best_xgboost_params(model)

    # --- Feature importance ---
    importance_metrics = get_feature_importance(model, X_test.columns.tolist())

    # --- Fairness metrics flattened for MLflow ---
    fairness_metrics = flatten_fairness_for_mlflow(fairness_df)

    # --- Model signature ---
    # Captures input feature schema and output probability schema.
    # SageMaker enforces this at inference time — mismatched inputs are rejected.
    signature = infer_signature(X_test, model.predict_proba(X_test))

    # --- MLflow run ---
    with mlflow.start_run(run_name="xgboost-production") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # Tags — searchable metadata in MLflow UI
        mlflow.set_tags({
            "model_type": "xgboost",
            "dataset": "ACS PUMS 2023",
            "state": "Virginia (FIPS 51)",
            "fairness_gate": "PASSED" if fairness_passed else "FAILED",
            "income_threshold": "75000",
            "pipeline_stage": "production",
        })

        # Hyperparameters
        mlflow.log_params(params)
        mlflow.log_params({
            "n_optuna_trials": 30,
            "cv_folds": 5,
            "test_size": 0.20,
            "random_state": 42,
            "income_threshold": 75000,
            "state_code": "51",
        })

        # Overall performance metrics
        mlflow.log_metrics({
            "auc_roc": metrics["auc_roc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "overall_ppr": overall_ppr,
            "fairness_gate_passed": int(fairness_passed),
        })

        # Feature importance
        mlflow.log_metrics(importance_metrics)

        # Fairness metrics — full per-group breakdown
        mlflow.log_metrics(fairness_metrics)

        # Model artifact with signature
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5],
        )

        # Preprocessing artifacts — required for inference reproducibility
        mlflow.log_artifact(encoder_path, artifact_path="preprocessing")
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        # Fairness report as artifact — keeps everything in one run
        mlflow.log_artifact(
            "docs/fairness_report.md",
            artifact_path="docs",
        )

        logger.info("All metrics, params, and artifacts logged")

    # --- Register in Model Registry ---
    model_name = "income-risk-xgboost"
    model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    version = registered.version

    # Set alias to "staging" — Production promotion requires manual approval
    # transition_model_version_stage is deprecated in MLflow 2.x
    # Aliases are the current recommended approach for version lifecycle management
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=version,
    )

    logger.info("=" * 55)
    logger.info(f"Model registered: {model_name} v{version} → alias: staging")
    logger.info(f"AUC-ROC: {metrics['auc_roc']} | F1: {metrics['f1']}")
    logger.info(f"Fairness gate: {'PASSED' if fairness_passed else 'FAILED'}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 55)

    print("\n--- MLflow Registration Complete ---")
    print(f"  Experiment:  {MLFLOW_EXPERIMENT_NAME}")
    print(f"  Run ID:      {run_id}")
    print(f"  Model:       {model_name} v{version}")
    print("  Alias:       staging")
    print(f"  AUC-ROC:     {metrics['auc_roc']}")
    print(f"  F1:          {metrics['f1']}")
    print(f"  Fairness:    {'PASSED' if fairness_passed else 'FAILED'}")
    print(f"\n  View at:     {MLFLOW_TRACKING_URI}")

    return run_id


if __name__ == "__main__":
    run_id = register_model()
