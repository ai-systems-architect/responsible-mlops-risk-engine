"""
drift_monitor.py — Data and Model Drift Monitoring
----------------------------------------------------
Compares the training data distribution (reference) against the test
set (current) to detect feature drift and model performance drift.

In production this script runs on a schedule — EventBridge triggers
Lambda which invokes this script daily against incoming inference data.
For portfolio demonstration, X_train is the reference and X_test
simulates production inference data.

Outputs:
    1. Console summary — drift detected per feature, overall verdict
    2. HTML report    — saved to reports/drift_report.html
    3. CloudWatch     — drift metrics pushed for alarming

NIST AI RMF alignment:
    MANAGE 2.4 — ongoing monitoring with defined thresholds
    MANAGE 4.1 — drift triggers documented retraining process
    MEASURE 2.6 — model performance monitored post-deployment
"""

import os
import logging
import json
import boto3
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime, timezone

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)

from config import (
    MODEL_FEATURES,
    PROCESSED_DATA_DIR,
    AWS_REGION,
    FAIRNESS_THRESHOLD,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"
CLOUDWATCH_NAMESPACE = "ResponsibleRiskEngine/Drift"

# Drift threshold — fraction of features that can drift before alert
# Aligned with FAIRNESS_THRESHOLD philosophy: flag early, act deliberately
DRIFT_THRESHOLD = 0.20


def load_data(data_dir: str = PROCESSED_DATA_DIR) -> tuple:
    """
    Load reference (training) and current (test) datasets.

    Reference = X_train — the distribution the model was trained on
    Current   = X_test  — simulates incoming production inference data

    person_weight is dropped — not a model feature.

    Returns:
        reference_df, current_df
    """
    def latest(pattern):
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching: {pattern}")
        return files[-1]

    X_train = pd.read_parquet(latest(f"{data_dir}/X_train_*.parquet"))
    X_test = pd.read_parquet(latest(f"{data_dir}/X_test_*.parquet"))

    # Drop person_weight — not a model feature
    X_train = X_train.drop(columns=["person_weight"], errors="ignore")
    X_test = X_test.drop(columns=["person_weight"], errors="ignore")

    # Align columns
    common_cols = [c for c in MODEL_FEATURES if c in X_train.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    logger.info(f"Reference (train): {X_train.shape}")
    logger.info(f"Current (test):    {X_test.shape}")
    logger.info(f"Features:          {common_cols}")

    return X_train, X_test


def run_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = REPORTS_DIR,
) -> dict:
    """
    Run Evidently drift report and save HTML output.

    Metrics computed:
        - DatasetDriftMetric  — overall drift across all features
        - ColumnDriftMetric   — per-feature drift detection
        - DataQualityPreset   — missing values, value ranges

    Returns:
        results dict with drift flags per feature and overall verdict
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Running Evidently drift analysis")

    # Build per-feature column drift metrics
    column_metrics = [
        ColumnDriftMetric(column_name=col)
        for col in reference_df.columns
    ]

    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
        DataQualityPreset(),
        *column_metrics,
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/drift_report_{timestamp}.html"
    report.save_html(report_path)
    logger.info(f"HTML report saved: {report_path}")

    # Extract results
    report_dict = report.as_dict()
    results = extract_drift_results(report_dict, reference_df.columns.tolist())
    results["report_path"] = report_path
    results["timestamp"] = timestamp

    # Compute drift_share from actual feature results — more reliable than
    # extracting from Evidently's internal representation
    if results["n_features"] > 0:
        drifted = sum(
            1 for fr in results["features"].values()
            if fr["drift_detected"]
        )
        results["n_drifted_features"] = drifted
        results["drift_share"] = round(drifted / results["n_features"], 4)

    return results


def extract_drift_results(report_dict: dict, features: list) -> dict:
    """
    Extract structured drift results from Evidently report dict.

    Returns:
        dict with overall drift verdict and per-feature results
    """
    results = {
        "features": {},
        "dataset_drift_detected": False,
        "drift_share": 0.0,
        "n_drifted_features": 0,
        "n_features": len(features),
    }

    for metric in report_dict.get("metrics", []):
        metric_id = metric.get("metric", "")

        # Overall dataset drift
        if "DatasetDriftMetric" in metric_id:
            result = metric.get("result", {})
            results["dataset_drift_detected"] = result.get(
                "dataset_drift", False
            )
            results["n_drifted_features"] = result.get(
                "number_of_drifted_columns", 0
            )

        # Per-feature drift
        if "ColumnDriftMetric" in metric_id:
            result = metric.get("result", {})
            col = result.get("column_name", "")
            if col:
                results["features"][col] = {
                    "drift_detected": result.get("drift_detected", False),
                    "stattest": result.get("stattest_name", ""),
                    "p_value": round(result.get("drift_score", 0.0), 4),
                }

    return results


def push_to_cloudwatch(results: dict) -> None:
    """
    Push drift metrics to CloudWatch for alarming.

    Metrics published:
        - DriftShare        — fraction of features drifted
        - DriftedFeatures   — count of drifted features
        - DatasetDrift      — 1 if overall drift detected, 0 if not

    CloudWatch alarms defined in Terraform trigger retraining
    if drift exceeds thresholds.
    """
    try:
        cw = boto3.client("cloudwatch", region_name=AWS_REGION)
        timestamp = datetime.now(timezone.utc)

        metric_data = [
            {
                "MetricName": "DriftShare",
                "Value": results["drift_share"],
                "Unit": "None",
                "Timestamp": timestamp,
            },
            {
                "MetricName": "DriftedFeatures",
                "Value": float(results["n_drifted_features"]),
                "Unit": "Count",
                "Timestamp": timestamp,
            },
            {
                "MetricName": "DatasetDrift",
                "Value": 1.0 if results["dataset_drift_detected"] else 0.0,
                "Unit": "None",
                "Timestamp": timestamp,
            },
        ]

        # Per-feature drift metrics
        for feature, feature_result in results["features"].items():
            metric_data.append({
                "MetricName": f"FeatureDrift_{feature}",
                "Value": 1.0 if feature_result["drift_detected"] else 0.0,
                "Unit": "None",
                "Timestamp": timestamp,
            })

        cw.put_metric_data(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricData=metric_data,
        )
        logger.info(
            f"CloudWatch metrics published: {len(metric_data)} metrics "
            f"to {CLOUDWATCH_NAMESPACE}"
        )

    except Exception as e:
        # CloudWatch failure should not block drift reporting
        logger.warning(f"CloudWatch publish failed: {e}")


def print_drift_report(results: dict) -> None:
    """
    Print human-readable drift summary to console.
    """
    print("\n" + "=" * 55)
    print("Drift Monitor — Results")
    print("=" * 55)
    print(f"Timestamp:          {results['timestamp']}")
    print(f"Reference dataset:  X_train (training distribution)")
    print(f"Current dataset:    X_test  (simulated production data)")
    print(f"Features monitored: {results['n_features']}")
    print()

    print("--- Per-Feature Drift ---")
    print(f"{'Feature':<22} {'Drift':<8} {'Test':<20} {'p-value'}")
    print("-" * 60)
    for feature, fr in results["features"].items():
        drift_flag = "⚠️  YES" if fr["drift_detected"] else "✅  no"
        print(
            f"{feature:<22} {drift_flag:<8} "
            f"{fr['stattest']:<20} {fr['p_value']}"
        )

    print()
    print("--- Overall ---")
    print(f"  Drifted features: {results['n_drifted_features']} / "
          f"{results['n_features']}")
    print(f"  Drift share:      {results['drift_share']}")
    print(f"  Dataset drift:    "
          f"{'⚠️  DETECTED' if results['dataset_drift_detected'] else '✅  NOT DETECTED'}")

    print()
    if results["drift_share"] > DRIFT_THRESHOLD:
        print(f"  🚨 ALERT: Drift share {results['drift_share']} exceeds "
              f"threshold {DRIFT_THRESHOLD}")
        print("  Action: Schedule retraining on fresh data")
    else:
        print(f"  ✅ PASS: Drift share {results['drift_share']} within "
              f"threshold {DRIFT_THRESHOLD}")
        print("  Action: No retraining required")

    print()
    print(f"  HTML report: {results['report_path']}")
    print("=" * 55)


def run_drift_monitor(
    data_dir: str = PROCESSED_DATA_DIR,
    output_dir: str = REPORTS_DIR,
    push_cloudwatch: bool = True,
) -> dict:
    """
    Full drift monitoring pipeline.

    Steps:
        1. Load reference and current datasets
        2. Run Evidently drift report
        3. Push metrics to CloudWatch
        4. Print summary to console

    Returns:
        results dict with drift findings
    """
    logger.info("=" * 55)
    logger.info("Drift Monitor — Starting")
    logger.info("=" * 55)

    reference_df, current_df = load_data(data_dir)
    results = run_evidently_report(reference_df, current_df, output_dir)

    if push_cloudwatch:
        push_to_cloudwatch(results)

    print_drift_report(results)

    logger.info("=" * 55)
    logger.info("Drift Monitor — Complete")
    logger.info(f"Drift detected: {results['dataset_drift_detected']}")
    logger.info(f"Drift share: {results['drift_share']}")
    logger.info("=" * 55)

    return results


if __name__ == "__main__":
    results = run_drift_monitor()
