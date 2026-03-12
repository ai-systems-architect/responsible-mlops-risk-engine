"""
config.py — Central Pipeline Configuration
-------------------------------------------
Single source of truth for all parameters across the entire pipeline.
Every script imports from here — no hardcoded values in pipeline code.

Sensitive values (API keys, account IDs, role ARNs) are loaded from
the .env file via environment variables — never stored in this file.

Config is organized into sections:
    - Data Source
    - Census API Variables
    - Business Decisions
    - Model Training
    - Fairness
    - AWS Infrastructure
"""

import os
from dotenv import load_dotenv

# Load .env file — contains sensitive values not stored in source code
# .env is listed in .gitignore and never committed to the repository
load_dotenv()


# =============================================================================
# DATA SOURCE
# =============================================================================

# ACS PUMS release year
ACS_YEAR = 2023

# ACS dataset — acs1 = 1-year estimates (most current)
# acs5 = 5-year estimates (larger sample, slightly older data)
ACS_DATASET = "acs/acs1/pums"

# State FIPS code controlling how much data is pulled from Census API
# "51" = Virginia (~60K records) — used during development for faster iteration
# "*"  = All 50 states (~1.5M records) — used for final model training
# Full FIPS reference: https://www.census.gov/library/reference/code-lists/ansi/ansi-codes-for-states.html
STATE_CODE = "51"

# Local directories for data artifacts
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


# =============================================================================
# CENSUS API VARIABLES
# =============================================================================
# Maps Census variable codes to human-readable column names.
# Full variable reference: https://api.census.gov/data/2023/acs/acs1/pums/variables.json
#
# Three categories:
#   Model features  — used as inputs to the model
#   Target source   — raw income field, converted to binary target in preprocess.py
#   Sensitive       — demographic fields used only for fairness audits

VARIABLES = {
    # Model features
    "AGEP":     "age",
    "SCHL":     "education",
    "OCCP":     "occupation",
    "WKHP":     "hours_per_week",
    "COW":      "class_of_worker",
    "MAR":      "marital_status",
    "PWGTP":    "person_weight",

    # Target source — converted to binary in preprocess.py, then removed from features
    "WAGP":     "wage_income",

    # Sensitive features — separated in preprocess.py, never used as model inputs
    "RAC1P":    "race",
    "SEX":      "sex",
    "NATIVITY": "nativity",
}

# Features used as model inputs
MODEL_FEATURES = [
    "age",
    "education",
    "occupation",
    "hours_per_week",
    "class_of_worker",
    "marital_status",
    "person_weight",
]

# Categorical features within MODEL_FEATURES — require label encoding
CATEGORICAL_FEATURES = [
    "education",
    "occupation",
    "class_of_worker",
    "marital_status",
]

# Numeric features within MODEL_FEATURES — require standard scaling
NUMERIC_FEATURES = [
    "age",
    "hours_per_week",
    "person_weight",
]

# Sensitive demographic features — preserved for fairness audits only
# Physically separated from model data in preprocess.py
SENSITIVE_FEATURES = ["race", "sex", "nativity"]

# Target column name used across all pipeline scripts
TARGET = "high_income"


# =============================================================================
# BUSINESS DECISIONS
# =============================================================================
# Parameters that represent deliberate analytical choices.
# Each decision is documented with rationale in docs/decision_log.md.

# Binary classification threshold for wage income
# $75,000 reflects approximately the 2023 US median household income
# and aligns with financial risk thresholds used in federal program eligibility
INCOME_THRESHOLD = 75_000

# Minimum age for inclusion in the dataset
# Pipeline is scoped to working-age adults only
MIN_AGE = 18


# =============================================================================
# MODEL TRAINING
# =============================================================================

# Train/test split proportion
TEST_SIZE = 0.20

# Fixed random seed ensures reproducibility across all runs
# Used in train/test split, cross-validation, and model initialization
RANDOM_STATE = 42

# Minimum AUC required to pass the CI/CD metrics gate
# Models below this threshold are not promoted to staging
MIN_AUC_THRESHOLD = 0.82

# Optuna hyperparameter tuning trial counts
# Lower count for CI/CD runs to keep pipeline execution time reasonable
# Higher count for local training runs to find better parameters
OPTUNA_TRIALS_CI = 5
OPTUNA_TRIALS_LOCAL = 30


# =============================================================================
# FAIRNESS
# =============================================================================

# Maximum allowable disparity in positive prediction rate across demographic groups
# Computed per group in evaluate.py after every training run
# Models exceeding this threshold do not proceed to the deployment approval step
FAIRNESS_THRESHOLD = 0.20


# =============================================================================
# AWS INFRASTRUCTURE
# =============================================================================
# Non-sensitive infrastructure configuration — instance types, region, endpoint names.
# Account-specific values (account ID, role ARN, bucket name) are loaded
# from the .env file below and never stored in this file.
AWS_REGION = "us-east-1"

# SageMaker real-time endpoint configuration
# ml.m5.xlarge selected for balance of memory and compute at reasonable cost
SAGEMAKER_INSTANCE = "ml.m5.xlarge"
SAGEMAKER_ENDPOINT_NAME = "responsible-risk-engine-prod-v1"
SAGEMAKER_FRAMEWORK_VERSION = "1.7-1"

# MLflow experiment name — groups all training runs for this project
MLFLOW_EXPERIMENT_NAME = "responsible-mlops-risk-engine"

# --- Values loaded from .env (account-specific, never committed) ---
S3_BUCKET = os.environ.get("S3_BUCKET")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")