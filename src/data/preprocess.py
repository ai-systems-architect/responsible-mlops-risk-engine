"""
preprocess.py — Feature Engineering and Data Preparation
----------------------------------------------------------
Second step in the pipeline. Takes raw ACS PUMS data produced by
ingest.py and prepares it for model training.

Pipeline steps:
    1. Load latest raw parquet from data/raw/
    2. Drop invalid rows (nulls, negative income, under minimum age)
    3. Create binary target variable from wage income
    4. Separate sensitive features from model features
    5. Encode categorical features
    6. Scale numeric features
    7. Stratified train/test split
    8. Save all artifacts to data/processed/

All parameters (threshold, split ratio, feature lists) are read
from config.py. No hardcoded values in this file.

Output artifacts:
    X_train, X_test        — model feature splits (parquet)
    y_train, y_test        — target label splits (parquet)
    sensitive              — demographic features for fairness audit (parquet)
    encoders               — fitted LabelEncoders for inference (joblib)
    scaler                 — fitted StandardScaler for inference (joblib)
"""

import os
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from glob import glob

# All parameters imported from central config
from config import (
    INCOME_THRESHOLD,
    MIN_AGE,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET,
    MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SENSITIVE_FEATURES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def load_latest_raw(data_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load the most recently saved raw parquet file from data/raw/.

    Selects the latest file by filename — timestamp is embedded in the
    filename by ingest.py e.g. acs_pums_2023_20260312_raw.parquet.

    Args:
        data_dir: Directory containing raw parquet files

    Returns:
        Raw dataframe as saved by ingest.py
    """
    files = glob(f"{data_dir}/acs_pums_*_raw.parquet")

    if not files:
        raise FileNotFoundError(
            f"No raw parquet files found in {data_dir}. "
            "Run src/data/ingest.py first."
        )

    # Sort by filename — timestamp suffix ensures latest file sorts last
    latest_file = sorted(files)[-1]
    logger.info(f"Loading raw data: {latest_file}")

    df = pd.read_parquet(latest_file)
    logger.info(f"Loaded {len(df):,} raw records — {df.shape[1]} columns")

    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records that are unsuitable for training.

    Filtering is intentionally deferred from ingest.py so the raw
    parquet file preserves the complete unmodified API response.
    Auditors can compare raw vs processed record counts.

    Rules applied:
        - wage_income must be present and non-negative
          (Census encodes not-applicable income as negative values)
        - age must be present and >= MIN_AGE (working-age adults only)
    """
    before = len(df)

    df = df[df["wage_income"].notna()]
    df = df[df["wage_income"] >= 0]
    df = df[df["age"].notna()]
    df = df[df["age"] >= MIN_AGE]

    after = len(df)
    dropped = before - after
    logger.info(
        f"Dropped {dropped:,} invalid rows ({dropped/before:.1%}) "
        f"— {after:,} records remaining"
    )

    return df.reset_index(drop=True)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target from raw wage income.

        high_income = 1  if wage_income >= INCOME_THRESHOLD
        high_income = 0  if wage_income <  INCOME_THRESHOLD

    Threshold is defined in config.INCOME_THRESHOLD ($75,000).
    Rationale for this value is documented in docs/decision_log.md DL-002.

    Class imbalance is logged here — informs the choice of
    scale_pos_weight in XGBoost and StratifiedKFold in cross-validation.
    """
    df[TARGET] = (df["wage_income"] >= INCOME_THRESHOLD).astype(int)

    positive_count = df[TARGET].sum()
    positive_rate = df[TARGET].mean()
    negative_count = len(df) - positive_count

    logger.info(f"Target created: {TARGET} (wage >= ${INCOME_THRESHOLD:,})")
    logger.info(f"  Positive: {positive_count:,} ({positive_rate:.1%})")
    logger.info(f"  Negative: {negative_count:,} ({1-positive_rate:.1%})")
    logger.info(f"  Imbalance ratio: {negative_count/positive_count:.1f}:1")

    if positive_rate < 0.3 or positive_rate > 0.7:
        logger.warning(
            f"Class imbalance detected ({positive_rate:.1%} positive). "
            "StratifiedKFold and scale_pos_weight will be applied in training."
        )

    return df


def separate_sensitive_features(df: pd.DataFrame) -> tuple:
    """
    Physically separate sensitive demographic features from model features.

    Sensitive features (race, sex, nativity) are never used as model inputs.
    They are preserved in a separate dataframe used exclusively by
    evaluate.py to compute per-group fairness metrics after prediction.

    Separation at ingestion time (rather than at training time) provides
    a clear audit trail demonstrating the model never had access to
    protected attributes during training.

    Returns:
        model_df:     Features for training + target column
        sensitive_df: Sensitive features + target column (fairness audit only)
    """
    # sensitive_df retains the target so evaluate.py can compute
    # per-group AUC, precision, recall without rejoining dataframes
    sensitive_df = df[SENSITIVE_FEATURES + [TARGET]].copy()

    # wage_income is excluded from model_df — it is the source of the target
    # variable. Including it would constitute direct data leakage.
    cols_to_drop = SENSITIVE_FEATURES + ["wage_income"]
    model_df = df.drop(columns=cols_to_drop)

    logger.info(f"Sensitive features separated: {SENSITIVE_FEATURES}")
    logger.info(f"Model features: {[c for c in MODEL_FEATURES]}")

    return model_df, sensitive_df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None) -> tuple:
    """
    Label encode categorical features.

    Census occupation and education values are ordinal integer codes.
    LabelEncoder maps them to a consistent 0-based integer range,
    which is required by XGBoost and Ridge regression.

    Two modes:
        Training mode (encoders=None): fits new encoders, returns them
        Inference mode (encoders=dict): applies pre-fit training encoders

    Inference mode ensures live data is encoded identically to training
    data — critical for model consistency in production.

    Args:
        df:       Dataframe containing categorical columns
        encoders: Pre-fit encoders from training run (None during training)

    Returns:
        df:       Dataframe with encoded categorical columns
        encoders: Dict of fitted LabelEncoders keyed by column name
    """
    fitting = encoders is None
    if fitting:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue

        # Fill nulls with mode before encoding
        # Prevents LabelEncoder from failing on NaN values
        if df[col].isnull().any():
            fill_value = df[col].mode()[0]
            df[col] = df[col].fillna(fill_value)
            logger.info(f"Filled {col} nulls with mode value: {fill_value}")

        if fitting:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
        else:
            df[col] = encoders[col].transform(df[col].astype(str))

    mode = "fit + transform" if fitting else "transform only"
    logger.info(f"Categorical encoding ({mode}): {CATEGORICAL_FEATURES}")

    return df, encoders


def scale_numerics(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple:
    """
    Standardize numeric features to mean=0, standard deviation=1.

    Scaling is required for Ridge regression — the L2 regularization
    penalty is scale-sensitive and would disproportionately penalize
    features with large magnitudes (e.g. person_weight in thousands).

    Applied to XGBoost as well for consistency — ensures the same
    preprocessed data is used across all three model stages.

    Two modes:
        Training mode (scaler=None): fits new scaler, returns it
        Inference mode (scaler=StandardScaler): applies pre-fit scaler

    Args:
        df:     Dataframe containing numeric columns
        scaler: Pre-fit scaler from training run (None during training)

    Returns:
        df:     Dataframe with scaled numeric columns
        scaler: Fitted StandardScaler
    """
    cols_to_scale = [c for c in NUMERIC_FEATURES if c in df.columns]
    fitting = scaler is None

    if fitting:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info(f"StandardScaler fit + applied to: {cols_to_scale}")
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        logger.info(f"Pre-fit StandardScaler applied to: {cols_to_scale}")

    return df, scaler


def split_data(model_df: pd.DataFrame) -> tuple:
    """
    Stratified train/test split.

    Stratification ensures both splits contain the same proportion of
    positive and negative labels as the full dataset. Essential when
    class imbalance is present — a random split could produce a test
    set with very few positive examples, making evaluation unreliable.

    Split ratio and random seed are defined in config.py.

    Args:
        model_df: Full feature dataframe including target column

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"Train: {len(X_train):,} records — {y_train.mean():.1%} positive")
    logger.info(f"Test:  {len(X_test):,} records — {y_test.mean():.1%} positive")

    return X_train, X_test, y_train, y_test


def save_processed(
    X_train, X_test,
    y_train, y_test,
    sensitive_df: pd.DataFrame,
    encoders: dict,
    scaler: StandardScaler,
    output_dir: str = PROCESSED_DATA_DIR,
) -> None:
    """
    Save all processed artifacts to data/processed/.

    Artifacts saved:
        X_train / X_test       — feature matrices for training and evaluation
        y_train / y_test       — label arrays for training and evaluation
        sensitive              — demographic data for fairness audit
        encoders               — fitted LabelEncoders (required for inference)
        scaler                 — fitted StandardScaler (required for inference)

    Encoders and scaler must be saved alongside the model so that
    inference data can be preprocessed identically to training data.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    X_train.to_parquet(f"{output_dir}/X_train_{timestamp}.parquet", index=False)
    X_test.to_parquet(f"{output_dir}/X_test_{timestamp}.parquet", index=False)
    y_train.to_frame().to_parquet(f"{output_dir}/y_train_{timestamp}.parquet", index=False)
    y_test.to_frame().to_parquet(f"{output_dir}/y_test_{timestamp}.parquet", index=False)
    sensitive_df.to_parquet(f"{output_dir}/sensitive_{timestamp}.parquet", index=False)
    joblib.dump(encoders, f"{output_dir}/encoders_{timestamp}.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler_{timestamp}.joblib")

    logger.info(f"All processed artifacts saved to: {output_dir}/")


def run_preprocessing(
    data_dir: str = RAW_DATA_DIR,
    output_dir: str = PROCESSED_DATA_DIR,
) -> tuple:
    """
    Orchestrates the full preprocessing pipeline.

    Steps:
        1. Load latest raw parquet from data/raw/
        2. Drop invalid rows
        3. Create binary target
        4. Separate sensitive features
        5. Encode categorical features (fit new encoders)
        6. Scale numeric features (fit new scaler)
        7. Stratified train/test split
        8. Save all artifacts to data/processed/

    Returns:
        X_train, X_test, y_train, y_test, sensitive_df
    """
    logger.info("=" * 55)
    logger.info("Preprocessing Pipeline — Starting")
    logger.info("=" * 55)

    df = load_latest_raw(data_dir=data_dir)
    df = drop_invalid_rows(df)
    df = create_target(df)
    model_df, sensitive_df = separate_sensitive_features(df)
    model_df, encoders = encode_categoricals(model_df)
    model_df, scaler = scale_numerics(model_df)
    X_train, X_test, y_train, y_test = split_data(model_df)

    # Split sensitive_df using the same index as X_train/X_test
    # so evaluate.py receives demographic features aligned with
    # the test set only — not the full dataset
    sensitive_test = sensitive_df.loc[X_test.index].reset_index(drop=True)

    save_processed(
        X_train, X_test,
        y_train, y_test,
        sensitive_test, encoders, scaler,
        output_dir=output_dir,
    )

    logger.info("=" * 55)
    logger.info("Preprocessing complete")
    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    logger.info("Next: src/training/baseline.py")
    logger.info("=" * 55)

    return X_train, X_test, y_train, y_test, sensitive_df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, sensitive_df = run_preprocessing()

    print("\n--- Feature Summary (X_train) ---")
    print(X_train.describe().to_string())

    print("\n--- Target Distribution (y_train) ---")
    print(y_train.value_counts(normalize=True).to_string())

    print("\n--- Sensitive Features Sample ---")
    print(sensitive_df.head(3).to_string())

    print("\n--- Processed Files ---")
    for f in sorted(os.listdir("data/processed")):
        size = os.path.getsize(f"data/processed/{f}") / 1024
        print(f"  {f} ({size:.0f} KB)")
