"""
ingest.py — ACS PUMS Data Ingestion
-------------------------------------
First step in the pipeline. Fetches raw person-level microdata from the
U.S. Census Bureau API and saves it exactly as received.

Scope:
    - Reads all parameters from config.py
    - Calls Census API with the variables defined in config.VARIABLES
    - Renames Census variable codes to human-readable column names
    - Casts numeric columns from string to correct types
    - Saves one timestamped parquet file to data/raw/

Out of scope (handled by preprocess.py):
    - Row filtering and null removal
    - Target variable creation
    - Sensitive feature separation
    - Categorical encoding and numeric scaling
    - Train/test split

Data Source:
    American Community Survey (ACS) Public Use Microdata Sample (PUMS)
    https://www.census.gov/programs-surveys/acs/microdata.html
    API reference: https://api.census.gov/data/2023/acs/acs1/pums/variables.json

Census API key is loaded from the CENSUS_API_KEY environment variable.
It is never stored in source code or config files.
"""

import os
import requests
import pandas as pd
import logging
from datetime import datetime

# All parameters come from config.py — no hardcoded values in this file
from config import (
    ACS_YEAR,
    ACS_DATASET,
    STATE_CODE,
    VARIABLES,
    RAW_DATA_DIR,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Census API base URL — constructed from config values
# Full URL example: https://api.census.gov/data/2023/acs/acs1/pums
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")
BASE_URL = f"https://api.census.gov/data/{ACS_YEAR}/{ACS_DATASET}"


def fetch_acs_pums(state_code: str = STATE_CODE) -> pd.DataFrame:
    """
    Fetch raw ACS PUMS person-level records from the Census Bureau API.

    The Census API returns a JSON list of lists:
        Row 0       = column headers (Census variable codes e.g. AGEP, SCHL)
        Rows 1..n   = one record per person, all values as strings

    Args:
        state_code: FIPS state code from config.py
                    "51" = Virginia (~60K records) for development
                    "*"  = All 50 states (~1.5M records) for final training

    Returns:
        Raw dataframe with Census variable codes as column names,
        all values still as strings — no transformations applied.
    """
    if not CENSUS_API_KEY:
        raise ValueError(
            "CENSUS_API_KEY environment variable is not set.\n"
            "Set it with: echo 'CENSUS_API_KEY=your-key' >> ~/.zshenv && source ~/.zshenv"
        )

    # Comma-separated variable string the API requires
    # Built from config.VARIABLES keys e.g. "AGEP,SCHL,OCCP,WKHP,WAGP,..."
    variable_list = ",".join(VARIABLES.keys())

    params = {
        "get": variable_list,
        "for": f"state:{state_code}",
        "key": CENSUS_API_KEY,
    }

    state_label = "all states" if state_code == "*" else f"state FIPS {state_code}"
    logger.info(f"Fetching ACS PUMS {ACS_YEAR} — {state_label}")
    logger.info(f"Variables requested: {list(VARIABLES.keys())}")

    # Timeout set to 120 seconds — national pulls can take 2-3 minutes
    response = requests.get(BASE_URL, params=params, timeout=120)

    if response.status_code != 200:
        raise ValueError(
            f"Census API returned status {response.status_code}.\n"
            f"Response: {response.text[:300]}"
        )

    # Parse response — first row is headers, remaining rows are records
    data = response.json()
    headers = data[0]
    records = data[1:]

    df = pd.DataFrame(records, columns=headers)
    logger.info(f"Received {len(df):,} raw records")

    return df


def rename_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Census variable codes to human-readable column names
    and cast numeric columns from string to float.

    Renaming is done here rather than in preprocess.py so the saved
    parquet file is readable without needing to cross-reference Census
    variable documentation.

    Casting is done here because the Census API returns all values as
    strings. Correct types must be set before saving to parquet.

    No rows are dropped. No new columns are created.
    All filtering and feature engineering is handled in preprocess.py.
    """
    # Rename Census codes using the mapping in config.VARIABLES
    # e.g. "AGEP" → "age", "WAGP" → "wage_income", "RAC1P" → "race"
    rename_map = {k: v for k, v in VARIABLES.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # The Census API automatically appends a "state" column containing
    # the FIPS code. It duplicates information already known from the
    # request parameters and is not used as a feature.
    df = df.drop(columns=["state"], errors="ignore")

    # Cast numeric columns from string to float
    # errors="coerce" converts any unparseable value to NaN
    # NaN handling is intentionally deferred to preprocess.py
    numeric_cols = [
        "age",
        "education",
        "hours_per_week",
        "wage_income",
        "class_of_worker",
        "marital_status",
        "person_weight",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Columns renamed and cast — shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def save_raw(df: pd.DataFrame, output_dir: str = RAW_DATA_DIR) -> str:
    """
    Save the raw dataframe as a parquet file to data/raw/.

    Parquet is used instead of CSV because:
        - Column data types are preserved without re-casting on load
        - Significantly faster read performance for large datasets
        - Columnar format allows efficient loading of specific features
        - Standard format across production ML and data engineering pipelines

    The filename includes the ACS year and a date timestamp so multiple
    ingestion runs produce distinct files rather than overwriting each other.
    Example: acs_pums_2023_20240315_raw.parquet

    Args:
        df:         Raw dataframe to save
        output_dir: Directory path from config.RAW_DATA_DIR

    Returns:
        filepath: Full path to the saved parquet file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    filepath = f"{output_dir}/acs_pums_{ACS_YEAR}_{timestamp}_raw.parquet"

    df.to_parquet(filepath, index=False)

    file_size_mb = os.path.getsize(filepath) / 1024 / 1024
    logger.info(f"Raw data saved: {filepath} ({file_size_mb:.1f} MB)")

    return filepath


def run_ingestion(
    state_code: str = STATE_CODE,
    output_dir: str = RAW_DATA_DIR,
) -> pd.DataFrame:
    """
    Orchestrates the full ingestion pipeline.

    Steps:
        1. Fetch raw records from Census API
        2. Rename columns and cast types
        3. Save parquet to data/raw/

    Args:
        state_code: Overrides config.STATE_CODE when called programmatically
        output_dir: Overrides config.RAW_DATA_DIR when called programmatically

    Returns:
        Raw dataframe — input for preprocess.py
    """
    logger.info("=" * 55)
    logger.info(f"ACS PUMS {ACS_YEAR} Ingestion — Starting")
    logger.info(f"Scope: {'National (all states)' if state_code == '*' else f'State FIPS {state_code}'}")
    logger.info("=" * 55)

    df = fetch_acs_pums(state_code=state_code)
    df = rename_and_cast(df)
    filepath = save_raw(df, output_dir=output_dir)

    logger.info("=" * 55)
    logger.info(f"Ingestion complete — {len(df):,} records")
    logger.info(f"Output: {filepath}")
    logger.info(f"Next: src/data/preprocess.py")
    logger.info("=" * 55)

    return df


if __name__ == "__main__":
    df = run_ingestion()

    print("\n--- Raw Data Sample ---")
    print(df.head(3).to_string())

    print("\n--- Column Types ---")
    print(df.dtypes.to_string())

    print("\n--- Shape ---")
    print(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n--- Null Counts ---")
    print(df.isnull().sum().to_string())

    print("\n--- Wage Income Range ---")
    print(f"Min:  ${df['wage_income'].min():,.0f}")
    print(f"Max:  ${df['wage_income'].max():,.0f}")
    print(f"Mean: ${df['wage_income'].mean():,.0f}")