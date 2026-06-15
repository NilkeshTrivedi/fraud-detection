"""
Transformer module — cleans, validates, and feature engineers merged data.
"""

import logging

import pandas as pd
import numpy as np

from etl.config import ETLConfig

logger = logging.getLogger(__name__)


def drop_high_null_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop columns exceeding null threshold.

    Args:
        df: Input dataframe.
        threshold: Null fraction above which column is dropped.

    Returns:
        Dataframe with high-null columns removed.
    """
    null_fractions = df.isnull().mean()
    cols_to_drop = null_fractions[null_fractions > threshold].index.tolist()

    logger.info(f"Dropping {len(cols_to_drop)} columns exceeding {threshold*100}% null threshold")
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Shape after dropping high-null columns: {df.shape}")

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill remaining missing values by dtype.
    Numeric columns → median, Categorical columns → 'Unknown'.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with no missing values.
    """
    logger.info("Filling remaining missing values...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    remaining_nulls = df.isnull().sum().sum()
    logger.info(f"Remaining nulls after fill: {remaining_nulls}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns to improve model signal.

    Args:
        df: Cleaned dataframe.

    Returns:
        Dataframe with engineered features.
    """
    logger.info("Engineering features...")
    df = df.copy()

    # Transaction amount bins
    df["amt_bin"] = pd.cut(
        df["TransactionAmt"],
        bins=[0, 50, 200, 1000, np.inf],
        labels=["low", "medium", "high", "very_high"],
    )

    # Log transform of transaction amount (reduces skew)
    df["log_transaction_amt"] = np.log1p(df["TransactionAmt"])

    # Transaction hour from TransactionDT (seconds since reference)
    df["transaction_hour"] = (df["TransactionDT"] // 3600) % 24

    # Weekend flag (TransactionDT in seconds, ref date is Black Friday 2017)
    df["transaction_day"] = (df["TransactionDT"] // (3600 * 24)) % 7
    df["is_weekend"] = df["transaction_day"].isin([5, 6]).astype(int)

    logger.info(f"Shape after feature engineering: {df.shape}")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label encode categorical columns for database storage and ML readiness.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with encoded categoricals.
    """
    logger.info("Encoding categorical columns...")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"Encoding {len(categorical_cols)} categorical columns")

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    return df


def validate_data(df: pd.DataFrame, target: str) -> None:
    """
    Run validation checks on transformed data.

    Args:
        df: Transformed dataframe.
        target: Target column name.

    Raises:
        ValueError: If any validation check fails.
    """
    logger.info("Running validation checks...")

    # Check no nulls remain
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Validation failed: {null_count} nulls remain after transformation")

    # Check target column exists
    if target not in df.columns:
        raise ValueError(f"Validation failed: target column '{target}' missing")

    # Check target is binary
    unique_targets = df[target].unique()
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"Validation failed: target column has non-binary values: {unique_targets}")

    # Check no negative transaction amounts
    if (df["TransactionAmt"] < 0).any():
        raise ValueError("Validation failed: negative transaction amounts found")

    logger.info("All validation checks passed ✅")


def run_transformation(df: pd.DataFrame, config: ETLConfig) -> pd.DataFrame:
    """
    Orchestrate full transformation pipeline.

    Args:
        df: Raw merged dataframe.
        config: ETLConfig instance.

    Returns:
        Fully transformed and validated dataframe.
    """
    logger.info("Starting transformation pipeline...")

    df = drop_high_null_columns(df, config.null_threshold)
    df = fill_missing_values(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    validate_data(df, config.target_column)

    logger.info("Transformation pipeline complete ✅")
    return df