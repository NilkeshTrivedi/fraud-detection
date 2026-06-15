"""
Extractor module — loads raw CSVs and merges transaction + identity data.
"""

import logging
from pathlib import Path

import pandas as pd

from etl.config import ETLConfig

logger = logging.getLogger(__name__)


def load_raw_data(config: ETLConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw transaction and identity CSV files.

    Args:
        config: ETLConfig instance.

    Returns:
        Tuple of (transaction_df, identity_df).

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    transaction_path = config.raw_dir / config.transaction_file
    identity_path = config.raw_dir / config.identity_file

    for path in [transaction_path, identity_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    logger.info("Extracting transaction data...")
    transaction_df = pd.read_csv(transaction_path)
    logger.info(f"Transaction data shape: {transaction_df.shape}")

    logger.info("Extracting identity data...")
    identity_df = pd.read_csv(identity_path)
    logger.info(f"Identity data shape: {identity_df.shape}")

    return transaction_df, identity_df


def merge_datasets(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left join transaction and identity data on TransactionID.
    Not all transactions have identity records — left join preserves all.

    Args:
        transaction_df: Raw transaction dataframe.
        identity_df: Raw identity dataframe.

    Returns:
        Merged dataframe.
    """
    logger.info("Merging transaction and identity datasets...")

    merged_df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
    )

    logger.info(f"Merged data shape: {merged_df.shape}")
    logger.info(
        f"Identity match rate: "
        f"{round(merged_df['id_01'].notna().sum() / len(merged_df) * 100, 2)}%"
    )

    return merged_df