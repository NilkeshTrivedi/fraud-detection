"""
Exploratory Data Analysis module for IEEE-CIS Fraud Detection dataset.
Provides statistical summaries, data quality checks, and fraud distribution insights.
"""

import logging
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EDAConfig:
    """Paths and settings for EDA."""
    raw_dir: Path = Path("data/raw")
    transaction_file: str = "train_transaction.csv"
    identity_file: str = "train_identity.csv"
    target_column: str = "isFraud"


# ── Loader ───────────────────────────────────────────────────────────────────
def load_data(config: EDAConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw transaction and identity CSVs.

    Args:
        config: EDAConfig instance with file paths.

    Returns:
        Tuple of (transaction_df, identity_df)

    Raises:
        FileNotFoundError: If either CSV is missing.
    """
    transaction_path = config.raw_dir / config.transaction_file
    identity_path = config.raw_dir / config.identity_file

    for path in [transaction_path, identity_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    logger.info("Loading transaction data...")
    transaction_df = pd.read_csv(transaction_path)
    logger.info(f"Transaction data loaded: {transaction_df.shape}")

    logger.info("Loading identity data...")
    identity_df = pd.read_csv(identity_path)
    logger.info(f"Identity data loaded: {identity_df.shape}")

    return transaction_df, identity_df


# ── Data Quality ─────────────────────────────────────────────────────────────
def check_data_quality(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Generate data quality report for a dataframe.

    Args:
        df: Input dataframe.
        name: Name label for logging.

    Returns:
        DataFrame with quality metrics per column.
    """
    logger.info(f"Running data quality checks on {name}...")

    quality_report = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "null_count": df.isnull().sum().values,
        "null_pct": (df.isnull().sum().values / len(df) * 100).round(2),
        "unique_values": df.nunique().values,
        "sample_value": [df[col].dropna().iloc[0] if not df[col].dropna().empty else None for col in df.columns],
    })

    high_null = quality_report[quality_report["null_pct"] > 50]
    logger.warning(f"{name} — columns with >50% nulls: {len(high_null)}")

    return quality_report


# ── Fraud Distribution ────────────────────────────────────────────────────────
def analyze_fraud_distribution(df: pd.DataFrame, target: str) -> dict:
    """
    Analyze fraud vs legitimate transaction distribution.

    Args:
        df: Transaction dataframe.
        target: Target column name.

    Returns:
        Dictionary with fraud statistics.
    """
    logger.info("Analyzing fraud distribution...")

    total = len(df)
    fraud_count = df[target].sum()
    legit_count = total - fraud_count
    fraud_pct = round((fraud_count / total) * 100, 4)

    stats = {
        "total_transactions": total,
        "fraud_transactions": int(fraud_count),
        "legitimate_transactions": int(legit_count),
        "fraud_percentage": fraud_pct,
        "class_imbalance_ratio": round(legit_count / fraud_count, 2),
    }

    logger.info(f"Total Transactions   : {total:,}")
    logger.info(f"Fraud Transactions   : {fraud_count:,} ({fraud_pct}%)")
    logger.info(f"Legitimate           : {legit_count:,}")
    logger.info(f"Class Imbalance Ratio: {stats['class_imbalance_ratio']}:1")

    return stats


# ── Transaction Amount Analysis ───────────────────────────────────────────────
def analyze_transaction_amounts(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Compare transaction amounts between fraud and legitimate transactions.

    Args:
        df: Transaction dataframe.
        target: Target column name.

    Returns:
        Summary statistics grouped by fraud label.
    """
    logger.info("Analyzing transaction amounts...")

    amount_stats = df.groupby(target)["TransactionAmt"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).round(2)

    amount_stats.index = ["Legitimate", "Fraud"]
    logger.info(f"\n{amount_stats.to_string()}")

    return amount_stats


# ── Missing Value Summary ─────────────────────────────────────────────────────
def summarize_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """
    Identify columns exceeding missing value threshold.

    Args:
        df: Input dataframe.
        threshold: Fraction above which column is flagged (default 0.5).

    Returns:
        List of column names exceeding threshold.
    """
    missing_frac = df.isnull().mean()
    high_missing = missing_frac[missing_frac > threshold].index.tolist()
    logger.info(f"Columns exceeding {threshold*100}% missing: {len(high_missing)}")
    return high_missing


# ── Main ──────────────────────────────────────────────────────────────────────
def run_eda() -> None:
    """Run full EDA pipeline."""
    logger.info("=" * 60)
    logger.info("Starting EDA Pipeline")
    logger.info("=" * 60)

    config = EDAConfig()

    # Load
    transaction_df, identity_df = load_data(config)

    # Quality checks
    txn_quality = check_data_quality(transaction_df, "Transaction")
    idn_quality = check_data_quality(identity_df, "Identity")

    # Fraud distribution
    fraud_stats = analyze_fraud_distribution(transaction_df, config.target_column)

    # Amount analysis
    amount_stats = analyze_transaction_amounts(transaction_df, config.target_column)

    # Missing values
    high_missing_cols = summarize_missing_values(transaction_df)

    logger.info("=" * 60)
    logger.info("EDA Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_eda()