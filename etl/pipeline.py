"""
ETL Pipeline orchestrator — runs Extract, Transform, Load in sequence.
"""

import logging
import sys

from etl.config import DatabaseConfig, ETLConfig
from etl.extractor import load_raw_data, merge_datasets
from etl.transformer import run_transformation
from etl.loader import get_engine, save_to_csv, load_to_postgres

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute full ETL pipeline."""
    logger.info("=" * 60)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 60)

    db_config = DatabaseConfig()
    etl_config = ETLConfig()

    # Extract
    transaction_df, identity_df = load_raw_data(etl_config)
    merged_df = merge_datasets(transaction_df, identity_df)

    # Transform
    processed_df = run_transformation(merged_df, etl_config)

    # Load
    save_to_csv(processed_df, etl_config)
    engine = get_engine(db_config)
    load_to_postgres(processed_df, engine, etl_config)

    logger.info("=" * 60)
    logger.info("ETL Pipeline Complete ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()