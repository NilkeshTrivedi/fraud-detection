"""
ETL Pipeline orchestrator — runs Extract, Transform, Load in sequence.
Skips stages already completed via checkpoint system.
"""

import argparse
import logging
import sys

from etl.checkpoint import is_completed, save_checkpoint, reset_checkpoint
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


def run_pipeline(force: bool = False) -> None:
    """
    Execute full ETL pipeline with checkpoint protection.

    Args:
        force: If True, ignores checkpoints and reruns everything.
    """
    logger.info("=" * 60)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 60)

    db_config = DatabaseConfig()
    etl_config = ETLConfig()

    # ── Extract ───────────────────────────────────────────────
    if not force and is_completed("extract"):
        logger.info("⏭️  Extract already completed — skipping")
        # Still need the data in memory for transform
        transaction_df, identity_df = load_raw_data(etl_config)
        merged_df = merge_datasets(transaction_df, identity_df)
    else:
        transaction_df, identity_df = load_raw_data(etl_config)
        merged_df = merge_datasets(transaction_df, identity_df)
        save_checkpoint("extract", {"rows": len(merged_df)})

    # ── Transform ─────────────────────────────────────────────
    if not force and is_completed("transform"):
        logger.info("⏭️  Transform already completed — skipping")
        import pandas as pd
        processed_df = pd.read_csv(etl_config.processed_dir / "processed_fraud_data.csv")
        logger.info(f"Loaded processed data: {processed_df.shape}")
    else:
        processed_df = run_transformation(merged_df, etl_config)
        save_to_csv(processed_df, etl_config)
        save_checkpoint("transform", {"rows": len(processed_df), "cols": len(processed_df.columns)})

    # ── Load ──────────────────────────────────────────────────
    if not force and is_completed("load"):
        logger.info("⏭️  Load already completed — skipping (data already in PostgreSQL)")
    else:
        engine = get_engine(db_config)
        load_to_postgres(processed_df, engine, etl_config)
        save_checkpoint("load", {"rows": len(processed_df), "table": etl_config.table_name})

    logger.info("=" * 60)
    logger.info("ETL Pipeline Complete ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun all stages ignoring checkpoints",
    )
    parser.add_argument(
        "--reset",
        type=str,
        default=None,
        help="Reset a specific checkpoint stage (extract/transform/load)",
    )
    args = parser.parse_args()

    if args.reset:
        reset_checkpoint(args.reset)
    else:
        run_pipeline(force=args.force)