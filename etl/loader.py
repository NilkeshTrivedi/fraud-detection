"""
Loader module — saves processed data to PostgreSQL using fast COPY command.
"""

import csv
import io
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from etl.config import DatabaseConfig, ETLConfig

logger = logging.getLogger(__name__)


def get_engine(db_config: DatabaseConfig) -> Engine:
    """
    Create SQLAlchemy engine from database config.

    Args:
        db_config: DatabaseConfig instance.

    Returns:
        SQLAlchemy Engine.
    """
    logger.info(f"Connecting to PostgreSQL at {db_config.host}:{db_config.port}...")
    engine = create_engine(db_config.url, pool_pre_ping=True)

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection successful ✅")

    return engine


def save_to_csv(df: pd.DataFrame, config: ETLConfig) -> None:
    """
    Save processed dataframe to CSV in processed directory.

    Args:
        df: Transformed dataframe.
        config: ETLConfig instance.
    """
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.processed_dir / "processed_fraud_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def create_table(engine: Engine, df: pd.DataFrame, table_name: str) -> None:
    """
    Create PostgreSQL table schema from dataframe dtypes.

    Args:
        engine: SQLAlchemy engine.
        df: Dataframe to derive schema from.
        table_name: Target table name.
    """
    logger.info(f"Creating table '{table_name}'...")

    # Map pandas dtypes to PostgreSQL types
    type_mapping = {
        "int64": "BIGINT",
        "int32": "INTEGER",
        "float64": "DOUBLE PRECISION",
        "float32": "REAL",
        "object": "TEXT",
        "bool": "BOOLEAN",
        "category": "TEXT",
    }

    columns_sql = []
    for col, dtype in df.dtypes.items():
        pg_type = type_mapping.get(str(dtype), "TEXT")
        safe_col = f'"{col}"'
        columns_sql.append(f"{safe_col} {pg_type}")

    create_sql = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            {", ".join(columns_sql)}
        );
    """

    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()

    logger.info(f"Table '{table_name}' created ✅")


def load_to_postgres(
    df: pd.DataFrame,
    engine: Engine,
    config: ETLConfig,
) -> None:
    """
    Load dataframe into PostgreSQL using fast COPY command.

    Args:
        df: Transformed dataframe.
        engine: SQLAlchemy engine.
        config: ETLConfig instance.
    """
    total_rows = len(df)
    logger.info(f"Loading {total_rows:,} rows into '{config.table_name}' using COPY...")

    # Create table first
    create_table(engine, df, config.table_name)

    # Write df to in-memory buffer as CSV
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
    buffer.seek(0)

    # Use raw psycopg2 connection for COPY
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()

        # Build column list
        columns = ", ".join([f'"{col}"' for col in df.columns])

        copy_sql = f"""
            COPY {config.table_name} ({columns})
            FROM STDIN
            WITH (FORMAT CSV, NULL '')
        """

        cursor.copy_expert(copy_sql, buffer)
        raw_conn.commit()

        # Verify row count
        cursor.execute(f"SELECT COUNT(*) FROM {config.table_name}")
        db_count = cursor.fetchone()[0]

        if db_count != total_rows:
            raise ValueError(
                f"Row count mismatch: expected {total_rows:,}, got {db_count:,}"
            )

        logger.info(f"Successfully loaded {db_count:,} rows into PostgreSQL ✅")

    except Exception as e:
        raw_conn.rollback()
        logger.error(f"COPY failed: {e}")
        raise
    finally:
        cursor.close()
        raw_conn.close()