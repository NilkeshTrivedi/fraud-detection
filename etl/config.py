"""
Configuration management for ETL pipeline.
Loads environment variables and provides typed settings.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Database Config ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL connection settings loaded from environment."""
    host: str = None
    port: int = None
    user: str = None
    password: str = None
    database: str = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", os.getenv("POSTGRES_HOST", "localhost"))
        object.__setattr__(self, "port", int(os.getenv("POSTGRES_PORT", 5432)))
        object.__setattr__(self, "user", os.getenv("POSTGRES_USER"))
        object.__setattr__(self, "password", os.getenv("POSTGRES_PASSWORD"))
        object.__setattr__(self, "database", os.getenv("POSTGRES_DB"))

        missing = [f for f, v in {
            "POSTGRES_USER": self.user,
            "POSTGRES_PASSWORD": self.password,
            "POSTGRES_DB": self.database,
        }.items() if not v]

        if missing:
            raise EnvironmentError(f"Missing environment variables: {missing}")

    @property
    def url(self) -> str:
        """SQLAlchemy connection URL."""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


# ── ETL Config ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ETLConfig:
    """Settings for ETL pipeline execution."""
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    transaction_file: str = "train_transaction.csv"
    identity_file: str = "train_identity.csv"
    target_column: str = "isFraud"
    null_threshold: float = 0.5
    chunk_size: int = 50_000
    table_name: str = "fraud_transactions"