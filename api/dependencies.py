"""
Shared FastAPI dependencies — model loader and DB session.
"""

import logging
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from etl.config import DatabaseConfig
from models.predictor import FraudPredictor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    """
    Load and cache fraud predictor.
    lru_cache ensures model is loaded only once.

    Returns:
        FraudPredictor instance.
    """
    logger.info("Loading FraudPredictor...")
    return FraudPredictor()


@lru_cache(maxsize=1)
def get_db_engine() -> Engine:
    """
    Create and cache database engine.

    Returns:
        SQLAlchemy Engine.
    """
    db_config = DatabaseConfig()
    engine = create_engine(db_config.url, pool_pre_ping=True)
    logger.info("Database engine created ✅")
    return engine