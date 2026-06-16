"""
Health check endpoint.
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from sqlalchemy import text

from api.dependencies import get_db_engine, get_predictor
from api.schemas.transaction import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_PATH = Path("models/artifacts/xgb_fraud_model.pkl")


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """
    Check API health — model loaded and database connected.
    """
    # Check model
    model_loaded = False
    try:
        get_predictor()
        model_loaded = True
    except Exception as e:
        logger.error(f"Model load failed: {e}")

    # Check DB
    db_connected = False
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"DB connection failed: {e}")

    return HealthResponse(
        status="healthy" if model_loaded and db_connected else "degraded",
        model_loaded=model_loaded,
        database_connected=db_connected,
    )