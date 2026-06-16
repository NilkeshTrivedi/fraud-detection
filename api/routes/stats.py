"""
Live fraud statistics endpoint from PostgreSQL.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.dependencies import get_db_engine
from api.schemas.transaction import StatsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats", response_model=StatsResponse, tags=["Statistics"])
def get_fraud_stats(
    engine: Engine = Depends(get_db_engine),
) -> StatsResponse:
    """
    Fetch live fraud statistics directly from PostgreSQL.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*)                                          AS total_transactions,
                    SUM("isFraud")                                    AS total_fraud,
                    COUNT(*) - SUM("isFraud")                        AS total_legitimate,
                    ROUND(AVG("isFraud") * 100, 4)                   AS fraud_percentage,
                    ROUND(AVG(CASE WHEN "isFraud" = 1
                        THEN "TransactionAmt" END)::numeric, 2)      AS avg_fraud_amount,
                    ROUND(AVG(CASE WHEN "isFraud" = 0
                        THEN "TransactionAmt" END)::numeric, 2)      AS avg_legitimate_amount,
                    ROUND(MAX(CASE WHEN "isFraud" = 1
                        THEN "TransactionAmt" END)::numeric, 2)      AS max_fraud_amount
                FROM fraud_transactions
            """)).fetchone()

        return StatsResponse(
            total_transactions=result[0],
            total_fraud=result[1],
            total_legitimate=result[2],
            fraud_percentage=float(result[3]),
            avg_fraud_amount=float(result[4]),
            avg_legitimate_amount=float(result[5]),
            max_fraud_amount=float(result[6]),
        )
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")