"""
Fraud prediction endpoints — single and batch scoring.
"""

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_predictor
from api.schemas.transaction import (
    BatchTransactionRequest,
    BatchTransactionResponse,
    TransactionRequest,
    TransactionResponse,
)
from models.predictor import FraudPredictor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=TransactionResponse, tags=["Prediction"])
def predict_fraud(
    transaction: TransactionRequest,
    predictor: FraudPredictor = Depends(get_predictor),
) -> TransactionResponse:
    """
    Score a single transaction for fraud probability.

    Returns fraud probability, decision, and risk level.
    """
    try:
        result = predictor.predict(transaction.model_dump())
        return TransactionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchTransactionResponse, tags=["Prediction"])
def predict_fraud_batch(
    request: BatchTransactionRequest,
    predictor: FraudPredictor = Depends(get_predictor),
) -> BatchTransactionResponse:
    """
    Score multiple transactions in one request.

    Accepts up to 1000 transactions per batch.
    """
    try:
        results = []
        for txn in request.transactions:
            txn_dict = txn.model_dump()
            # Apply custom threshold if provided
            predictor.threshold = request.threshold
            result = predictor.predict(txn_dict)
            results.append(TransactionResponse(**result))

        fraud_count = sum(1 for r in results if r.is_fraud)

        return BatchTransactionResponse(
            total=len(results),
            fraud_count=fraud_count,
            legitimate_count=len(results) - fraud_count,
            results=results,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")