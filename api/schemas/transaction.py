"""
Pydantic schemas for request/response validation.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Single transaction input for fraud scoring."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "TransactionAmt": 150.00,
                "ProductCD": 1,
                "card1": 9500,
                "card2": 111.0,
                "card3": 150.0,
                "card5": 226.0,
                "addr1": 299.0,
                "addr2": 87.0,
                "dist1": 0.0,
                "TransactionDT": 86400,
                "P_emaildomain": 0,
                "R_emaildomain": 0,
            }
        }
    )

    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: Optional[int] = Field(0, description="Product code")
    card1: Optional[int] = Field(0, description="Card parameter 1")
    card2: Optional[float] = Field(0.0, description="Card parameter 2")
    card3: Optional[float] = Field(0.0, description="Card parameter 3")
    card5: Optional[float] = Field(0.0, description="Card parameter 5")
    addr1: Optional[float] = Field(0.0, description="Address parameter 1")
    addr2: Optional[float] = Field(0.0, description="Address parameter 2")
    dist1: Optional[float] = Field(0.0, description="Distance parameter 1")
    TransactionDT: Optional[int] = Field(86400, description="Transaction datetime offset")
    P_emaildomain: Optional[int] = Field(0, description="Purchaser email domain")
    R_emaildomain: Optional[int] = Field(0, description="Recipient email domain")


class TransactionResponse(BaseModel):
    """Fraud scoring result for a single transaction."""

    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Fraud decision based on threshold")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    threshold_used: float = Field(..., description="Decision threshold applied")


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for bulk scoring."""

    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of transactions to score"
    )
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Custom decision threshold")


class BatchTransactionResponse(BaseModel):
    """Batch fraud scoring results."""

    total: int
    fraud_count: int
    legitimate_count: int
    results: list[TransactionResponse]


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_loaded: bool
    database_connected: bool
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    """Live fraud statistics from database."""

    total_transactions: int
    total_fraud: int
    total_legitimate: int
    fraud_percentage: float
    avg_fraud_amount: float
    avg_legitimate_amount: float
    max_fraud_amount: float