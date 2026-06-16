"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)


# ── Mock predictor ────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock model and DB dependencies for API tests."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        "fraud_probability": 0.85,
        "is_fraud": True,
        "risk_level": "HIGH",
        "threshold_used": 0.5,
    }
    mock_predictor.threshold = 0.5

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("api.dependencies.get_predictor", return_value=mock_predictor), \
         patch("api.dependencies.get_db_engine", return_value=mock_engine):
        yield mock_predictor, mock_engine


# ── Health Tests ──────────────────────────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "database_connected" in data
        assert "version" in data


# ── Predict Tests ─────────────────────────────────────────────────────────────
class TestPredictEndpoint:

    def test_predict_returns_200(self):
        """Predict endpoint should return 200 with valid input."""
        payload = {
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
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_structure(self):
        """Predict response should have required fields."""
        payload = {"TransactionAmt": 150.00}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "threshold_used" in data

    def test_predict_invalid_amount(self):
        """Negative transaction amount should return 422."""
        payload = {"TransactionAmt": -100.00}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_amount(self):
        """Missing required field should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ── Batch Predict Tests ───────────────────────────────────────────────────────
class TestBatchPredictEndpoint:

    def test_batch_predict_returns_200(self):
        """Batch predict should return 200."""
        payload = {
            "transactions": [
                {"TransactionAmt": 150.00},
                {"TransactionAmt": 500.00},
            ],
            "threshold": 0.5,
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_predict_response_counts(self):
        """Batch response should have correct total count."""
        payload = {
            "transactions": [
                {"TransactionAmt": 150.00},
                {"TransactionAmt": 500.00},
                {"TransactionAmt": 75.00},
            ],
            "threshold": 0.5,
        }
        response = client.post("/predict/batch", json=payload)
        data = response.json()
        assert data["total"] == 3