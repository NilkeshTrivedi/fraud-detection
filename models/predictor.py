"""
Predictor module — loads trained model and scores new transactions.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/artifacts")


class FraudPredictor:
    """
    Loads trained fraud detection model and provides prediction interface.
    """

    def __init__(self) -> None:
        """Load model artifacts from disk."""
        logger.info("Loading fraud detection model...")

        self.model = joblib.load(MODELS_DIR / "xgb_fraud_model.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.feature_names: list[str] = joblib.load(MODELS_DIR / "feature_names.pkl")
        self.threshold: float = 0.5

        logger.info("Model loaded ✅")

    def predict(self, transaction: dict) -> dict:
        """
        Score a single transaction for fraud probability.

        Args:
            transaction: Dictionary of transaction features.

        Returns:
            Dictionary with fraud_probability, is_fraud, risk_level.

        Raises:
            ValueError: If required features are missing.
        """
        # Build feature vector
        input_df = pd.DataFrame([transaction])

        # Align to training features
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[self.feature_names]

        # Scale
        input_scaled = self.scaler.transform(input_df)

        # Predict
        fraud_prob = float(self.model.predict_proba(input_scaled)[0][1])
        is_fraud = fraud_prob >= self.threshold

        # Risk level
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.6:
            risk_level = "MEDIUM"
        elif fraud_prob < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": bool(is_fraud),
            "risk_level": risk_level,
            "threshold_used": self.threshold,
        }