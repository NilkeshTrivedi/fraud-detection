"""
Run model evaluation on test set.
"""

import logging
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from etl.config import DatabaseConfig, ETLConfig
from etl.loader import get_engine
from models.evaluator import evaluate_model
from models.trainer import prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODELS_DIR = __import__("pathlib").Path("models/artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def run_evaluation() -> None:
    """Load model and evaluate on held-out test set."""
    logger.info("=" * 60)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 60)

    # Load data
    db_config = DatabaseConfig()
    etl_config = ETLConfig()
    engine = get_engine(db_config)

    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {etl_config.table_name}"), conn)

    # Prepare features
    X, y = prepare_features(df)

    # Recreate same test split
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Load scaler and model
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    model = joblib.load(MODELS_DIR / "xgb_fraud_model.pkl")

    X_test_scaled = scaler.transform(X_test)

    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test.values)

    logger.info("=" * 60)
    logger.info("Evaluation Complete ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_evaluation()