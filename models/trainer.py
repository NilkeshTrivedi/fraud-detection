"""
Model training module — loads processed data, applies SMOTE,
trains XGBoost fraud detection model, and saves to disk.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier

from etl.config import DatabaseConfig, ETLConfig
from etl.loader import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models/artifacts")
TARGET = "isFraud"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_SAMPLING = 0.3  # bring minority to 30% of majority


def load_data_from_db(db_config: DatabaseConfig, etl_config: ETLConfig) -> pd.DataFrame:
    """
    Load processed fraud data from PostgreSQL.

    Args:
        db_config: DatabaseConfig instance.
        etl_config: ETLConfig instance.

    Returns:
        DataFrame with all processed transactions.
    """
    logger.info("Loading data from PostgreSQL...")
    engine = get_engine(db_config)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(f"SELECT * FROM {etl_config.table_name}"),
            conn,
        )

    logger.info(f"Loaded {len(df):,} rows from database")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target, drop non-ML columns.

    Args:
        df: Full dataframe.

    Returns:
        Tuple of (X features, y target).
    """
    logger.info("Preparing features...")

    # Columns to drop — IDs and non-predictive
    drop_cols = ["TransactionID", TARGET]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[TARGET]

    # Keep only numeric columns for model
    X = X.select_dtypes(include=[np.number])

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    return X, y


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to handle class imbalance.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Resampled (X_resampled, y_resampled).
    """
    logger.info(f"Applying SMOTE — before: {dict(y_train.value_counts())}")

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING,
        random_state=RANDOM_STATE,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    unique, counts = np.unique(y_resampled, return_counts=True)
    logger.info(f"After SMOTE: {dict(zip(unique, counts))}")

    return X_resampled, y_resampled


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """
    Train XGBoost fraud detection model.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained XGBClassifier.
    """
    logger.info("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # handled by SMOTE already
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",  # faster training
    )

    model.fit(
        X_train,
        y_train,
        verbose=True,
    )

    logger.info("Model training complete ✅")
    return model


def save_artifacts(
    model: XGBClassifier,
    scaler: StandardScaler,
    feature_names: list[str],
) -> None:
    """
    Save model, scaler, and feature names to disk.

    Args:
        model: Trained XGBClassifier.
        scaler: Fitted StandardScaler.
        feature_names: List of feature column names.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "xgb_fraud_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    logger.info(f"Artifacts saved to {MODELS_DIR} ✅")


def run_training() -> tuple[XGBClassifier, StandardScaler, list, dict]:
    """
    Orchestrate full model training pipeline.

    Returns:
        Tuple of (model, scaler, feature_names, data_splits)
    """
    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)

    db_config = DatabaseConfig()
    etl_config = ETLConfig()

    # Load
    df = load_data_from_db(db_config, etl_config)

    # Prepare
    X, y = prepare_features(df)
    feature_names = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)

    # Train
    model = train_model(X_train_resampled, y_train_resampled)

    # Save
    save_artifacts(model, scaler, feature_names)

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete ✅")
    logger.info("=" * 60)

    return model, scaler, feature_names, {
        "X_test": X_test_scaled,
        "y_test": y_test,
    }


if __name__ == "__main__":
    run_training()