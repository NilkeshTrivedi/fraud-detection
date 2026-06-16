"""
Model evaluation module — computes fraud detection metrics and reports.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/artifacts")


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model performance with fraud-specific metrics.

    Args:
        model: Trained XGBClassifier.
        X_test: Test features.
        y_test: True test labels.
        threshold: Decision threshold for fraud classification.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Evaluating model...")

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "average_precision": round(avg_precision, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fraud_catch_rate": round(tp / (tp + fn) * 100, 2),
        "false_alarm_rate": round(fp / (fp + tn) * 100, 2),
    }

    logger.info("=" * 50)
    logger.info(f"AUC-ROC Score      : {metrics['auc_roc']}")
    logger.info(f"Average Precision  : {metrics['average_precision']}")
    logger.info(f"Fraud Catch Rate   : {metrics['fraud_catch_rate']}%")
    logger.info(f"False Alarm Rate   : {metrics['false_alarm_rate']}%")
    logger.info(f"True Positives     : {metrics['true_positives']:,}")
    logger.info(f"False Negatives    : {metrics['false_negatives']:,}")
    logger.info("=" * 50)
    logger.info(f"\nClassification Report:\n{report}")

    # Save metrics
    joblib.dump(metrics, MODELS_DIR / "metrics.pkl")
    logger.info("Metrics saved ✅")

    return metrics