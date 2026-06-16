"""
Tests for ETL pipeline modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from etl.transformer import (
    drop_high_null_columns,
    fill_missing_values,
    engineer_features,
    validate_data,
)
from etl.extractor import merge_datasets


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_transaction_df() -> pd.DataFrame:
    """Create sample transaction dataframe for testing."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionAmt": [100.0, 250.0, 50.0, 1000.0, 75.0],
        "TransactionDT": [86400, 172800, 259200, 345600, 432000],
        "isFraud": [0, 1, 0, 1, 0],
        "ProductCD": [1, 2, 1, 3, 1],
        "high_null_col": [None, None, None, None, 1.0],
    })


@pytest.fixture
def sample_identity_df() -> pd.DataFrame:
    """Create sample identity dataframe for testing."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "id_01": [-5.0, -10.0, 0.0],
        "id_02": [70.0, 80.0, 90.0],
    })


# ── Extractor Tests ───────────────────────────────────────────────────────────
class TestExtractor:

    def test_merge_datasets_shape(self, sample_transaction_df, sample_identity_df):
        """Merged df should have all transaction rows."""
        merged = merge_datasets(sample_transaction_df, sample_identity_df)
        assert len(merged) == len(sample_transaction_df)

    def test_merge_datasets_left_join(self, sample_transaction_df, sample_identity_df):
        """Left join should preserve all transaction rows."""
        merged = merge_datasets(sample_transaction_df, sample_identity_df)
        assert merged["TransactionID"].nunique() == len(sample_transaction_df)

    def test_merge_identity_nulls_for_unmatched(self, sample_transaction_df, sample_identity_df):
        """Transactions without identity should have null identity columns."""
        merged = merge_datasets(sample_transaction_df, sample_identity_df)
        unmatched = merged[~merged["TransactionID"].isin(sample_identity_df["TransactionID"])]
        assert unmatched["id_01"].isnull().all()


# ── Transformer Tests ─────────────────────────────────────────────────────────
class TestTransformer:

    def test_drop_high_null_columns(self, sample_transaction_df):
        """Columns exceeding null threshold should be dropped."""
        result = drop_high_null_columns(sample_transaction_df, threshold=0.5)
        assert "high_null_col" not in result.columns

    def test_drop_high_null_keeps_valid_columns(self, sample_transaction_df):
        """Valid columns should be retained after null drop."""
        result = drop_high_null_columns(sample_transaction_df, threshold=0.5)
        assert "TransactionAmt" in result.columns
        assert "isFraud" in result.columns

    def test_fill_missing_values_no_nulls(self, sample_transaction_df):
        """No nulls should remain after filling."""
        df = drop_high_null_columns(sample_transaction_df, threshold=0.5)
        df.loc[0, "TransactionAmt"] = None
        result = fill_missing_values(df)
        assert result.isnull().sum().sum() == 0

    def test_engineer_features_adds_columns(self, sample_transaction_df):
        """Feature engineering should add expected columns."""
        df = drop_high_null_columns(sample_transaction_df, threshold=0.5)
        df = fill_missing_values(df)
        result = engineer_features(df)
        assert "log_transaction_amt" in result.columns
        assert "transaction_hour" in result.columns
        assert "is_weekend" in result.columns

    def test_validate_data_passes(self, sample_transaction_df):
        """Validation should pass on clean data."""
        df = drop_high_null_columns(sample_transaction_df, threshold=0.5)
        df = fill_missing_values(df)
        df = engineer_features(df)
        # Should not raise
        validate_data(df, "isFraud")

    def test_validate_data_fails_on_nulls(self, sample_transaction_df):
        """Validation should raise on remaining nulls."""
        df = sample_transaction_df.copy()
        with pytest.raises(ValueError):
            validate_data(df, "isFraud")