"""Tests for financial data validation."""

import pandas as pd
import pytest

from src.validate_data import validate_financial_data


def create_valid_data() -> pd.DataFrame:
    """Create a small valid financial dataset for testing."""
    return pd.DataFrame(
        {
            "end": ["2024-09-28", "2025-09-27"],
            "revenue_billions": [391.04, 416.16],
            "operating_income_billions": [123.22, 133.05],
            "net_income_billions": [93.74, 112.01],
            "operating_cash_flow_billions": [118.25, 111.48],
            "operating_margin_pct": [31.51, 31.97],
            "net_profit_margin_pct": [23.97, 26.92],
        }
    )


def test_valid_financial_data() -> None:
    """Valid financial data should pass validation."""
    data = create_valid_data()

    validate_financial_data(data)


def test_negative_revenue_is_rejected() -> None:
    """Negative revenue should fail validation."""
    data = create_valid_data()
    data.loc[0, "revenue_billions"] = -1

    with pytest.raises(
        ValueError,
        match="Revenue must be positive",
    ):
        validate_financial_data(data)