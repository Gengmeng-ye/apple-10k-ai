"""Validate Apple's processed financial data."""

from pathlib import Path

import duckdb
import pandas as pd


DATABASE_FILE = Path("warehouse/apple_finance.duckdb")

REQUIRED_COLUMNS = {
    "end",
    "revenue_billions",
    "operating_income_billions",
    "net_income_billions",
    "operating_cash_flow_billions",
    "total_assets_billions",
    "total_liabilities_billions",
    "cash_and_cash_equivalents_billions",
    "operating_margin_pct",
    "net_profit_margin_pct",
}


def load_financial_data() -> pd.DataFrame:
    """Load financial data from DuckDB."""
    connection = duckdb.connect(str(DATABASE_FILE), read_only=True)

    data = connection.execute(
        "SELECT * FROM apple_financial_summary").fetchdf()
    connection.close()
    return data


def validate_financial_data(data: pd.DataFrame) -> None:
    """Run basic data quality checks."""
    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if data.empty:
        raise ValueError("Financial data is empty.")

    if data.isnull().any().any():
        raise ValueError("Financial data contains missing values.")

    if data["end"].duplicated().any():
        raise ValueError("Financial data contains duplicate fiscal years.")

    if (data["revenue_billions"] <= 0).any():
        raise ValueError("Revenue must be positive.")

    positive_balance_sheet_columns = {
        "total_assets_billions",
        "total_liabilities_billions",
        "cash_and_cash_equivalents_billions",
    }

    for column in positive_balance_sheet_columns:
        if (data[column] <= 0).any():
            raise ValueError(f"{column} must be positive.")

    if not data["operating_margin_pct"].between(0, 100).all():
        raise ValueError("Operating margin is outside the expected range.")

    print("All data quality checks passed.")


def main() -> None:
    """Load and validate the financial data."""
    data = load_financial_data()
    validate_financial_data(data)


if __name__ == "__main__":
    main()
