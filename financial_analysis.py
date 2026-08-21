"""Prepare Apple financial data for analysis."""

from pathlib import Path

import duckdb
import pandas as pd


DATABASE_FILE = Path("warehouse/apple_finance.duckdb")


def load_financial_data(years: int = 5) -> pd.DataFrame:
    """Load the latest financial data from DuckDB."""
    connection = duckdb.connect(str(DATABASE_FILE), read_only=True)

    data = connection.execute(
        """
        SELECT *
        FROM apple_financial_summary
        ORDER BY "end"
        """
    ).fetchdf()

    connection.close()
    return data.tail(years).reset_index(drop=True)


def calculate_financial_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate financial growth metrics."""
    data = data.copy()

    data["revenue_growth_pct"] = (data["revenue_billions"].pct_change() * 100)
    return data


def main() -> None:
    """Run the financial analysis."""
    data = load_financial_data()
    data = calculate_financial_metrics(data)

    print("\nApple financial analysis:")
    print(data.round(2).to_string(index=False))


if __name__ == "__main__":
    main()