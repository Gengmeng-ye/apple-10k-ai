"""Load processed Apple financial data into DuckDB."""

from pathlib import Path

import duckdb
import pandas as pd


PROCESSED_DATA_FILE = Path(
    "data/processed/apple_financial_summary.parquet"
)

DATABASE_FILE = Path(
    "warehouse/apple_finance.duckdb"
)

TABLE_NAME = "apple_financial_summary"


def load_processed_data() -> pd.DataFrame:
    """Load processed financial data from Parquet."""

    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {PROCESSED_DATA_FILE}"
        )

    return pd.read_parquet(PROCESSED_DATA_FILE)


def save_to_duckdb(financial_summary: pd.DataFrame) -> None:
    """Load processed financial data into DuckDB."""

    DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = duckdb.connect(
        str(DATABASE_FILE)
    )

    try:
        connection.register(
            "financial_summary_df",
            financial_summary,
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {TABLE_NAME} AS
            SELECT * FROM financial_summary_df
            """
        )
    finally:
        connection.close()

    print(
        f"Loaded {len(financial_summary)} rows "
        f"into DuckDB table: {TABLE_NAME}"
    )

    print(
        f"DuckDB database: {DATABASE_FILE}"
    )


def main() -> None:
    """Run the warehouse loading process."""

    financial_summary = load_processed_data()
    save_to_duckdb(financial_summary)


if __name__ == "__main__":
    main()