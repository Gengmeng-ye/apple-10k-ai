"""Transform Apple SEC data into structured financial metrics."""

import json
from pathlib import Path

import pandas as pd


RAW_DATA_FILE = Path("data/raw/apple_companyfacts.json")
PROCESSED_DATA_FILE = Path("data/processed/apple_financial_summary.parquet")


# Map output columns to SEC US-GAAP concepts
FINANCIAL_CONCEPTS = {
    "revenue_billions": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "operating_income_billions": "OperatingIncomeLoss",
    "net_income_billions": "NetIncomeLoss",
    "operating_cash_flow_billions": "NetCashProvidedByUsedInOperatingActivities",
}


def load_company_facts() -> dict:
    """Load Apple's raw SEC Company Facts JSON."""

    with RAW_DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_annual_metric(
    company_facts: dict,
    concept_name: str,
    column_name: str,
) -> pd.DataFrame:
    """Extract and clean one annual financial metric."""

    records = company_facts["facts"]["us-gaap"][concept_name]["units"]["USD"]
    data = pd.DataFrame(records)

    # Keep annual records from 10-K filings
    data = data[(data["form"] == "10-K") & (data["fp"] == "FY")]

    # Keep the latest filing for each fiscal year
    data = data.sort_values("filed")
    data = data.drop_duplicates(subset="end", keep="last")
    data = data.sort_values("end")

    # Convert dollars to billions of dollars
    data[column_name] = data["val"] / 1_000_000_000

    return data[["end", column_name]]


def build_financial_summary(company_facts: dict) -> pd.DataFrame:
    """Build Apple's annual financial summary."""

    financial_tables = []

    for column_name, concept_name in FINANCIAL_CONCEPTS.items():
        table = extract_annual_metric(
            company_facts,
            concept_name,
            column_name,
        )
        financial_tables.append(table)

    financial_summary = financial_tables[0]

    for table in financial_tables[1:]:
        financial_summary = financial_summary.merge(table, on="end")

    financial_summary["operating_margin_pct"] = (
        financial_summary["operating_income_billions"]
        / financial_summary["revenue_billions"]
        * 100
    )

    financial_summary["net_profit_margin_pct"] = (
        financial_summary["net_income_billions"]
        / financial_summary["revenue_billions"]
        * 100
    )

    return financial_summary


def save_to_parquet(financial_summary: pd.DataFrame) -> None:
    """Save processed financial data as Parquet."""

    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    financial_summary.to_parquet(PROCESSED_DATA_FILE, index=False)

    print(f"\nSaved to Parquet: {PROCESSED_DATA_FILE}")


def main() -> None:
    """Run the financial transformation process."""

    company_facts = load_company_facts()
    financial_summary = build_financial_summary(company_facts)

    print("\nApple financial summary:")
    print(financial_summary.tail(5).round(2).to_string(index=False))
    print(f"\nTotal rows: {len(financial_summary)}")

    save_to_parquet(financial_summary)


if __name__ == "__main__":
    main()