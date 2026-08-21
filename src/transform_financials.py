"""Transform raw SEC data into structured financial data."""

import json
from pathlib import Path
import pandas as pd

RAW_DATA_FILE = Path("data/raw/apple_companyfacts.json")

REVENUE_CONCEPT = ( "RevenueFromContractWithCustomerExcludingAssessedTax")
OPERATING_INCOME_CONCEPT = "OperatingIncomeLoss"

def load_company_facts() -> dict:
    """Load Apple's raw SEC Company Facts JSON."""

    with RAW_DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_annual_revenue(company_facts: dict) -> pd.DataFrame:
    """Extract and clean Apple's annual revenue records."""

    records = company_facts["facts"]["us-gaap"][REVENUE_CONCEPT]["units"]["USD"]
    revenue = pd.DataFrame(records)

    revenue = revenue[
        (revenue["form"] == "10-K")
        & (revenue["fp"] == "FY")]

    revenue = (
        revenue.sort_values("filed")
        .drop_duplicates(subset="end", keep="last")
        .sort_values("end")
    )

    revenue["revenue_billions"] = revenue["val"] / 1_000_000_000

    return revenue[["end", "revenue_billions", "filed", "form", "accn"]]

def extract_annual_operating_income(
    company_facts: dict,
) -> pd.DataFrame:
    """Extract and clean Apple's annual operating income."""

    records = company_facts["facts"]["us-gaap"][OPERATING_INCOME_CONCEPT]["units"]["USD"]

    operating_income = pd.DataFrame(records)

    operating_income = operating_income[
        (operating_income["form"] == "10-K")
        & (operating_income["fp"] == "FY")]

    operating_income = (
        operating_income.sort_values("filed")
        .drop_duplicates(subset="end", keep="last")
        .sort_values("end")
    )

    operating_income["operating_income_billions"] = (operating_income["val"] / 1_000_000_000)

    return operating_income[["end", "operating_income_billions"]]

def main():
    """Run the financial transformation pipeline."""
    company_facts = load_company_facts()
    annual_revenue = extract_annual_revenue(company_facts)
    annual_operating_income = extract_annual_operating_income(company_facts)

    financial_summary = annual_revenue.merge(annual_operating_income, on="end",)
    financial_summary["operating_margin_pct"] = (financial_summary["operating_income_billions"]/ financial_summary["revenue_billions"]* 100)

    print("\nApple financial summary:")
    print(financial_summary[
        ["end", "revenue_billions", "operating_income_billions", "operating_margin_pct"]].tail(5).round(2).to_string(index=False))


if __name__ == "__main__":
    main()