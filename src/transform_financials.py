"""Transform raw SEC data into structured financial data."""

import json
from pathlib import Path
import pandas as pd

RAW_DATA_FILE = Path("data/raw/apple_companyfacts.json")

REVENUE_CONCEPT = ( "RevenueFromContractWithCustomerExcludingAssessedTax")

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
        & (revenue["fp"] == "FY")
    ]

    revenue = (
        revenue.sort_values("filed")
        .drop_duplicates(subset="end", keep="last")
        .sort_values("end")
    )

    revenue["revenue_billions"] = revenue["val"] / 1_000_000_000

    return revenue[
        ["end", "revenue_billions", "filed", "form", "accn"]
    ]


def main():
    """Run the financial transformation pipeline."""

    company_facts = load_company_facts()
    annual_revenue = extract_annual_revenue(company_facts)

    print("\nApple annual revenue:")
    print(annual_revenue.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()