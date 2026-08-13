import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Read the SEC user identity from the environment
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

if not SEC_USER_AGENT:
    raise ValueError(
        "SEC_USER_AGENT is missing. Add your name and email to the .env file."
    )


# Apple's Central Index Key assigned by the SEC
APPLE_CIK = "0000320193"

# SEC API endpoints
COMPANY_FACTS_URL = (
    f"https://data.sec.gov/api/xbrl/companyfacts/"
    f"CIK{APPLE_CIK}.json"
)

SUBMISSIONS_URL = (
    f"https://data.sec.gov/submissions/"
    f"CIK{APPLE_CIK}.json"
)

# Identify the application when requesting data from the SEC
HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


def download_apple_financial_data():
    """Download Apple's financial facts from the SEC and save them as JSON."""

    print("Downloading Apple financial data from the SEC...")

    response = requests.get(
        COMPANY_FACTS_URL,
        headers=HEADERS,
        timeout=30,
    )

    # Raise an error if the request was unsuccessful
    response.raise_for_status()

    apple_data = response.json()

    # Create the raw data directory if it does not exist
    output_folder = Path("data/raw")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "apple_companyfacts.json"

    # Save the API response as a formatted JSON file
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            apple_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Company Facts download completed successfully!")
    print(f"Company: {apple_data.get('entityName')}")
    print(f"CIK: {apple_data.get('cik')}")
    print(f"Saved to: {output_file}")


def find_latest_10k():
    """Find Apple's latest 10-K filing in the SEC submissions data."""

    print("\nSearching for Apple's latest 10-K filing...")

    response = requests.get(
        SUBMISSIONS_URL,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    submissions_data = response.json()
    recent_filings = submissions_data["filings"]["recent"]

    forms = recent_filings["form"]
    accession_numbers = recent_filings["accessionNumber"]
    primary_documents = recent_filings["primaryDocument"]
    filing_dates = recent_filings["filingDate"]
    report_dates = recent_filings["reportDate"]

    # Search through recent filings and select the first 10-K
    for index, form in enumerate(forms):
        if form == "10-K":
            filing_information = {
                "form": form,
                "accession_number": accession_numbers[index],
                "primary_document": primary_documents[index],
                "filing_date": filing_dates[index],
                "report_date": report_dates[index],
            }

            print("Latest 10-K found!")
            print(f"Filing date: {filing_information['filing_date']}")
            print(f"Report date: {filing_information['report_date']}")

            return filing_information

    raise ValueError("No 10-K filing was found in Apple's recent SEC filings.")


def download_latest_10k(filing_information):
    """Download Apple's latest 10-K HTML document from the SEC."""

    accession_number = filing_information["accession_number"]
    primary_document = filing_information["primary_document"]

    # SEC archive URLs use the accession number without hyphens
    accession_number_clean = accession_number.replace("-", "")
    cik_without_leading_zeros = str(int(APPLE_CIK))

    filing_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_without_leading_zeros}/"
        f"{accession_number_clean}/"
        f"{primary_document}"
    )

    print("\nDownloading Apple's latest 10-K HTML document...")

    response = requests.get(
        filing_url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    # Create the filings directory if it does not exist
    output_folder = Path("data/filings")
    output_folder.mkdir(parents=True, exist_ok=True)

    html_output_file = output_folder / "apple_latest_10k.html"
    metadata_output_file = output_folder / "apple_latest_10k_metadata.json"

    # Save the original 10-K HTML document
    with html_output_file.open("w", encoding="utf-8") as file:
        file.write(response.text)

    # Save the filing metadata and original SEC source URL
    filing_metadata = {
        "company": "Apple Inc.",
        "cik": APPLE_CIK,
        "form": filing_information["form"],
        "filing_date": filing_information["filing_date"],
        "report_date": filing_information["report_date"],
        "accession_number": accession_number,
        "primary_document": primary_document,
        "source_url": filing_url,
    }

    with metadata_output_file.open("w", encoding="utf-8") as file:
        json.dump(
            filing_metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("10-K download completed successfully!")
    print(f"HTML saved to: {html_output_file}")
    print(f"Metadata saved to: {metadata_output_file}")
    print(f"SEC source: {filing_url}")


def main():
    """Run the SEC data extraction pipeline."""

    download_apple_financial_data()

    latest_10k = find_latest_10k()
    download_latest_10k(latest_10k)

    print("\nAll SEC extraction tasks completed successfully!")


if __name__ == "__main__":
    main()