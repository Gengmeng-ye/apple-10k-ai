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

# SEC Company Facts API endpoint for Apple
SEC_URL = (
    f"https://data.sec.gov/api/xbrl/companyfacts/"
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
        SEC_URL,
        headers=HEADERS,
        timeout=30,
    )

    # Raise an error if the request was unsuccessful
    response.raise_for_status()

    apple_data = response.json()

    # Create the raw data directory if it does not exist
    output_folder = Path("data/raw")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define the output file path
    output_file = output_folder / "apple_companyfacts.json"

    # Save the API response as a formatted JSON file
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            apple_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Download completed successfully!")
    print(f"Company: {apple_data.get('entityName')}")
    print(f"CIK: {apple_data.get('cik')}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    download_apple_financial_data()