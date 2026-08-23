"""Extract Item 1A Risk Factors from Apple's latest 10-K."""

import re
from pathlib import Path

from bs4 import BeautifulSoup


FILING_FILE = Path("data/filings/apple_latest_10k.html")
OUTPUT_FILE = Path("data/processed/apple_risk_factors.txt")

START_PATTERN = r"Item\s+1A\.\s+Risk Factors"
END_PATTERN = r"Item\s+1B\.\s+Unresolved Staff Comments"

SPACING_FIXES = {
    "resultsof": "results of",
    "andthe": "and the",
    "canbe": "can be",
    "adverselyaffect": "adversely affect",
    "timeand": "time and",
}


def clean_filing_text(text: str) -> str:
    """Normalize whitespace and fix known SEC formatting artifacts."""

    text = re.sub(r"\s+", " ", text)

    for incorrect, corrected in SPACING_FIXES.items():
        text = re.sub(
            rf"\b{re.escape(incorrect)}\b",
            corrected,
            text,
            flags=re.IGNORECASE,
        )

    return text.strip()


def load_filing_text() -> str:
    """Load the 10-K HTML and convert it to clean plain text."""

    html = FILING_FILE.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    return clean_filing_text(text)


def find_risk_factors_start(text: str) -> re.Match[str]:
    """Find the real Item 1A heading instead of the table of contents."""

    matches = re.finditer(
        START_PATTERN,
        text,
        flags=re.IGNORECASE,
    )

    for match in matches:
        preview = text[match.end():match.end() + 500].lower()

        if "the following summarizes factors" in preview:
            return match

    raise ValueError("Could not find the Risk Factors section.")


def extract_risk_factors(text: str) -> str:
    """Extract text from Item 1A up to Item 1B."""

    start_match = find_risk_factors_start(text)
    remaining_text = text[start_match.end():]

    end_match = re.search(
        END_PATTERN,
        remaining_text,
        flags=re.IGNORECASE,
    )

    if end_match is None:
        raise ValueError(
            "Could not find the end of the Risk Factors section."
        )

    start_position = start_match.start()
    end_position = start_match.end() + end_match.start()

    return text[start_position:end_position].strip()


def main() -> None:
    """Extract and save Apple's Risk Factors section."""

    text = load_filing_text()
    risk_factors = extract_risk_factors(text)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        risk_factors,
        encoding="utf-8",
    )

    print(f"Risk Factors saved to: {OUTPUT_FILE}")
    print(f"Total characters: {len(risk_factors):,}")


if __name__ == "__main__":
    main()