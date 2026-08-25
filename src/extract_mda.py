"""Extract Item 7 MD&A from Apple's latest 10-K."""

import re
from pathlib import Path

from src.extract_risk_factors import load_filing_text


OUTPUT_FILE = Path("data/processed/apple_mda.txt")

START_PATTERN = (
    r"Item\s+7\.\s+Management.s Discussion and Analysis "
    r"of Financial Condition and Results of Operations"
)

END_PATTERN = (
    r"Item\s+7A\.\s+Quantitative and Qualitative "
    r"Disclosures About Market Risk"
)


def find_mda_start(text: str) -> re.Match[str]:
    """Find the real Item 7 heading instead of the table of contents."""

    matches = re.finditer(
        START_PATTERN,
        text,
        flags=re.IGNORECASE,
    )

    for match in matches:
        preview = text[match.end():match.end() + 500].lower()

        if "the following discussion should be read in conjunction" in preview:
            return match

    raise ValueError("Could not find the MD&A section.")


def extract_mda(text: str) -> str:
    """Extract Item 7 MD&A up to Item 7A."""

    start_match = find_mda_start(text)
    remaining_text = text[start_match.end():]

    end_match = re.search(
        END_PATTERN,
        remaining_text,
        flags=re.IGNORECASE,
    )

    if end_match is None:
        raise ValueError("Could not find the end of the MD&A section.")

    start_position = start_match.start()
    end_position = start_match.end() + end_match.start()

    return text[start_position:end_position].strip()


def main() -> None:
    """Extract and save Apple's MD&A section."""

    text = load_filing_text()
    mda = extract_mda(text)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        mda,
        encoding="utf-8",
    )

    print(f"MD&A saved to: {OUTPUT_FILE}")
    print(f"Total characters: {len(mda):,}")


if __name__ == "__main__":
    main()