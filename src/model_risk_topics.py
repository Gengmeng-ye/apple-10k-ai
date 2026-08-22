"""Prepare Apple's Risk Factors for topic analysis."""

import json
import re
from pathlib import Path


RISK_FACTORS_FILE = Path("data/processed/apple_risk_factors.txt")
CHUNKS_FILE = Path("data/processed/apple_risk_chunks.json")


def load_risk_factors() -> str:
    """Load the extracted Risk Factors text."""
    return RISK_FACTORS_FILE.read_text(encoding="utf-8")


def split_into_chunks(text: str, max_characters: int = 1500,) -> list[str]:
    """Split text into sentence-based chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text,)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        candidate = " ".join(current_chunk + [sentence])

        if (current_chunk and len(candidate) > max_characters):
            chunks.append( " ".join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append( " ".join(current_chunk))

    return chunks


def save_chunks(chunks: list[str]) -> None:
    """Save numbered Risk Factors chunks as JSON."""
    records = [
        {
            "chunk_id": index,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1,)
    ]

    CHUNKS_FILE.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Load, split, and save Apple's Risk Factors text."""
    text = load_risk_factors()
    chunks = split_into_chunks(text)

    save_chunks(chunks)

    print(f"Total characters: {len(text):,}")
    print(f"Total chunks: {len(chunks)}")
    print(
        f"First chunk length: "
        f"{len(chunks[0]):,}"
    )
    print(f"Chunks saved to: {CHUNKS_FILE}")

    print("\nFirst chunk preview:")
    print(chunks[0][:500])


if __name__ == "__main__":
    main()