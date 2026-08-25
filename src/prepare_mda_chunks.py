"""Split Apple's MD&A into sentence-based retrieval chunks."""

import json
import re
from pathlib import Path


MDA_FILE = Path("data/processed/apple_mda.txt")
CHUNKS_FILE = Path("data/processed/apple_mda_chunks.json")

MAX_CHARACTERS = 1500
OVERLAP_SENTENCES = 1


def load_mda() -> str:
    """Load the extracted MD&A text."""

    return MDA_FILE.read_text(encoding="utf-8")


def split_into_chunks(
    text: str,
    max_characters: int = MAX_CHARACTERS,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[str]:
    """Split text into sentence-based chunks with a small overlap."""

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        candidate = " ".join(current_chunk + [sentence])

        if current_chunk and len(candidate) > max_characters:
            chunks.append(" ".join(current_chunk))

            overlap = current_chunk[-overlap_sentences:]
            current_chunk = overlap + [sentence]
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def save_chunks(chunks: list[str]) -> None:
    """Save MD&A chunks for retrieval."""

    records = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        records.append(
            {
                "chunk_id": chunk_index,
                "section": "Item 7. MD&A",
                "text": chunk,
            }
        )

    CHUNKS_FILE.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Load, split, and save the MD&A text."""

    text = load_mda()
    chunks = split_into_chunks(text)
    save_chunks(chunks)

    print(f"MD&A characters: {len(text):,}")
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0]):,}")
    print(f"Chunks saved to: {CHUNKS_FILE}")
    print("\nFirst chunk preview:")
    print(chunks[0][:500])


if __name__ == "__main__":
    main()