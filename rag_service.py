"""Retrieve evidence from Apple's 10-K Risk Factors."""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNKS_FILE = Path("data/processed/apple_risk_chunks.json")


def load_risk_chunks() -> list[dict]:
    """Load the processed Risk Factors chunks."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Risk chunks file not found: {CHUNKS_FILE}"
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_evidence(
    question: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Retrieve the most relevant chunks for a question."""

    chunk_texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    chunk_matrix = vectorizer.fit_transform(chunk_texts)
    question_vector = vectorizer.transform([question])

    similarity_scores = cosine_similarity(
        question_vector,
        chunk_matrix,
    ).flatten()

    ranked_results = sorted(
        enumerate(similarity_scores),
        key=lambda item: item[1],
        reverse=True,
    )

    evidence = []

    for chunk_index, score in ranked_results[:top_k]:
        result = chunks[chunk_index].copy()
        result["retrieval_score"] = round(float(score), 4)
        evidence.append(result)

    return evidence


def main() -> None:
    """Test the Risk Factors retrieval process."""

    question = "What are Apple's supply chain risks?"
    chunks = load_risk_chunks()
    evidence = retrieve_evidence(question, chunks)

    print(f"\nQuestion: {question}")
    print("\nRetrieved evidence:")

    for result in evidence:
        print(
            f"\nChunk {result['chunk_id']} "
            f"| {result['topic_label']} "
            f"| Score: {result['retrieval_score']}"
        )
        print(result["text"][:500])


if __name__ == "__main__":
    main()