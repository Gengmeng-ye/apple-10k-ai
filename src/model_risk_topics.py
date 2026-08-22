"""Prepare Apple's Risk Factors for topic analysis."""

import json
import re
from pathlib import Path

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)


RISK_FACTORS_FILE = Path("data/processed/apple_risk_factors.txt")
CHUNKS_FILE = Path("data/processed/apple_risk_chunks.json")

NUMBER_OF_TOPICS = 6
WORDS_PER_TOPIC = 8


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

def build_topic_model(chunks: list[str],):
    """Build an NMF topic model from the risk chunks."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(chunks)

    model = NMF(
        n_components=NUMBER_OF_TOPICS,
        init="nndsvda",
        random_state=42,
        max_iter=500,
    )

    chunk_topic_matrix = model.fit_transform(tfidf_matrix)

    return (model, vectorizer, chunk_topic_matrix,)


def get_weight(word_and_weight):
    """Return the weight from a word-weight pair."""
    return word_and_weight[1]


def print_topic_keywords( model: NMF, vectorizer: TfidfVectorizer,) -> None:
    """Print the most important words for each topic."""
    feature_names = (vectorizer.get_feature_names_out())

    number_of_topics = len(model.components_)

    for topic_index in range(number_of_topics):
        topic_number = topic_index + 1

        topic_weights = model.components_[topic_index]

        words_with_weights = []

        for word_index in range(len(feature_names)):
            word = feature_names[word_index]
            weight = topic_weights[word_index]

            words_with_weights.append((word, weight))

        words_with_weights.sort(
            key=get_weight,
            reverse=True,
        )

        top_words_with_weights = (words_with_weights[:WORDS_PER_TOPIC])

        keywords = []

        for word, weight in top_words_with_weights:
            keywords.append(word)

        print(
            f"Topic {topic_number}: "
            f"{', '.join(keywords)}"
        )


def print_representative_chunks(chunks: list[str], chunk_topic_matrix,) -> None:
    """Print the strongest chunk for each topic."""
    print("\nRepresentative chunks:")

    for topic_index in range(NUMBER_OF_TOPICS):
        topic_scores = chunk_topic_matrix[:,topic_index,]

        best_chunk_index = (topic_scores.argmax())

        best_score = topic_scores[best_chunk_index]

        print(
            f"\nTopic {topic_index + 1} "
            f"| Chunk {best_chunk_index + 1} "
            f"| Score: {best_score:.3f}"
        )

        print(chunks[best_chunk_index][:500])


def main() -> None:
    """Load, split, and save Apple's Risk Factors text."""
    text = load_risk_factors()
    chunks = split_into_chunks(text)

    save_chunks(chunks)

    model, vectorizer, chunk_topic_matrix = (build_topic_model(chunks))
    print("\nDiscovered risk topics:")

    print_topic_keywords(model, vectorizer,)
    print_representative_chunks(chunks, chunk_topic_matrix,)


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