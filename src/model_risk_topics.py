"""Prepare Apple's Risk Factors for topic analysis."""

import json
import itertools
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

TOPIC_LABELS = {
    1: "Cybersecurity and Data Privacy",
    2: "Products and Competition",
    3: "Foreign Exchange",
    4: "Credit, Investment and Tax Risk",
    5: "Legal and Regulatory",
    6: "Trade and Supply Chain",
}

# Anchor terms are used only to attach human-readable labels to the topics NMF
# discovers.  NMF component numbers have no stable semantic meaning, so labels
# must not be tied directly to the component order.
TOPIC_ANCHORS = {
    1: ("information", "attacks", "confidential", "unauthorized", "security"),
    2: ("products", "services", "competition", "software", "intellectual"),
    3: ("foreign", "dollar", "currency", "currencies", "denominated"),
    4: ("tax", "taxes", "receivables", "credit", "investment"),
    5: ("laws", "legal", "regulations", "investigations", "litigation"),
    6: ("tariffs", "trade", "supply", "imports", "partners"),
}

LOW_CONFIDENCE_SHARE = 0.50
MIXED_TOPIC_LABEL = "Other / Mixed Risks"
PERIOD_TOKEN = "<PERIOD>"


def split_sentences(text: str) -> list[str]:
    """Split prose without breaking common filing abbreviations such as U.S."""
    protected = text
    for abbreviation in ("U.S.", "D.C.", "Inc.", "No."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", PERIOD_TOKEN))
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [sentence.replace(PERIOD_TOKEN, ".") for sentence in sentences]


def load_risk_factors() -> str:
    """Load the extracted Risk Factors text."""
    return RISK_FACTORS_FILE.read_text(encoding="utf-8")


def split_into_chunks(text: str, max_characters: int = 1500,) -> list[str]:
    """Split text into sentence-based chunks."""
    # Preserve the original, reproducible chunk boundaries used to fit NMF.
    sentences = re.split(r"(?<=[.!?])\s+", text)

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


def infer_topic_ids(model: NMF, vectorizer: TfidfVectorizer) -> list[int]:
    """Match unordered NMF components to the six human-readable themes."""
    feature_names = vectorizer.get_feature_names_out()
    feature_indexes = {word: index for index, word in enumerate(feature_names)}
    scores = []

    for component in model.components_:
        scores.append({
            topic_id: sum(
                component[feature_indexes[word]]
                for word in anchors
                if word in feature_indexes
            )
            for topic_id, anchors in TOPIC_ANCHORS.items()
        })

    topic_ids = tuple(TOPIC_LABELS)
    return list(max(
        itertools.permutations(topic_ids),
        key=lambda assignment: sum(
            scores[index][topic_id]
            for index, topic_id in enumerate(assignment)
        ),
    ))


def build_topic_excerpt(chunk: str, topic_id: int, max_sentences: int = 4) -> str:
    """Return complete sentences that best express a chunk's assigned theme."""
    cleaned = re.sub(r"Apple Inc\. \| \d{4} Form 10-K \| \d+", "", chunk)
    sentences = split_sentences(cleaned.strip())
    anchors = TOPIC_ANCHORS[topic_id]
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            sum(item[1].lower().count(anchor) for anchor in anchors),
            -item[0],
        ),
        reverse=True,
    )
    selected_indexes = sorted(
        index
        for index, sentence in ranked[:max_sentences]
        if any(anchor in sentence.lower() for anchor in anchors)
    )
    if not selected_indexes:
        selected_indexes = list(range(min(max_sentences, len(sentences))))
    return " ".join(sentences[index].strip() for index in selected_indexes)


def save_chunks(
    chunks: list[str],
    chunk_topic_matrix,
    component_topic_ids: list[int],
) -> None:
    """Save chunks with their strongest topics."""
    records = []

    for chunk_index, chunk in enumerate(chunks):
        topic_scores = chunk_topic_matrix[chunk_index]

        topic_index = int(topic_scores.argmax())
        topic_number = component_topic_ids[topic_index]

        topic_score = topic_scores[topic_index]
        score_total = float(topic_scores.sum())
        topic_share = float(topic_score / score_total) if score_total else 0.0
        classification_status = (
            "review" if topic_share < LOW_CONFIDENCE_SHARE else "confident"
        )
        display_topic_label = (
            MIXED_TOPIC_LABEL
            if classification_status == "review"
            else TOPIC_LABELS[topic_number]
        )

        record = {
            "chunk_id": chunk_index + 1,
            "topic_id": topic_number,
            "topic_label": TOPIC_LABELS[topic_number],
            "display_topic_label": display_topic_label,
            "topic_score": round(float(topic_score),4,),
            "topic_share": round(topic_share, 4),
            "classification_status": classification_status,
            "excerpt": build_topic_excerpt(chunk, topic_number),
            "text": chunk,
        }

        records.append(record)

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


    model, vectorizer, chunk_topic_matrix = (build_topic_model(chunks))
    component_topic_ids = infer_topic_ids(model, vectorizer)
    save_chunks(chunks, chunk_topic_matrix, component_topic_ids)

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
