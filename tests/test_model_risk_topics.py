"""Test Risk Factors text processing."""

from src.model_risk_topics import (
    NUMBER_OF_TOPICS,
    build_topic_model,
    build_topic_excerpt,
    infer_topic_ids,
    split_sentences,
    split_into_chunks,
)


def test_split_into_chunks_preserves_text() -> None:
    """Verify that chunking preserves all sentences."""
    text = (
        "Apple faces competition. "
        "Demand may decline. "
        "Costs may increase."
    )

    chunks = split_into_chunks(
        text,
        max_characters=35,
    )

    assert len(chunks) == 3
    assert " ".join(chunks) == text
    assert all(len(chunk) <= 35 for chunk in chunks)


def test_topic_model_output_shape() -> None:
    """Verify the topic model output dimensions."""
    chunks = [
        "cyber attack data security systems",
        "cyber security unauthorized data access",
        "product competition software services",
        "product services software competition",
        "foreign currency dollar exchange sales",
        "foreign exchange currency dollar sales",
        "tax rates receivables financial credit",
        "tax credit receivables financial rates",
        "legal laws regulations investigations",
        "legal regulations laws litigation",
        "tariffs imports supply chain partners",
        "tariffs supply imports trade partners",
    ]

    model, vectorizer, chunk_topic_matrix = (build_topic_model(chunks))

    assert chunk_topic_matrix.shape == (len(chunks),NUMBER_OF_TOPICS,)

    assert model.components_.shape[0] == (NUMBER_OF_TOPICS)

    assert model.components_.shape[1] == len(vectorizer.get_feature_names_out())

    labels = infer_topic_ids(model, vectorizer)
    assert sorted(labels) == list(range(1, NUMBER_OF_TOPICS + 1))

    feature_names = vectorizer.get_feature_names_out()
    cyber_component = next(
        index
        for index, component in enumerate(model.components_)
        if feature_names[component.argmax()] in {"access", "cyber", "data", "security"}
    )
    assert labels[cyber_component] == 1


def test_topic_excerpt_removes_unrelated_leading_sentence() -> None:
    chunk = (
        "Demand for one product could decline. "
        "Foreign exchange rates can affect sales. "
        "The U.S. dollar may strengthen relative to foreign currencies."
    )
    excerpt = build_topic_excerpt(chunk, topic_id=3)
    assert "one product" not in excerpt
    assert "Foreign exchange" in excerpt
    assert "foreign currencies" in excerpt


def test_sentence_splitter_preserves_filing_abbreviations() -> None:
    sentences = split_sentences(
        "Apple Inc. has exposure to the U.S. dollar. Foreign currencies affect sales."
    )
    assert sentences == [
        "Apple Inc. has exposure to the U.S. dollar.",
        "Foreign currencies affect sales.",
    ]
