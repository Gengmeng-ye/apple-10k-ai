"""Test Risk Factors text processing."""

from src.model_risk_topics import (
    NUMBER_OF_TOPICS,
    build_topic_model,
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