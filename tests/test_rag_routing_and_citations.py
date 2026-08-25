"""Regression tests for question routing and grounded citations."""

from rag_service import (
    build_deterministic_financial_answer,
    classify_question,
    should_retrieve_mda,
)


METADATA = {
    "company": "Apple Inc.",
    "form": "10-K",
    "filing_date": "2025-10-31",
    "source_url": "https://www.sec.gov/example-10-k.htm",
}


def test_question_routes_cover_financial_risk_and_unsupported():
    assert classify_question("What was Apple's revenue in 2025?") == "financial"
    assert classify_question("What supply chain risks does Apple disclose?") == "risk"
    assert classify_question("Who designed the first iPhone?") == "unsupported"


def test_mda_is_only_requested_for_explanatory_financial_questions():
    assert not should_retrieve_mda("What was Apple's revenue in 2025?")
    assert not should_retrieve_mda(
        "Compare Apple's revenue and operating margin in 2023, 2024, and 2025."
    )
    assert should_retrieve_mda("Why did Apple's revenue increase in 2025?")


def test_multi_year_financial_answer_uses_one_group_citation():
    evidence = """Financial metrics by fiscal year:

FY2024
Revenue: $391.04B
Operating Margin: 31.51%

FY2025
Revenue: $416.16B
Operating Margin: 31.97%

Overall change from FY2024 to FY2025:
Revenue: +$25.12B
Revenue growth: +6.42%
Operating Margin: +0.46 percentage points
"""
    answer = build_deterministic_financial_answer(
        "Compare Apple's revenue and operating margin in 2024 and 2025.",
        evidence,
        METADATA,
    )

    body, references = answer.split("\n\nReferences:\n", 1)

    assert body.count("[F1]") == 1
    assert "Financial results [F1]:" in body
    assert "FY2024:\n" in body
    assert "FY2025:\n" in body
    assert references.count("[F1]") == 1
    assert METADATA["source_url"] in references


def test_single_value_answer_keeps_one_inline_citation():
    evidence = "Revenue in FY2025: $416.16B"
    answer = build_deterministic_financial_answer(
        "What was Apple's revenue in 2025?",
        evidence,
        METADATA,
    )

    body, references = answer.split("\n\nReferences:\n", 1)

    assert body.count("[F1]") == 1
    assert references.count("[F1]") == 1
