"""Regression tests for question routing and grounded citations."""

from rag_service import (
    build_deterministic_financial_answer,
    classify_question,
    clean_generated_answer,
    find_financial_metrics,
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


def test_balance_sheet_questions_are_routed_as_financial():
    questions = [
        "What were Apple's total assets in 2025?",
        "Compare Apple's liabilities in 2024 and 2025.",
        "How did Apple's cash position change over the past five years?",
    ]

    for question in questions:
        assert classify_question(question) == "financial"


def test_balance_sheet_metric_aliases_map_to_expected_columns():
    metrics = find_financial_metrics(
        "Compare Apple's assets, liabilities, and cash in 2024 and 2025."
    )
    columns = [column for column, _, _ in metrics]

    assert columns == [
        "total_assets_billions",
        "total_liabilities_billions",
        "cash_and_cash_equivalents_billions",
    ]
    assert len(columns) == len(set(columns))


def test_operating_cash_flow_is_not_misread_as_cash_balance():
    metrics = find_financial_metrics(
        "What was Apple's operating cash flow in 2025?"
    )

    assert metrics == [
        (
            "operating_cash_flow_billions",
            "Operating Cash Flow",
            "$B",
        )
    ]


def test_plural_balance_sheet_metric_uses_correct_grammar():
    answer = build_deterministic_financial_answer(
        "What were Apple's total assets in 2025?",
        "Total Assets in FY2025: $359.24B",
        METADATA,
    )

    assert "total assets in FY2025 were $359.24B [F1]" in answer


def test_mixed_multi_year_trend_states_net_direction():
    evidence = """Financial metrics by fiscal year:

FY2023
Total Liabilities: $290.44B

FY2024
Total Liabilities: $308.03B

FY2025
Total Liabilities: $285.51B

Overall change from FY2023 to FY2025:
Total Liabilities: -$4.93B
Total Liabilities growth: -1.70%
"""
    answer = build_deterministic_financial_answer(
        "Compare Apple's total liabilities in 2023, 2024, and 2025.",
        evidence,
        METADATA,
    )

    assert "Varied across the reported intervals" in answer
    assert "decreased by $4.93B (1.70%) overall" in answer


def test_generated_answer_cleaner_preserves_compact_sections_and_bullets():
    raw = (
        "**Summary:**\nApple discloses supply-chain exposure [1].\n\n"
        "**Key risks:**\n- **Supplier dependence:** Limited suppliers may disrupt output [2].\n\n"
        "**Why it matters:**\nInterruptions can raise costs [3]."
    )

    cleaned = clean_generated_answer(raw)

    assert "Summary:" in cleaned
    assert "- Supplier dependence:" in cleaned
    assert "Why it matters:" in cleaned
    assert "**" not in cleaned


def test_generated_answer_cleaner_adds_missing_space_before_citation():
    cleaned = clean_generated_answer("A disclosed risk[1].")

    assert cleaned == "A disclosed risk [1]."
