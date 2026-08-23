"""Answer Apple financial and 10-K risk questions from local data."""

import json
import re
from pathlib import Path

import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATABASE_FILE = Path("warehouse/apple_finance.duckdb")
CHUNKS_FILE = Path("data/processed/apple_risk_chunks.json")

FINANCIAL_METRICS = {
    "operating cash flow": ("operating_cash_flow_billions", "Operating Cash Flow", "$B"),
    "operating margin": ("operating_margin_pct", "Operating Margin", "%"),
    "net profit margin": ("net_profit_margin_pct", "Net Profit Margin", "%"),
    "net margin": ("net_profit_margin_pct", "Net Profit Margin", "%"),
    "operating income": ("operating_income_billions", "Operating Income", "$B"),
    "net income": ("net_income_billions", "Net Income", "$B"),
    "revenue": ("revenue_billions", "Revenue", "$B"),
}

RISK_KEYWORDS = [
    "risk",
    "cybersecurity",
    "privacy",
    "ransomware",
    "supply chain",
    "supplier",
    "manufacturing",
    "tariff",
    "foreign exchange",
    "currency",
    "competition",
    "regulation",
    "regulatory",
    "legal",
    "intellectual property",
]


def find_financial_metric(question: str):
    """Find the financial metric mentioned in the question."""

    question = question.lower()

    for keyword, metric in FINANCIAL_METRICS.items():
        if keyword in question:
            return metric

    return None


def find_years(question: str) -> list[int]:
    """Find all four-digit years in the question."""

    years = re.findall(r"\b20\d{2}\b", question)
    return [int(year) for year in years]


def classify_question(question: str) -> str:
    """Classify the question as financial, risk, or unsupported."""

    question_lower = question.lower()

    if any(keyword in question_lower for keyword in RISK_KEYWORDS):
        return "risk"

    if find_financial_metric(question):
        return "financial"

    return "unsupported"


def query_financial_data(question: str) -> None:
    """Query or compare a financial metric from DuckDB."""

    metric = find_financial_metric(question)

    if metric is None:
        print("Please specify a supported financial metric.")
        return

    column, label, unit = metric
    years = find_years(question)

    connection = duckdb.connect(str(DATABASE_FILE), read_only=True)

    if len(years) >= 2:
        selected_years = years[:2]

        query = f"""
            SELECT
                YEAR(CAST("end" AS DATE)) AS fiscal_year,
                {column}
            FROM apple_financial_summary
            WHERE YEAR(CAST("end" AS DATE)) IN (?, ?)
            ORDER BY fiscal_year
        """

        rows = connection.execute(
            query,
            selected_years,
        ).fetchall()

        connection.close()

        if len(rows) != 2:
            print("Financial data was not found for both fiscal years.")
            return

        first_year, first_value = rows[0]
        second_year, second_value = rows[1]
        
        value_change = second_value - first_value
        percentage_change = (value_change / first_value) * 100
        
        if unit == "%":
            print(f"{label} in FY{int(first_year)}: {first_value:.2f}%")
            print(f"{label} in FY{int(second_year)}: {second_value:.2f}%")
            print(f"Change: {value_change:+.2f} percentage points")
            
        else:
            sign = "+" if value_change >= 0 else "-"
            print(f"{label} in FY{int(first_year)}: ${first_value:.2f}B")
            print(f"{label} in FY{int(second_year)}: ${second_value:.2f}B")
            print(f"Change: {sign}${abs(value_change):.2f}B")
            print(f"Growth: {percentage_change:+.2f}%")
        
        print("Source: SEC Company Facts API via DuckDB")
        return

    if len(years) == 1:
        query = f"""
            SELECT
                YEAR(CAST("end" AS DATE)) AS fiscal_year,
                {column}
            FROM apple_financial_summary
            WHERE YEAR(CAST("end" AS DATE)) = ?
        """

        result = connection.execute(
            query,
            [years[0]],
        ).fetchone()
    else:
        query = f"""
            SELECT
                YEAR(CAST("end" AS DATE)) AS fiscal_year,
                {column}
            FROM apple_financial_summary
            ORDER BY CAST("end" AS DATE) DESC
            LIMIT 1
        """

        result = connection.execute(query).fetchone()

    connection.close()

    if result is None:
        requested_year = years[0] if years else "the latest year"
        print(f"No financial data was found for {requested_year}.")
        return

    fiscal_year, value = result

    if unit == "%":
        formatted_value = f"{value:.2f}%"
    else:
        formatted_value = f"${value:.2f}B"

    print(f"{label} in FY{int(fiscal_year)}: {formatted_value}")
    print("Source: SEC Company Facts API via DuckDB")


def load_risk_chunks() -> list[dict]:
    """Load the processed Risk Factors chunks."""

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_risk_evidence(question: str, top_k: int = 3) -> list[dict]:
    """Return the most relevant Risk Factors chunks."""

    chunks = load_risk_chunks()
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    chunk_matrix = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(question_vector, chunk_matrix).flatten()
    ranked_indexes = scores.argsort()[::-1][:top_k]

    evidence = []

    for index in ranked_indexes:
        result = chunks[index].copy()
        result["retrieval_score"] = round(float(scores[index]), 4)
        evidence.append(result)

    return evidence


def print_risk_evidence(question: str) -> None:
    """Print the most relevant Risk Factors evidence."""

    evidence = retrieve_risk_evidence(question)

    print("Retrieved Risk Factors evidence:")

    for result in evidence:
        print(
            f"\nChunk {result['chunk_id']} | "
            f"{result['topic_label']} | "
            f"Score: {result['retrieval_score']}"
        )
        print(result["text"][:500])


def answer_question(question: str) -> None:
    """Send the question to the correct local data source."""

    question_type = classify_question(question)

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Route: {question_type}\n")

    if question_type == "financial":
        query_financial_data(question)
    elif question_type == "risk":
        print_risk_evidence(question)
    else:
        print(
            "This question is outside the scope of Apple financial "
            "and 10-K risk analysis."
        )


def main() -> None:
    """Test the three question routes."""

    questions = [
        "What was Apple's revenue in 2025?",
        "Compare Apple's revenue between 2024 and 2025.",
        "Compare Apple's operating margin between 2024 and 2025.",
        "What supply chain risks does Apple disclose?",
        "What is the weather in Los Angeles?",
    ]

    for question in questions:
        answer_question(question)


if __name__ == "__main__":
    main()