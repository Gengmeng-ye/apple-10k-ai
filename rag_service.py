"""Answer Apple financial and 10-K risk questions from local data."""

import json
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATABASE_FILE = Path("warehouse/apple_finance.duckdb")
CHUNKS_FILE = Path("data/processed/apple_risk_chunks.json")

FINANCIAL_METRICS = {
    "operating cash flow": (
        "operating_cash_flow_billions",
        "Operating Cash Flow",
        "$B",
    ),
    "operating margin": (
        "operating_margin_pct",
        "Operating Margin",
        "%",
    ),
    "net profit margin": (
        "net_profit_margin_pct",
        "Net Profit Margin",
        "%",
    ),
    "net margin": (
        "net_profit_margin_pct",
        "Net Profit Margin",
        "%",
    ),
    "operating income": (
        "operating_income_billions",
        "Operating Income",
        "$B",
    ),
    "net income": (
        "net_income_billions",
        "Net Income",
        "$B",
    ),
    "revenue": (
        "revenue_billions",
        "Revenue",
        "$B",
    ),
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


def find_financial_metrics(question: str) -> list[tuple]:
    """Find all financial metrics mentioned in the question."""

    question_lower = question.lower()
    matches = []

    for keyword, metric in FINANCIAL_METRICS.items():
        position = question_lower.find(keyword)

        if position != -1:
            matches.append((position, metric))

    matches.sort(key=lambda item: item[0])

    metrics = []
    used_columns = set()

    for _, metric in matches:
        column = metric[0]

        if column not in used_columns:
            metrics.append(metric)
            used_columns.add(column)

    return metrics


def find_financial_metric(question: str):
    """Find the first financial metric mentioned in the question."""

    metrics = find_financial_metrics(question)
    return metrics[0] if metrics else None


def find_years(question: str) -> list[int]:
    """Find unique four-digit fiscal years in the question."""

    years = re.findall(r"\b20\d{2}\b", question)
    return list(dict.fromkeys(int(year) for year in years))


def classify_question(question: str) -> str:
    """Classify the question as financial, risk, or unsupported."""

    question_lower = question.lower()

    if any(keyword in question_lower for keyword in RISK_KEYWORDS):
        return "risk"

    if find_financial_metrics(question):
        return "financial"

    return "unsupported"


def query_financial_data(question: str) -> None:
    """Query any combination of financial metrics and fiscal years."""

    metrics = find_financial_metrics(question)
    years = find_years(question)
    question_lower = question.lower()

    if not metrics:
        print("Please specify a supported financial metric.")
        return

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    trend_match = re.search(
        r"(?:last|past|over the last|over)\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:fiscal\s+)?years?",
        question_lower,
    )

    if trend_match:
        year_count_text = trend_match.group(1)

        if year_count_text.isdigit():
            trend_year_count = int(year_count_text)
        else:
            trend_year_count = number_words[year_count_text]

    elif "trend" in question_lower:
        trend_year_count = 5

    else:
        trend_year_count = None

    show_trend = trend_year_count is not None
    columns = ", ".join(metric[0] for metric in metrics)

    connection = duckdb.connect(
        str(DATABASE_FILE),
        read_only=True,
    )

    if years:
        placeholders = ", ".join("?" for _ in years)

        query = f"""
            SELECT
                YEAR(CAST("end" AS DATE)) AS fiscal_year,
                {columns}
            FROM apple_financial_summary
            WHERE YEAR(CAST("end" AS DATE)) IN ({placeholders})
            ORDER BY fiscal_year
        """

        rows = connection.execute(query, years).fetchall()

    elif show_trend:
        query = f"""
            SELECT
                YEAR(CAST("end" AS DATE)) AS fiscal_year,
                {columns}
            FROM apple_financial_summary
            ORDER BY CAST("end" AS DATE) DESC
            LIMIT {trend_year_count}
        """

        rows = connection.execute(query).fetchall()
        rows.reverse()

    else:
        query = f"""
            SELECT
                YEAR(CAST("end" AS DATE)) AS fiscal_year,
                {columns}
            FROM apple_financial_summary
            ORDER BY CAST("end" AS DATE) DESC
            LIMIT 1
        """

        rows = connection.execute(query).fetchall()

    connection.close()

    if not rows:
        print("No matching financial data was found.")
        return

    if years:
        returned_years = {int(row[0]) for row in rows}
        missing_years = [year for year in years if year not in returned_years]

        if missing_years:
            missing_text = ", ".join(f"FY{year}" for year in missing_years)
            print(f"No financial data was found for: {missing_text}\n")

    if len(rows) == 1 and len(metrics) == 1:
        fiscal_year = int(rows[0][0])
        value = rows[0][1]
        _, label, unit = metrics[0]
        formatted_value = f"{value:.2f}%" if unit == "%" else f"${value:.2f}B"
        print(f"{label} in FY{fiscal_year}: {formatted_value}")

    else:
        print("Financial metrics by fiscal year:")

        for row in rows:
            fiscal_year = int(row[0])
            values = row[1:]
            print(f"\nFY{fiscal_year}")

            for metric, value in zip(metrics, values):
                _, label, unit = metric
                formatted_value = f"{value:.2f}%" if unit == "%" else f"${value:.2f}B"
                print(f"{label}: {formatted_value}")

    if len(rows) > 1:
        first_row = rows[0]
        last_row = rows[-1]
        first_year = int(first_row[0])
        last_year = int(last_row[0])

        print(f"\nOverall change from FY{first_year} to FY{last_year}:")

        for metric_index, metric in enumerate(metrics, start=1):
            _, label, unit = metric
            first_value = first_row[metric_index]
            last_value = last_row[metric_index]
            value_change = last_value - first_value

            if unit == "%":
                print(f"{label}: {value_change:+.2f} percentage points")
            else:
                sign = "+" if value_change >= 0 else "-"
                growth = (value_change / first_value) * 100
                print(f"{label}: {sign}${abs(value_change):.2f}B")
                print(f"{label} growth: {growth:+.2f}%")

    if show_trend and len(metrics) == 1:
        highest_row = max(rows, key=lambda row: row[1])
        lowest_row = min(rows, key=lambda row: row[1])
        print(f"Highest year: FY{int(highest_row[0])}")
        print(f"Lowest year: FY{int(lowest_row[0])}")

    print("\nSource: SEC Company Facts API via DuckDB")


def load_risk_chunks() -> list[dict]:
    """Load the processed Risk Factors chunks."""

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_risk_evidence(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """Return the most relevant Risk Factors chunks."""

    chunks = load_risk_chunks()
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

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

    if not evidence or evidence[0]["retrieval_score"] == 0:
        print("No relevant Risk Factors evidence was found.")
        return

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


def get_answer(question: str) -> str:
    """Return the local financial or risk answer as text."""

    output = StringIO()

    with redirect_stdout(output):
        answer_question(question)

    return output.getvalue().strip()


def main() -> None:
    """Test financial, risk, and unsupported questions."""

    questions = [
        "What was Apple's revenue in 2025?",
        "Compare Apple's revenue in 2023, 2024, and 2025.",
        "Compare Apple's revenue and operating margin in 2023, 2024, and 2025.",
        "Show Apple's revenue over the past 2 years.",
        "Show Apple's revenue trend over the past three years.",
        "Show Apple's revenue trend.",
        "What supply chain risks does Apple disclose?",
        "What is the weather in Los Angeles?",
    ]

    for question in questions:
        answer_question(question)


if __name__ == "__main__":
    main()