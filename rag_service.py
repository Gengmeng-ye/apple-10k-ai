"""Answer Apple financial and 10-K risk questions from local data."""

import json
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATABASE_FILE = Path("warehouse/apple_finance.duckdb")
CHUNKS_FILE = Path("data/processed/apple_risk_chunks.json")
MDA_CHUNKS_FILE = Path("data/processed/apple_mda_chunks.json")
METADATA_FILE = Path("data/filings/apple_latest_10k_metadata.json")
OPENAI_MODEL = "gpt-5.6"


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
    "net sales": (
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


def load_filing_metadata() -> dict:
    """Load metadata for Apple's latest 10-K filing."""

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from .env."""

    load_dotenv()
    return OpenAI()


def clean_generated_answer(answer: str) -> str:
    """Normalize model output for the dashboard chat renderer."""

    spacing_replacements = {
        "iPhoneup": "iPhone up",
        "andiPad": "and iPad",
        "growthat": "growth at",
        "orforce": "or force",
        "oncommercially": "on commercially",
        "andproduct": "and product",
        "riskscould": "risks could",
        "higherServices": "higher Services",
        "orlimited": "or limited",
    }

    for joined_text, corrected_text in spacing_replacements.items():
        answer = answer.replace(joined_text, corrected_text)

    answer = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", answer)
    answer = re.sub(
        r"(?<!\s)(\[(?:F|M)?\d+\])",
        r" \1",
        answer,
    )
    answer = re.sub(
        r"\b(up|down)(?=\d)",
        r"\1 ",
        answer,
    )
    answer = re.sub(
        r"(?<=\d)([BM])(?=[A-Za-z])",
        r"\1 ",
        answer,
    )
    answer = re.sub(r"(?<=%)(?=[A-Za-z])", " ", answer)
    cleaned_lines = []

    for raw_line in answer.splitlines():
        line = raw_line.strip().replace("**", "")

        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue

        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and cells[0].lower() in {"fiscal year", "year"}:
                continue
            if len(cells) >= 2:
                details = ", ".join(cell for cell in cells[1:] if cell)
                cleaned_lines.append(f"{cells[0]}: {details}")
                continue

        cleaned_lines.append(line.lstrip("# "))

    return "\n".join(cleaned_lines).strip()


def parse_financial_evidence(
    financial_evidence: str,
) -> tuple[
    dict[int, dict[str, str]],
    tuple[str, int, str] | None,
    dict[str, str],
]:
    """Parse deterministic DuckDB evidence into display-ready values."""

    yearly_values: dict[int, dict[str, str]] = {}
    overall_values: dict[str, str] = {}
    single_value = None
    current_year = None
    in_overall = False

    single_match = re.search(
        r"^(.+?) in FY(20\d{2}): (.+)$",
        financial_evidence,
        flags=re.MULTILINE,
    )

    if single_match:
        label, year, value = single_match.groups()
        single_value = (label, int(year), value)

    for raw_line in financial_evidence.splitlines():
        line = raw_line.strip()
        year_match = re.fullmatch(r"FY(20\d{2})", line)

        if year_match:
            current_year = int(year_match.group(1))
            yearly_values[current_year] = {}
            in_overall = False
            continue

        if line.startswith("Overall change"):
            current_year = None
            in_overall = True
            continue

        if line.startswith("Source:"):
            current_year = None
            in_overall = False
            continue

        if current_year is not None and ":" in line:
            label, value = line.split(":", 1)
            yearly_values[current_year][label.strip()] = value.strip()

        elif in_overall and ":" in line:
            label, value = line.split(":", 1)
            overall_values[label.strip()] = value.strip()

    return yearly_values, single_value, overall_values


def numeric_financial_value(value: str) -> float:
    """Return the numeric component of a formatted financial value."""

    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else 0.0


def describe_metric_change(
    label: str,
    unit: str,
    years: list[int],
    yearly_values: dict[int, dict[str, str]],
    overall_values: dict[str, str],
) -> str:
    """Build one deterministic comparison sentence for a metric."""

    values = [numeric_financial_value(yearly_values[year][label]) for year in years]
    changes = [later - earlier for earlier, later in zip(values, values[1:])]
    total_change = values[-1] - values[0]

    largest_index = max(range(len(changes)), key=lambda index: abs(changes[index]))
    largest_start = years[largest_index]
    largest_end = years[largest_index + 1]

    if unit == "%":
        verb = "expanded" if total_change >= 0 else "contracted"
        change_text = overall_values.get(
            label,
            f"{abs(total_change):.2f} percentage points",
        ).lstrip("+-")
        interval_text = f"{abs(changes[largest_index]):.2f} percentage points"
    else:
        verb = "increased" if total_change >= 0 else "decreased"
        growth = (total_change / values[0]) * 100 if values[0] else 0.0
        reported_change = overall_values.get(label, f"${abs(total_change):.2f}B")
        reported_growth = overall_values.get(
            f"{label} growth",
            f"{abs(growth):.2f}%",
        )
        change_text = (
            f"{reported_change.lstrip('+-')} "
            f"({reported_growth.lstrip('+-')})"
        )
        interval_text = f"${abs(changes[largest_index]):.2f}B"

    if all(change >= 0 for change in changes):
        pattern = f"{verb.capitalize()} in every reported interval"
    elif all(change <= 0 for change in changes):
        pattern = f"{verb.capitalize()} in every reported interval"
    else:
        pattern = "Varied across the reported intervals"

    return (
        f"- {label}: {pattern}, with an overall change of {change_text}; "
        f"the largest annual change was FY{largest_start}–FY{largest_end} "
        f"({interval_text})"
    )


def build_deterministic_financial_answer(
    question: str,
    financial_evidence: str,
    metadata: dict,
) -> str:
    """Format numeric financial questions without a generative API call."""

    yearly_values, single_value, overall_values = parse_financial_evidence(
        financial_evidence
    )
    reference = (
        f"[F1] {metadata['company']} {metadata['form']} — "
        f"Financial Statements, filed {metadata['filing_date']}"
    )

    if single_value:
        label, year, value = single_value
        body = f"Apple’s {label.lower()} in FY{year} was {value} [F1]."
    elif yearly_values:
        years = sorted(yearly_values)
        metrics = find_financial_metrics(question)
        sections = ["Financial results [F1]:"]

        for year in years:
            metric_lines = [
                f"- {label}: {yearly_values[year][label]}"
                for _, label, _ in metrics
                if label in yearly_values[year]
            ]
            sections.append(f"FY{year}:\n" + "\n".join(metric_lines))

        if len(years) > 1:
            summaries = [
                describe_metric_change(
                    label,
                    unit,
                    years,
                    yearly_values,
                    overall_values,
                )
                for _, label, unit in metrics
                if all(label in yearly_values[year] for year in years)
            ]
            sections.append("Overall:\n" + "\n".join(summaries))

        body = "\n\n".join(sections)
    else:
        body = financial_evidence

    return (
        f"{body}\n\n"
        f"References:\n{reference}\n"
        f"{metadata['source_url']}"
    )


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


def load_mda_chunks() -> list[dict]:
    """Load the processed MD&A chunks."""

    with MDA_CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def should_retrieve_mda(question: str) -> bool:
    """Decide whether a financial question needs management explanation."""

    question_lower = question.lower()

    explanation_terms = [
        "why",
        "reason",
        "driver",
        "driven",
        "cause",
        "contribute",
        "explain",
        "because",
        "due to",
        "attribut",
    ]

    return any(
        term in question_lower
        for term in explanation_terms
    )


def expand_mda_query(question: str) -> str:
    """Add SEC terminology related to common user wording."""

    normalized_question = re.sub(
        r"(?<=[A-Za-z])(?=\d)",
        " ",
        question,
    )

    expansions = {
        "revenue": "net sales sales",
        "operating margin": "operating income net sales margin",
        "net margin": "net income net sales margin",
        "cash flow": "cash generated by operating activities liquidity",
        "iphone": "iPhone net sales",
        "services": "Services net sales",
    }

    expanded_terms = [normalized_question]
    question_lower = normalized_question.lower()

    for user_term, sec_terms in expansions.items():
        if user_term in question_lower:
            expanded_terms.append(sec_terms)

    return " ".join(expanded_terms)


def retrieve_mda_evidence(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """Return the most relevant MD&A chunks."""

    chunks = load_mda_chunks()
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    chunk_matrix = vectorizer.fit_transform(texts)
    expanded_question = expand_mda_query(question)
    question_vector = vectorizer.transform([expanded_question])
    scores = cosine_similarity(
        question_vector,
        chunk_matrix,
    ).flatten()

    ranked_indexes = scores.argsort()[::-1][:top_k]

    evidence = []

    for index in ranked_indexes:
        result = chunks[index].copy()
        result["retrieval_score"] = round(float(scores[index]), 4)
        evidence.append(result)

    return evidence


def build_mda_context(
    question: str,
) -> tuple[str, list[dict]]:
    """Build cited MD&A context when a question needs explanation."""

    if not should_retrieve_mda(question):
        return "", []

    evidence = retrieve_mda_evidence(question)

    if not evidence or evidence[0]["retrieval_score"] == 0:
        return "", []

    context = "\n\n".join(
        (
            f"[M{index}] Item 7. MD&A, "
            f"retrieved chunk {result['chunk_id']}\n"
            f"{result['text']}"
        )
        for index, result in enumerate(evidence, start=1)
    )

    return context, evidence


def generate_risk_answer(question: str) -> str:
    """Generate an evidence-grounded risk answer with SEC citations."""

    evidence = retrieve_risk_evidence(question)
    metadata = load_filing_metadata()

    if not evidence or evidence[0]["retrieval_score"] == 0:
        return "No relevant Risk Factors evidence was found."

    evidence_text = "\n\n".join(
        (
            f"[{index}] Topic: {result['topic_label']}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"SEC excerpt:\n{result['text']}"
        )
        for index, result in enumerate(evidence, start=1)
    )

    instructions = """
You are a financial research assistant analyzing Apple SEC filings.

Answer only from the supplied SEC excerpts.
Do not add facts that are not supported by the excerpts.
Distinguish a disclosed risk from an event that actually occurred.
Write a concise but analytical answer.
Cite supporting evidence using [1], [2], or [3].
If the evidence is insufficient, clearly say so.
Do not invent citations, URLs, numbers, or dates.
Return plain text only.
Do not use Markdown tables, Markdown headings, or bold markers.
Start with a direct answer, then briefly explain the significance of the disclosed risks.
Keep the response compact for a dashboard chat interface.
Ensure normal spacing between all words.
"""

    prompt = f"""
Question:
{question}

SEC Risk Factors evidence:
{evidence_text}
"""

    client = create_openai_client()
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=prompt,
    )

    generated_answer = clean_generated_answer(response.output_text)
    citation_lines = []

    for index, result in enumerate(evidence, start=1):
        citation = f"[{index}]"

        if citation in generated_answer:
            citation_lines.append(
                f"{citation} {metadata['company']} "
                f"{metadata['form']} — Item 1A. Risk Factors · "
                f"{result['topic_label']}"
            )

    citations = "\n".join(citation_lines)

    source = (
        f"{metadata['company']} {metadata['form']}, "
        f"filed {metadata['filing_date']}\n"
        f"{metadata['source_url']}"
    )

    return (
        f"{generated_answer}\n\n"
        f"References:\n{citations}\n\n"
        f"SEC filing:\n{source}"
    )


def generate_financial_answer(question: str) -> str:
    """Generate a financial answer grounded in local DuckDB results."""

    output = StringIO()

    with redirect_stdout(output):
        query_financial_data(question)

    financial_evidence = output.getvalue().strip()

    if (
        "No matching financial data was found." in financial_evidence
        or "Please specify a supported financial metric." in financial_evidence
    ):
        return financial_evidence

    metadata = load_filing_metadata()
    mda_context, mda_evidence = build_mda_context(question)

    if not mda_context:
        return build_deterministic_financial_answer(
            question,
            financial_evidence,
            metadata,
        )

    answer_format_instructions = ""

    if mda_context:
        answer_format_instructions = """
Because MD&A evidence is available, organize the answer using exactly these labels:

What changed:
State the reported financial result using the precise values from the financial evidence.
For year-over-year change questions, include the beginning value, ending value,
absolute change, and exact calculated percentage when they are available.
Prefer the precise calculated percentage from financial evidence over a rounded
percentage reported in MD&A.

What drove it:
Summarize management's explanation using only the supplied MD&A evidence.

Keep each section concise.
"""

    instructions = """
You are a financial research assistant analyzing Apple financial data.

Answer only from the supplied financial evidence and MD&A evidence.
Do not invent financial values, dates, explanations, or business causes.
Preserve the distinction between percentage change and percentage-point change.
Use fiscal years, such as FY2025, when discussing annual results.
Provide a concise but analytical interpretation of the numbers.
Cite financial claims using [F1].
Cite management explanations using [M1], [M2], or [M3].
Use MD&A citations only when MD&A evidence is supplied.
Clearly distinguish reported financial results from management's explanation.
If the supplied evidence does not explain why a metric changed, do not speculate about causes.
Do not use Markdown tables.
Do not use Markdown headings.
Use no more than two short paragraphs.
Use simple bullet points only when they improve readability.
Keep the response compact for display in a dashboard chat interface.
Return plain text only and do not use bold markers.
For a single-year, single-metric question, answer in one concise sentence.
Do not add a sentence saying that no comparison was supplied.
For comparison or trend questions, start with a direct answer, followed by one
sentence interpreting the direction and magnitude of the change.
For multi-year questions, identify whether the
trend was consistent and which interval changed the most. Do not speculate
about business causes unless they appear in the supplied evidence.
Avoid subjective labels such as moderate, strong, or significant unless the supplied evidence includes a relevant benchmark.
"""

    mda_prompt = ""

    if mda_context:
        mda_prompt = f"""

Management's Discussion and Analysis evidence:
{mda_context}
"""

    prompt = f"""
Question:
{question}

Financial evidence retrieved from the local DuckDB database:
{financial_evidence}
{mda_prompt}

Additional answer-format instructions:
{answer_format_instructions}

Follow the additional answer-format instructions over the general paragraph
limit whenever they are supplied.
"""

    client = create_openai_client()
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=prompt,
    )

    generated_answer = clean_generated_answer(response.output_text)
    reference_lines = [
        (
            f"[F1] {metadata['company']} {metadata['form']} — "
            f"Financial Statements, filed {metadata['filing_date']}"
        )
    ]

    for index, result in enumerate(mda_evidence, start=1):
        citation = f"[M{index}]"

        if citation in generated_answer:
            reference_lines.append(
                f"{citation} {metadata['company']} "
                f"{metadata['form']} — "
                f"Item 7. Management’s Discussion and Analysis"
            )


    references = "\n".join(reference_lines)

    return (
        f"{generated_answer}\n\n"
        f"References:\n{references}\n"
        f"{metadata['source_url']}"
    )


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
    """Return an AI-generated answer grounded in local evidence."""

    question_type = classify_question(question)

    if question_type == "financial":
        return generate_financial_answer(question)

    if question_type == "risk":
        return generate_risk_answer(question)

    return (
        "This question is outside the scope of Apple financial "
        "and 10-K risk analysis."
    )


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
