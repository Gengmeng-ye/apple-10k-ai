# Apple 10-K Financial Analyst

An end-to-end SEC analytics application that transforms Apple filings into an interactive financial dashboard and evidence-grounded question-answering experience.

![Apple 10-K Financial Analyst dashboard hero](assets/apple_energy_hero.png)

## Overview

Apple 10-K Financial Analyst combines data engineering, financial analysis, natural-language retrieval, and responsive data visualization in one Streamlit application. It uses official SEC filings and Company Facts data to help users explore Apple’s financial performance, compare fiscal years, review balance-sheet trends, and investigate management commentary and disclosed risks.

The application currently covers Apple fiscal years 2021–2025 and is designed for both desktop and mobile screens.

## Key Features

### Interactive financial dashboard

- Fiscal-year selector with KPI cards for revenue, operating income, net income, and operating cash flow
- Five-year financial trends for FY2021–FY2025
- Profitability analysis using operating margin and net profit margin
- Balance-sheet analysis for total assets, total liabilities, and cash and cash equivalents
- Responsive desktop, tablet, and mobile layouts
- Underlying financial data table for transparent review

### Evidence-grounded financial assistant

- Deterministic DuckDB answers for reported financial values and comparisons
- Single-year, multi-year, trend, and multi-metric queries
- Automatic calculations for absolute change, percentage change, and percentage-point change
- MD&A retrieval for questions asking why a financial result changed
- Risk Factors retrieval for questions about Apple’s disclosed business risks
- Direct links and structured citations back to Apple’s SEC filing
- Clear rejection of questions outside the supported financial and filing scope

### Risk analysis

- Extraction of Item 1A, Risk Factors, from Apple’s Form 10-K
- Sentence-based chunking of filing text
- TF-IDF feature engineering and NMF topic modeling
- Six interpretable risk themes:
  - Legal and Regulatory
  - Products and Competition
  - Trade and Supply Chain
  - Credit and Financial Risk
  - Cybersecurity and Data Privacy
  - Foreign Exchange
- Interactive theme coverage chart and filing excerpts

## Supported Financial Metrics

| Category | Metrics |
| --- | --- |
| Income statement | Revenue, operating income, net income |
| Cash flow | Operating cash flow |
| Balance sheet | Total assets, total liabilities, cash and cash equivalents |
| Profitability | Operating margin, net profit margin |
| Growth | Revenue growth, five-year revenue CAGR |

Financial figures are presented in USD billions. Margins and growth rates are presented as percentages.

## Example Questions

```text
What were Apple's total assets in 2025?
Compare Apple's revenue in 2024 and 2025.
Compare Apple's assets, liabilities, and cash in 2023, 2024, and 2025.
Show Apple's cash position over the past five years.
Why did Apple's revenue change in 2025?
What supply-chain risks does Apple disclose?
```

Reported-value questions are answered deterministically from DuckDB. Questions requiring management explanation or risk synthesis use retrieved SEC evidence with the OpenAI API.

## OpenAI Integration

The application uses the OpenAI API selectively rather than sending every question to a language model:

- Reported financial values, year comparisons, and trend calculations are generated deterministically from DuckDB and Python without an OpenAI call.
- Questions asking why a financial result changed retrieve relevant Item 7 MD&A evidence before OpenAI produces a concise, cited explanation.
- Risk questions retrieve relevant Item 1A Risk Factors excerpts before OpenAI synthesizes an evidence-grounded answer.
- The model is instructed to use only the supplied SEC evidence, preserve the distinction between reported results and management commentary, and state when the evidence is insufficient.

This design limits unnecessary API usage while keeping numerical answers reproducible and narrative answers grounded in SEC disclosures.

## Architecture

```text
SEC Company Facts API ──> Financial transformation ──> Validation
                                                        │
                                                        v
                                              Parquet + DuckDB
                                                        │
                                                        v
                                                Financial Q&A

Apple Form 10-K ──> Item 7 MD&A extraction ──> TF-IDF retrieval ──┐
                                                                  ├─> Cited answers
Apple Form 10-K ──> Item 1A extraction ──> NMF topics + retrieval ─┘

Parquet + DuckDB + processed filing chunks ──> Streamlit dashboard
```

### Question routing

1. The question is classified as financial, risk-related, or unsupported.
2. Financial metrics and requested fiscal years are identified.
3. Reported values are queried from the local DuckDB warehouse.
4. Explanation questions retrieve relevant MD&A chunks.
5. Risk questions retrieve relevant Item 1A chunks.
6. The response includes SEC-grounded citations and a filing link.

## Data Pipeline

`run_pipeline.py` executes the main financial and risk workflow:

1. Transform SEC Company Facts into an annual financial summary.
2. Load the curated dataset into DuckDB.
3. Validate required columns, data types, missing values, and value ranges.
4. Extract Risk Factors from Apple’s latest Form 10-K.
5. Build and label the NMF risk-topic model.

MD&A extraction and chunk preparation are handled by `src/extract_mda.py` and `src/prepare_mda_chunks.py`.

## Technology Stack

- **Application:** Streamlit
- **Visualization:** Plotly
- **Data processing:** Python, pandas, PyArrow
- **Analytics warehouse:** DuckDB and SQL
- **SEC ingestion:** Requests, Beautiful Soup, lxml
- **Text retrieval and modeling:** scikit-learn, TF-IDF, cosine similarity, NMF
- **Grounded response generation:** OpenAI API
- **Testing:** pytest
- **Version control:** Git and GitHub

## Project Structure

```text
apple-10k-ai/
├── app.py
├── rag_service.py
├── financial_analysis.py
├── run_pipeline.py
├── requirements.txt
├── assets/
│   └── apple_energy_hero.png
├── data/
│   ├── filings/
│   ├── raw/
│   └── processed/
│       ├── apple_financial_summary.parquet
│       ├── apple_mda_chunks.json
│       └── apple_risk_chunks.json
├── src/
│   ├── extract_sec_data.py
│   ├── transform_financials.py
│   ├── load_warehouse.py
│   ├── validate_data.py
│   ├── extract_mda.py
│   ├── prepare_mda_chunks.py
│   ├── extract_risk_factors.py
│   └── model_risk_topics.py
├── tests/
│   ├── test_validate_data.py
│   ├── test_model_risk_topics.py
│   └── test_rag_routing_and_citations.py
└── warehouse/
    └── apple_finance.duckdb
```

## Local Setup

### 1. Clone the repository

```bash
git clone --branch feature/mda-analysis https://github.com/Gengmeng-ye/apple-10k-ai.git
cd apple-10k-ai
```

### 2. Create the Python environment

```bash
conda create -n apple10k python=3.12 -y
conda activate apple10k
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```text
SEC_USER_AGENT=Your Name your.email@example.com
OPENAI_API_KEY=your_openai_api_key
```

The SEC requests an identifiable user agent. The OpenAI API key is required for MD&A explanations and synthesized Risk Factors answers. Do not commit the `.env` file.

### 4. Launch the application

The repository includes lightweight processed data and a DuckDB database for the deployed application:

```bash
streamlit run app.py
```

To rebuild the financial and risk datasets from locally downloaded SEC source files:

```bash
python run_pipeline.py
```

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

The current suite contains 21 passing tests covering:

- Financial-data validation
- Risk-topic processing
- Question classification and routing
- Financial metric aliases
- MD&A retrieval decisions
- Citation and answer-format behavior

Additional checks used before deployment:

```bash
python -m py_compile app.py rag_service.py
git diff --check
```

## Data Sources and Provenance

All reported financial figures and narrative evidence in the application come from official U.S. Securities and Exchange Commission sources. The project does not use third-party financial estimates, forecasts, or market-data providers.

| Source | How it is used |
| --- | --- |
| [SEC Company Facts API](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json) | Structured XBRL facts used to build Apple’s FY2021–FY2025 annual financial dataset |
| [Apple FY2025 Form 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm) | Item 7 MD&A, Item 1A Risk Factors, filing metadata, evidence excerpts, and answer citations |

Apple’s Central Index Key is `0000320193`. The FY2025 Form 10-K covers the fiscal year ended September 27, 2025, and was filed on October 31, 2025.

### SEC data ingestion

The project retrieves SEC data through two ingestion paths:

1. **Structured financial data**  
   `src/extract_sec_data.py` sends an HTTP request to the SEC Company Facts API using an identifiable `SEC_USER_AGENT`. The response is saved locally as raw JSON before transformation.

2. **Narrative filing data**  
   The same ingestion module identifies Apple’s latest Form 10-K from SEC filing metadata and downloads the filing’s primary HTML document from EDGAR.

The downloaded Form 10-K is parsed with Beautiful Soup:

- `src/extract_mda.py` extracts Item 7, Management’s Discussion and Analysis.
- `src/extract_risk_factors.py` extracts Item 1A, Risk Factors.
- The extracted sections are normalized and divided into retrieval-ready chunks.

The subsequent processing workflow:

- selects annual USD facts and resolves duplicate SEC observations using fiscal-period metadata and period-end dates;
- converts reported dollar values to USD billions;
- calculates margins, annual growth, and five-year CAGR in Python rather than asking a language model to calculate them;
- validates required columns, missing values, data types, and acceptable value ranges;
- stores the curated financial summary in Parquet and DuckDB;
- normalizes and chunks MD&A and Risk Factors text for retrieval, topic modeling, and citations; and
- preserves the filing URL and metadata used to trace answers back to the SEC source.

Raw SEC responses and filing HTML are retained locally for reproducibility but excluded from Git because they are generated source files. Curated Parquet, JSON chunks, and DuckDB assets are included for application deployment.

## Design Decisions

- Financial values are queried locally instead of generated by a language model.
- OpenAI is used only when a question requires narrative synthesis from retrieved filing evidence.
- Cash is visualized separately because its scale is substantially smaller than total assets and liabilities.
- Lightweight processed assets are included so the application can start in a hosted environment without rebuilding the full SEC pipeline.
- Raw filings, secrets, caches, logs, and large generated artifacts remain excluded from version control.

## Limitations

- The application currently supports Apple only.
- Annual financial analysis covers FY2021–FY2025; it does not yet include quarterly 10-Q data.
- Risk-theme coverage represents the number of filing excerpts, not risk severity.
- Retrieved filing evidence may not explain every financial change; the assistant does not speculate when support is insufficient.
- The project does not forecast Apple’s stock price or future financial results.
- The application is intended for analysis and education, not investment decision-making.

## Future Improvements

- Add Microsoft and Alphabet for peer comparison
- Add quarterly 10-Q ingestion and analysis
- Add hybrid retrieval across Risk Factors, MD&A, and financial statements for questions connecting disclosed risks with realized financial impacts
- Add retrieval evaluation metrics and broader question test cases
- Automate SEC data refreshes for future filings

## Disclaimer

This project is for educational and portfolio demonstration purposes only. It does not constitute financial or investment advice.

## Author

**Gengmeng Ye**

MS in Business Analytics, USC Marshall School of Business
