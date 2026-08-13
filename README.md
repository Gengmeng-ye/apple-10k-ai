# Apple 10-K AI Financial Analyst

An end-to-end financial analytics application that combines data engineering, data analysis, data science, and retrieval-augmented generation to analyze Apple’s SEC filings.

## Project Goal

Public-company filings contain valuable financial and risk information, but they are often long and difficult to analyze efficiently.

This project builds a reproducible SEC data pipeline, analyzes Apple’s financial performance, identifies major risk themes from 10-K filings, and answers financial questions with supporting source evidence.

## Planned Features

- Extract structured financial data from the SEC EDGAR API
- Build a reproducible ETL pipeline
- Validate and store processed data in Parquet and DuckDB
- Analyze revenue, operating income, net income, operating cash flow, and operating margin
- Visualize three-year financial trends
- Extract Risk Factors from Apple’s 10-K filings
- Identify risk themes using TF-IDF and K-Means clustering
- Evaluate clustering results using silhouette scores and manual review
- Answer questions from 10-K filings using retrieval-augmented generation
- Display supporting source evidence with each answer

## Project Architecture

### Financial Data Pipeline

```text
SEC Company Facts API
        ↓
Raw JSON
        ↓
Data Transformation
        ↓
Data Quality Validation
        ↓
Parquet
        ↓
DuckDB
        ↓
SQL Analysis
        ↓
Streamlit Dashboard
```

### Risk Topic Modeling

```text
Apple 10-K Filings
        ↓
Risk Factor Extraction
        ↓
Text Preprocessing
        ↓
TF-IDF Feature Engineering
        ↓
K-Means Clustering
        ↓
Model Evaluation
        ↓
Risk Topic Analysis
```

### RAG Question Answering

```text
User Question
        ↓
Search Apple 10-K Filings
        ↓
Retrieve Relevant Evidence
        ↓
Generate an Evidence-Based Answer
        ↓
Display Answer and Source Evidence
```

## Technology Stack

- Python 3.12.13
- SEC EDGAR API
- pandas and PyArrow
- DuckDB and SQL
- Streamlit and Plotly
- scikit-learn
- OpenAI API and File Search
- pytest
- Git and GitHub

## Project Structure

```text
apple-10k-ai/
├── app.py
├── financial_analysis.py
├── rag_service.py
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── extract_sec_data.py
│   ├── transform_financials.py
│   ├── validate_data.py
│   ├── load_warehouse.py
│   ├── extract_risk_factors.py
│   └── model_risk_topics.py
├── data/
│   ├── filings/
│   ├── raw/
│   └── processed/
├── warehouse/
├── models/
├── outputs/
├── logs/
├── assets/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## Core Financial Metrics

The first version of the project focuses on five financial metrics:

- Revenue / Net Sales
- Operating Income
- Net Income
- Operating Cash Flow
- Operating Margin

Operating margin is calculated as:

```text
Operating Margin = Operating Income / Revenue × 100%
```

Revenue growth is calculated as:

```text
Revenue Growth = (Current Revenue / Previous Revenue - 1) × 100%
```

## Data Engineering Design

The project follows a layered data architecture:

- `data/raw`: unchanged responses retrieved from the SEC API
- `data/processed`: cleaned and standardized Parquet data
- `warehouse`: curated financial data stored in DuckDB
- `logs`: pipeline execution and validation records

The pipeline is designed to be:

- Reproducible
- Traceable
- Idempotent
- Validated before loading
- Safe to rerun without creating duplicate records

## Data Science Design

Risk paragraphs from Apple’s 10-K filings will be converted into numerical features using TF-IDF.

K-Means will group similar risk paragraphs into topics. The number of clusters will be selected using silhouette scores together with manual interpretation.

Truncated SVD will be used to visualize the high-dimensional text features in two dimensions.

Because the project only covers three fiscal years, it will not use those three observations to make unreliable stock-price or revenue forecasts.

## RAG Design

The RAG component will retrieve relevant evidence from Apple’s 10-K filings before generating an answer.

The application will:

- Answer only from the provided SEC filings
- Display the filing used as evidence
- Show retrieved excerpts when available
- Avoid inventing page numbers or unsupported citations
- State clearly when the available evidence is insufficient

## Current Status

- [x] Project scope defined
- [x] Python 3.12 environment created
- [x] Project folder structure initialized
- [x] Core dependencies installed and tested
- [x] Git repository initialized
- [x] Environment files protected
- [ ] SEC data extraction
- [ ] Financial data transformation
- [ ] Data quality validation
- [ ] DuckDB loading
- [ ] Financial dashboard
- [ ] RAG question answering
- [ ] Risk Factor extraction
- [ ] Risk topic modeling
- [ ] Full-pipeline testing and evaluation

## Data Source

Financial data and company filings will be obtained from the U.S. Securities and Exchange Commission’s EDGAR system.

Apple’s Central Index Key (CIK) is:

```text
0000320193
```

## Environment Setup

Create and activate the project environment:

```bash
conda create -n apple10k python=3.12 -y
conda activate apple10k
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and provide the required values:

```text
OPENAI_API_KEY=
OPENAI_VECTOR_STORE_ID=
SEC_USER_AGENT=Your Name your.email@example.com
```

Do not commit the real `.env` file to GitHub.

## Planned Usage

Run the financial data pipeline:

```bash
python run_pipeline.py
```

Launch the application:

```bash
streamlit run app.py
```

These commands will become functional as the corresponding components are implemented.

## Limitations

- The initial version supports only Apple
- Financial analysis covers FY2023–FY2025
- Risk-topic clusters require human interpretation
- RAG answers depend on the quality of retrieved evidence
- The application does not provide investment recommendations
- The current version does not predict stock prices or future financial results

## Disclaimer

This project is for educational and portfolio purposes only. It does not constitute financial or investment advice.