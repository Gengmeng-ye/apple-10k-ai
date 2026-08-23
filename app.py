"""Modern Streamlit dashboard for Apple financial and risk analysis."""

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from financial_analysis import calculate_financial_metrics, load_financial_data


HERO_IMAGE = Path("assets/apple_energy_hero.png")
RISK_CHUNKS_FILE = Path("data/processed/apple_risk_chunks.json")

st.set_page_config(
    page_title="Apple 10-K Financial Analyst",
    page_icon="🍎",
    layout="wide",
)


def render_html(html: str) -> None:
    """Render compact HTML without Markdown indentation issues."""
    compact_html = " ".join(line.strip() for line in html.splitlines())
    st.markdown(compact_html, unsafe_allow_html=True)


def load_risk_chunks() -> pd.DataFrame:
    """Load the modeled Risk Factors chunks."""
    with RISK_CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def create_financial_trend_chart(data: pd.DataFrame) -> go.Figure:
    """Create the five-year financial trend chart."""
    years = pd.to_datetime(data["end"]).dt.year
    figure = go.Figure()
    series = [
        ("Revenue", "revenue_billions", "#2F5FD0", 3.2, "circle"),
        ("Operating Income", "operating_income_billions", "#2FA8E0", 3.0, "square"),
        ("Net Income", "net_income_billions", "#6F8FBF", 2.8, "diamond"),
    ]
    for name, column, color, width, symbol in series:
        figure.add_trace(
            go.Scatter(
                x=years,
                y=data[column],
                name=name,
                mode="lines+markers",
                line={"color": color, "width": width},
                marker={
                    "size": 8,
                    "color": color,
                    "symbol": symbol,
                    "line": {"color": "#FFFFFF", "width": 1.2},
                },
                hovertemplate=f"{name}: $%{{y:.2f}}B<extra></extra>",
            )
        )
    figure.update_layout(
        height=380,
        margin={"l": 25, "r": 18, "t": 52, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis={
            "tickmode": "array",
            "tickvals": years.tolist(),
            "ticktext": [f"FY{year}" for year in years],
            "showgrid": False,
            "linecolor": "#E3E3E3",
        },
        yaxis={
            "title": "USD billions",
            "gridcolor": "#EEEEEE",
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "center",
            "x": 0.5,
        },
        font={
            "family": "Avenir Next, Helvetica Neue, Arial, sans-serif",
            "color": "#4E555A",
            "size": 13,
        },
    )
    return figure


def create_margin_chart(data: pd.DataFrame) -> go.Figure:
    """Create the five-year profitability chart."""
    years = pd.to_datetime(data["end"]).dt.year
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=years,
            y=data["operating_margin_pct"],
            name="Operating Margin",
            marker_color="#C5C9EF",
            opacity=0.88,
            hovertemplate="Operating Margin: %{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=years,
            y=data["net_profit_margin_pct"],
            name="Net Profit Margin",
            mode="lines+markers",
            line={"color": "#198F88", "width": 4},
            marker={
                "size": 11,
                "color": "#198F88",
                "line": {"color": "#FFFFFF", "width": 1.8},
            },
            hovertemplate="Net Profit Margin: %{y:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=380,
        margin={"l": 25, "r": 18, "t": 52, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        bargap=0.52,
        xaxis={
            "tickmode": "array",
            "tickvals": years.tolist(),
            "ticktext": [f"FY{year}" for year in years],
            "showgrid": False,
            "linecolor": "#E3E3E3",
        },
        yaxis={"title": "Margin (%)", "gridcolor": "#EEEEEE", "zeroline": False},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "center",
            "x": 0.5,
        },
        font={
            "family": "Avenir Next, Helvetica Neue, Arial, sans-serif",
            "color": "#4E555A",
            "size": 13,
        },
    )
    return figure


def create_risk_topic_chart(risk_summary: pd.DataFrame) -> go.Figure:
    """Create a compact chart showing Risk Factors topic coverage."""
    chart_data = risk_summary.sort_values("chunk_count", ascending=True)
    maximum_count = int(chart_data["chunk_count"].max())
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=chart_data["chunk_count"],
            y=chart_data["topic_label"],
            orientation="h",
            marker={"color": "#91BCE8"},
            text=[f"{count} excerpts" for count in chart_data["chunk_count"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>%{x} filing excerpts<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 8, "r": 82, "t": 12, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.5,
        xaxis={"visible": False, "range": [0, maximum_count * 1.22]},
        yaxis={
            "title": None,
            "showgrid": False,
            "tickfont": {"size": 12, "color": "#525960"},
        },
        font={
            "family": "Avenir Next, Helvetica Neue, Arial, sans-serif",
            "color": "#4E555A",
            "size": 12,
        },
    )
    return figure


# Load and prepare data.
data = calculate_financial_metrics(load_financial_data()).reset_index(drop=True)
risk_chunks = load_risk_chunks()
risk_summary = (
    risk_chunks.groupby(["topic_id", "topic_label"], as_index=False)
    .agg(
        chunk_count=("chunk_id", "count"),
        average_score=("topic_score", "mean"),
    )
    .sort_values("chunk_count", ascending=False)
    .reset_index(drop=True)
)

latest = data.iloc[-1]
previous = data.iloc[-2]
first = data.iloc[0]
latest_year = pd.to_datetime(latest["end"]).year
revenue_cagr = (
    (latest["revenue_billions"] / first["revenue_billions"])
    ** (1 / (len(data) - 1))
    - 1
) * 100
margin_change = latest["operating_margin_pct"] - previous["operating_margin_pct"]
latest_cash_flow_growth = (
    latest["operating_cash_flow_billions"]
    / previous["operating_cash_flow_billions"]
    - 1
) * 100


render_html(
    """
    <style>
        :root {
            --ink: #182126; --ink-soft: #676D71; --paper: #FFFFFF;
            --line: #D9DAE7; --primary: #5B63C9;
            --primary-dark: #41488F; --positive: #2E9D78;
            --negative: #E05A5A;
        }
        .stApp {
            background: var(--paper); color: var(--ink);
            font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
        }
        .block-container { max-width: 1220px; padding-top: 3rem; padding-bottom: 4rem; }
        header[data-testid="stHeader"] { background: rgba(255,255,255,.72); }
        .topbar {
            display:flex; align-items:center; justify-content:space-between;
            padding:.35rem 0 1.25rem; border-bottom:1px solid var(--line);
        }
        .brand { display:flex; align-items:center; gap:.75rem; font-size:.8rem; font-weight:650; }
        .brand-mark {
            width:34px; height:34px; display:inline-flex; align-items:center;
            justify-content:center; border:1px solid #B9B7B1; border-radius:50%;
            font-size:.7rem; font-weight:700;
        }
        .data-status { color:var(--ink-soft); font-size:.78rem; }
        div[data-testid="stImage"] { margin-top:1.35rem; }
        div[data-testid="stImage"] img {
            width:100%; height:300px; object-fit:cover; object-position:center 49%;
            border-radius:10px;
        }
        .hero-copy { max-width:1120px; margin:2.25rem auto 0; text-align:center; }
        .eyebrow {
            color:var(--primary); font-size:.76rem; font-weight:700;
            letter-spacing:.15rem; text-transform:uppercase; margin-bottom:.95rem;
        }
        .hero-title {
            color:var(--ink); font-size:clamp(2.45rem,3.7vw,3.15rem);
            font-weight:680; line-height:1.08; letter-spacing:-.105rem;
            margin-bottom:1.05rem; white-space:nowrap;
        }
        .hero-title em { color:var(--primary); font:inherit; font-style:normal; }
        .hero-description {
            max-width:680px; margin:0 auto; color:var(--ink-soft);
            font-size:1rem; line-height:1.65;
        }
        .hero-tags {
            display:flex; justify-content:center; flex-wrap:wrap;
            gap:.55rem; margin-top:1.35rem;
        }
        .hero-tag {
            padding:.38rem .68rem; border:1px solid var(--line);
            border-radius:999px; background:#FFF; color:#565C60; font-size:.76rem;
        }
        .section-heading { margin:2.5rem auto 1.25rem; text-align:center; }
        .section-kicker {
            color:var(--primary); font-size:1.9rem; font-weight:650;
            letter-spacing:-.035rem; text-transform:uppercase; margin-bottom:.45rem;
        }
        .section-title { color:var(--ink); font-size:1.9rem; font-weight:650; letter-spacing:-.035rem; }
        .section-description {
            max-width:680px; margin:.6rem auto 0; color:var(--ink-soft);
            font-size:.96rem; line-height:1.6;
        }
        .metric-card {
            min-height:142px; padding:1.35rem; background:#FFF;
            border:1px solid var(--line); border-radius:10px;
        }
        .metric-label {
            color:#6D7377; font-size:.73rem; font-weight:700;
            letter-spacing:.075rem; text-transform:uppercase;
        }
        .metric-value { color:var(--ink); font-size:1.85rem; font-weight:650; margin-top:.72rem; }
        .metric-note { color:var(--ink-soft); font-size:.79rem; margin-top:.42rem; }
        .year-selector-label {
            margin:-.2rem 0 .42rem; color:var(--ink-soft); font-size:.76rem;
            font-weight:700; letter-spacing:.06rem; text-align:center; text-transform:uppercase;
        }
        div[data-testid="stSelectbox"] { margin:0 auto .75rem; }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height:52px; background:#FFF; border:1.5px solid var(--line)!important;
            border-radius:12px; box-shadow:0 8px 22px rgba(65,72,143,.08);
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color:var(--primary)!important;
            box-shadow:0 0 0 3px rgba(91,99,201,.14)!important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child {
            flex:1 1 auto!important; justify-content:center!important;
            text-align:center!important; padding-left:38px!important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child * {
            text-align:center!important;
        }
        div[data-testid="stPlotlyChart"] {
            padding:.75rem; background:#FFF; border:1px solid var(--line); border-radius:10px;
        }
        div[data-testid="stTabs"] button { color:var(--ink-soft); font-size:.92rem; }
        .stApp button[role="tab"][aria-selected="true"],
        .stApp div[data-baseweb="tab-list"] button[aria-selected="true"],
        .stApp div[data-testid="stTabs"] button[aria-selected="true"] {
            color:var(--primary-dark)!important; font-weight:700;
            box-shadow:inset 0 -2px 0 var(--primary)!important;
        }
        .stApp div[data-baseweb="tab-highlight"] { background-color:var(--primary)!important; }
        .takeaway-wrap {
            margin-top:1rem; padding:1.15rem 1.25rem; background:#FFF;
            border:1px solid var(--line); border-radius:10px;
        }
        .takeaway-header {
            display:flex; align-items:baseline; justify-content:space-between; gap:1rem;
            padding-bottom:.85rem; border-bottom:1px solid var(--line);
        }
        .takeaway-title { color:var(--ink); font-size:1.05rem; font-weight:700; }
        .takeaway-summary { color:var(--ink-soft); font-size:.9rem; }
        .takeaway-grid { display:grid; grid-template-columns:repeat(4,1fr); margin-top:1rem; }
        .takeaway-item { padding:0 1rem; border-left:1px solid var(--line); }
        .takeaway-item:first-child { padding-left:0; border-left:0; }
        .takeaway-value { color:var(--ink); font-size:1.55rem; font-weight:680; }
        .takeaway-label { color:var(--ink-soft); font-size:.86rem; margin-top:.3rem; line-height:1.45; }
        .positive { color:var(--positive); } .negative { color:var(--negative); }
        .table-shell { width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:10px; }
        .financial-table {
            width:100%; table-layout:fixed; border-collapse:collapse; color:var(--ink);
            font-family:"Avenir Next","Helvetica Neue",Arial,sans-serif; font-size:.81rem;
        }
        .financial-table th,.financial-table td {
            padding:.9rem .65rem; border-right:1px solid var(--line);
            border-bottom:1px solid var(--line); text-align:center!important;
        }
        .financial-table th { background:#F6F6F8; color:#555C61; font-weight:700; font-size:.76rem; }
        .financial-table th:nth-child(1),.financial-table td:nth-child(1) { width:9%; }
        .financial-table th:nth-child(2),.financial-table td:nth-child(2) { width:10%; }
        .financial-table th:nth-child(3),.financial-table td:nth-child(3) { width:14%; }
        .financial-table th:nth-child(4),.financial-table td:nth-child(4) { width:11%; }
        .financial-table th:nth-child(5),.financial-table td:nth-child(5) { width:16%; }
        .financial-table th:nth-child(6),.financial-table td:nth-child(6) { width:15%; }
        .financial-table th:nth-child(7),.financial-table td:nth-child(7) { width:11%; }
        .financial-table th:nth-child(8),.financial-table td:nth-child(8) { width:14%; }
        .financial-table th:last-child,.financial-table td:last-child { border-right:0; }
        .financial-table tbody tr:last-child td { border-bottom:0; }
        .risk-column-label {
            color:var(--primary-dark); font-size:.75rem; font-weight:700;
            letter-spacing:.07rem; text-transform:uppercase; margin:.15rem 0 .65rem;
        }
        .risk-summary-control {
            height:40px!important; min-height:40px!important;
            max-height:40px!important; box-sizing:border-box; display:flex;
            align-items:center; justify-content:space-between; gap:1rem;
            padding:0 1rem; background:#F0F2F6; border: none;
            border-radius:12px; box-shadow: none; color:#31333F; font-size:.82rem;
        }
        .risk-summary-control strong { color:var(--ink); font-size:.95rem; }
        .risk-detail {
            height:356px!important; max-height:356px!important;
            box-sizing:border-box; overflow:hidden!important;
            display:flex; flex-direction:column;
            padding:1.2rem 1.25rem; background:#F8FAFD;
            border:1px solid #D8E4F0; border-radius:10px;
        }
        .risk-detail-title { color:var(--ink); font-size:1rem; font-weight:700; margin-bottom:.65rem; }
        .risk-detail-text {
            flex:1 1 auto; min-height:0; overflow-y:auto; padding-right:.4rem;
            color:#555D63; font-size:.84rem; line-height:1.62;
        }
        .risk-detail-source {
            flex:0 0 auto; color:#7A8288; font-size:.72rem;
            margin-top:.75rem; padding-top:.75rem;
            border-top:1px solid #D8E4F0;
        }
        .risk-note { color:#7A8288; font-size:.74rem; line-height:1.5; margin-top:.65rem; }
        .footer {
            margin-top:3.5rem; padding-top:1.25rem; border-top:1px solid var(--line);
            color:#747A7E; font-size:.76rem; line-height:1.7; text-align:center;
        }
        @media (max-width:900px) {
            div[data-testid="stImage"] img { height:230px; }
            .hero-title { font-size:2.55rem; white-space:normal; }
            .takeaway-grid { grid-template-columns:repeat(2,1fr); gap:1rem 0; }
            .data-status { display:none; }
            .financial-table { min-width:920px; table-layout:auto; }
        }
    </style>
    """
)


# Top bar and hero.
render_html(
    f"""
    <div class="topbar">
        <div class="brand"><span class="brand-mark">GY</span><span>Apple 10-K Financial Analyst</span></div>
        <div class="data-status">Official SEC data · Updated through FY{latest_year}</div>
    </div>
    """
)
st.image(str(HERO_IMAGE), use_container_width=True)
render_html(
    """
    <div class="hero-copy">
        <div class="eyebrow">Finance × Data × Visualization</div>
        <div class="hero-title"><em>Apple’s</em> Financial Performance, Visualized.</div>
        <div class="hero-description">
            This dashboard turns Apple’s SEC filings into an interactive view of its financial performance.
            Select a fiscal year to review the results and see how they have changed over time.
        </div>
        <div class="hero-tags">
            <span class="hero-tag">SEC EDGAR</span><span class="hero-tag">Python ETL</span>
            <span class="hero-tag">DuckDB</span><span class="hero-tag">Financial Analysis</span>
        </div>
    </div>
    """
)


# Financial snapshot.
render_html(
    """
    <div class="section-heading">
        <div class="section-kicker">01 · Overview</div>
        <div class="section-title">Financial snapshot</div>
        <div class="section-description">
            Choose a fiscal year to review Apple’s scale, profitability, and operating cash generation.
        </div>
    </div>
    """
)
fiscal_years = pd.to_datetime(data["end"]).dt.year.tolist()
_, selector_center, _ = st.columns([1.45, 0.55, 1.45])
with selector_center:
    render_html('<div class="year-selector-label">Fiscal year</div>')
    selected_year = st.selectbox(
        "Fiscal year",
        options=list(reversed(fiscal_years)),
        index=0,
        label_visibility="collapsed",
    )

selected_position = fiscal_years.index(selected_year)
selected = data.iloc[selected_position]
selected_previous = data.iloc[selected_position - 1] if selected_position > 0 else None
selected_revenue_growth = selected["revenue_growth_pct"]
revenue_growth_note = (
    f"{selected_revenue_growth:+.2f}% year-over-year"
    if pd.notna(selected_revenue_growth)
    else "First year in selected range"
)
selected_cash_flow_growth = (
    (selected["operating_cash_flow_billions"] / selected_previous["operating_cash_flow_billions"] - 1) * 100
    if selected_previous is not None
    else None
)
cash_flow_note = (
    f"{selected_cash_flow_growth:+.2f}% year-over-year"
    if selected_cash_flow_growth is not None
    else "First year in selected range"
)

metrics = [
    ("Revenue", f"${selected['revenue_billions']:.2f}B", revenue_growth_note),
    ("Operating Income", f"${selected['operating_income_billions']:.2f}B", f"{selected['operating_margin_pct']:.2f}% operating margin"),
    ("Net Income", f"${selected['net_income_billions']:.2f}B", f"{selected['net_profit_margin_pct']:.2f}% net margin"),
    ("Operating Cash Flow", f"${selected['operating_cash_flow_billions']:.2f}B", cash_flow_note),
]
for column, (label, value, note) in zip(st.columns(4), metrics):
    with column:
        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """
        )


# Five-year financial analysis.
render_html(
    """
    <div class="section-heading">
        <div class="section-kicker">02 · Analysis</div>
        <div class="section-title">Five-year financial direction</div>
        <div class="section-description">
            Explore Apple’s growth, profitability, and underlying financial data across the latest five fiscal years.
        </div>
    </div>
    """
)
trend_tab, margin_tab, data_tab = st.tabs(["Financial trends", "Profitability", "Underlying data"])

with trend_tab:
    st.plotly_chart(create_financial_trend_chart(data), use_container_width=True, config={"displayModeBar": False})
    render_html(
        f"""
        <div class="takeaway-wrap">
            <div class="takeaway-header">
                <div class="takeaway-title">Key Takeaways</div>
                <div class="takeaway-summary">Growth strengthened in FY{latest_year}, while cash generation softened.</div>
            </div>
            <div class="takeaway-grid">
                <div class="takeaway-item"><div class="takeaway-value positive">{latest['revenue_growth_pct']:+.2f}%</div><div class="takeaway-label">Latest annual revenue growth</div></div>
                <div class="takeaway-item"><div class="takeaway-value">{revenue_cagr:.2f}%</div><div class="takeaway-label">Five-year revenue CAGR</div></div>
                <div class="takeaway-item"><div class="takeaway-value positive">{margin_change:+.2f} pp</div><div class="takeaway-label">Annual operating-margin change</div></div>
                <div class="takeaway-item"><div class="takeaway-value negative">{latest_cash_flow_growth:+.2f}%</div><div class="takeaway-label">Annual operating-cash-flow change</div></div>
            </div>
        </div>
        """
    )

with margin_tab:
    st.plotly_chart(create_margin_chart(data), use_container_width=True, config={"displayModeBar": False})

with data_tab:
    display_data = data.copy()
    display_data["Fiscal Year"] = pd.to_datetime(display_data["end"]).dt.year
    display_data = display_data[
        ["Fiscal Year", "revenue_billions", "operating_income_billions", "net_income_billions",
         "operating_cash_flow_billions", "operating_margin_pct", "net_profit_margin_pct", "revenue_growth_pct"]
    ].rename(
        columns={
            "revenue_billions": "Revenue ($B)",
            "operating_income_billions": "Operating Income ($B)",
            "net_income_billions": "Net Income ($B)",
            "operating_cash_flow_billions": "Operating Cash Flow ($B)",
            "operating_margin_pct": "Operating Margin (%)",
            "net_profit_margin_pct": "Net Margin (%)",
            "revenue_growth_pct": "Revenue Growth (%)",
        }
    )
    table_html = display_data.round(2).to_html(
        index=False, classes="financial-table", border=0, na_rep="—"
    )
    render_html(f'<div class="table-shell">{table_html}</div>')


# FY2025 Risk Analysis.
render_html(
    f"""
    <div class="section-heading">
        <div class="section-kicker">
            03 · Risk Analysis
        </div>

        <div class="section-title">
            Key risk themes in Apple’s {latest_year} 10-K
        </div>

        <div class="section-description">
            Explore recurring Item 1A risk themes
            and the filing excerpts behind them.
        </div>
    </div>
    """
)

risk_control_left, risk_control_right = st.columns([1, 1], gap="large")
with risk_control_left:
    render_html('<div class="risk-column-label">Theme coverage</div>')
    render_html(
        f"""
        <div class="risk-summary-control">
            <span><strong>{len(risk_chunks)}</strong> filing excerpts</span>
            <span><strong>{len(risk_summary)}</strong> themes</span>
        </div>
        """
    )

with risk_control_right:
    render_html('<div class="risk-column-label">Explore a theme</div>')
    selected_topic = st.selectbox(
        "Risk theme",
        options=risk_summary["topic_label"].tolist(),
        label_visibility="collapsed",
        key="risk_topic_selector",
    )

selected_topic_chunks = risk_chunks[
    risk_chunks["topic_label"] == selected_topic
].sort_values("topic_score", ascending=False)
representative_chunk = selected_topic_chunks.iloc[0]
excerpt = str(representative_chunk["text"]).strip()

risk_chart_column, risk_detail_column = st.columns([1, 1], gap="large")
with risk_chart_column:
    st.plotly_chart(
        create_risk_topic_chart(risk_summary),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    render_html(
        '<div class="risk-note">Bars show the number of filing excerpts assigned to each theme, not the severity of the risk.</div>'
    )

with risk_detail_column:
    render_html(
        f"""
        <div class="risk-detail">
            <div class="risk-detail-title">{escape(selected_topic)}</div>
            <div class="risk-detail-text">{escape(excerpt)}</div>
            <div class="risk-detail-source">Source: Apple FY{latest_year} Form 10-K · Item 1A</div>
        </div>
        """
    )


render_html(
    """
    <div class="footer">
        Data source: U.S. Securities and Exchange Commission Company Facts API and Apple Form 10-K.
        Financial figures are presented in USD billions.<br>
        Built by Gengmeng Ye for educational and demonstration purposes.
        This application does not constitute investment advice.
    </div>
    """
)