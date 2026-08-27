"""Apple financial and 10-K risk analysis dashboard.

Version: refined conversation workspace v21
"""

import json
import re
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from financial_analysis import calculate_financial_metrics, load_financial_data
from rag_service import get_answer

HERO_IMAGE = Path("assets/apple_energy_hero.png")
RISK_FILE = Path("data/processed/apple_risk_chunks.json")

st.set_page_config(page_title="Apple 10-K Financial Analyst", page_icon="🍎", layout="wide")


@st.cache_data(show_spinner=False, ttl=900)
def get_cached_answer(question: str) -> str:
    """Reuse identical grounded answers briefly to avoid repeat API waits."""
    return get_answer(question)


def render_html(html: str) -> None:
    """Render compact HTML."""
    st.markdown(" ".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def fill_example_question(question: str) -> None:
    """Copy a suggested question into the question input."""
    st.session_state.qa_input = question


def queue_question() -> None:
    """Queue one question from either Enter or the send button."""
    question = st.session_state.get("qa_input", "").strip()

    if question:
        st.session_state.pending_question = question
        st.session_state.qa_input = ""


def trend_chart(data: pd.DataFrame, mobile: bool = False) -> go.Figure:
    """Create the five-year financial chart."""
    years = pd.to_datetime(data["end"]).dt.year
    figure = go.Figure()
    series = [
        ("Revenue", "revenue_billions", "#2F5FD0", "circle"),
        ("Operating Income", "operating_income_billions", "#2FA8E0", "square"),
        ("Net Income", "net_income_billions", "#6F8FBF", "diamond"),
    ]
    for name, column, color, symbol in series:
        figure.add_scatter(
            x=years, y=data[column], name=name, mode="lines+markers",
            line={"color": color, "width": 3},
            marker={"size": 8, "color": color, "symbol": symbol, "line": {"color": "white", "width": 1}},
            hovertemplate=f"{name}: $%{{y:.2f}}B<extra></extra>",
        )
    year_values = years.tolist()
    figure.update_layout(
        height=300 if mobile else 400,
        margin={"l": 8, "r": 12, "t": 54, "b": 38} if mobile else {"l": 20, "r": 32, "t": 56, "b": 68},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        xaxis={
            "tickvals": year_values,
            "ticktext": [f"FY{year}" for year in years],
            "range": [min(year_values) - .55, max(year_values) + .55],
            "showgrid": False,
            "automargin": True,
        },
        yaxis={"gridcolor": "#EEEEEE", "zeroline": False, "automargin": True},
        annotations=[{
            "text": "USD billions", "xref": "paper", "yref": "paper",
            "x": 0, "y": 1.08 if mobile else 1.12, "xanchor": "center", "yanchor": "bottom",
            "showarrow": False, "font": {"size": 9 if mobile else 12, "color": "#7A8297"},
        }],
        legend={
            "orientation": "h", "y": 1.13 if mobile else 1.16,
            "x": .56 if mobile else .5, "xanchor": "center",
            "font": {"size": 9 if mobile else 13},
        },
        font={"family": "Avenir Next, Helvetica Neue, Arial", "color": "#4E555A", "size": 9 if mobile else 13},
    )
    return figure


def margin_chart(data: pd.DataFrame, mobile: bool = False) -> go.Figure:
    """Create the profitability chart."""
    years = pd.to_datetime(data["end"]).dt.year
    figure = go.Figure()
    figure.add_bar(
        x=years, y=data["operating_margin_pct"], name="Operating Margin",
        marker_color="#C5C9EF", opacity=.88,
        hovertemplate="Operating Margin: %{y:.2f}%<extra></extra>",
    )
    figure.add_scatter(
        x=years, y=data["net_profit_margin_pct"], name="Net Profit Margin", mode="lines+markers",
        line={"color": "#198F88", "width": 4},
        marker={"size": 10, "color": "#198F88", "line": {"color": "white", "width": 1.5}},
        hovertemplate="Net Profit Margin: %{y:.2f}%<extra></extra>",
    )
    year_values = years.tolist()
    figure.update_layout(
        height=300 if mobile else 400,
        margin={"l": 8, "r": 12, "t": 48, "b": 38} if mobile else {"l": 20, "r": 32, "t": 56, "b": 68},
        bargap=.58 if mobile else .52,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        xaxis={
            "tickvals": year_values,
            "ticktext": [f"FY{year}" for year in years],
            "range": [min(year_values) - .55, max(year_values) + .55],
            "showgrid": False,
            "automargin": True,
        },
        yaxis={"gridcolor": "#EEEEEE", "zeroline": False, "automargin": True},
        annotations=[{
            "text": "Margin (%)", "xref": "paper", "yref": "paper",
            "x": 0, "y": 1.07 if mobile else 1.12, "xanchor": "center", "yanchor": "bottom",
            "showarrow": False, "font": {"size": 9 if mobile else 12, "color": "#7A8297"},
        }],
        legend={"orientation": "h", "y": 1.12 if mobile else 1.16, "x": .56 if mobile else .5, "xanchor": "center", "font": {"size": 9 if mobile else 13}},
        font={"family": "Avenir Next, Helvetica Neue, Arial", "color": "#4E555A", "size": 9 if mobile else 13},
    )
    return figure


def assets_liabilities_chart(data: pd.DataFrame, mobile: bool = False) -> go.Figure:
    """Compare total assets and total liabilities by fiscal year."""
    years = pd.to_datetime(data["end"]).dt.year
    year_values = years.tolist()
    figure = go.Figure()
    figure.add_bar(
        x=years,
        y=data["total_assets_billions"],
        name="Total Assets",
        marker_color="#82A5DF",
        hovertemplate="Total Assets: $%{y:.2f}B<extra></extra>",
    )
    figure.add_bar(
        x=years,
        y=data["total_liabilities_billions"],
        name="Total Liabilities",
        marker_color="#BDCCEB",
        hovertemplate="Total Liabilities: $%{y:.2f}B<extra></extra>",
    )
    figure.update_layout(
        height=285 if mobile else 390,
        margin={"l": 8, "r": 10, "t": 64, "b": 34} if mobile else {"l": 20, "r": 34, "t": 60, "b": 62},
        barmode="group",
        bargap=.34,
        bargroupgap=.08,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "tickvals": year_values,
            "ticktext": [f"FY{year}" for year in years],
            "range": [min(year_values) - .58, max(year_values) + .58],
            "showgrid": False,
            "automargin": True,
        },
        yaxis={
            "gridcolor": "#EEEEEE",
            "zeroline": False,
            "automargin": True,
        },
        annotations=([{
            "text": "<b>Assets and liabilities</b>", "xref": "paper", "yref": "paper",
            "x": .5, "y": 1.25, "xanchor": "center", "yanchor": "top",
            "showarrow": False, "font": {"size": 11, "color": "#182126"},
        }] if mobile else []) + [{
            "text": "USD billions", "xref": "paper", "yref": "paper",
            "x": 0, "y": 1.03 if mobile else 1.12, "xanchor": "left", "yanchor": "bottom",
            "showarrow": False, "font": {"size": 9 if mobile else 12, "color": "#7A8297"},
        }],
        legend={"orientation": "h", "y": 1.14 if mobile else 1.16, "x": .56 if mobile else .5, "xanchor": "center", "font": {"size": 9 if mobile else 12}},
        font={
            "family": "Avenir Next, Helvetica Neue, Arial",
            "color": "#4E555A",
            "size": 9 if mobile else 12,
        },
    )
    return figure


def cash_position_chart(data: pd.DataFrame, mobile: bool = False) -> go.Figure:
    """Show the five-year cash and cash equivalents trend."""
    years = pd.to_datetime(data["end"]).dt.year
    year_values = years.tolist()
    figure = go.Figure()
    figure.add_scatter(
        x=years,
        y=data["cash_and_cash_equivalents_billions"],
        name="Cash and Cash Equivalents",
        mode="lines+markers",
        line={"color": "#198F88", "width": 4},
        marker={
            "size": 10,
            "color": "#198F88",
            "line": {"color": "white", "width": 1.5},
        },
        fill="tozeroy",
        fillcolor="rgba(25,143,136,.06)",
        hovertemplate="Cash: $%{y:.2f}B<extra></extra>",
    )
    figure.update_layout(
        height=285 if mobile else 390,
        margin={"l": 8, "r": 10, "t": 50, "b": 34} if mobile else {"l": 20, "r": 30, "t": 54, "b": 62},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "tickvals": year_values,
            "ticktext": [f"FY{year}" for year in years],
            "range": [min(year_values) - .58, max(year_values) + .58],
            "showgrid": False,
            "automargin": True,
        },
        yaxis={
            "range": [0, 42],
            "dtick": 10,
            "showgrid": False,
            "zeroline": False,
            "automargin": True,
        },
        shapes=[{
            "type": "line",
            "xref": "paper",
            "yref": "y",
            "x0": 0,
            "x1": 1,
            "y0": tick,
            "y1": tick,
            "layer": "below",
            "line": {"color": "#E7E9ED", "width": 1},
        } for tick in [0, 10, 20, 30, 40]],
        annotations=([{
            "text": "<b>Cash position</b>",
            "xref": "paper", "yref": "paper",
            "x": .5, "y": 1.20,
            "xanchor": "center", "yanchor": "top",
            "showarrow": False,
            "font": {"size": 11, "color": "#182126"},
        }] if mobile else []) + [{
            "text": "USD billions",
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": 1.04 if mobile else 1.12,
            "xanchor": "left",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 9 if mobile else 12, "color": "#7A8297"},
        }],
        font={
            "family": "Avenir Next, Helvetica Neue, Arial",
            "color": "#4E555A",
            "size": 9 if mobile else 12,
        },
    )
    return figure


def risk_chart(summary: pd.DataFrame, mobile: bool = False) -> go.Figure:
    """Create the risk-theme coverage chart."""
    chart_data = summary.sort_values("chunk_count")
    maximum = int(chart_data["chunk_count"].max())
    figure = go.Figure(go.Bar(
        x=chart_data["chunk_count"], y=chart_data["topic_label"], orientation="h",
        marker_color="#91BCE8", text=[f"{count} excerpts" for count in chart_data["chunk_count"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x} filing excerpts<extra></extra>",
    ))
    figure.update_layout(
        height=235 if mobile else 330,
        margin={"l": 4, "r": 46, "t": 14, "b": 14} if mobile else {"l": 28, "r": 82, "t": 4, "b": 38},
        bargap=.34 if mobile else .42, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [0, maximum * 1.22]},
        yaxis={
            "showgrid": False,
            "automargin": True,
            "tickfont": {"size": 9 if mobile else 12, "color": "#525960"},
        },
        font={"family": "Avenir Next, Helvetica Neue, Arial", "color": "#4E555A", "size": 9 if mobile else 12},
    )
    return figure


def format_answer_html(answer_text: str, message_index: int) -> str:
    """Convert grounded AI output into clean dashboard HTML."""
    parts = []

    def citation_target(label: str) -> str:
        """Return a unique reference anchor for one conversation turn."""
        return f"answer-{message_index}-reference-{label.lower()}"

    def format_inline_citations(text: str) -> str:
        """Render grounded citations as linked superscripts."""
        safe_text = escape(text)
        safe_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe_text)

        def replace_citation(match: re.Match) -> str:
            label = match.group(1)
            punctuation = match.group(2) or ""
            target = citation_target(label)
            return (
                '<span class="answer-citation-cluster">'
                f'<a class="answer-inline-citation" href="#{target}" '
                f'aria-label="Reference {label}"><sup>[{label}]</sup></a>'
                f'{escape(punctuation)}</span>'
            )

        return re.sub(
            r"\[((?:F|M)?\d+)\]([.,;:]?)",
            replace_citation,
            safe_text,
        )

    def format_bullet_text(text: str) -> str:
        """Emphasize a short bullet label without styling the whole sentence."""
        label_match = re.match(r"^([^:]{2,45}):\s+(.+)$", text)
        if not label_match:
            return format_inline_citations(text)
        label, details = label_match.groups()
        return (
            f'<span class="answer-bullet-label">{escape(label)}:</span> '
            f'{format_inline_citations(details)}'
        )

    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line or set(line) == {"="}:
            continue
        if line.startswith("Question:") or line.startswith("Route:"):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue

        reference_match = re.match(r"^\[((?:F|M)?\d+)\]\s*(.*)$", line)
        year_heading_match = re.match(r"^(FY20\d{2}):?(?:\s+(.*))?$", line)
        safe_line = format_inline_citations(line)

        if line.startswith("http://") or line.startswith("https://"):
            safe_url = escape(line, quote=True)
            parts.append(
                '<div class="answer-source-link">'
                f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">'
                "View Apple’s SEC filing &#8599;</a></div>"
            )
        elif line.startswith("Source:"):
            parts.append(f'<div class="answer-source">{safe_line}</div>')
        elif reference_match:
            label, description = reference_match.groups()
            target = citation_target(label)
            parts.append(
                f'<div class="answer-citation" id="{target}">'
                f'<span class="answer-reference-label">[{escape(label)}]</span> '
                f'{escape(description)}</div>'
            )
        elif line.startswith("Chunk "):
            parts.append(f'<div class="answer-evidence-title">{safe_line}</div>')
        elif year_heading_match:
            year_label, citation_text = year_heading_match.groups()
            parts.append(
                '<div class="answer-year-row">'
                f'<span class="answer-year">{escape(year_label)}</span>'
                f'{format_inline_citations(citation_text or "")}'
                '</div>'
            )
        elif line.endswith(":") or line in {
            "Retrieved Risk Factors evidence:",
            "Financial metrics by fiscal year:",
        }:
            parts.append(f'<div class="answer-heading">{safe_line}</div>')
        elif line.startswith("- "):
            parts.append(
                f'<div class="answer-bullet">{format_bullet_text(line[2:])}</div>'
            )
        else:
            parts.append(f'<div class="answer-line">{safe_line}</div>')
    return "".join(parts)


def render_chat_page() -> None:
    """Render the dedicated filing Q&A page and preserve conversation history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_input" not in st.session_state:
        st.session_state.qa_input = ""

    pending_question = st.session_state.pop("pending_question", None)

    render_html("""
    <div class="chat-page-heading">
        <div class="chat-page-title"><em>Apple’s</em> Financial Performance, Explained.</div>
        <div class="chat-page-description">Ask about financial results, fiscal-year comparisons, or disclosures from Apple’s latest 10-K.</div>
    </div>
    """)

    chat_workspace = st.container(border=True, key="chat_workspace",)
    with chat_workspace:
        if st.session_state.chat_history:
            history_heading, clear_column = st.columns([1, .13], vertical_alignment="center")
            with history_heading:
                render_html('<div class="conversation-label">Conversation</div>')
            with clear_column:
                clear_clicked = st.button("Clear", use_container_width=True, key="clear_chat")
            if clear_clicked:
                st.session_state.chat_history = []
                st.rerun()

            with st.container(border=False, key="conversation_panel"):
                for message_index, message in enumerate(st.session_state.chat_history):
                    render_html(
                        f"""
                        <div class="conversation-turn">
                            <div class="chat-user-row">
                                <div class="chat-user-bubble">{escape(message['question'])}</div>
                            </div>
                            <div class="chat-assistant-row">
                                <div class="chat-avatar" aria-hidden="true">
                                    <svg viewBox="0 0 48 48">
                                        <path d="M24 9v5"/><circle cx="24" cy="7" r="2.5"/>
                                        <rect x="10" y="14" width="28" height="24" rx="10"/>
                                        <circle cx="19" cy="25" r="2.4"/><circle cx="29" cy="25" r="2.4"/>
                                        <path d="M19 31c2.8 2.2 7.2 2.2 10 0"/>
                                    </svg>
                                </div>
                                <div class="chat-answer">
                                    <div class="chat-name">Apple Financial Analyst</div>
                                    {format_answer_html(message['answer'], message_index)}
                                </div>
                            </div>
                        </div>
                        """
                    )
        else:
            with st.container(height=125, border=False, key="conversation_empty"):
                render_html('<div class="chat-ready">Choose a prompt or ask your own question below.</div>')

        with st.container(key="chat_controls"):
            with st.container(key="prompt_suggestions"):
                prompt_left, example_one, example_two, example_three, prompt_right = st.columns([.35, 1, 1, 1, .35], gap="small")
                examples = [
                    (example_one, "Revenue: 2024 vs 2025", "Compare Apple's revenue in 2024 and 2025."),
                    (example_two, "Assets, liabilities & cash", "Compare Apple's assets, liabilities, and cash in 2023, 2024, and 2025."),
                    (example_three, "Supply-chain risks", "What supply chain risks does Apple disclose?"),
                ]
                for column, label, example_question in examples:
                    with column:
                        st.button(
                            label,
                            use_container_width=True,
                            on_click=fill_example_question,
                            args=(example_question,),
                            key=f"example_{example_question}",
                        )

            with st.container(key="question_composer"):
                input_column, send_column = st.columns([8, .46], gap="small")
                with input_column:
                    st.text_input(
                        "Question",
                        placeholder="Message Apple Financial Analyst...",
                        label_visibility="collapsed",
                        key="qa_input",
                        on_change=queue_question,
                    )
                with send_column:
                    st.button(
                        "↑",
                        use_container_width=True,
                        key="send_question",
                        on_click=queue_question,
                    )

    if pending_question:
        with st.spinner("Reviewing Apple’s financial data and 10-K..."):
            answer = get_cached_answer(pending_question)
        st.session_state.chat_history.append(
            {"question": pending_question, "answer": answer}
        )
        st.rerun()


data = calculate_financial_metrics(load_financial_data()).reset_index(drop=True)
with RISK_FILE.open("r", encoding="utf-8") as file:
    risk_chunks = pd.DataFrame(json.load(file))
risk_summary = (
    risk_chunks.groupby(["topic_id", "topic_label"], as_index=False)
    .agg(chunk_count=("chunk_id", "count"), average_score=("topic_score", "mean"))
    .sort_values("chunk_count", ascending=False).reset_index(drop=True)
)
latest, previous, first = data.iloc[-1], data.iloc[-2], data.iloc[0]
latest_year = pd.to_datetime(latest["end"]).year
revenue_cagr = ((latest["revenue_billions"] / first["revenue_billions"]) ** (1 / (len(data) - 1)) - 1) * 100
margin_change = latest["operating_margin_pct"] - previous["operating_margin_pct"]
cash_flow_growth = (latest["operating_cash_flow_billions"] / previous["operating_cash_flow_billions"] - 1) * 100


render_html("""
<style>
:root{--ink:#182126;--soft:#676D71;--line:#D9DAE7;--primary:#5B63C9;--dark:#41488F;--positive:#2E9D78;--negative:#E05A5A}
.stAppDeployButton,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],#MainMenu,footer{display:none!important;visibility:hidden!important}
.stApp{background:#FFF;color:var(--ink);font-family:"Avenir Next","Helvetica Neue",Arial,sans-serif}.block-container{max-width:1220px;padding-top:3.75rem!important;padding-bottom:3rem}
.topbar{min-height:48px;display:flex;align-items:center;justify-content:space-between;padding:.35rem 0 .8rem;border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:.75rem;font-size:.8rem;font-weight:650;line-height:1.25}.brand-mark{width:34px;height:34px;flex:0 0 34px;display:inline-flex;align-items:center;justify-content:center;border:1px solid #B9B7B1;border-radius:50%;font-size:.7rem;font-weight:700}.data-status{color:var(--soft);font-size:.78rem;line-height:1.25}
div[data-testid="stRadio"]{width:440px;margin:1.1rem auto .2rem}div[data-testid="stRadio"]>label{display:none!important}div[data-testid="stRadio"] div[role="radiogroup"]{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px;background:#F1F2F6;border:1px solid #E1E3EB;border-radius:11px}div[data-testid="stRadio"] div[role="radiogroup"]>label{min-height:38px;display:flex!important;align-items:center;justify-content:center;padding:.42rem .85rem!important;background:transparent;border:0;border-radius:8px;color:#666C72;font-size:.82rem;font-weight:620;cursor:pointer}div[data-testid="stRadio"] div[role="radiogroup"]>label:has(input:checked){background:#FFF;color:var(--dark);box-shadow:0 2px 8px rgba(37,42,65,.10)}div[data-testid="stRadio"] div[role="radiogroup"]>label>div:first-child{display:none!important}
div[data-testid="stImage"]{margin-top:.85rem}div[data-testid="stImage"] img{width:100%;height:270px;object-fit:cover;object-position:center 49%;border-radius:10px}.hero-copy{max-width:1120px;margin:1.35rem auto 0;text-align:center}.hero-title{font-size:clamp(2.45rem,3.7vw,3.15rem);font-weight:680;line-height:1.08;letter-spacing:-.105rem;white-space:nowrap}.hero-title em{color:var(--primary);font:inherit;font-style:normal}.hero-description{max-width:680px;margin:.8rem auto 0;color:var(--soft);font-size:1rem;line-height:1.65}
.section-heading{margin:1.9rem auto 1rem;text-align:center}.section-kicker{color:var(--primary);font-size:1.9rem;font-weight:650;text-transform:uppercase}.section-title{font-size:1.9rem;font-weight:650;letter-spacing:-.035rem;margin-top:.35rem}.section-description{max-width:680px;margin:.6rem auto 0;color:var(--soft);font-size:.96rem;line-height:1.6}
div[data-testid="stTabs"] button{color:var(--soft)!important;font-size:.92rem;border-bottom-color:transparent!important}div[data-testid="stTabs"] button p{color:inherit!important}div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--dark)!important;font-weight:700!important;border-bottom-color:var(--primary)!important;box-shadow:inset 0 -2px 0 var(--primary)!important}div[data-testid="stTabs"] button[aria-selected="true"] p{color:var(--dark)!important}
.year-label{text-align:center;color:var(--soft);font-size:.76rem;font-weight:700;letter-spacing:.06rem;text-transform:uppercase;margin:.5rem 0}.metric-card{min-height:142px;padding:1.35rem;border:1px solid var(--line);border-radius:10px}.metric-label{color:#6D7377;font-size:.73rem;font-weight:700;letter-spacing:.075rem;text-transform:uppercase}.metric-value{font-size:1.85rem;font-weight:650;margin-top:.72rem}.metric-note{color:var(--soft);font-size:.79rem;margin-top:.42rem}
div[data-testid="stSelectbox"] [data-baseweb="select"]>div{min-height:52px;background:#F0F2F6;border:0!important;border-radius:12px}div[data-testid="stSelectbox"] [data-baseweb="select"]>div>div:first-child{flex:1!important;justify-content:center!important;padding-left:38px!important}
.chat-page-heading{max-width:720px;margin:2.1rem auto 1.15rem;text-align:center}.chat-page-title{margin-top:.4rem;font-size:1.7rem;font-weight:680;letter-spacing:-.03rem}.chat-page-description{max-width:580px;margin:.45rem auto 0;color:var(--soft);font-size:.88rem;line-height:1.5}.conversation-label{max-width:800px;margin:1.25rem auto .2rem;color:#858A8E;font-size:.68rem;font-weight:700;letter-spacing:.08rem;text-transform:uppercase}.conversation-stream{max-width:800px;margin:0 auto}.conversation-turn{margin:0;padding:1.35rem 0 1.15rem;border-bottom:1px solid #ECEDEF}.conversation-turn:last-child{border-bottom:0}.chat-user-row{display:flex;justify-content:flex-end;margin-bottom:1.15rem}.chat-user-bubble{max-width:70%;padding:.65rem .9rem;background:#F0F1F6;border-radius:18px 18px 5px 18px;color:#30343A;font-size:.94rem;line-height:1.55}.chat-assistant-row{display:grid;grid-template-columns:30px minmax(0,1fr);gap:.75rem;align-items:start}.chat-avatar{width:30px;height:30px;display:flex;align-items:center;justify-content:center;background:var(--primary);border-radius:50%;color:#FFF;font-size:.61rem;font-weight:750}.chat-answer{max-width:720px;padding:.02rem 0;color:#343A3F;font-size:.96rem;line-height:1.72}.chat-name{margin-bottom:.42rem;color:var(--ink);font-size:.82rem;font-weight:700}.chat-ready{max-width:800px;margin:2.2rem auto 1rem;padding:1rem;text-align:center;color:#8A8F93;font-size:.84rem}.answer-heading{font-size:.9rem;font-weight:650;color:var(--ink);margin:.85rem 0 .34rem}.answer-year-row{display:flex;align-items:flex-start;gap:.06rem;margin:.78rem 0 .3rem}.answer-year{display:inline-flex;align-items:center;margin:0;padding:.18rem .56rem;background:#ECEEFB;border-radius:999px;color:var(--dark);font-size:.76rem;font-weight:650;letter-spacing:.01em}.answer-line{color:#454C51;margin:.24rem 0;line-height:1.62}.answer-bullet{position:relative;color:#454C51;margin:.28rem 0;padding-left:1rem;line-height:1.58}.answer-bullet:before{content:"";position:absolute;left:.1rem;top:.67rem;width:4px;height:4px;border-radius:50%;background:#6C73CF}.answer-evidence-title{margin:.85rem 0 .32rem;padding-top:.7rem;border-top:1px solid #ECEDEF;color:var(--dark);font-size:.82rem;font-weight:650}.answer-source,.answer-citation{color:#7A8288;font-size:.74rem;font-weight:400;line-height:1.5;margin:.2rem 0}.answer-inline-citation{display:inline-block;margin-left:.05rem;color:#6971B7!important;font-size:.66em;font-weight:500;line-height:0;text-decoration:none!important;border:0!important;vertical-align:super}.answer-inline-citation sup{font-size:inherit;font-weight:inherit;text-decoration:none!important}.answer-inline-citation:hover{color:#4D559E!important;text-decoration:none!important}.answer-reference-label{color:inherit;font-weight:inherit}.answer-source-link{margin-top:.38rem}.answer-source-link a{color:#626996;font-size:.74rem;font-weight:500;text-decoration:none}.answer-source-link a:hover{color:var(--dark);text-decoration:underline}.snapshot-subheading{margin:2.4rem auto 1rem;text-align:center}.snapshot-title{font-size:1.45rem;font-weight:680}.snapshot-description{margin:.45rem auto 0;color:var(--soft);font-size:.9rem}div[data-testid="stButton"] button{min-height:38px;background:#FFF;border:1px solid var(--line);border-radius:10px;color:#596066;font-size:.76rem}div[data-testid="stButton"] button:hover{background:#F7F7FA;border-color:#BFC4E8;color:var(--dark)}div[data-testid="stForm"]{position:sticky;bottom:1rem;z-index:20;max-width:800px;margin:1rem auto 0;padding:.42rem;background:#FFF;border:1px solid #D8DAE2;border-radius:18px;box-shadow:0 10px 32px rgba(29,34,55,.12)}div[data-testid="stTextInput"] input{min-height:48px;border:0!important;background:#FFF;border-radius:14px;font-size:.92rem}div[data-testid="stFormSubmitButton"] button{width:48px!important;height:48px;min-height:48px;margin:0;background:var(--primary);border:0;border-radius:14px;color:white;font-size:1.15rem;font-weight:700}
div[data-testid="stPlotlyChart"]{padding:.75rem;background:#FFF;border:1px solid var(--line);border-radius:10px}.takeaway-wrap{margin-top:1rem;padding:1.15rem 1.25rem;border:1px solid var(--line);border-radius:10px}.takeaway-header{display:flex;justify-content:space-between;padding-bottom:.85rem;border-bottom:1px solid var(--line)}.takeaway-title{font-size:1.05rem;font-weight:700}.takeaway-summary{color:var(--soft);font-size:.9rem}.takeaway-grid{display:grid;grid-template-columns:repeat(4,1fr);margin-top:1rem}.takeaway-item{padding:0 1.45rem;border-left:1px solid var(--line)}.takeaway-item:first-child{border:0;padding-left:.65rem}.takeaway-item:last-child{padding-right:.65rem}.takeaway-value{font-size:1.55rem;font-weight:680}.takeaway-label{color:var(--soft);font-size:.86rem;margin-top:.3rem}.positive{color:var(--positive)}.negative{color:var(--negative)}
.st-key-risk_coverage_chart,.st-key-risk_coverage_chart>div,.st-key-risk_coverage_chart div[data-testid="stPlotlyChart"]{overflow:hidden!important}
.balance-chart-title{margin:.15rem 0 .65rem;text-align:center;color:var(--ink);font-size:.9rem;font-weight:680}.balance-note{margin:.25rem auto 0;text-align:center;color:var(--soft);font-size:.74rem;line-height:1.5}
.table-shell{width:100%;overflow:hidden;border:1px solid var(--line);border-radius:10px}.financial-table{width:100%;border:0;border-collapse:separate;border-spacing:0;font-size:.81rem}.financial-table th,.financial-table td{padding:.9rem .55rem;border-right:1px solid var(--line);border-bottom:1px solid var(--line);text-align:center!important}.financial-table tbody tr:last-child td{border-bottom:0}.financial-table th:last-child,.financial-table td:last-child{border-right:0}.financial-table th{background:#F6F6F8;color:#555C61}.risk-label{color:var(--dark);font-size:.75rem;font-weight:700;letter-spacing:.07rem;text-transform:uppercase;margin:.15rem 0 .65rem}.risk-summary{height:40px;display:flex;align-items:center;justify-content:space-between;padding:0 1rem;background:#F0F2F6;border-radius:12px;font-size:.82rem}.risk-detail{box-sizing:border-box;height:330px;overflow:hidden;display:flex;flex-direction:column;padding:1.2rem 1.25rem;background:#F8FAFD;border:1px solid #D8E4F0;border-radius:10px}.risk-detail-title{font-size:1rem;font-weight:700;margin-bottom:.65rem}.risk-detail-text{flex:1;overflow-y:auto;color:#555D63;font-size:.84rem;line-height:1.62}.risk-source{color:#7A8288;font-size:.72rem;margin-top:.75rem;padding-top:.75rem;border-top:1px solid #D8E4F0}.risk-note{color:#7A8288;font-size:.74rem;margin-top:.65rem}.footer{margin-top:3.5rem;padding-top:1.25rem;border-top:1px solid var(--line);color:#747A7E;font-size:.76rem;line-height:1.7;text-align:center}
/* Keep the same small gap below both desktop risk controls. */
@media(min-width:601px){
    .st-key-risk_desktop{margin-top:.35rem!important}
    .risk-detail{position:relative;top:-.65rem;margin-bottom:-.65rem}
}
.conversation-turn{max-width:800px!important;margin-left:auto!important;margin-right:auto!important}
div[data-testid="stSegmentedControl"]{width:440px;margin:.8rem auto .05rem}div[data-testid="stSegmentedControl"]>label{display:none!important}div[data-testid="stSegmentedControl"] [data-baseweb="button-group"]{width:100%;padding:4px;background:#F1F2F6;border:1px solid #E1E3EB;border-radius:11px}div[data-testid="stSegmentedControl"] button{flex:1;min-height:38px;border:0;border-radius:8px;color:#666C72;font-size:.82rem;font-weight:620}div[data-testid="stSegmentedControl"] button[aria-pressed="true"]{background:#FFF;color:var(--dark);box-shadow:0 2px 8px rgba(37,42,65,.10)}
div[data-testid="stForm"]{position:static;bottom:auto;z-index:auto;margin:1.45rem auto 0;box-shadow:0 8px 24px rgba(29,34,55,.10)}
div[data-testid="stVerticalBlockBorderWrapper"]{width:100%;max-width:none;margin:0;background:linear-gradient(180deg,#FAFAFD 0%,#F7F8FC 100%);border:1px solid #E1E3EC!important;border-radius:18px!important;box-shadow:0 8px 26px rgba(29,34,55,.04)}
.conversation-label{max-width:920px;margin-top:1rem}.conversation-turn{max-width:840px!important;padding:1.15rem 0}.chat-user-bubble{max-width:62%}.chat-answer{max-width:760px;font-size:1rem}.chat-ready{margin:3.4rem auto 0;font-size:.9rem}
div[data-testid="stForm"]{width:100%;max-width:none;margin:.7rem 0 0;padding:.38rem;border-color:#D8DAE4;box-shadow:0 8px 24px rgba(29,34,55,.08)}
.chat-page-heading{max-width:1120px;margin:1.8rem auto 1rem}.chat-heading-lockup{display:flex;align-items:center;justify-content:center;gap:1rem}.chat-page-title{display:block;margin:0;font-size:clamp(2.45rem,3.7vw,3.15rem);font-weight:680;line-height:1.08;letter-spacing:-.105rem;white-space:nowrap}.chat-page-title em{color:var(--primary);font:inherit;font-style:normal}.chat-bot-icon{width:3.5rem;height:3.5rem;flex:0 0 3.5rem;display:inline-flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#F5F6FF,#E8EBFF);border:1px solid #D5D9F7;border-radius:17px;box-shadow:0 7px 18px rgba(65,72,143,.12)}.chat-bot-icon svg{width:2.15rem;height:2.15rem;fill:none;stroke:var(--primary);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}.chat-bot-icon svg circle{fill:#FFF}.chat-page-description{max-width:700px;margin:.55rem auto 0;font-size:.86rem}
.chat-assistant-row{grid-template-columns:44px minmax(0,1fr);gap:.85rem}.chat-avatar{width:44px;height:44px;background:linear-gradient(145deg,#F5F6FF,#E8EBFF);border:1px solid #D5D9F7;border-radius:14px;box-shadow:0 5px 14px rgba(65,72,143,.09)}.chat-avatar svg{width:28px;height:28px;fill:none;stroke:var(--primary);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}.chat-avatar svg circle{fill:#FFF}
.st-key-conversation_panel{background:#F4F7FA!important;border-color:#DDE4EC!important;border-radius:16px!important}.st-key-conversation_panel>div{background:#F4F7FA!important}.st-key-conversation_panel .conversation-turn{max-width:1080px!important;padding:1.4rem 2rem}.st-key-conversation_panel .chat-user-bubble{max-width:56%;background:#E5EAF3}.st-key-conversation_panel .chat-answer{max-width:880px}.prompt-label{margin:.75rem 0 .3rem;text-align:center;color:#8A8F98;font-size:.66rem;font-weight:700;letter-spacing:.07rem;text-transform:uppercase}.st-key-prompt_suggestions div[data-testid="stButton"] button{min-height:34px;background:#F3F4FA;border:1px solid #E2E4F0;border-radius:999px;color:#626875;font-size:.71rem;box-shadow:none}.st-key-prompt_suggestions div[data-testid="stButton"] button:hover{background:#EDEFFA;border-color:#CACFEC;color:var(--dark)}
.chat-ready{margin:2.85rem auto 0;font-size:.9rem}
div[data-testid="stForm"]{width:100%;max-width:none;margin:.9rem 0 1.4rem;padding:0;background:transparent;border:0!important;border-radius:0;box-shadow:none!important}
div[data-testid="stForm"] div[data-testid="stTextInput"]{padding:0;background:transparent;border:0;border-radius:0;box-shadow:none}
div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="input"]{min-height:58px;background:#FFF!important;border:1px solid #D7DCE6!important;border-radius:18px!important;box-shadow:0 9px 26px rgba(37,48,73,.10)!important}
div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within{border-color:#AEB6E8!important;box-shadow:0 10px 30px rgba(75,86,160,.14)!important}
.chat-source-note{width:100%;max-width:none;margin:0 auto;padding:1rem 0 0;border-top:1px solid var(--line);color:#7A8085;font-size:.74rem;line-height:1.45;text-align:center}
.chat-page-heading+div div[data-testid="stButton"] button{border-radius:999px;background:#FCFCFE;border-color:#D9DCEF}.chat-page-heading+div div[data-testid="stButton"] button:hover{background:#F2F3FB;border-color:#C5CAEB}.conversation-label{max-width:1120px;margin:.75rem auto .25rem}div[data-testid="stVerticalBlockBorderWrapper"]{background:#F8FAFC;border-color:#DDE4EC!important;box-shadow:0 8px 24px rgba(29,34,55,.035)}

/* Shared chat alignment and clearer visual hierarchy. */
.chat-page-heading,.st-key-chat_workspace,.chat-source-note,
.st-key-chat_workspace input,.st-key-chat_workspace button{
    font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif!important;
}
.st-key-conversation_panel,div[data-testid="stForm"]{width:100%!important;max-width:1120px!important;margin-left:auto!important;margin-right:auto!important;box-sizing:border-box!important}
.st-key-conversation_panel,.st-key-conversation_panel>div{background:#F8FAFC!important}
.st-key-conversation_panel .chat-user-bubble{background:#E7EAF4!important;border:1px solid #DDE1EE}
.st-key-prompt_suggestions{width:100%;max-width:1120px;margin-left:auto!important;margin-right:auto!important}
.st-key-prompt_suggestions div[data-testid="stButton"] button{background:#FCFCFE!important;border-color:#D9DCEF!important;box-shadow:0 2px 7px rgba(45,51,88,.035)!important}
div[data-testid="stForm"]{margin-top:1.15rem!important;margin-bottom:1.65rem!important}
div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="input"]{min-height:46px!important;height:46px!important;border-radius:14px!important;box-shadow:0 7px 20px rgba(37,48,73,.09)!important}
div[data-testid="stForm"] div[data-testid="stTextInput"] input{min-height:44px!important;height:44px!important}
div[data-testid="stFormSubmitButton"] button{width:44px!important;height:44px!important;min-height:44px!important;border-radius:13px!important;font-size:1rem!important;box-shadow:0 4px 12px rgba(91,99,201,.18)!important}
/* Unified Q&A workspace */
.st-key-chat_workspace .st-key-prompt_suggestions{
    width:100%;
    max-width:none;
    margin:1.5rem auto 0!important;
    padding:0;
    border-top:0!important;
    transform:translateY(14px);
}

/* One composer handles both Enter and button submission. */
.st-key-question_composer{
    width:100%;
    max-width:none;
    margin:.45rem 0 1.4rem;
}
.st-key-question_composer div[data-testid="stTextInput"] div[data-baseweb="input"]{
    min-height:46px!important;
    height:46px!important;
    background:#FFF!important;
    border:1px solid #D7DCE6!important;
    border-radius:14px!important;
    box-shadow:0 7px 20px rgba(37,48,73,.09)!important;
}
.st-key-question_composer div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within{
    border-color:#AEB6E8!important;
    box-shadow:0 10px 30px rgba(75,86,160,.14)!important;
}
.st-key-question_composer div[data-testid="stTextInput"] input{
    min-height:44px!important;
    height:44px!important;
}
.st-key-question_composer div[data-testid="stButton"] button{
    width:44px!important;
    height:44px!important;
    min-height:44px!important;
    margin:0!important;
    background:var(--primary)!important;
    border:0!important;
    border-radius:13px!important;
    color:#FFF!important;
    font-size:1rem!important;
    font-weight:700!important;
    box-shadow:0 4px 12px rgba(91,99,201,.18)!important;
}

.st-key-chat_workspace div[data-testid="stForm"]{
    width:100%!important;
    max-width:none!important;
    margin:.45rem 0 0!important;
    padding:0!important;
}

/* Final chat alignment cleanup */
.st-key-chat_workspace .conversation-label{
    height:38px;
    display:flex;
    align-items:center;
    margin:0!important;
}

.chat-source-note{
    margin:.65rem auto 0!important;
    padding:.35rem 0 0!important;
    border-top:0!important;
}
/* Reduce the empty space below conversation history */
.st-key-chat_workspace .st-key-conversation_panel{
    margin-bottom:0!important;
}
.st-key-conversation_empty,
.st-key-conversation_empty>div,
.st-key-conversation_empty div[data-testid="stVerticalBlock"]{
    height:220px!important;
    min-height:220px!important;
    max-height:220px!important;
}
.st-key-conversation_empty .chat-ready{
    width:100%!important;
    max-width:none!important;
    box-sizing:border-box!important;
    text-align:center!important;
    height:220px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin:0!important;
}
.st-key-chat_workspace{
    width:100%!important;
    max-width:1180px!important;
    margin-left:auto!important;
    margin-right:auto!important;
}
.st-key-conversation_panel{
    max-height:560px!important;
    overflow-y:auto!important;
    overflow-x:hidden!important;
}
.st-key-trend_mobile,
.st-key-profitability_mobile,
.st-key-balance_mobile,
.st-key-risk_mobile,
.mobile-data-table{display:none!important}
@media(max-width:900px){div[data-testid="stImage"] img{height:220px}.hero-title{font-size:2.55rem;white-space:normal}.takeaway-grid{grid-template-columns:repeat(2,1fr)}.data-status{display:none}.table-shell{overflow-x:auto}.financial-table{min-width:920px}div[data-testid="stSegmentedControl"]{width:100%}}
@media(max-width:600px){
    .block-container{padding:2.55rem .8rem 1.75rem!important}
    .topbar{min-height:36px;padding:.1rem 0 .5rem}.brand{gap:.45rem;font-size:.66rem}.brand-mark{width:27px;height:27px;flex-basis:27px;font-size:.6rem}
    div[data-testid="stImage"]{margin-top:.55rem}div[data-testid="stImage"] img{height:175px;border-radius:8px}
    .st-key-active_page{width:100%!important;max-width:310px!important;margin:.55rem auto .05rem!important}.st-key-active_page div[data-testid="stSegmentedControl"]{width:100%!important;max-width:310px!important;margin:0 auto!important}div[data-testid="stSegmentedControl"] button{min-height:32px;padding:.18rem .28rem!important;font-size:.64rem}
    .hero-copy{margin:.72rem auto 0}.hero-title{font-size:1.72rem;line-height:1.08;letter-spacing:-.045rem;white-space:normal}
    .section-heading{margin:.78rem auto .5rem}.section-kicker{font-size:1.22rem}.section-title{font-size:1.2rem;letter-spacing:-.018rem;margin-top:.14rem}
    .year-label{margin:.18rem 0;font-size:.6rem}.st-key-snapshot_year{width:210px!important;margin:0 auto!important}div[data-testid="stSelectbox"] [data-baseweb="select"]>div{min-height:38px;border-radius:9px}

    /* Keep the desktop KPI design, but make the four cards a compact 2-by-2 group. */
    div[data-testid="stHorizontalBlock"]:has(.metric-card){display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr))!important;column-gap:.55rem!important;row-gap:.7rem!important}
    div[data-testid="stHorizontalBlock"]:has(.metric-card)>div[data-testid="stColumn"]{width:auto!important;min-width:0!important;margin:0!important;flex:unset!important}
    .metric-card{box-sizing:border-box;min-height:108px;margin:0!important;padding:.78rem}.metric-label{font-size:.56rem;letter-spacing:.045rem}.metric-value{font-size:1.22rem;margin-top:.42rem}.metric-note{font-size:.6rem;margin-top:.25rem;line-height:1.3}

    div[data-testid="stTabs"] [role="tablist"]{overflow-x:auto!important;overflow-y:hidden!important;justify-content:flex-start!important;scrollbar-width:none}
    div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar{display:none}
    div[data-testid="stTabs"] button{flex:0 0 auto!important;width:auto!important;min-width:0!important;padding-left:.34rem!important;padding-right:.34rem!important;font-size:.62rem!important}
    div[data-testid="stTabs"] button p{font-size:.62rem!important;white-space:nowrap!important}
    .st-key-trend_desktop,.st-key-profitability_desktop,.st-key-balance_desktop,.st-key-risk_desktop,.desktop-data-table{display:none!important}
    .st-key-trend_mobile,.st-key-profitability_mobile,.st-key-balance_mobile,.st-key-risk_mobile,.mobile-data-table{display:block!important}
    .st-key-trend_mobile,.st-key-profitability_mobile,.st-key-balance_mobile,.st-key-risk_mobile{margin:0!important}
    .st-key-trend_mobile div[data-testid="stPlotlyChart"],
    .st-key-profitability_mobile div[data-testid="stPlotlyChart"],
    .st-key-balance_mobile div[data-testid="stPlotlyChart"],
    .st-key-risk_mobile div[data-testid="stPlotlyChart"]{touch-action:pan-y!important}
    div[data-testid="stPlotlyChart"]{padding:.3rem}
    div[data-testid="stHorizontalBlock"]:has(.balance-chart-title){row-gap:.45rem!important}
    .balance-chart-title{position:relative;z-index:3;display:block;line-height:1.4;margin:.2rem 0 .42rem;font-size:.76rem}.st-key-balance_mobile div[data-testid="stMarkdownContainer"]:has(.mobile-cash-title){display:block!important;margin:1.15rem 0 .7rem!important;padding:0!important}.mobile-cash-title{margin:0!important}.balance-note{padding:0 .3rem;font-size:.62rem}

    .takeaway-wrap{margin-top:.65rem;padding:.78rem}.takeaway-header{display:block;padding-bottom:.6rem}.takeaway-title{font-size:.88rem}.takeaway-summary{margin-top:.25rem;font-size:.72rem;line-height:1.4}
    .takeaway-grid{gap:0;margin-top:.6rem}.takeaway-item{min-height:76px;padding:.55rem .62rem;border-left:1px solid var(--line);border-top:1px solid var(--line)}
    .takeaway-item:first-child{padding:.55rem .62rem;border-top:0;border-left:0}.takeaway-item:nth-child(2){border-top:0}.takeaway-item:nth-child(3){border-left:0}
    .takeaway-value{font-size:1.08rem}.takeaway-label{font-size:.63rem;line-height:1.3}

    .risk-label{margin:.2rem 0 .4rem;font-size:.64rem}.risk-summary{height:36px;margin-bottom:.3rem;padding:0 .75rem;font-size:.7rem}.st-key-risk_mobile div[data-testid="stPlotlyChart"]{padding:.35rem .25rem}.risk-note{margin:.3rem 0 .65rem;font-size:.64rem}.risk-detail{height:290px;min-height:0;padding:.9rem}.risk-detail-title{font-size:.86rem}.risk-detail-text{overflow-y:auto;font-size:.72rem;line-height:1.5}.risk-source{font-size:.62rem;margin-top:.55rem;padding-top:.55rem}
    .table-shell{overflow:hidden}.financial-table{min-width:0!important;font-size:.61rem;table-layout:fixed}.financial-table th,.financial-table td{padding:.48rem .18rem;white-space:normal;line-height:1.2}.financial-table th:first-child,.financial-table td:first-child{width:31%;text-align:left!important;padding-left:.48rem}.financial-table th:not(:first-child),.financial-table td:not(:first-child){width:13.8%}
    .footer{margin-top:1.8rem;font-size:.65rem}

    .chat-page-heading{width:100%;margin:.8rem auto .55rem}.chat-heading-lockup{gap:.5rem}.chat-page-title{font-size:1.55rem;line-height:1.1;letter-spacing:-.035rem;white-space:normal}.chat-page-description{margin:.35rem auto 0;font-size:.69rem;line-height:1.4}.chat-bot-icon{width:2.6rem;height:2.6rem;flex-basis:2.6rem}
    .st-key-chat_workspace{width:100%!important;max-width:none!important;margin:0 auto!important;padding:0!important;background:linear-gradient(180deg,#FAFAFD 0%,#F7F8FC 100%)!important;border:1px solid #E1E3EC!important;border-radius:14px!important;box-shadow:0 6px 18px rgba(29,34,55,.045)!important}.st-key-chat_workspace>div{padding:.62rem!important}
    .st-key-chat_workspace>div>div[data-testid="stVerticalBlock"]{gap:.35rem!important}
    .st-key-conversation_empty,.st-key-conversation_empty>div,.st-key-conversation_empty div[data-testid="stVerticalBlock"]{height:105px!important;min-height:105px!important;max-height:105px!important;margin-bottom:0!important}.st-key-conversation_empty{background:#F4F7FA!important;border:1px solid #E4E8EF!important;border-radius:11px!important}.st-key-conversation_empty .chat-ready{margin:0!important;height:105px;display:flex;align-items:center;justify-content:center;padding:0 .8rem;font-size:.66rem;line-height:1.35}
    .st-key-chat_workspace div[data-testid="stHorizontalBlock"]:has(.conversation-label){display:flex!important;flex-wrap:nowrap!important;align-items:center!important;gap:.4rem!important}
    .st-key-chat_workspace div[data-testid="stHorizontalBlock"]:has(.conversation-label)>div[data-testid="stColumn"]:first-child{width:auto!important;min-width:0!important;flex:1 1 auto!important}
    .st-key-chat_workspace div[data-testid="stHorizontalBlock"]:has(.conversation-label)>div[data-testid="stColumn"]:last-child{width:52px!important;min-width:52px!important;flex:0 0 52px!important}
    .st-key-chat_workspace .conversation-label{height:28px!important;font-size:.6rem!important}
    .st-key-chat_workspace div[data-testid="stHorizontalBlock"]:has(.conversation-label) div[data-testid="stButton"] button{width:52px!important;min-height:28px!important;height:28px!important;padding:.1rem .35rem!important;font-size:.58rem!important;border-radius:8px!important}
    .st-key-conversation_panel,.st-key-conversation_panel>div,.st-key-conversation_panel div[data-testid="stVerticalBlock"],.st-key-conversation_panel div[data-testid="stVerticalBlockBorderWrapper"]{height:auto!important;min-height:0!important;max-height:none!important;overflow:visible!important}
    .st-key-conversation_panel{margin-bottom:0!important}
    .st-key-chat_workspace .st-key-prompt_suggestions{width:100%!important;margin:.45rem auto 0!important;transform:none;overflow:hidden!important}
    .st-key-chat_controls{width:100%!important;margin:.35rem 0 0!important}
        .st-key-chat_controls>div[data-testid="stVerticalBlock"],
        .st-key-chat_controls>div>div[data-testid="stVerticalBlock"]{gap:.5rem!important}
    .st-key-chat_controls .st-key-prompt_suggestions,.st-key-chat_controls .st-key-question_composer{margin:0!important}
    .st-key-prompt_suggestions div[data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:.42rem!important;width:100%!important;min-width:0!important}
    .st-key-prompt_suggestions div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]{width:auto!important;min-width:0!important;flex:unset!important}.st-key-prompt_suggestions div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]:not(:has(button)){display:none!important}
    .st-key-prompt_suggestions div[data-testid="stButton"] button{width:100%!important;min-height:27px!important;height:27px!important;padding:.18rem .25rem!important;white-space:nowrap!important;font-size:.53rem!important}
    .st-key-prompt_suggestions div[data-testid="stButton"] button p{font-size:.56rem!important;line-height:1!important;white-space:nowrap!important}
    .st-key-question_composer{width:100%!important;margin:.3rem 0 .1rem!important}.st-key-question_composer div[data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:minmax(0,1fr) 38px!important;gap:.38rem!important;align-items:center!important}.st-key-question_composer div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]{width:auto!important;min-width:0!important;flex:unset!important}.st-key-question_composer div[data-testid="stTextInput"] div[data-baseweb="input"],.st-key-question_composer div[data-testid="stTextInput"] input{height:38px!important;min-height:38px!important;font-size:.72rem!important}.st-key-question_composer div[data-testid="stButton"] button{width:38px!important;height:38px!important;min-height:38px!important;margin:0!important;border-radius:11px!important}
    .st-key-conversation_panel .conversation-turn{padding:.8rem .45rem}.st-key-conversation_panel .chat-user-bubble{max-width:88%;padding:.52rem .68rem;font-size:.76rem;line-height:1.42}.chat-assistant-row{grid-template-columns:30px minmax(0,1fr);gap:.5rem}.chat-avatar{width:30px;height:30px;border-radius:10px}.chat-avatar svg{width:20px;height:20px}.chat-answer{font-size:.8rem;line-height:1.58}.chat-name{font-size:.7rem;margin-bottom:.32rem}.answer-heading{font-size:.78rem}.answer-line,.answer-bullet{line-height:1.5}.answer-year{font-size:.66rem}.answer-source,.answer-citation,.answer-source-link a{font-size:.64rem}
}
.st-key-conversation_panel .chat-answer{max-width:820px!important;font-size:.94rem!important;line-height:1.66!important;color:#3E454B}.st-key-conversation_panel .answer-line,.st-key-conversation_panel .answer-bullet{max-width:78ch}.answer-heading{margin:1rem 0 .36rem;font-size:.86rem;font-weight:700}.answer-bullet{margin:.42rem 0;line-height:1.58}.answer-bullet-label{color:#2F363B;font-weight:650}.answer-citation-cluster{display:inline-flex;align-items:baseline;white-space:nowrap}.answer-source,.answer-citation{color:#687178;font-size:.78rem;line-height:1.55}.answer-source-link a{color:#5F6891;font-size:.78rem}.st-key-conversation_panel{max-height:620px!important}
@media(max-width:768px){.st-key-conversation_panel{max-height:none!important}.st-key-conversation_panel .chat-answer{font-size:.8rem!important;line-height:1.58!important}.st-key-conversation_panel .answer-line,.st-key-conversation_panel .answer-bullet{max-width:none}.answer-source,.answer-citation,.answer-source-link a{font-size:.66rem}}
</style>
""")



# Shared header and page navigation.
render_html(f'<div class="topbar"><div class="brand"><span class="brand-mark">GY</span><span>Apple 10-K Financial Analyst</span></div><div class="data-status">Official SEC data · Updated through FY{latest_year}</div></div>')
st.image(str(HERO_IMAGE), use_container_width=True)
nav_left, nav_center, nav_right = st.columns([1, 1.15, 1])
with nav_center:
    active_page = st.segmented_control(
        "Page",
        ["Financial Dashboard", "Ask Apple’s Filings"],
        default="Financial Dashboard",
        selection_mode="single",
        label_visibility="collapsed",
        key="active_page",
    )

if active_page == "Ask Apple’s Filings":
    render_chat_page()
    render_html('<div class="chat-source-note">Answers are based on SEC Company Facts data and Apple’s latest Form 10-K disclosures.</div>')
    st.stop()


# Financial dashboard hero.
render_html("""
<div class="hero-copy"><div class="hero-title"><em>Apple’s</em> Financial Performance, Visualized.</div></div>
""")


# Financial overview.
render_html("""
<div class="section-heading"><div class="section-kicker">01 · Overview</div>
<div class="section-title">Financial snapshot</div></div>
""")
years = pd.to_datetime(data["end"]).dt.year.tolist()
_, center, _ = st.columns([1.45, .55, 1.45])
with center:
    render_html('<div class="year-label">Fiscal year</div>')
    selected_year = st.selectbox("Fiscal year", list(reversed(years)), label_visibility="collapsed", key="snapshot_year")
position = years.index(selected_year)
selected = data.iloc[position]
prior = data.iloc[position - 1] if position > 0 else None
revenue_note = f"{selected['revenue_growth_pct']:+.2f}% year-over-year" if pd.notna(selected["revenue_growth_pct"]) else "First year in range"
cash_change = (selected["operating_cash_flow_billions"] / prior["operating_cash_flow_billions"] - 1) * 100 if prior is not None else None
cash_note = f"{cash_change:+.2f}% year-over-year" if cash_change is not None else "First year in range"
metrics = [
    ("Revenue", f"${selected['revenue_billions']:.2f}B", revenue_note),
    ("Operating Income", f"${selected['operating_income_billions']:.2f}B", f"{selected['operating_margin_pct']:.2f}% operating margin"),
    ("Net Income", f"${selected['net_income_billions']:.2f}B", f"{selected['net_profit_margin_pct']:.2f}% net margin"),
    ("Operating Cash Flow", f"${selected['operating_cash_flow_billions']:.2f}B", cash_note),
]
for column, (label, value, note) in zip(st.columns(4), metrics):
    with column:
        render_html(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>'
        )


# Analysis.
render_html("""
<div class="section-heading"><div class="section-kicker">02 · Analysis</div><div class="section-title">Five-year financial direction</div></div>
""")
trend_tab, profitability_tab, balance_sheet_tab, data_tab = st.tabs(
    ["Financial trends", "Profitability", "Balance sheet", "Underlying data"]
)
with trend_tab:
    with st.container(key="trend_desktop"):
        st.plotly_chart(trend_chart(data), use_container_width=True, config={"displayModeBar": False})
    with st.container(key="trend_mobile"):
        st.plotly_chart(trend_chart(data, mobile=True), use_container_width=True, config={"displayModeBar": False, "staticPlot": True, "responsive": True})
    render_html(f"""
    <div class="takeaway-wrap"><div class="takeaway-header"><div class="takeaway-title">Key Takeaways</div><div class="takeaway-summary">Growth strengthened in FY{latest_year}, while cash generation softened.</div></div>
    <div class="takeaway-grid"><div class="takeaway-item"><div class="takeaway-value positive">{latest['revenue_growth_pct']:+.2f}%</div><div class="takeaway-label">Latest annual revenue growth</div></div>
    <div class="takeaway-item"><div class="takeaway-value">{revenue_cagr:.2f}%</div><div class="takeaway-label">Five-year revenue CAGR</div></div>
    <div class="takeaway-item"><div class="takeaway-value positive">{margin_change:+.2f} pp</div><div class="takeaway-label">Annual operating-margin change</div></div>
    <div class="takeaway-item"><div class="takeaway-value negative">{cash_flow_growth:+.2f}%</div><div class="takeaway-label">Annual operating-cash-flow change</div></div></div></div>
    """)
with profitability_tab:
    with st.container(key="profitability_desktop"):
        st.plotly_chart(margin_chart(data), use_container_width=True, config={"displayModeBar": False})
    with st.container(key="profitability_mobile"):
        st.plotly_chart(margin_chart(data, mobile=True), use_container_width=True, config={"displayModeBar": False, "staticPlot": True, "responsive": True})
with balance_sheet_tab:
    with st.container(key="balance_desktop"):
        balance_left, balance_right = st.columns(2, gap="large")
        with balance_left:
            render_html('<div class="balance-chart-title">Assets and liabilities</div>')
            st.plotly_chart(
                assets_liabilities_chart(data),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with balance_right:
            render_html('<div class="balance-chart-title">Cash position</div>')
            st.plotly_chart(
                cash_position_chart(data),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with st.container(key="balance_mobile"):
        st.plotly_chart(
            assets_liabilities_chart(data, mobile=True),
            use_container_width=True,
            config={"displayModeBar": False, "staticPlot": True, "responsive": True},
        )
        st.plotly_chart(
            cash_position_chart(data, mobile=True),
            use_container_width=True,
            config={"displayModeBar": False, "staticPlot": True, "responsive": True},
        )
    render_html(
        '<div class="balance-note">Cash is shown separately because its scale is '
        'much smaller than total assets and liabilities.</div>'
    )
with data_tab:
    table = data.copy()
    table["Fiscal Year"] = pd.to_datetime(table["end"]).dt.year
    table = table[["Fiscal Year", "revenue_billions", "operating_income_billions", "net_income_billions", "operating_cash_flow_billions", "total_assets_billions", "total_liabilities_billions", "cash_and_cash_equivalents_billions", "operating_margin_pct", "net_profit_margin_pct", "revenue_growth_pct"]]
    table.columns = ["Fiscal Year", "Revenue ($B)", "Operating Income ($B)", "Net Income ($B)", "Operating Cash Flow ($B)", "Total Assets ($B)", "Total Liabilities ($B)", "Cash ($B)", "Operating Margin (%)", "Net Margin (%)", "Revenue Growth (%)"]
    mobile_table = table.set_index("Fiscal Year").transpose().reset_index()
    mobile_table.columns = ["Metric"] + [f"FY{year}" for year in table["Fiscal Year"]]
    mobile_table["Metric"] = [
        "Revenue ($B)", "Operating income ($B)", "Net income ($B)",
        "Operating cash flow ($B)", "Total assets ($B)",
        "Total liabilities ($B)", "Cash ($B)", "Operating margin (%)",
        "Net margin (%)", "Revenue growth (%)",
    ]
    render_html(
        f'<div class="desktop-data-table"><div class="table-shell">'
        f'{table.round(2).to_html(index=False, classes="financial-table", border=0, na_rep="—")}'
        '</div></div>'
        f'<div class="mobile-data-table"><div class="table-shell mobile-table-shell">'
        f'{mobile_table.round(2).to_html(index=False, classes="financial-table mobile-financial-table", border=0, na_rep="—")}'
        '</div></div>'
    )


# Risk analysis.
render_html(f"""
<div class="section-heading"><div class="section-kicker">03 · Risk Analysis</div><div class="section-title">Key risk themes in Apple’s {latest_year} 10-K</div></div>
""")
left, right = st.columns(2, gap="large")
with left:
    render_html('<div class="risk-label">Theme coverage</div>')
    render_html(f'<div class="risk-summary"><span><strong>{len(risk_chunks)}</strong> filing excerpts</span><span><strong>{len(risk_summary)}</strong> themes</span></div>')
    with st.container(key="risk_desktop"):
        st.plotly_chart(
            risk_chart(risk_summary),
            use_container_width=True,
            config={"displayModeBar": False},
            key="risk_coverage_chart_desktop",
        )
    with st.container(key="risk_mobile"):
        st.plotly_chart(
            risk_chart(risk_summary, mobile=True),
            use_container_width=True,
            config={"displayModeBar": False, "staticPlot": True, "responsive": True},
            key="risk_coverage_chart_mobile",
        )
    render_html('<div class="risk-note">Bars show filing excerpt coverage, not risk severity.</div>')
with right:
    render_html('<div class="risk-label">Explore a theme</div>')
    selected_topic = st.selectbox("Risk theme", risk_summary["topic_label"].tolist(), label_visibility="collapsed", key="risk_topic")
    topic_rows = risk_chunks[risk_chunks["topic_label"] == selected_topic].sort_values("topic_score", ascending=False)
    excerpt = str(topic_rows.iloc[0]["text"]).strip()
    render_html(f'<div class="risk-detail"><div class="risk-detail-title">{escape(selected_topic)}</div><div class="risk-detail-text">{escape(excerpt)}</div><div class="risk-source">Source: Apple FY{latest_year} Form 10-K · Item 1A</div></div>')


render_html("""
<div class="footer">Data source: U.S. Securities and Exchange Commission Company Facts API and Apple Form 10-K. Financial figures are presented in USD billions.<br>
Built by Gengmeng Ye for educational and demonstration purposes. This application does not constitute investment advice.</div>
""")
