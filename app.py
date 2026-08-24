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


def render_html(html: str) -> None:
    """Render compact HTML."""
    st.markdown(" ".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def fill_example_question(question: str) -> None:
    """Copy a suggested question into the question input."""
    st.session_state.qa_input = question


def trend_chart(data: pd.DataFrame) -> go.Figure:
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
    figure.update_layout(
        height=380, margin={"l": 25, "r": 18, "t": 52, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        xaxis={"tickvals": years.tolist(), "ticktext": [f"FY{year}" for year in years], "showgrid": False},
        yaxis={"title": "USD billions", "gridcolor": "#EEEEEE", "zeroline": False},
        legend={"orientation": "h", "y": 1.04, "x": 0.5, "xanchor": "center"},
        font={"family": "Avenir Next, Helvetica Neue, Arial", "color": "#4E555A", "size": 13},
    )
    return figure


def margin_chart(data: pd.DataFrame) -> go.Figure:
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
    figure.update_layout(
        height=380, margin={"l": 25, "r": 18, "t": 52, "b": 20}, bargap=.52,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        xaxis={"tickvals": years.tolist(), "ticktext": [f"FY{year}" for year in years], "showgrid": False},
        yaxis={"title": "Margin (%)", "gridcolor": "#EEEEEE", "zeroline": False},
        legend={"orientation": "h", "y": 1.04, "x": .5, "xanchor": "center"},
        font={"family": "Avenir Next, Helvetica Neue, Arial", "color": "#4E555A", "size": 13},
    )
    return figure


def risk_chart(summary: pd.DataFrame) -> go.Figure:
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
        height=330, margin={"l": 8, "r": 82, "t": 12, "b": 8}, bargap=.5, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [0, maximum * 1.22]},
        yaxis={"showgrid": False, "tickfont": {"size": 12, "color": "#525960"}},
        font={"family": "Avenir Next, Helvetica Neue, Arial", "color": "#4E555A", "size": 12},
    )
    return figure


def format_answer_html(answer_text: str) -> str:
    """Convert grounded AI output into clean dashboard HTML."""
    parts = []
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line or set(line) == {"="}:
            continue
        if line.startswith("Question:") or line.startswith("Route:"):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue

        safe_line = escape(line)
        safe_line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe_line)

        if line.startswith("http://") or line.startswith("https://"):
            safe_url = escape(line, quote=True)
            parts.append(
                '<div class="answer-source-link">'
                f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">'
                "View Apple' s SEC filing &#8599;</a></div>"
            )
        elif line.startswith("Source:"):
            parts.append(f'<div class="answer-source">{safe_line}</div>')
        elif line.startswith("[F") or re.match(r"^\[\d+\]", line):
            parts.append(f'<div class="answer-citation">{safe_line}</div>')
        elif line.startswith("Chunk "):
            parts.append(f'<div class="answer-evidence-title">{safe_line}</div>')
        elif line.startswith("FY") and line[2:6].isdigit():
            parts.append(f'<div class="answer-year">{safe_line}</div>')
        elif line.endswith(":") or line in {
            "Retrieved Risk Factors evidence:",
            "Financial metrics by fiscal year:",
        }:
            parts.append(f'<div class="answer-heading">{safe_line}</div>')
        elif line.startswith("- "):
            parts.append(f'<div class="answer-bullet">{safe_line[2:]}</div>')
        else:
            parts.append(f'<div class="answer-line">{safe_line}</div>')
    return "".join(parts)


def render_chat_page() -> None:
    """Render the dedicated filing Q&A page and preserve conversation history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_input" not in st.session_state:
        st.session_state.qa_input = ""

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

            with st.container(height=560, border=False, key="conversation_panel"):
                for message in st.session_state.chat_history:
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
                                    {format_answer_html(message['answer'])}
                                </div>
                            </div>
                        </div>
                        """
                    )
        else:
            with st.container(height=240, border=False, key="conversation_panel"):
                render_html('<div class="chat-ready">Choose a prompt or ask your own question below.</div>')

        with st.container(key="prompt_suggestions"):
            prompt_left, example_one, example_two, example_three, prompt_right = st.columns([.35, 1, 1, 1, .35], gap="small")
            examples = [
                (example_one, "Compare revenue in 2024 and 2025", "Compare Apple's revenue in 2024 and 2025."),
                (example_two, "Show the past three years", "Show Apple's revenue over the past three years."),
                (example_three, "What are Apple’s supply-chain risks?", "What supply chain risks does Apple disclose?"),
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

        with st.form("apple_question_form", clear_on_submit=False):
            input_column, send_column = st.columns([8, .46], gap="small")
            with input_column:
                question = st.text_input(
                    "Question",
                    placeholder="Message Apple Financial Analyst...",
                    label_visibility="collapsed",
                    key="qa_input",
                )
            with send_column:
                submitted = st.form_submit_button("↑", use_container_width=True)

    if submitted:
        if question.strip():
            with st.spinner("Reviewing Apple’s financial data and 10-K..."):
                answer = get_answer(question.strip())
            st.session_state.chat_history.append({"question": question.strip(), "answer": answer})
            st.rerun()
        else:
            st.warning("Please enter a question.")


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
.stApp{background:#FFF;color:var(--ink);font-family:"Avenir Next","Helvetica Neue",Arial,sans-serif}.block-container{max-width:1220px;padding-top:3.75rem!important;padding-bottom:3rem}
.topbar{min-height:48px;display:flex;align-items:center;justify-content:space-between;padding:.35rem 0 .8rem;border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:.75rem;font-size:.8rem;font-weight:650;line-height:1.25}.brand-mark{width:34px;height:34px;flex:0 0 34px;display:inline-flex;align-items:center;justify-content:center;border:1px solid #B9B7B1;border-radius:50%;font-size:.7rem;font-weight:700}.data-status{color:var(--soft);font-size:.78rem;line-height:1.25}
div[data-testid="stRadio"]{width:440px;margin:1.1rem auto .2rem}div[data-testid="stRadio"]>label{display:none!important}div[data-testid="stRadio"] div[role="radiogroup"]{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:4px;background:#F1F2F6;border:1px solid #E1E3EB;border-radius:11px}div[data-testid="stRadio"] div[role="radiogroup"]>label{min-height:38px;display:flex!important;align-items:center;justify-content:center;padding:.42rem .85rem!important;background:transparent;border:0;border-radius:8px;color:#666C72;font-size:.82rem;font-weight:620;cursor:pointer}div[data-testid="stRadio"] div[role="radiogroup"]>label:has(input:checked){background:#FFF;color:var(--dark);box-shadow:0 2px 8px rgba(37,42,65,.10)}div[data-testid="stRadio"] div[role="radiogroup"]>label>div:first-child{display:none!important}
div[data-testid="stImage"]{margin-top:.85rem}div[data-testid="stImage"] img{width:100%;height:270px;object-fit:cover;object-position:center 49%;border-radius:10px}.hero-copy{max-width:1120px;margin:1.35rem auto 0;text-align:center}.hero-title{font-size:clamp(2.45rem,3.7vw,3.15rem);font-weight:680;line-height:1.08;letter-spacing:-.105rem;white-space:nowrap}.hero-title em{color:var(--primary);font:inherit;font-style:normal}.hero-description{max-width:680px;margin:.8rem auto 0;color:var(--soft);font-size:1rem;line-height:1.65}
.section-heading{margin:1.9rem auto 1rem;text-align:center}.section-kicker{color:var(--primary);font-size:1.9rem;font-weight:650;text-transform:uppercase}.section-title{font-size:1.9rem;font-weight:650;letter-spacing:-.035rem;margin-top:.35rem}.section-description{max-width:680px;margin:.6rem auto 0;color:var(--soft);font-size:.96rem;line-height:1.6}
div[data-testid="stTabs"] button{color:var(--soft);font-size:.92rem}.stApp button[role="tab"][aria-selected="true"]{color:var(--dark)!important;font-weight:700;box-shadow:inset 0 -2px 0 var(--primary)!important}
.year-label{text-align:center;color:var(--soft);font-size:.76rem;font-weight:700;letter-spacing:.06rem;text-transform:uppercase;margin:.5rem 0}.metric-card{min-height:142px;padding:1.35rem;border:1px solid var(--line);border-radius:10px}.metric-label{color:#6D7377;font-size:.73rem;font-weight:700;letter-spacing:.075rem;text-transform:uppercase}.metric-value{font-size:1.85rem;font-weight:650;margin-top:.72rem}.metric-note{color:var(--soft);font-size:.79rem;margin-top:.42rem}
div[data-testid="stSelectbox"] [data-baseweb="select"]>div{min-height:52px;background:#F0F2F6;border:0!important;border-radius:12px}div[data-testid="stSelectbox"] [data-baseweb="select"]>div>div:first-child{flex:1!important;justify-content:center!important;padding-left:38px!important}
.chat-page-heading{max-width:720px;margin:2.1rem auto 1.15rem;text-align:center}.chat-page-title{margin-top:.4rem;font-size:1.7rem;font-weight:680;letter-spacing:-.03rem}.chat-page-description{max-width:580px;margin:.45rem auto 0;color:var(--soft);font-size:.88rem;line-height:1.5}.conversation-label{max-width:800px;margin:1.25rem auto .2rem;color:#858A8E;font-size:.68rem;font-weight:700;letter-spacing:.08rem;text-transform:uppercase}.conversation-stream{max-width:800px;margin:0 auto}.conversation-turn{margin:0;padding:1.35rem 0 1.15rem;border-bottom:1px solid #ECEDEF}.conversation-turn:last-child{border-bottom:0}.chat-user-row{display:flex;justify-content:flex-end;margin-bottom:1.15rem}.chat-user-bubble{max-width:70%;padding:.65rem .9rem;background:#F0F1F6;border-radius:18px 18px 5px 18px;color:#30343A;font-size:.94rem;line-height:1.55}.chat-assistant-row{display:grid;grid-template-columns:30px minmax(0,1fr);gap:.75rem;align-items:start}.chat-avatar{width:30px;height:30px;display:flex;align-items:center;justify-content:center;background:var(--primary);border-radius:50%;color:#FFF;font-size:.61rem;font-weight:750}.chat-answer{max-width:720px;padding:.02rem 0;color:#343A3F;font-size:.96rem;line-height:1.72}.chat-name{margin-bottom:.42rem;color:var(--ink);font-size:.82rem;font-weight:700}.chat-ready{max-width:800px;margin:2.2rem auto 1rem;padding:1rem;text-align:center;color:#8A8F93;font-size:.84rem}.answer-heading{font-size:.9rem;font-weight:700;color:var(--ink);margin:.8rem 0 .32rem}.answer-year{display:inline-block;margin:.65rem 0 .22rem;padding:.16rem .48rem;background:#ECEEFB;border-radius:999px;color:var(--dark);font-size:.78rem;font-weight:700}.answer-line{color:#454C51;margin:.26rem 0;line-height:1.65}.answer-bullet{position:relative;color:#454C51;margin:.3rem 0;padding-left:1rem;line-height:1.6}.answer-bullet:before{content:"";position:absolute;left:.1rem;top:.7rem;width:4px;height:4px;border-radius:50%;background:var(--primary)}.answer-evidence-title{margin:.85rem 0 .32rem;padding-top:.7rem;border-top:1px solid #ECEDEF;color:var(--dark);font-size:.82rem;font-weight:700}.answer-source,.answer-citation{color:#7A8288;font-size:.74rem;line-height:1.5;margin:.2rem 0}.answer-source-link{margin-top:.3rem}.answer-source-link a{color:var(--dark);font-size:.75rem;font-weight:650;text-decoration:none}.answer-source-link a:hover{text-decoration:underline}.snapshot-subheading{margin:2.4rem auto 1rem;text-align:center}.snapshot-title{font-size:1.45rem;font-weight:680}.snapshot-description{margin:.45rem auto 0;color:var(--soft);font-size:.9rem}div[data-testid="stButton"] button{min-height:38px;background:#FFF;border:1px solid var(--line);border-radius:10px;color:#596066;font-size:.76rem}div[data-testid="stButton"] button:hover{background:#F7F7FA;border-color:#BFC4E8;color:var(--dark)}div[data-testid="stForm"]{position:sticky;bottom:1rem;z-index:20;max-width:800px;margin:1rem auto 0;padding:.42rem;background:#FFF;border:1px solid #D8DAE2;border-radius:18px;box-shadow:0 10px 32px rgba(29,34,55,.12)}div[data-testid="stTextInput"] input{min-height:48px;border:0!important;background:#FFF;border-radius:14px;font-size:.92rem}div[data-testid="stFormSubmitButton"] button{width:48px!important;height:48px;min-height:48px;margin:0;background:var(--primary);border:0;border-radius:14px;color:white;font-size:1.15rem;font-weight:700}
div[data-testid="stPlotlyChart"]{padding:.75rem;background:#FFF;border:1px solid var(--line);border-radius:10px}.takeaway-wrap{margin-top:1rem;padding:1.15rem 1.25rem;border:1px solid var(--line);border-radius:10px}.takeaway-header{display:flex;justify-content:space-between;padding-bottom:.85rem;border-bottom:1px solid var(--line)}.takeaway-title{font-size:1.05rem;font-weight:700}.takeaway-summary{color:var(--soft);font-size:.9rem}.takeaway-grid{display:grid;grid-template-columns:repeat(4,1fr);margin-top:1rem}.takeaway-item{padding:0 1rem;border-left:1px solid var(--line)}.takeaway-item:first-child{border:0;padding-left:0}.takeaway-value{font-size:1.55rem;font-weight:680}.takeaway-label{color:var(--soft);font-size:.86rem;margin-top:.3rem}.positive{color:var(--positive)}.negative{color:var(--negative)}
.table-shell{width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:10px}.financial-table{width:100%;border-collapse:collapse;font-size:.81rem}.financial-table th,.financial-table td{padding:.9rem .55rem;border-right:1px solid var(--line);border-bottom:1px solid var(--line);text-align:center!important}.financial-table th{background:#F6F6F8;color:#555C61}.risk-label{color:var(--dark);font-size:.75rem;font-weight:700;letter-spacing:.07rem;text-transform:uppercase;margin:.15rem 0 .65rem}.risk-summary{height:40px;display:flex;align-items:center;justify-content:space-between;padding:0 1rem;background:#F0F2F6;border-radius:12px;font-size:.82rem}.risk-detail{height:356px;overflow:hidden;display:flex;flex-direction:column;padding:1.2rem 1.25rem;background:#F8FAFD;border:1px solid #D8E4F0;border-radius:10px}.risk-detail-title{font-size:1rem;font-weight:700;margin-bottom:.65rem}.risk-detail-text{flex:1;overflow-y:auto;color:#555D63;font-size:.84rem;line-height:1.62}.risk-source{color:#7A8288;font-size:.72rem;margin-top:.75rem;padding-top:.75rem;border-top:1px solid #D8E4F0}.risk-note{color:#7A8288;font-size:.74rem;margin-top:.65rem}.footer{margin-top:3.5rem;padding-top:1.25rem;border-top:1px solid var(--line);color:#747A7E;font-size:.76rem;line-height:1.7;text-align:center}
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
    margin-bottom:-1.75rem!important;
}
@media(max-width:900px){div[data-testid="stImage"] img{height:220px}.hero-title{font-size:2.55rem;white-space:normal}.takeaway-grid{grid-template-columns:repeat(2,1fr)}.data-status{display:none}.financial-table{min-width:920px}div[data-testid="stSegmentedControl"]{width:100%}}
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
<div class="hero-copy"><div class="hero-title"><em>Apple’s</em> Financial Performance, Visualized.</div>
<div class="hero-description">This dashboard turns Apple’s SEC filings into an interactive view of its financial performance. Select a fiscal year to review the results and see how they have changed over time.</div>
</div>
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
        render_html(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>')


# Analysis.
render_html("""
<div class="section-heading"><div class="section-kicker">02 · Analysis</div><div class="section-title">Five-year financial direction</div></div>
""")
trend_tab, profitability_tab, data_tab = st.tabs(["Financial trends", "Profitability", "Underlying data"])
with trend_tab:
    st.plotly_chart(trend_chart(data), use_container_width=True, config={"displayModeBar": False})
    render_html(f"""
    <div class="takeaway-wrap"><div class="takeaway-header"><div class="takeaway-title">Key Takeaways</div><div class="takeaway-summary">Growth strengthened in FY{latest_year}, while cash generation softened.</div></div>
    <div class="takeaway-grid"><div class="takeaway-item"><div class="takeaway-value positive">{latest['revenue_growth_pct']:+.2f}%</div><div class="takeaway-label">Latest annual revenue growth</div></div>
    <div class="takeaway-item"><div class="takeaway-value">{revenue_cagr:.2f}%</div><div class="takeaway-label">Five-year revenue CAGR</div></div>
    <div class="takeaway-item"><div class="takeaway-value positive">{margin_change:+.2f} pp</div><div class="takeaway-label">Annual operating-margin change</div></div>
    <div class="takeaway-item"><div class="takeaway-value negative">{cash_flow_growth:+.2f}%</div><div class="takeaway-label">Annual operating-cash-flow change</div></div></div></div>
    """)
with profitability_tab:
    st.plotly_chart(margin_chart(data), use_container_width=True, config={"displayModeBar": False})
with data_tab:
    table = data.copy()
    table["Fiscal Year"] = pd.to_datetime(table["end"]).dt.year
    table = table[["Fiscal Year", "revenue_billions", "operating_income_billions", "net_income_billions", "operating_cash_flow_billions", "operating_margin_pct", "net_profit_margin_pct", "revenue_growth_pct"]]
    table.columns = ["Fiscal Year", "Revenue ($B)", "Operating Income ($B)", "Net Income ($B)", "Operating Cash Flow ($B)", "Operating Margin (%)", "Net Margin (%)", "Revenue Growth (%)"]
    render_html(f'<div class="table-shell">{table.round(2).to_html(index=False, classes="financial-table", border=0, na_rep="—")}</div>')


# Risk analysis.
render_html(f"""
<div class="section-heading"><div class="section-kicker">03 · Risk Analysis</div><div class="section-title">Key risk themes in Apple’s {latest_year} 10-K</div></div>
""")
left, right = st.columns(2, gap="large")
with left:
    render_html('<div class="risk-label">Theme coverage</div>')
    render_html(f'<div class="risk-summary"><span><strong>{len(risk_chunks)}</strong> filing excerpts</span><span><strong>{len(risk_summary)}</strong> themes</span></div>')
with right:
    render_html('<div class="risk-label">Explore a theme</div>')
    selected_topic = st.selectbox("Risk theme", risk_summary["topic_label"].tolist(), label_visibility="collapsed", key="risk_topic")

topic_rows = risk_chunks[risk_chunks["topic_label"] == selected_topic].sort_values("topic_score", ascending=False)
excerpt = str(topic_rows.iloc[0]["text"]).strip()
left, right = st.columns(2, gap="large")
with left:
    st.plotly_chart(risk_chart(risk_summary), use_container_width=True, config={"displayModeBar": False})
    render_html('<div class="risk-note">Bars show filing excerpt coverage, not risk severity.</div>')
with right:
    render_html(f'<div class="risk-detail"><div class="risk-detail-title">{escape(selected_topic)}</div><div class="risk-detail-text">{escape(excerpt)}</div><div class="risk-source">Source: Apple FY{latest_year} Form 10-K · Item 1A</div></div>')


render_html("""
<div class="footer">Data source: U.S. Securities and Exchange Commission Company Facts API and Apple Form 10-K. Financial figures are presented in USD billions.<br>
Built by Gengmeng Ye for educational and demonstration purposes. This application does not constitute investment advice.</div>
""")