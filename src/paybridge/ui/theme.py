"""PayBridge design system: navy + teal enterprise FinTech styling."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
NAVY_900 = "#0A1F33"
NAVY_800 = "#102A43"
NAVY_600 = "#243B53"
NAVY_400 = "#486581"
NAVY_200 = "#9FB3C8"
TEAL_600 = "#0B6E6F"
TEAL_500 = "#0F8B8D"
TEAL_400 = "#14A8A4"
TEAL_200 = "#7FD1CE"
SURFACE = "#F4F7FA"
SURFACE_ALT = "#EAF0F6"
BORDER = "#DCE4EC"
WHITE = "#FFFFFF"

SUCCESS = "#0E7C66"
WARNING = "#B7791F"
DANGER = "#C0392B"
INFO = "#2B6CB0"
PURPLE = "#6B46C1"

CATEGORICAL = [TEAL_500, NAVY_600, "#E08A2E", INFO, PURPLE, SUCCESS, NAVY_200, DANGER]
SEQUENTIAL = ["#E6F2F2", "#BFE0E0", "#8CC9C8", "#4FADAC", "#1F8F8E", "#0B6E6F", "#064F50"]

RAIL_COLORS = {"ACH_PARTNER": NAVY_600, "SWIFT": "#E08A2E", "USDC": TEAL_500}
BAND_COLORS = {"low": SUCCESS, "medium": "#C9A227", "high": "#D97706", "critical": DANGER}
STATUS_COLORS = {
    "completed": SUCCESS,
    "settled": SUCCESS,
    "approved": SUCCESS,
    "pending": INFO,
    "in_transit": INFO,
    "screening": INFO,
    "submitted": INFO,
    "under_review": WARNING,
    "review": WARNING,
    "more_info_required": WARNING,
    "failed": DANGER,
    "returned": DANGER,
    "rejected": DANGER,
    "blocked": DANGER,
}

PLOTLY_TEMPLATE = "paybridge"


def register_plotly_template() -> None:
    """Register and activate the PayBridge Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif", size=13, color=NAVY_800),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        colorway=CATEGORICAL,
        title=dict(font=dict(size=16, color=NAVY_800), x=0.01, xanchor="left"),
        xaxis=dict(
            gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(size=12, color=NAVY_400), title=dict(font=dict(size=12, color=NAVY_400)),
        ),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(size=12, color=NAVY_400), title=dict(font=dict(size=12, color=NAVY_400)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=12, color=NAVY_600), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=16, r=16, t=48, b=16),
        hoverlabel=dict(bgcolor=NAVY_800, font=dict(color=WHITE, size=12), bordercolor=NAVY_800),
    )
    pio.templates[PLOTLY_TEMPLATE] = template
    pio.templates.default = PLOTLY_TEMPLATE


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {{
    font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.stApp {{ background: {SURFACE}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY_900} 0%, {NAVY_800} 55%, {NAVY_600} 100%);
    border-right: 1px solid {NAVY_600};
}}
section[data-testid="stSidebar"] * {{ color: #E6EDF4 !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.93rem; padding: 0.18rem 0; letter-spacing: 0.01em;
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.14); }}
.pb-sidebar-brand {{ padding: 0.2rem 0 0.9rem 0; }}
.pb-sidebar-brand .pb-logo {{
    font-size: 1.32rem; font-weight: 700; letter-spacing: -0.02em; color: {WHITE} !important;
}}
.pb-sidebar-brand .pb-logo span {{ color: {TEAL_200} !important; }}
.pb-sidebar-brand .pb-tag {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: {TEAL_200} !important; margin-top: 0.15rem;
}}

/* ---------------- Page header ---------------- */
.pb-page-header {{
    background: linear-gradient(100deg, {NAVY_900} 0%, {NAVY_800} 48%, {TEAL_600} 140%);
    border-radius: 14px; padding: 1.15rem 1.4rem; margin-bottom: 1.1rem;
    box-shadow: 0 10px 24px -18px rgba(16,42,67,0.75);
}}
.pb-page-header h1 {{
    color: {WHITE}; font-size: 1.52rem; font-weight: 650; margin: 0; letter-spacing: -0.02em;
}}
.pb-page-header p {{ color: #C9DAE8; font-size: 0.93rem; margin: 0.32rem 0 0 0; max-width: 92ch; }}
.pb-page-header .pb-eyebrow {{
    color: {TEAL_200}; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.3rem;
}}

/* ---------------- Metric cards ---------------- */
.pb-kpi-grid {{ display: grid; gap: 0.7rem; margin-bottom: 0.3rem; }}
.pb-kpi {{
    background: {WHITE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 0.85rem 0.95rem 0.9rem 0.95rem; position: relative; overflow: hidden;
    box-shadow: 0 1px 2px rgba(16,42,67,0.04);
}}
.pb-kpi::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: {TEAL_500};
}}
.pb-kpi.tone-navy::before {{ background: {NAVY_600}; }}
.pb-kpi.tone-success::before {{ background: {SUCCESS}; }}
.pb-kpi.tone-warning::before {{ background: {WARNING}; }}
.pb-kpi.tone-danger::before {{ background: {DANGER}; }}
.pb-kpi .pb-kpi-label {{
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.085em; color: {NAVY_400}; margin-bottom: 0.3rem;
}}
.pb-kpi .pb-kpi-value {{
    font-size: 1.48rem; font-weight: 680; color: {NAVY_800};
    line-height: 1.12; letter-spacing: -0.025em; font-variant-numeric: tabular-nums;
}}
.pb-kpi .pb-kpi-sub {{ font-size: 0.76rem; color: {NAVY_400}; margin-top: 0.3rem; line-height: 1.35; }}
.pb-kpi .pb-kpi-delta {{ font-size: 0.78rem; font-weight: 600; margin-top: 0.28rem; }}
.pb-kpi .pb-kpi-delta.up {{ color: {SUCCESS}; }}
.pb-kpi .pb-kpi-delta.down {{ color: {DANGER}; }}
.pb-kpi .pb-kpi-delta.flat {{ color: {NAVY_400}; }}

/* ---------------- Panels ---------------- */
.pb-panel {{
    background: {WHITE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1rem 1.1rem; margin-bottom: 0.9rem;
}}
.pb-section-title {{
    font-size: 1.02rem; font-weight: 650; color: {NAVY_800};
    margin: 1.15rem 0 0.1rem 0; letter-spacing: -0.012em;
}}
.pb-section-sub {{ font-size: 0.84rem; color: {NAVY_400}; margin-bottom: 0.65rem; }}

/* ---------------- Badges & notices ---------------- */
.pb-badge {{
    display: inline-block; padding: 0.14rem 0.52rem; border-radius: 999px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em; border: 1px solid transparent;
}}
.pb-badge.low, .pb-badge.success {{ background: #E3F3EF; color: {SUCCESS}; border-color: #BFE3DA; }}
.pb-badge.medium, .pb-badge.info {{ background: #E8F0FA; color: {INFO}; border-color: #C9DCF2; }}
.pb-badge.high, .pb-badge.warning {{ background: #FDF3E2; color: {WARNING}; border-color: #F3DFB8; }}
.pb-badge.critical, .pb-badge.danger {{ background: #FBEAE7; color: {DANGER}; border-color: #F3C9C2; }}
.pb-badge.neutral {{ background: {SURFACE_ALT}; color: {NAVY_600}; border-color: {BORDER}; }}

.pb-notice {{
    border-radius: 10px; padding: 0.68rem 0.85rem; font-size: 0.83rem;
    border: 1px solid {BORDER}; background: {WHITE}; color: {NAVY_600}; margin-bottom: 0.8rem;
}}
.pb-notice.synthetic {{
    background: #FFF9E8; border-color: #F0DFAE; color: #7A5A12;
}}
.pb-notice.modelled {{ background: #EEF4FB; border-color: #CFE0F2; color: #1C4E80; }}
.pb-notice.danger {{ background: #FBEAE7; border-color: #F3C9C2; color: {DANGER}; }}
.pb-notice.success {{ background: #E9F5F1; border-color: #BFE3DA; color: {SUCCESS}; }}
.pb-notice strong {{ font-weight: 650; }}

/* ---------------- Tabs, inputs, tables ---------------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.15rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    height: 2.4rem; padding: 0 0.95rem; font-size: 0.88rem; font-weight: 550; color: {NAVY_400};
}}
.stTabs [aria-selected="true"] {{ color: {TEAL_600} !important; }}
.stButton > button {{
    border-radius: 8px; font-weight: 600; font-size: 0.86rem;
    border: 1px solid {BORDER}; padding: 0.38rem 0.95rem;
}}
.stButton > button[kind="primary"] {{
    background: {TEAL_600}; border-color: {TEAL_600}; color: {WHITE};
}}
.stButton > button[kind="primary"]:hover {{ background: {TEAL_500}; border-color: {TEAL_500}; }}
div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; color: {NAVY_800}; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
.pb-footer {{
    margin-top: 2rem; padding-top: 0.8rem; border-top: 1px solid {BORDER};
    font-size: 0.76rem; color: {NAVY_400};
}}
.pb-kv {{ font-size: 0.85rem; color: {NAVY_600}; line-height: 1.75; }}
.pb-kv b {{ color: {NAVY_800}; font-weight: 600; }}
.pb-mono {{
    font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.82rem; color: {NAVY_800};
}}
</style>
"""


def inject_css() -> None:
    """Inject the design system CSS (idempotent per Streamlit rerun)."""
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
