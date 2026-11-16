"""Reusable Streamlit presentation components."""
from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from paybridge.config import SYNTHETIC_DATA_NOTICE
from paybridge.ui import theme


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def fmt_usd(value: float | None, decimals: int = 0, compact: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if compact:
        for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(value) >= threshold:
                return f"${value / threshold:,.2f}{suffix}"
        return f"${value:,.0f}"
    return f"${value:,.{decimals}f}"


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}%"


def fmt_hours(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if value < 1:
        return f"{value * 60:.0f} min"
    if value < 48:
        return f"{value:.1f} h"
    return f"{value / 24:.1f} d"


def fmt_int(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


# --------------------------------------------------------------------------
# Layout primitives
# --------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    parts = ['<div class="pb-page-header">']
    if eyebrow:
        parts.append(f'<div class="pb-eyebrow">{html.escape(eyebrow)}</div>')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    if subtitle:
        parts.append(f"<p>{html.escape(subtitle)}</p>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="pb-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="pb-section-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="pb-badge {html.escape(tone)}">{html.escape(str(text))}</span>'


def notice(message: str, tone: str = "info", icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f'<div class="pb-notice {html.escape(tone)}">{prefix}{message}</div>', unsafe_allow_html=True
    )


def synthetic_notice() -> None:
    notice(f"<strong>Synthetic demonstration data.</strong> {SYNTHETIC_DATA_NOTICE}", "synthetic", "⚠️")


def modelled_notice(message: str) -> None:
    notice(f"<strong>Modelled, not realised.</strong> {message}", "modelled", "📐")


def kpi_cards(cards: Sequence[Mapping[str, Any]], columns: int = 4) -> None:
    """Render a responsive grid of metric cards.

    Each card accepts: label, value, sub, delta, delta_dir ('up'/'down'/'flat'),
    tone ('teal'|'navy'|'success'|'warning'|'danger').
    """
    items = []
    for card in cards:
        tone = f"tone-{card.get('tone', 'teal')}"
        bits = [f'<div class="pb-kpi {tone}">']
        bits.append(f'<div class="pb-kpi-label">{html.escape(str(card.get("label", "")))}</div>')
        bits.append(f'<div class="pb-kpi-value">{html.escape(str(card.get("value", "-")))}</div>')
        if card.get("delta"):
            direction = card.get("delta_dir", "flat")
            bits.append(
                f'<div class="pb-kpi-delta {direction}">{html.escape(str(card["delta"]))}</div>'
            )
        if card.get("sub"):
            bits.append(f'<div class="pb-kpi-sub">{html.escape(str(card["sub"]))}</div>')
        bits.append("</div>")
        items.append("".join(bits))
    grid = (
        f'<div class="pb-kpi-grid" style="grid-template-columns: repeat({columns}, minmax(0, 1fr));">'
        + "".join(items)
        + "</div>"
    )
    st.markdown(grid, unsafe_allow_html=True)


def key_values(pairs: Sequence[tuple[str, Any]]) -> None:
    rows = "".join(
        f"<div><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</div>" for k, v in pairs
    )
    st.markdown(f'<div class="pb-kv">{rows}</div>', unsafe_allow_html=True)


def chart(fig: go.Figure, height: int = 320, title: str = "") -> None:
    fig.update_layout(height=height, template=theme.PLOTLY_TEMPLATE)
    if title:
        fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def dataframe(
    df: pd.DataFrame,
    height: int | None = None,
    column_config: Mapping[str, Any] | None = None,
    hide_index: bool = True,
) -> None:
    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        hide_index=hide_index,
        column_config=dict(column_config or {}),
    )


def empty_state(message: str, icon: str = "🔍") -> None:
    notice(f"{icon} {html.escape(message)}", "info")


def footer(extra: str = "") -> None:
    text = (
        "PayBridge - independent portfolio project. Synthetic data only; no real payments, "
        "screening or regulatory verification."
    )
    if extra:
        text = f"{text} {extra}"
    st.markdown(f'<div class="pb-footer">{html.escape(text)}</div>', unsafe_allow_html=True)


def band_badge(band: str) -> str:
    return badge(band.upper(), band if band in theme.BAND_COLORS else "neutral")


def status_badge(status: str) -> str:
    tone_map = {
        "completed": "success", "settled": "success", "approved": "success",
        "pending": "info", "in_transit": "info", "screening": "info", "submitted": "info",
        "under_review": "warning", "review": "warning", "more_info_required": "warning",
        "failed": "danger", "returned": "danger", "rejected": "danger", "blocked": "danger",
    }
    return badge(status.replace("_", " ").title(), tone_map.get(status, "neutral"))
