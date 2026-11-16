"""Shared dashboard filter widgets."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

from paybridge.config import CORRIDORS, RAIL_CODES, TRANSACTION_STATUSES


@dataclass
class FilterState:
    date_range: tuple[date, date] | None = None
    rails: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    corridors: list[str] = field(default_factory=list)
    business_ids: list[str] = field(default_factory=list)


def filter_panel(
    transactions: pd.DataFrame,
    businesses: pd.DataFrame,
    key_prefix: str = "flt",
    show_business: bool = True,
) -> FilterState:
    """Render the standard filter row and return the selected state."""
    min_date = transactions["transaction_timestamp"].min().date()
    max_date = transactions["transaction_timestamp"].max().date()

    cols = st.columns([2.2, 1.3, 1.3, 1.5, 1.9] if show_business else [2.2, 1.3, 1.3, 1.5])
    with cols[0]:
        picked = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_dates",
        )
    with cols[1]:
        rails = st.multiselect("Payment rail", list(RAIL_CODES), key=f"{key_prefix}_rails")
    with cols[2]:
        statuses = st.multiselect("Status", list(TRANSACTION_STATUSES), key=f"{key_prefix}_status")
    with cols[3]:
        corridors = st.multiselect(
            "Corridor", [c.code for c in CORRIDORS], key=f"{key_prefix}_corridors"
        )

    business_ids: list[str] = []
    if show_business:
        with cols[4]:
            name_by_id = dict(zip(businesses["business_id"], businesses["legal_name"]))
            picked_names = st.multiselect(
                "Business customer",
                list(name_by_id),
                format_func=lambda b: f"{name_by_id.get(b, b)} ({b})",
                key=f"{key_prefix}_biz",
            )
            business_ids = list(picked_names)

    date_range: tuple[date, date] | None = None
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        date_range = (picked[0], picked[1])
    elif isinstance(picked, date):
        date_range = (picked, picked)

    return FilterState(
        date_range=date_range,
        rails=rails,
        statuses=statuses,
        corridors=corridors,
        business_ids=business_ids,
    )
