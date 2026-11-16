"""PayBridge - Cross-Border Payments Platform with AI Fraud Detection.

Streamlit entry point. Run with:  streamlit run app.py

All data is synthetic. No real payment is ever initiated by this application.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from paybridge import __version__, config  # noqa: E402
from paybridge.bootstrap import bootstrap, reset_demo  # noqa: E402
from paybridge.ui import theme  # noqa: E402

st.set_page_config(
    page_title="PayBridge | Cross-Border Payments & AI Fraud Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES: tuple[tuple[str, str, str], ...] = (
    ("Executive Overview", "📊", "overview"),
    ("Payments & Payouts", "💸", "payments"),
    ("Compliance Operations", "🛡️", "compliance"),
    ("FraudLens AI", "🧠", "fraudlens"),
    ("Investigation Workbench", "🔎", "workbench"),
    ("AI Copilot", "🤖", "copilot"),
    ("Model Evaluation", "📈", "evaluation"),
    ("Program Command Center", "🗂️", "program"),
    ("Product Analytics", "📐", "analytics"),
    ("About PayBridge", "ℹ️", "about"),
)


@st.cache_resource(show_spinner="Preparing synthetic PayBridge environment...")
def _bootstrap() -> dict:
    return bootstrap()


def _sidebar(boot: dict) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="pb-sidebar-brand">'
            '<div class="pb-logo">Pay<span>Bridge</span></div>'
            '<div class="pb-tag">Payments &amp; Fraud Intelligence</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        labels = [f"{icon}  {name}" for name, icon, _ in PAGES]
        default = st.session_state.get("pb_nav_index", 0)
        choice = st.radio("Navigation", labels, index=default, label_visibility="collapsed")
        selected = labels.index(choice)
        st.session_state["pb_nav_index"] = selected

        st.divider()
        manifest = boot.get("manifest", {})
        counts = manifest.get("row_counts", {})
        window = manifest.get("window", {})
        st.caption("**Synthetic dataset**")
        st.caption(
            f"{counts.get('transactions', 0):,} transactions · "
            f"{counts.get('businesses', 0):,} businesses\n\n"
            f"Window: {str(window.get('start', ''))[:10]} → {str(window.get('end', ''))[:10]} "
            f"({window.get('days', 0)} days) · seed {manifest.get('seed', '-')}"
        )

        st.divider()
        key_present = config.anthropic_api_key() is not None
        if key_present:
            st.caption("🤖 **AI Copilot:** Claude API key detected — LLM explanation layer enabled.")
        else:
            st.caption(
                "🤖 **AI Copilot:** no API key — running in deterministic mode. "
                "All metrics, rules and explanations still work."
            )

        with st.expander("Demo controls"):
            st.caption(
                "Resetting clears simulated payouts, investigation decisions, compliance "
                "decisions and program edits, then reseeds the demo."
            )
            regen = st.checkbox("Also regenerate the synthetic dataset", value=False)
            if st.button("Reset demo state", use_container_width=True):
                reset_demo(regenerate_data=regen)
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Demo state reset.")
                st.rerun()

        st.divider()
        st.caption(f"v{__version__} · Independent portfolio project · Synthetic data only")
    return PAGES[selected][2]


def main() -> None:
    theme.register_plotly_template()
    theme.inject_css()
    boot = _bootstrap()
    page_key = _sidebar(boot)

    import importlib

    module = importlib.import_module(f"paybridge.ui.pages.{page_key}")
    module.render()


if __name__ == "__main__":
    main()
