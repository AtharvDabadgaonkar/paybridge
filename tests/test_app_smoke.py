"""Smoke tests: every page renders without raising."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGE_LABELS = [
    "📊  Executive Overview",
    "💸  Payments & Payouts",
    "🛡️  Compliance Operations",
    "🧠  FraudLens AI",
    "🔎  Investigation Workbench",
    "🤖  AI Copilot",
    "📈  Model Evaluation",
    "🗂️  Program Command Center",
    "📐  Product Analytics",
    "ℹ️  About PayBridge",
]


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    return at


def test_app_boots_without_exception(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_navigation_lists_every_module(app):
    assert len(app.sidebar.radio) == 1
    assert list(app.sidebar.radio[0].options) == PAGE_LABELS


@pytest.mark.parametrize("label", PAGE_LABELS)
def test_each_page_renders(label):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(label).run()
    assert not at.exception, f"{label} raised: {[str(e) for e in at.exception]}"
    assert at.markdown, f"{label} rendered no content"


def test_no_error_elements_on_the_default_page(app):
    assert not app.error, [e.value for e in app.error]
