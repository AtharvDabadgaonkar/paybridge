"""Central configuration: paths, simulation constants, rail economics.

Every number in this module is a DOCUMENTED MODELLING ASSUMPTION for a
synthetic simulation. None of it is sourced from a real payment provider's
production rate card.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = Path(os.getenv("PAYBRIDGE_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("PAYBRIDGE_MODEL_DIR", PROJECT_ROOT / "models"))
DB_PATH = Path(os.getenv("PAYBRIDGE_DB_PATH", DATA_DIR / "paybridge.db"))

TRANSACTIONS_FILE = DATA_DIR / "transactions.parquet"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
BUSINESSES_FILE = DATA_DIR / "businesses.parquet"
COMPLIANCE_FILE = DATA_DIR / "compliance_cases.parquet"
EVENTS_FILE = DATA_DIR / "product_events.parquet"
WATCHLIST_FILE = DATA_DIR / "synthetic_watchlist.parquet"
DATASET_MANIFEST = DATA_DIR / "dataset_manifest.json"

MODEL_BUNDLE = MODEL_DIR / "fraudlens_model.joblib"
MODEL_METRICS = MODEL_DIR / "fraudlens_metrics.json"

for _d in (DATA_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------------
RANDOM_SEED = 42
SIM_DAYS = 90
SIM_END_DATE = "2026-11-01"          # fixed so the dataset is reproducible
N_BUSINESSES = 200
TARGET_TRANSACTIONS = 25_000
BASE_FRAUD_RATE = 0.015              # ~1.5% of transactions are fraudulent

# --------------------------------------------------------------------------
# Corridors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Corridor:
    code: str
    sender_country: str
    recipient_country: str
    currency: str
    weight: float                    # share of transaction volume
    partner_payout_fee_usd: float    # local partner payout cost (platform cost)
    fx_wholesale_spread_bps: float   # cost PayBridge pays to source FX
    corridor_risk: float             # 0..1 synthetic geographic risk weight
    swift_hops: float                # avg correspondent hops (SWIFT only)


CORRIDORS: tuple[Corridor, ...] = (
    Corridor("US-MX", "US", "MX", "MXN", 0.22, 1.40, 18.0, 0.35, 1.0),
    Corridor("US-IN", "US", "IN", "INR", 0.19, 1.10, 15.0, 0.30, 1.0),
    Corridor("US-PH", "US", "PH", "PHP", 0.14, 1.75, 22.0, 0.45, 2.0),
    Corridor("US-GB", "US", "GB", "GBP", 0.13, 0.90, 10.0, 0.15, 1.0),
    Corridor("US-NG", "US", "NG", "NGN", 0.09, 2.60, 45.0, 0.75, 2.0),
    Corridor("US-BR", "US", "BR", "BRL", 0.09, 1.95, 28.0, 0.50, 2.0),
    Corridor("US-DE", "US", "DE", "EUR", 0.08, 0.85, 9.0, 0.12, 1.0),
    Corridor("US-SG", "US", "SG", "SGD", 0.06, 1.05, 12.0, 0.18, 1.0),
)

CORRIDOR_BY_CODE = {c.code: c for c in CORRIDORS}
CURRENCIES = tuple(sorted({c.currency for c in CORRIDORS}))

# Indicative synthetic mid-market rates (units of local currency per 1 USD).
FX_MID_RATES = {
    "MXN": 17.10,
    "INR": 83.40,
    "PHP": 56.80,
    "GBP": 0.79,
    "NGN": 1480.0,
    "BRL": 5.12,
    "EUR": 0.92,
    "SGD": 1.34,
    "USD": 1.00,
}

# --------------------------------------------------------------------------
# Payment rails
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RailConfig:
    """Economics and operational profile for one simulated payment rail.

    Customer-facing price components (what the business pays PayBridge):
        customer_fixed_fee_usd, customer_variable_pct, customer_fx_markup_bps
    Platform direct cost components (what PayBridge pays providers):
        provider_fixed_cost_usd, provider_variable_pct, partner cost, FX spread
    """

    code: str
    label: str
    description: str
    customer_fixed_fee_usd: float
    customer_variable_pct: float      # as a fraction of amount, e.g. 0.0035
    customer_fx_markup_bps: float
    provider_fixed_cost_usd: float
    provider_variable_pct: float
    settlement_hours_p50: float
    settlement_hours_p90: float
    max_amount_usd: float
    min_amount_usd: float
    success_rate: float
    supports_corridor: tuple[str, ...] = field(default=())   # empty == all

    def eligible_for(self, corridor_code: str, amount_usd: float) -> bool:
        if self.supports_corridor and corridor_code not in self.supports_corridor:
            return False
        return self.min_amount_usd <= amount_usd <= self.max_amount_usd


# NOTE ON ACH: ACH is a US domestic network. It is modelled here as the USD
# funding/collection leg plus a local-partner payout leg in the destination
# market ("partner-assisted payout"). It is NOT modelled as a standalone
# international settlement network.
RAILS: tuple[RailConfig, ...] = (
    RailConfig(
        code="ACH_PARTNER",
        label="ACH funding + local partner payout",
        description=(
            "USD collected from the business by ACH debit, then paid out in the "
            "destination market by a local banking partner. ACH covers only the "
            "domestic US funding leg."
        ),
        customer_fixed_fee_usd=2.50,
        customer_variable_pct=0.0035,
        customer_fx_markup_bps=90.0,
        provider_fixed_cost_usd=0.80,     # ACH debit origination cost
        provider_variable_pct=0.0008,
        settlement_hours_p50=28.0,
        settlement_hours_p90=62.0,
        max_amount_usd=250_000.0,
        min_amount_usd=1.0,
        success_rate=0.975,
    ),
    RailConfig(
        code="SWIFT",
        label="SWIFT correspondent wire",
        description=(
            "Traditional international bank wire routed through correspondent "
            "banks. Carries an originating wire fee plus per-hop lifting fees."
        ),
        customer_fixed_fee_usd=25.00,
        customer_variable_pct=0.0010,
        customer_fx_markup_bps=250.0,
        provider_fixed_cost_usd=12.00,    # originating bank wire cost
        provider_variable_pct=0.0002,
        settlement_hours_p50=52.0,
        settlement_hours_p90=108.0,
        max_amount_usd=5_000_000.0,
        min_amount_usd=100.0,
        success_rate=0.962,
    ),
    RailConfig(
        code="USDC",
        label="USDC stablecoin settlement",
        description=(
            "SIMULATED stablecoin settlement: USD on-ramped to USDC, moved to a "
            "destination-market partner, then off-ramped to local currency. No "
            "wallet, key, chain or exchange is touched by this application."
        ),
        customer_fixed_fee_usd=1.00,
        customer_variable_pct=0.0060,
        customer_fx_markup_bps=50.0,
        provider_fixed_cost_usd=0.50,     # simulated network/transfer fee
        provider_variable_pct=0.0025,     # on-ramp + off-ramp provider take
        settlement_hours_p50=1.5,
        settlement_hours_p90=7.0,
        max_amount_usd=500_000.0,
        min_amount_usd=10.0,
        success_rate=0.981,
    ),
)

RAIL_BY_CODE = {r.code: r for r in RAILS}
RAIL_CODES = tuple(r.code for r in RAILS)

# Per-hop SWIFT correspondent ("lifting") fee passed through to the customer.
SWIFT_LIFTING_FEE_PER_HOP_USD = 15.0

# --------------------------------------------------------------------------
# Status vocabularies
# --------------------------------------------------------------------------
TRANSACTION_STATUSES = ("completed", "pending", "under_review", "failed", "returned")
ONBOARDING_STATUSES = (
    "draft",
    "submitted",
    "screening",
    "review",
    "approved",
    "rejected",
    "more_info_required",
)
COMPLIANCE_STATUSES = ("cleared", "pending_review", "escalated", "blocked")
INVESTIGATION_DECISIONS = ("approve", "reject", "escalate", "pending")
RISK_BANDS = ("low", "medium", "high", "critical")

# --------------------------------------------------------------------------
# Fraud / risk operating assumptions
# --------------------------------------------------------------------------
# Average economic loss, as a fraction of amount, when a fraudulent payment
# settles undetected. Modelling assumption used in loss simulations only.
FRAUD_LOSS_RATE = 0.85
# Fully-loaded analyst cost of one manual investigation (modelling assumption).
MANUAL_REVIEW_COST_USD = 18.0

RISK_BAND_THRESHOLDS = {"low": 0.0, "medium": 0.25, "high": 0.55, "critical": 0.80}

# --------------------------------------------------------------------------
# AI copilot
# --------------------------------------------------------------------------
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1200"))


def anthropic_api_key() -> str | None:
    """Return the Anthropic key from the environment, or None.

    Streamlit secrets are also consulted when available so the app can run on
    Streamlit Community Cloud without a committed .env file.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if key and key.strip():
        return key.strip()
    try:  # pragma: no cover - depends on a running Streamlit runtime
        import streamlit as st

        secret = st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[union-attr]
        if secret and str(secret).strip():
            return str(secret).strip()
    except Exception:
        pass
    return None


SYNTHETIC_DATA_NOTICE = (
    "All data in PayBridge is synthetic and generated for demonstration. "
    "No real payments are initiated and no real sanctions, identity or "
    "regulatory verification is performed."
)
