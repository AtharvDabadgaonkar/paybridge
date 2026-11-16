"""Experiment design: sample size, power and rollout criteria.

These are experiment DESIGNS. No experiment has been run against real users,
so no result is reported anywhere in the application. Where a design is
evaluated against synthetic data, the output is labelled as a simulation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy import stats

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def sample_size_two_proportions(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    two_sided: bool = True,
) -> int:
    """Per-arm sample size for a two-proportion test.

    n = (z_(1-a/2) * sqrt(2*p_bar*(1-p_bar)) + z_power * sqrt(p1(1-p1)+p2(1-p2)))^2 / delta^2

    baseline_rate: control conversion rate (0..1)
    mde_relative:  minimum detectable effect, relative to baseline (0.10 = +10%)
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be strictly between 0 and 1")
    if mde_relative <= 0:
        raise ValueError("mde_relative must be positive")

    p1 = baseline_rate
    p2 = min(baseline_rate * (1.0 + mde_relative), 0.999999)
    delta = p2 - p1
    if delta <= 0:
        raise ValueError("Minimum detectable effect resolves to zero")

    z_alpha = stats.norm.ppf(1.0 - alpha / (2.0 if two_sided else 1.0))
    z_power = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2.0
    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_power * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return int(math.ceil(numerator / (delta**2)))


def runtime_days(sample_size_per_arm: int, daily_eligible_units: float, n_arms: int = 2) -> int:
    """Days needed to accrue the required sample across all arms."""
    if daily_eligible_units <= 0:
        raise ValueError("daily_eligible_units must be positive")
    return int(math.ceil(sample_size_per_arm * n_arms / daily_eligible_units))


@dataclass(frozen=True)
class ExperimentDesign:
    experiment_id: str
    name: str
    hypothesis: str
    primary_kpi: str
    guardrails: tuple[str, ...]
    baseline_rate: float
    mde_relative: float
    alpha: float
    power: float
    sample_size_per_arm: int
    runtime_days: int
    daily_eligible_units: float
    rollout_criteria: str
    unit_of_randomisation: str

    def as_dict(self) -> dict:
        data = asdict(self)
        data["guardrails"] = "; ".join(self.guardrails)
        return data


def design_experiment(
    experiment_id: str,
    name: str,
    hypothesis: str,
    primary_kpi: str,
    guardrails: tuple[str, ...],
    baseline_rate: float,
    mde_relative: float,
    daily_eligible_units: float,
    rollout_criteria: str,
    unit_of_randomisation: str = "business account",
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> ExperimentDesign:
    """Build a fully specified experiment design with a computed sample size."""
    n = sample_size_two_proportions(baseline_rate, mde_relative, alpha, power)
    return ExperimentDesign(
        experiment_id=experiment_id,
        name=name,
        hypothesis=hypothesis,
        primary_kpi=primary_kpi,
        guardrails=guardrails,
        baseline_rate=baseline_rate,
        mde_relative=mde_relative,
        alpha=alpha,
        power=power,
        sample_size_per_arm=n,
        runtime_days=runtime_days(n, daily_eligible_units),
        daily_eligible_units=daily_eligible_units,
        rollout_criteria=rollout_criteria,
        unit_of_randomisation=unit_of_randomisation,
    )


# Seed designs. Baseline rates are stated as design inputs; where a baseline is
# observable in the synthetic data the Product Analytics page shows the measured
# value next to it.
EXPERIMENT_SEEDS: tuple[dict, ...] = (
    {
        "experiment_id": "EXP-001",
        "name": "Simplified KYB onboarding",
        "hypothesis": "Collapsing beneficial-ownership capture into a single progressive step increases onboarding completion, because the current multi-step form is where applicants drop out.",
        "primary_kpi": "Onboarding completion rate (submitted -> approved)",
        "guardrails": ("Screening alert rate", "Manual review rate", "Rejected-after-approval rate"),
        "baseline_rate": 0.72,
        "mde_relative": 0.08,
        "daily_eligible_units": 14.0,
        "rollout_criteria": "Ship to 100% if completion improves with p < 0.05 and neither screening alert rate nor manual review rate degrades by more than 2 percentage points.",
        "unit_of_randomisation": "onboarding application",
    },
    {
        "experiment_id": "EXP-002",
        "name": "Default rail recommendation",
        "hypothesis": "Pre-selecting the lowest modelled-cost eligible rail increases adoption of cheaper rails, because businesses default to the familiar wire option when no recommendation is shown.",
        "primary_kpi": "Share of payouts created on the recommended rail",
        "guardrails": ("Payout success rate", "Settlement time p90", "Payout abandonment rate"),
        "baseline_rate": 0.45,
        "mde_relative": 0.12,
        "daily_eligible_units": 260.0,
        "rollout_criteria": "Ship if recommended-rail share improves with p < 0.05 and payout success rate does not fall by more than 0.5 percentage points.",
        "unit_of_randomisation": "business account",
    },
    {
        "experiment_id": "EXP-003",
        "name": "Risk-ranked alert queue",
        "hypothesis": "Ordering the alert queue by combined risk score instead of recency raises the share of confirmed fraud caught in the first ten reviews per analyst shift.",
        "primary_kpi": "Confirmed-fraud hit rate in the first ten reviews per shift",
        "guardrails": ("Median time to decision", "Alert ageing beyond 24h", "Analyst override rate"),
        "baseline_rate": 0.18,
        "mde_relative": 0.25,
        "daily_eligible_units": 40.0,
        "rollout_criteria": "Ship if hit rate improves with p < 0.05 and no alert cohort ages beyond the 24-hour service target.",
        "unit_of_randomisation": "analyst shift",
    },
    {
        "experiment_id": "EXP-004",
        "name": "Reduced investigation steps",
        "hypothesis": "Surfacing the top SHAP drivers and prior beneficiary history on the case header reduces the number of steps an analyst takes before deciding, without reducing decision quality.",
        "primary_kpi": "Share of cases decided within the target step budget",
        "guardrails": ("Decision reversal rate", "Confirmed fraud approved in error", "Notes completeness"),
        "baseline_rate": 0.55,
        "mde_relative": 0.15,
        "daily_eligible_units": 36.0,
        "rollout_criteria": "Ship if the step-budget share improves with p < 0.05 and the decision reversal rate does not increase.",
        "unit_of_randomisation": "investigation case",
    },
)


def seed_designs() -> list[ExperimentDesign]:
    return [design_experiment(**seed) for seed in EXPERIMENT_SEEDS]
