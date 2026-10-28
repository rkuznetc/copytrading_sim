# src/copytrading_sim/simulator/user_generator.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


USER_GENERATOR_ARCHETYPES = (
    "conservative",
    "balanced",
    "aggressive",
    "social_driven",
)

ASSET_CLASSES = (
    "stocks",
    "crypto",
    "forex",
    "commodities",
    "mixed",
)

HORIZONS = (
    "short",
    "medium",
    "long",
)

CAPITAL_BUCKETS = (
    "low",
    "medium",
    "high",
)

EXPERIENCE_LEVELS = (
    "beginner",
    "intermediate",
    "advanced",
)

LIQUIDITY_NEEDS = (
    "low",
    "medium",
    "high",
)


USER_ARCHETYPE_PARAMS: dict[str, dict[str, float]] = {
    "conservative": {
        "risk_tolerance_mean": 0.25,
        "drawdown_tolerance_mean": 0.25,
        "social_sensitivity_mean": 0.25,
    },
    "balanced": {
        "risk_tolerance_mean": 0.50,
        "drawdown_tolerance_mean": 0.50,
        "social_sensitivity_mean": 0.40,
    },
    "aggressive": {
        "risk_tolerance_mean": 0.75,
        "drawdown_tolerance_mean": 0.70,
        "social_sensitivity_mean": 0.35,
    },
    "social_driven": {
        "risk_tolerance_mean": 0.55,
        "drawdown_tolerance_mean": 0.45,
        "social_sensitivity_mean": 0.80,
    },
}


def generate_users(
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic copy-trading users.

    Each user later corresponds to one federated client.
    The generated profile is intentionally compact and suitability-oriented:
    risk tolerance, drawdown tolerance, horizon, capital bucket,
    experience level, liquidity need, social sensitivity, and preferred asset class.
    """
    if rng is None:
        seed = config["random"]["seed"]
        rng = np.random.default_rng(seed)

    n_users = int(config["simulation"]["n_users"])
    generator_archetypes = _sample_user_generator_archetypes(n_users, rng)

    rows: list[dict[str, Any]] = []

    for user_id, generator_archetype in enumerate(generator_archetypes):
        generator_archetype = str(generator_archetype)
        params = USER_ARCHETYPE_PARAMS[generator_archetype]

        risk_tolerance = _sample_beta_around_mean(
            mean=params["risk_tolerance_mean"],
            concentration=24.0,
            rng=rng,
        )
        drawdown_tolerance = _sample_beta_around_mean(
            mean=params["drawdown_tolerance_mean"],
            concentration=24.0,
            rng=rng,
        )
        social_sensitivity = _sample_beta_around_mean(
            mean=params["social_sensitivity_mean"],
            concentration=18.0,
            rng=rng,
        )

        investment_horizon = _sample_horizon(generator_archetype, rng)
        capital_bucket = _sample_capital_bucket(generator_archetype, rng)
        preferred_asset_class = _sample_preferred_asset_class(generator_archetype, rng)
        experience_level = _sample_experience_level(generator_archetype, rng)
        liquidity_need = _sample_liquidity_need(
            generator_archetype=generator_archetype,
            investment_horizon=investment_horizon,
            rng=rng,
        )

        user_type = _classify_user_type(
            risk_tolerance=risk_tolerance,
            drawdown_tolerance=drawdown_tolerance,
            social_sensitivity=social_sensitivity,
        )

        rows.append(
            {
                "user_id": user_id,
                "user_generator_archetype": generator_archetype,
                "user_type": user_type,
                "risk_tolerance": risk_tolerance,
                "drawdown_tolerance": drawdown_tolerance,
                "investment_horizon": investment_horizon,
                "capital_bucket": capital_bucket,
                "experience_level": experience_level,
                "liquidity_need": liquidity_need,
                "social_sensitivity": social_sensitivity,
                "preferred_asset_class": preferred_asset_class,

                # Backward-compatible aliases for older validation / experiments.
                "user_archetype": user_type,
                "drawdown_sensitivity": 1.0 - drawdown_tolerance,
                "social_bias": social_sensitivity,
            }
        )

    return pd.DataFrame(rows)


def _sample_user_generator_archetypes(
    n_users: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample generator priors for users.

    These labels are generation priors, not final behavioral truth.
    Final user_type is assigned by explicit rules.
    """
    probabilities = [0.30, 0.35, 0.20, 0.15]
    return rng.choice(USER_GENERATOR_ARCHETYPES, size=n_users, p=probabilities)


def _sample_horizon(archetype: str, rng: np.random.Generator) -> str:
    """Sample investment horizon conditional on generator archetype."""
    if archetype == "conservative":
        return str(rng.choice(HORIZONS, p=[0.10, 0.35, 0.55]))

    if archetype == "aggressive":
        return str(rng.choice(HORIZONS, p=[0.45, 0.40, 0.15]))

    if archetype == "social_driven":
        return str(rng.choice(HORIZONS, p=[0.30, 0.45, 0.25]))

    return str(rng.choice(HORIZONS, p=[0.25, 0.45, 0.30]))


def _sample_capital_bucket(archetype: str, rng: np.random.Generator) -> str:
    """Sample rough capital category instead of exact balance."""
    if archetype == "conservative":
        return str(rng.choice(CAPITAL_BUCKETS, p=[0.35, 0.45, 0.20]))

    if archetype == "aggressive":
        return str(rng.choice(CAPITAL_BUCKETS, p=[0.50, 0.35, 0.15]))

    return str(rng.choice(CAPITAL_BUCKETS, p=[0.45, 0.40, 0.15]))


def _sample_preferred_asset_class(archetype: str, rng: np.random.Generator) -> str:
    """Sample preferred asset class conditional on generator archetype."""
    if archetype == "conservative":
        return str(rng.choice(ASSET_CLASSES, p=[0.45, 0.05, 0.10, 0.15, 0.25]))

    if archetype == "aggressive":
        return str(rng.choice(ASSET_CLASSES, p=[0.20, 0.40, 0.20, 0.05, 0.15]))

    if archetype == "social_driven":
        return str(rng.choice(ASSET_CLASSES, p=[0.25, 0.30, 0.20, 0.05, 0.20]))

    return str(rng.choice(ASSET_CLASSES, p=[0.30, 0.15, 0.20, 0.10, 0.25]))


def _sample_experience_level(archetype: str, rng: np.random.Generator) -> str:
    """Sample investment experience level."""
    if archetype == "conservative":
        return str(rng.choice(EXPERIENCE_LEVELS, p=[0.45, 0.40, 0.15]))

    if archetype == "aggressive":
        return str(rng.choice(EXPERIENCE_LEVELS, p=[0.25, 0.45, 0.30]))

    if archetype == "social_driven":
        return str(rng.choice(EXPERIENCE_LEVELS, p=[0.40, 0.45, 0.15]))

    return str(rng.choice(EXPERIENCE_LEVELS, p=[0.35, 0.45, 0.20]))


def _sample_liquidity_need(
    generator_archetype: str,
    investment_horizon: str,
    rng: np.random.Generator,
) -> str:
    """Sample liquidity need.

    Short-horizon and conservative users are slightly more likely
    to have higher liquidity needs.
    """
    if investment_horizon == "short":
        return str(rng.choice(LIQUIDITY_NEEDS, p=[0.15, 0.35, 0.50]))

    if generator_archetype == "conservative":
        return str(rng.choice(LIQUIDITY_NEEDS, p=[0.20, 0.45, 0.35]))

    if investment_horizon == "long":
        return str(rng.choice(LIQUIDITY_NEEDS, p=[0.45, 0.40, 0.15]))

    return str(rng.choice(LIQUIDITY_NEEDS, p=[0.30, 0.45, 0.25]))


def _classify_user_type(
    risk_tolerance: float,
    drawdown_tolerance: float,
    social_sensitivity: float,
) -> str:
    """Classify final user type by interpretable rules.

    Priority is intentional:
    social_driven -> aggressive -> conservative -> balanced.
    """
    if social_sensitivity >= 0.70:
        return "social_driven"

    if risk_tolerance >= 0.65 and drawdown_tolerance >= 0.60:
        return "aggressive"

    if risk_tolerance < 0.35 and drawdown_tolerance < 0.35:
        return "conservative"

    return "balanced"


def _sample_beta_around_mean(
    mean: float,
    concentration: float,
    rng: np.random.Generator,
) -> float:
    """Sample a value in [0, 1] around a target mean using Beta distribution."""
    mean = float(np.clip(mean, 0.01, 0.99))
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    return float(rng.beta(alpha, beta))