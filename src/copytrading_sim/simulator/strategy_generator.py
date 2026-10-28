# src/copytrading_sim/simulator/strategy_generator.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STRATEGY_GENERATOR_ARCHETYPES = (
    "conservative",
    "balanced",
    "aggressive",
    "high_volatility",
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

STYLE_TYPES = (
    "trend_following",
    "mean_reversion",
    "scalping",
    "swing",
    "mixed",
)


STRATEGY_ARCHETYPE_PARAMS: dict[str, dict[str, float]] = {
    "conservative": {
        "return_mean": 0.35,
        "volatility_mean": 0.25,
        "drawdown_mean": 0.20,
        "downside_risk_mean": 0.22,
        "concentration_mean": 0.25,
        "turnover_mean": 0.25,
        "track_record_months_mean": 30.0,
    },
    "balanced": {
        "return_mean": 0.50,
        "volatility_mean": 0.45,
        "drawdown_mean": 0.38,
        "downside_risk_mean": 0.40,
        "concentration_mean": 0.40,
        "turnover_mean": 0.45,
        "track_record_months_mean": 24.0,
    },
    "aggressive": {
        "return_mean": 0.68,
        "volatility_mean": 0.68,
        "drawdown_mean": 0.60,
        "downside_risk_mean": 0.62,
        "concentration_mean": 0.55,
        "turnover_mean": 0.65,
        "track_record_months_mean": 18.0,
    },
    "high_volatility": {
        "return_mean": 0.58,
        "volatility_mean": 0.86,
        "drawdown_mean": 0.75,
        "downside_risk_mean": 0.78,
        "concentration_mean": 0.65,
        "turnover_mean": 0.80,
        "track_record_months_mean": 12.0,
    },
}


def generate_strategies(
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic copy-trading strategies.

    Strategies are recommendation objects with public and derived attributes.
    risk_score and stability_score are computed from interpretable components,
    not sampled directly.
    """
    if rng is None:
        seed = config["random"]["seed"]
        rng = np.random.default_rng(seed)

    n_strategies = int(config["simulation"]["n_strategies"])
    generator_archetypes = _sample_strategy_generator_archetypes(n_strategies, rng)

    rows: list[dict[str, Any]] = []

    for strategy_id, generator_archetype in enumerate(generator_archetypes):
        generator_archetype = str(generator_archetype)
        params = STRATEGY_ARCHETYPE_PARAMS[generator_archetype]

        volatility_proxy = _sample_beta_around_mean(
            params["volatility_mean"], concentration=22.0, rng=rng
        )
        downside_risk_proxy = _sample_beta_around_mean(
            params["downside_risk_mean"], concentration=22.0, rng=rng
        )
        hhi_norm = _sample_beta_around_mean(
            params["concentration_mean"], concentration=18.0, rng=rng
        )
        turnover_proxy = _sample_beta_around_mean(
            params["turnover_mean"], concentration=18.0, rng=rng
        )

        # Drawdown is correlated with volatility, downside risk and concentration.
        raw_drawdown = (
            0.45 * volatility_proxy
            + 0.35 * downside_risk_proxy
            + 0.20 * hhi_norm
            + rng.normal(loc=0.0, scale=0.06)
        )
        archetype_drawdown = _sample_beta_around_mean(
            params["drawdown_mean"], concentration=22.0, rng=rng
        )
        max_drawdown_proxy = _clip01(0.55 * raw_drawdown + 0.45 * archetype_drawdown)

        # Return is related to archetype and risk, but not perfectly.
        risk_premium = 0.10 * (volatility_proxy + downside_risk_proxy) / 2.0
        raw_return = (
            params["return_mean"]
            + risk_premium
            - 0.08 * max_drawdown_proxy
            + rng.normal(loc=0.0, scale=0.10)
        )
        return_proxy = _clip01(raw_return)

        sortino_proxy = _compute_sortino_proxy(
            return_proxy=return_proxy,
            downside_risk_proxy=downside_risk_proxy,
        )

        track_record_months = _sample_track_record_months(
            mean_months=params["track_record_months_mean"],
            rng=rng,
        )
        track_record_confidence = _track_record_confidence(track_record_months)

        risk_score = _compute_risk_score(
            volatility_proxy=volatility_proxy,
            max_drawdown_proxy=max_drawdown_proxy,
            downside_risk_proxy=downside_risk_proxy,
            hhi_norm=hhi_norm,
        )
        complexity_score = _compute_complexity_score(
            turnover_proxy=turnover_proxy,
            hhi_norm=hhi_norm,
        )
        stability_score = _compute_stability_score(
            volatility_proxy=volatility_proxy,
            max_drawdown_proxy=max_drawdown_proxy,
            sortino_proxy=sortino_proxy,
            track_record_confidence=track_record_confidence,
            complexity_score=complexity_score,
        )

        popularity_score = _generate_popularity_score(
            return_proxy=return_proxy,
            stability_score=stability_score,
            max_drawdown_proxy=max_drawdown_proxy,
            track_record_confidence=track_record_confidence,
            turnover_proxy=turnover_proxy,
            rng=rng,
        )
        copiers_count = _generate_copiers_count(popularity_score, rng)
        aum_proxy = _generate_aum_proxy(popularity_score, copiers_count, rng)

        main_asset_class = _sample_main_asset_class(generator_archetype, rng)
        horizon_type = _sample_horizon(generator_archetype, turnover_proxy, rng)

        strategy_type = _classify_strategy_type(
            risk_score=risk_score,
            stability_score=stability_score,
            volatility_proxy=volatility_proxy,
            max_drawdown_proxy=max_drawdown_proxy,
            return_proxy=return_proxy,
            horizon_type=horizon_type,
        )

        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_generator_archetype": generator_archetype,
                "strategy_type": strategy_type,
                "return_proxy": return_proxy,
                "roi_365d_proxy": return_proxy,
                "volatility_proxy": volatility_proxy,
                "max_drawdown_proxy": max_drawdown_proxy,
                "downside_risk_proxy": downside_risk_proxy,
                "sortino_proxy": sortino_proxy,
                "hhi_norm": hhi_norm,
                "turnover_proxy": turnover_proxy,
                "avg_holding_period_days": _turnover_to_holding_period_days(turnover_proxy),
                "risk_score": risk_score,
                "stability_score": stability_score,
                "complexity_score": complexity_score,
                "main_asset_class": main_asset_class,
                "asset_class": main_asset_class,  # Backward-compatible alias.
                "horizon_type": horizon_type,
                "style_type": _sample_style_type(generator_archetype, rng),
                "track_record_months": track_record_months,
                "track_record_confidence": track_record_confidence,
                "platform_risk_number": _platform_risk_number(risk_score),
                "popularity_score": popularity_score,
                "copiers_count": copiers_count,
                "aum_proxy": aum_proxy,

                # Backward-compatible aliases for older validation / experiments.
                "strategy_archetype": strategy_type,
                "risk_level": risk_score,
                "volatility": volatility_proxy,
                "max_drawdown": max_drawdown_proxy,
            }
        )

    return pd.DataFrame(rows)


def _sample_strategy_generator_archetypes(
    n_strategies: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample generator priors for strategies."""
    probabilities = [0.25, 0.35, 0.25, 0.15]
    return rng.choice(STRATEGY_GENERATOR_ARCHETYPES, size=n_strategies, p=probabilities)


def _sample_main_asset_class(archetype: str, rng: np.random.Generator) -> str:
    """Sample primary asset class conditional on generator archetype."""
    if archetype == "conservative":
        return str(rng.choice(ASSET_CLASSES, p=[0.45, 0.05, 0.15, 0.10, 0.25]))

    if archetype in {"aggressive", "high_volatility"}:
        return str(rng.choice(ASSET_CLASSES, p=[0.20, 0.35, 0.25, 0.05, 0.15]))

    return str(rng.choice(ASSET_CLASSES, p=[0.30, 0.15, 0.20, 0.10, 0.25]))


def _sample_horizon(
    archetype: str,
    turnover_proxy: float,
    rng: np.random.Generator,
) -> str:
    """Sample strategy horizon using archetype and turnover.

    Higher turnover makes short horizon more likely.
    """
    if turnover_proxy >= 0.70:
        return str(rng.choice(HORIZONS, p=[0.60, 0.30, 0.10]))

    if archetype == "conservative":
        return str(rng.choice(HORIZONS, p=[0.10, 0.35, 0.55]))

    if archetype == "aggressive":
        return str(rng.choice(HORIZONS, p=[0.45, 0.40, 0.15]))

    return str(rng.choice(HORIZONS, p=[0.25, 0.50, 0.25]))


def _sample_style_type(archetype: str, rng: np.random.Generator) -> str:
    """Sample trading style conditional on generator archetype."""
    if archetype == "conservative":
        return str(rng.choice(STYLE_TYPES, p=[0.20, 0.35, 0.05, 0.25, 0.15]))

    if archetype == "high_volatility":
        return str(rng.choice(STYLE_TYPES, p=[0.30, 0.10, 0.35, 0.15, 0.10]))

    if archetype == "aggressive":
        return str(rng.choice(STYLE_TYPES, p=[0.35, 0.10, 0.20, 0.25, 0.10]))

    return str(rng.choice(STYLE_TYPES))


def _classify_strategy_type(
    risk_score: float,
    stability_score: float,
    volatility_proxy: float,
    max_drawdown_proxy: float,
    return_proxy: float,
    horizon_type: str,
) -> str:
    """Classify final strategy type using interpretable rules."""
    if volatility_proxy >= 0.75 and max_drawdown_proxy >= 0.55:
        return "high_volatility"

    if risk_score < 0.35 and stability_score >= 0.60 and horizon_type in {"medium", "long"}:
        return "conservative"

    if risk_score >= 0.60 and return_proxy >= 0.50:
        return "aggressive"

    return "balanced"


def _compute_risk_score(
    volatility_proxy: float,
    max_drawdown_proxy: float,
    downside_risk_proxy: float,
    hhi_norm: float,
) -> float:
    """Compute interpretable aggregated strategy risk score."""
    risk_score = (
        0.35 * volatility_proxy
        + 0.35 * max_drawdown_proxy
        + 0.15 * downside_risk_proxy
        + 0.15 * hhi_norm
    )
    return _clip01(risk_score)


def _compute_stability_score(
    volatility_proxy: float,
    max_drawdown_proxy: float,
    sortino_proxy: float,
    track_record_confidence: float,
    complexity_score: float,
) -> float:
    """Compute strategy stability score.

    Stability is not exactly 1 - risk_score because track record and
    operational complexity matter separately.
    """
    stability_score = (
        0.30 * (1.0 - volatility_proxy)
        + 0.30 * (1.0 - max_drawdown_proxy)
        + 0.15 * sortino_proxy
        + 0.15 * track_record_confidence
        + 0.10 * (1.0 - complexity_score)
    )
    return _clip01(stability_score)


def _compute_complexity_score(
    turnover_proxy: float,
    hhi_norm: float,
) -> float:
    """Compute strategy operational complexity."""
    return _clip01(0.70 * turnover_proxy + 0.30 * hhi_norm)


def _compute_sortino_proxy(
    return_proxy: float,
    downside_risk_proxy: float,
) -> float:
    """Compute bounded Sortino-like proxy without an equity curve."""
    return _clip01(0.60 * return_proxy + 0.40 * (1.0 - downside_risk_proxy))


def _generate_popularity_score(
    return_proxy: float,
    stability_score: float,
    max_drawdown_proxy: float,
    track_record_confidence: float,
    turnover_proxy: float,
    rng: np.random.Generator,
) -> float:
    """Generate noisy platform/social popularity.

    Popularity is correlated with visible performance and history,
    but it is not identical to strategy quality.
    """
    raw_score = (
        0.40 * return_proxy
        + 0.20 * stability_score
        - 0.15 * max_drawdown_proxy
        + 0.15 * track_record_confidence
        + 0.10 * turnover_proxy
        + rng.normal(loc=0.0, scale=0.12)
    )
    return _clip01(raw_score)


def _generate_copiers_count(
    popularity_score: float,
    rng: np.random.Generator,
) -> int:
    """Generate rough social proof count from popularity."""
    log_copiers = rng.normal(loc=1.2 + 6.0 * popularity_score, scale=0.85)
    return int(np.clip(np.exp(log_copiers), 0, 100_000))


def _generate_aum_proxy(
    popularity_score: float,
    copiers_count: int,
    rng: np.random.Generator,
) -> float:
    """Generate synthetic AUM proxy.

    This is not a real currency amount. It is only a scale/social proof proxy.
    """
    average_copy_size = rng.lognormal(mean=7.0 + 1.2 * popularity_score, sigma=0.7)
    return float(copiers_count * average_copy_size)


def _sample_track_record_months(
    mean_months: float,
    rng: np.random.Generator,
) -> int:
    """Sample track record length in months."""
    value = rng.normal(loc=mean_months, scale=6.0)
    return int(np.clip(round(value), 1, 60))


def _track_record_confidence(
    track_record_months: int,
    target_months: int = 12,
) -> float:
    """Convert track record length into confidence score."""
    return float(np.clip(track_record_months / target_months, 0.0, 1.0))


def _turnover_to_holding_period_days(turnover_proxy: float) -> int:
    """Convert turnover proxy into rough holding period proxy."""
    holding_days = 3 + (1.0 - turnover_proxy) * 180
    return int(np.clip(round(holding_days), 1, 365))


def _platform_risk_number(risk_score: float) -> int:
    """Convert risk_score in [0, 1] into a platform-like 1-10 risk number."""
    return int(np.clip(round(1 + 9 * risk_score), 1, 10))


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


def _clip01(value: float) -> float:
    """Clip numeric value to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))