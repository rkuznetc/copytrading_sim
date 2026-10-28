# src/copytrading_sim/simulator/copy_decision.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_COPY_DECISION_PARAMS = {
    "compatibility_weight": 0.65,
    "social_popularity_weight": 0.15,
    "return_weight": 0.10,
    "drawdown_penalty_weight": 0.05,
    "liquidity_penalty_weight": 0.05,
    "noise_std": 0.04,
    "temperature": 0.25,
    "no_copy_bias": 0.35,
}

LIQUIDITY_NEED_TO_SCORE = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
}

HORIZON_TO_LIQUIDITY_LOCK = {
    "short": 0.20,
    "medium": 0.50,
    "long": 0.80,
}

REQUIRED_EXPOSURE_COLUMNS = {
    "slate_id",
    "user_id",
    "strategy_id",
    "timestamp_step",
    "position",
    "compatibility_score",
    "strategy_popularity_score",
    "social_sensitivity",
    "strategy_return_proxy",
    "strategy_max_drawdown_proxy",
    "strategy_horizon_type",
    "liquidity_need",
}


def simulate_copy_decisions(
    exposures: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate copy decisions from logged exposure slates.

    The user can copy at most one strategy from each shown slate.
    A no-copy option is included, so some slates may produce no copied strategy.
    """
    _validate_exposures(exposures)

    if rng is None:
        seed = int(config["random"]["seed"])
        rng = np.random.default_rng(seed)

    params = _get_copy_decision_params(config)

    decisions = exposures.copy()

    decisions["decision_score"] = np.nan
    decisions["copy_probability"] = np.nan
    decisions["no_copy_probability"] = np.nan
    decisions["copied"] = 0

    for slate_id, group in decisions.groupby("slate_id", sort=False):
        group_index = group.index

        decision_scores = _compute_decision_scores(
            slate=group,
            params=params,
            rng=rng,
        )

        no_copy_score = float(params["no_copy_bias"]) + rng.normal(
            loc=0.0,
            scale=float(params["noise_std"]),
        )

        all_scores = np.concatenate(
            [
                decision_scores,
                np.array([no_copy_score], dtype=float),
            ]
        )

        probabilities = _softmax(
            scores=all_scores,
            temperature=float(params["temperature"]),
        )

        strategy_probabilities = probabilities[:-1]
        no_copy_probability = float(probabilities[-1])

        choice_index = int(rng.choice(np.arange(len(probabilities)), p=probabilities))

        decisions.loc[group_index, "decision_score"] = decision_scores
        decisions.loc[group_index, "copy_probability"] = strategy_probabilities
        decisions.loc[group_index, "no_copy_probability"] = no_copy_probability

        if choice_index < len(group):
            chosen_row_index = group_index[choice_index]
            decisions.loc[chosen_row_index, "copied"] = 1

    decisions["copied"] = decisions["copied"].astype(int)

    return decisions


def build_copy_events(copy_decisions: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-copy-event table from exposure-level decisions."""
    copied = copy_decisions[copy_decisions["copied"] == 1].copy()

    if copied.empty:
        return pd.DataFrame(
            columns=[
                "copy_event_id",
                "slate_id",
                "user_id",
                "strategy_id",
                "timestamp_step",
                "position",
                "compatibility_score",
                "copy_probability",
                "decision_score",
            ]
        )

    copied = copied.reset_index(drop=True)
    copied.insert(0, "copy_event_id", np.arange(len(copied), dtype=int))

    preferred_columns = [
        "copy_event_id",
        "slate_id",
        "user_id",
        "strategy_id",
        "timestamp_step",
        "position",
        "compatibility_score",
        "copy_probability",
        "decision_score",
        "strategy_popularity_score",
        "strategy_return_proxy",
        "strategy_max_drawdown_proxy",
        "strategy_risk_score",
        "user_type",
        "strategy_type",
    ]

    existing_preferred_columns = [col for col in preferred_columns if col in copied.columns]
    remaining_columns = [col for col in copied.columns if col not in existing_preferred_columns]

    return copied[existing_preferred_columns + remaining_columns]


def _compute_decision_scores(
    slate: pd.DataFrame,
    params: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute per-strategy decision scores inside one slate."""
    compatibility = slate["compatibility_score"].to_numpy(dtype=float)
    popularity = slate["strategy_popularity_score"].to_numpy(dtype=float)
    social_sensitivity = slate["social_sensitivity"].to_numpy(dtype=float)
    return_proxy = slate["strategy_return_proxy"].to_numpy(dtype=float)
    max_drawdown = slate["strategy_max_drawdown_proxy"].to_numpy(dtype=float)

    liquidity_penalty = np.array(
        [
            _liquidity_penalty(
                liquidity_need=str(row["liquidity_need"]),
                strategy_horizon_type=str(row["strategy_horizon_type"]),
                strategy_max_drawdown_proxy=float(row["strategy_max_drawdown_proxy"]),
            )
            for _, row in slate.iterrows()
        ],
        dtype=float,
    )

    noise = rng.normal(
        loc=0.0,
        scale=float(params["noise_std"]),
        size=len(slate),
    )

    scores = (
        float(params["compatibility_weight"]) * compatibility
        + float(params["social_popularity_weight"]) * social_sensitivity * popularity
        + float(params["return_weight"]) * return_proxy
        - float(params["drawdown_penalty_weight"]) * max_drawdown
        - float(params["liquidity_penalty_weight"]) * liquidity_penalty
        + noise
    )

    return np.clip(scores, 0.0, 1.0)


def _liquidity_penalty(
    liquidity_need: str,
    strategy_horizon_type: str,
    strategy_max_drawdown_proxy: float,
) -> float:
    """Penalty for high-liquidity users choosing illiquid/risky strategies."""
    if liquidity_need not in LIQUIDITY_NEED_TO_SCORE:
        raise ValueError(f"Unknown liquidity_need: {liquidity_need}")

    if strategy_horizon_type not in HORIZON_TO_LIQUIDITY_LOCK:
        raise ValueError(f"Unknown strategy_horizon_type: {strategy_horizon_type}")

    liquidity_need_score = LIQUIDITY_NEED_TO_SCORE[liquidity_need]
    horizon_lock_score = HORIZON_TO_LIQUIDITY_LOCK[strategy_horizon_type]

    penalty = (
        0.50 * liquidity_need_score * horizon_lock_score
        + 0.50 * liquidity_need_score * strategy_max_drawdown_proxy
    )

    return float(np.clip(penalty, 0.0, 1.0))


def _get_copy_decision_params(config: dict[str, Any]) -> dict[str, float]:
    """Read copy decision params from config or use defaults.

    Supported optional config section:

    copy_decision:
      compatibility_weight: 0.65
      social_popularity_weight: 0.15
      return_weight: 0.10
      drawdown_penalty_weight: 0.05
      liquidity_penalty_weight: 0.05
      noise_std: 0.04
      temperature: 0.25
      no_copy_bias: 0.35
    """
    params = DEFAULT_COPY_DECISION_PARAMS.copy()
    params.update(config.get("copy_decision", {}))

    for key in params:
        params[key] = float(params[key])

    positive_keys = ["temperature"]
    non_negative_keys = [
        "compatibility_weight",
        "social_popularity_weight",
        "return_weight",
        "drawdown_penalty_weight",
        "liquidity_penalty_weight",
        "noise_std",
        "no_copy_bias",
    ]

    for key in positive_keys:
        if params[key] <= 0:
            raise ValueError(f"copy_decision.{key} must be positive.")

    for key in non_negative_keys:
        if params[key] < 0:
            raise ValueError(f"copy_decision.{key} must be non-negative.")

    return params


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Stable softmax with temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    scaled = scores / temperature
    scaled = scaled - np.max(scaled)
    exp_scores = np.exp(scaled)
    total = exp_scores.sum()

    if total <= 0 or not np.isfinite(total):
        return np.full(len(scores), 1.0 / len(scores))

    return exp_scores / total


def _validate_exposures(exposures: pd.DataFrame) -> None:
    """Validate required exposure columns."""
    missing_columns = REQUIRED_EXPOSURE_COLUMNS - set(exposures.columns)

    if missing_columns:
        raise ValueError(f"Missing exposure columns: {sorted(missing_columns)}")