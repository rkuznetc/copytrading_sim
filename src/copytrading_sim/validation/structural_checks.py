"""Structural invariants of the current, one-pass simulation (not realism tests)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from copytrading_sim.config import generation_config_digest, resolve_config
from copytrading_sim.randomness import RNG_SCHEME

TABLE_NAMES = (
    "users", "strategies", "compatibility", "exposures",
    "copy_decisions", "copy_events", "metadata",
)
REQUIRED_COLUMNS = {
    "users": {"user_id", "user_type", "risk_tolerance", "drawdown_tolerance"},
    "strategies": {"strategy_id", "strategy_type", "risk_score", "max_drawdown_proxy"},
    "compatibility": {"user_id", "strategy_id", "compatibility_score"},
    "exposures": {
        "exposure_id", "slate_id", "user_id", "strategy_id", "timestamp_step",
        "position", "was_shown", "policy_name", "position_probability",
        "slate_probability_prefix", "compatibility_score",
    },
    "copy_decisions": {
        "exposure_id", "slate_id", "user_id", "strategy_id", "copied",
        "decision_score", "copy_probability", "no_copy_probability",
    },
    "copy_events": {"copy_event_id", "exposure_id", "slate_id", "user_id", "strategy_id"},
    "metadata": {
        "seed", "rng_scheme", "generation_config_sha256", "n_users", "n_strategies",
        "n_compatibility_rows", "n_exposure_rows", "n_copy_events", "n_slates",
        "copy_rate_per_exposed_item", "copy_rate_per_slate",
        "exposures_per_user", "slate_size",
    },
}
UNIT_INTERVAL_COLUMNS = {
    "risk_tolerance", "drawdown_tolerance", "social_sensitivity", "drawdown_sensitivity",
    "social_bias", "return_proxy", "roi_365d_proxy", "volatility_proxy",
    "max_drawdown_proxy", "downside_risk_proxy", "sortino_proxy", "hhi_norm",
    "turnover_proxy", "risk_score", "stability_score", "complexity_score",
    "track_record_confidence", "popularity_score", "risk_level", "volatility", "max_drawdown",
    "risk_match", "drawdown_match", "horizon_match", "asset_match", "complexity_match",
    "compatibility_score", "strategy_risk_score", "strategy_max_drawdown_proxy",
    "strategy_popularity_score", "strategy_return_proxy", "strategy_complexity_score",
    "logging_score", "position_probability", "slate_probability_prefix", "decision_score",
    "copy_probability", "no_copy_probability", "copy_rate_per_exposed_item", "copy_rate_per_slate",
}
INTEGER_COLUMNS = {
    "user_id", "strategy_id", "exposure_id", "copy_event_id", "timestamp_step", "position",
    "was_shown", "copied", "seed", "n_users", "n_strategies", "n_compatibility_rows",
    "n_exposure_rows", "n_copy_events", "n_slates", "exposures_per_user", "slate_size",
    "track_record_months", "avg_holding_period_days", "platform_risk_number", "copiers_count",
}


class SimulationValidationError(ValueError):
    """A generated or loaded dataset violates the simulation table contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Simulation validation failed:\n- " + "\n- ".join(errors))


def validate_tables(
    tables: dict[str, pd.DataFrame], config: dict[str, Any],
) -> dict[str, Any]:
    """Check structure, references, slates and conditional choice probabilities.

    Raises SimulationValidationError on failure. Passing these checks does NOT
    establish financial realism, recommendation quality or OPE validity.
    """
    config = resolve_config(config)
    errors: list[str] = []
    checks = 0

    def check(condition: Any, message: str) -> None:
        nonlocal checks
        checks += 1
        if not bool(condition):
            errors.append(message)

    check(set(tables) == set(TABLE_NAMES), "Expected exactly the seven simulation tables")
    for name in TABLE_NAMES:
        table = tables.get(name)
        check(isinstance(table, pd.DataFrame), f"{name}: expected a DataFrame")
        if not isinstance(table, pd.DataFrame):
            continue
        check(table.columns.is_unique, f"{name}: duplicate column names")
        missing = REQUIRED_COLUMNS[name] - set(table.columns)
        check(not missing, f"{name}: missing columns {sorted(missing)}")
    if errors:
        raise SimulationValidationError(errors)

    sim = config["simulation"]
    n_users, n_strategies = sim["n_users"], sim["n_strategies"]
    steps, k = sim["exposures_per_user"], sim["slate_size"]
    expected_rows = {
        "users": n_users, "strategies": n_strategies, "compatibility": n_users * n_strategies,
        "exposures": n_users * steps * k, "copy_decisions": n_users * steps * k, "metadata": 1,
    }
    for name, expected in expected_rows.items():
        check(len(tables[name]) == expected, f"{name}: expected {expected} rows")
    for name, table in tables.items():
        check(not table.isna().any().any(), f"{name}: missing values")
        for column in table.columns:
            series = table[column]
            if is_numeric_dtype(series):
                check(np.isfinite(series.to_numpy()).all(), f"{name}.{column}: non-finite values")
            if column in INTEGER_COLUMNS:
                check(is_integer_dtype(series) and not is_bool_dtype(series),
                      f"{name}.{column}: expected integer dtype")
                if is_numeric_dtype(series):
                    check((series >= 0).all(), f"{name}.{column}: negative values")
            if column in UNIT_INTERVAL_COLUMNS:
                valid_type = is_numeric_dtype(series) and not is_bool_dtype(series)
                check(valid_type, f"{name}.{column}: expected a numeric score")
                if valid_type:
                    check(series.between(0, 1).all(), f"{name}.{column}: outside [0, 1]")
    # Stop before joins/grouping if the basic schema is unusable.
    if errors:
        raise SimulationValidationError(errors)

    users, strategies = tables["users"], tables["strategies"]
    pairs, exposures = tables["compatibility"], tables["exposures"]
    decisions, events = tables["copy_decisions"], tables["copy_events"]
    for name, key in (
        ("users", "user_id"), ("strategies", "strategy_id"),
        ("exposures", "exposure_id"), ("copy_decisions", "exposure_id"),
        ("copy_events", "copy_event_id"),
    ):
        check(tables[name][key].is_unique, f"{name}.{key}: duplicate identifiers")
    for name in ("compatibility", "exposures", "copy_decisions", "copy_events"):
        check(tables[name]["user_id"].isin(users["user_id"]).all(), f"{name}: unknown user_id")
        check(tables[name]["strategy_id"].isin(strategies["strategy_id"]).all(),
              f"{name}: unknown strategy_id")
    check(not pairs.duplicated(["user_id", "strategy_id"]).any(),
          "compatibility: duplicate user-strategy pairs")
    check(events["exposure_id"].is_unique, "copy_events: duplicate exposure_id")
    if errors:
        raise SimulationValidationError(errors)

    groups = exposures.groupby("slate_id", sort=False)
    check(groups.ngroups == n_users * steps, "exposures: wrong number of slates")
    check(groups.size().eq(k).all(), "exposures: wrong slate size")
    check(groups["user_id"].nunique().eq(1).all(), "exposures: multiple users in one slate")
    check(groups["timestamp_step"].nunique().eq(1).all(), "exposures: multiple steps in one slate")
    check(groups["position"].nunique().eq(k).all() and exposures["position"].between(1, k).all(),
          "exposures: positions must be 1..slate_size without duplicates")
    check(not exposures.duplicated(["slate_id", "strategy_id"]).any(),
          "exposures: repeated strategy within a slate")
    check(exposures["timestamp_step"].between(0, steps - 1).all(), "exposures: invalid step")
    check(exposures.groupby("user_id")["slate_id"].nunique().eq(steps).all(),
          "exposures: wrong number of slates per user")
    check(exposures.groupby(["user_id", "timestamp_step"])["slate_id"].nunique().eq(1).all(),
          "exposures: multiple slates for the same user-step")
    check(exposures["was_shown"].eq(1).all(), "exposures.was_shown must equal 1")
    check(exposures["policy_name"].eq(config["exposure_policy"]["policy_name"]).all(),
          "exposures.policy_name differs from config")

    ordered = exposures.sort_values(["slate_id", "position"])
    prefixes = ordered.groupby("slate_id", sort=False)["position_probability"].cumprod()
    check(np.allclose(ordered["slate_probability_prefix"], prefixes, rtol=1e-9, atol=1e-12),
          "exposures: incorrect probability prefixes")
    check(exposures["position_probability"].gt(0).all(),
          "exposures: a selected item must have positive conditional probability")
    reference = pairs.set_index(["user_id", "strategy_id"])["compatibility_score"]
    lookup = pd.MultiIndex.from_frame(exposures[["user_id", "strategy_id"]])
    check(np.allclose(exposures["compatibility_score"], reference.reindex(lookup).to_numpy()),
          "exposures: compatibility scores differ from the pair table")

    check(set(exposures.columns).issubset(decisions.columns),
          "copy_decisions must preserve all exposure columns")
    if set(exposures.columns).issubset(decisions.columns):
        left = exposures.sort_values("exposure_id").reset_index(drop=True)
        right = decisions.sort_values("exposure_id")[exposures.columns].reset_index(drop=True)
        check(left.equals(right), "copy_decisions: exposure rows were lost or modified")
    check(decisions["copied"].isin([0, 1]).all(), "copy_decisions.copied must be 0 or 1")
    decision_groups = decisions.groupby("slate_id", sort=False)
    check(decision_groups["copied"].sum().le(1).all(), "More than one copy in a slate")
    check(decision_groups["no_copy_probability"].nunique().eq(1).all(),
          "no_copy_probability must be constant within a slate")
    totals = (decision_groups["copy_probability"].sum()
              + decision_groups["no_copy_probability"].first())
    check(np.allclose(totals, 1.0, atol=1e-9, rtol=0),
          "Copy probabilities plus one no-copy probability must sum to 1 per slate")
    check(set(events.columns) == set(decisions.columns) | {"copy_event_id"},
          "copy_events: schema must be stable even when empty")
    if set(decisions.columns).issubset(events.columns):
        chosen = decisions.loc[decisions["copied"].eq(1)].sort_values("exposure_id")
        actual = events.sort_values("exposure_id")[decisions.columns]
        check(chosen.reset_index(drop=True).equals(actual.reset_index(drop=True)),
              "copy_events must match exactly the copied decisions")

    meta = tables["metadata"].iloc[0]
    counts = {
        "n_users": n_users, "n_strategies": n_strategies, "n_slates": n_users * steps,
        "n_compatibility_rows": len(pairs), "n_exposure_rows": len(exposures),
        "n_copy_events": len(events), "exposures_per_user": steps, "slate_size": k,
        "seed": config["random"]["seed"],
    }
    for column, value in counts.items():
        check(meta[column] == value, f"metadata.{column}: inconsistent value")
    check(meta["rng_scheme"] == RNG_SCHEME, "metadata: unsupported RNG scheme")
    check(meta["generation_config_sha256"] == generation_config_digest(config),
          "metadata: configuration differs from the one used to generate the tables")
    check(np.isclose(meta["copy_rate_per_exposed_item"], len(events) / len(exposures)),
          "metadata: inconsistent per-item copy rate")
    check(np.isclose(meta["copy_rate_per_slate"], len(events) / (n_users * steps)),
          "metadata: inconsistent per-slate copy rate")
    if errors:
        raise SimulationValidationError(errors)
    warnings = [] if len(events) else ["No copy events: valid structurally, but check learning data."]
    return {"passed": True, "checks_run": checks, "errors": [], "warnings": warnings}
