from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from copytrading_sim.simulator.copy_decision import build_copy_events
from copytrading_sim.simulator.pipeline import run_pipeline
from copytrading_sim.validation.structural_checks import SimulationValidationError, validate_tables


@pytest.fixture
def valid_tables(tiny_config):
    return run_pipeline(tiny_config)


def test_valid_run(tiny_config, valid_tables):
    report = validate_tables(valid_tables, tiny_config)
    assert report["passed"]
    assert report["checks_run"] > 50


@pytest.mark.parametrize("table,column,value,match", [
    ("users", "user_id", 1, "duplicate identifiers"),
    ("strategies", "risk_score", np.nan, "missing values"),
    ("strategies", "risk_score", np.inf, "non-finite"),
    ("strategies", "risk_score", 1.2, r"outside \[0, 1\]"),
    ("exposures", "strategy_id", 9999, "unknown strategy_id"),
    ("exposures", "position", 0, "positions must be"),
    ("exposures", "slate_probability_prefix", 0.99, "incorrect probability prefixes"),
    ("copy_decisions", "copy_probability", 0.99, "sum to 1"),
    ("copy_decisions", "no_copy_probability", 0.99, "constant within a slate"),
    ("metadata", "n_copy_events", 9999, "inconsistent value"),
])
def test_corruption_is_detected(tiny_config, valid_tables, table, column, value, match):
    valid_tables[table].loc[0, column] = value
    with pytest.raises(SimulationValidationError, match=match):
        validate_tables(valid_tables, tiny_config)


def test_duplicate_slate_strategy(tiny_config, valid_tables):
    frame = valid_tables["exposures"]
    frame.loc[1, "strategy_id"] = frame.loc[0, "strategy_id"]
    with pytest.raises(SimulationValidationError, match="repeated strategy"):
        validate_tables(valid_tables, tiny_config)


def test_two_copies_in_one_slate(tiny_config, valid_tables):
    valid_tables["copy_decisions"].loc[[0, 1], "copied"] = 1
    with pytest.raises(SimulationValidationError, match="More than one copy"):
        validate_tables(valid_tables, tiny_config)


def test_empty_copy_events_have_full_typed_schema(tiny_config):
    tiny_config["copy_decision"] = {"no_copy_bias": 100, "noise_std": 0}
    tables = run_pipeline(tiny_config)
    events = tables["copy_events"]
    assert events.empty
    assert set(events.columns) == set(tables["copy_decisions"].columns) | {"copy_event_id"}
    assert events["user_id"].dtype == tables["copy_decisions"]["user_id"].dtype
    assert validate_tables(tables, tiny_config)["warnings"]


def test_one_user_one_strategy(tiny_config):
    tiny_config["simulation"].update(n_users=1, n_strategies=1, slate_size=1)
    tables = run_pipeline(tiny_config)
    assert tables["exposures"]["position_probability"].eq(1).all()
    assert len(tables["copy_events"]) <= 3


def test_copy_events_are_subset_not_independent_generation(valid_tables):
    decisions = valid_tables["copy_decisions"]
    events = build_copy_events(decisions)
    pd.testing.assert_frame_equal(
        events[decisions.columns], decisions.loc[decisions["copied"].eq(1)].reset_index(drop=True)
    )


def test_config_mismatch_is_detected(tiny_config, valid_tables):
    changed = deepcopy(tiny_config)
    changed["copy_decision"] = {"temperature": 0.7}
    with pytest.raises(SimulationValidationError, match="configuration differs"):
        validate_tables(valid_tables, changed)


def test_missing_table_fails_clearly(tiny_config, valid_tables):
    del valid_tables["users"]
    with pytest.raises(SimulationValidationError, match="seven simulation tables"):
        validate_tables(valid_tables, tiny_config)
