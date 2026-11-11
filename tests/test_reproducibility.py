from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from copytrading_sim.config import generation_config_digest, load_config, resolve_config
from copytrading_sim.paths import CONFIGS_DIR
from copytrading_sim.randomness import make_rng
from copytrading_sim.simulator.pipeline import run_pipeline
from copytrading_sim.simulator.strategy_generator import generate_strategies
from copytrading_sim.simulator.user_generator import generate_users


def test_resolution_is_idempotent_and_does_not_mutate(tiny_config):
    original = deepcopy(tiny_config)
    resolved = resolve_config(tiny_config)
    assert tiny_config == original
    assert resolved == resolve_config(resolved)
    assert resolved["exposure_policy"]["temperature"] == 0.25
    assert resolved["copy_decision"]["no_copy_bias"] == 0.35
    assert sum(resolved["compatibility"]["weights"].values()) == 1.0


@pytest.mark.parametrize("section,key,value", [
    ("simulation", "n_users", 0),
    ("simulation", "n_users", True),
    ("simulation", "exposures_per_user", -1),
    ("simulation", "slate_size", 1.5),
    ("simulation", "slate_size", 99),
    ("random", "seed", -1),
    ("random", "seed", True),
    ("output", "format", "parquet"),
    ("output", "dir", ""),
    ("exposure_policy", "temperature", 0),
    ("exposure_policy", "noise_std", float("nan")),
    ("exposure_policy", "compatibility_weight", -0.1),
    ("copy_decision", "temperature", float("inf")),
])
def test_invalid_values_fail_early(tiny_config, section, key, value):
    tiny_config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        run_pipeline(tiny_config)


def test_unknown_config_keys_are_not_silently_ignored(tiny_config):
    tiny_config["logging_policy"] = {"temperature": 1}
    with pytest.raises(ValueError, match="logging_policy"):
        resolve_config(tiny_config)


def test_overlay_is_explicit_and_preserves_missing_defaults(tmp_path):
    overlay = tmp_path / "policy.yaml"
    overlay.write_text("exposure_policy:\n  temperature: 0.7\n")
    result = load_config(CONFIGS_DIR / "small.yaml", [overlay])
    assert result["exposure_policy"]["temperature"] == 0.7
    assert result["exposure_policy"]["noise_std"] == 0.05
    assert load_config(CONFIGS_DIR / "small.yaml")["exposure_policy"]["temperature"] == 0.25
    load_config(CONFIGS_DIR / "small.yaml", [CONFIGS_DIR / "logging_policy.yaml"])


@pytest.mark.parametrize("contents", ["", "null", "[]", "42", "simulation: {}"])
def test_bad_yaml_shape(tmp_path, contents):
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_config(path)


def test_same_seed_repeats_all_tables(tiny_config):
    original = deepcopy(tiny_config)
    first, second = run_pipeline(tiny_config), run_pipeline(tiny_config)
    assert tiny_config == original
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)


def test_changing_seed_changes_data(tiny_config):
    first = run_pipeline(tiny_config)
    tiny_config["random"]["seed"] += 1
    second = run_pipeline(tiny_config)
    assert not first["users"].equals(second["users"])
    assert not first["strategies"].equals(second["strategies"])


def test_stages_are_isolated(tiny_config):
    first = run_pipeline(tiny_config)
    tiny_config["simulation"]["n_users"] += 1
    second = run_pipeline(tiny_config)
    pd.testing.assert_frame_equal(first["strategies"], second["strategies"], check_exact=True)
    tiny_config["copy_decision"] = {"noise_std": 0.2}
    third = run_pipeline(tiny_config)
    pd.testing.assert_frame_equal(second["exposures"], third["exposures"], check_exact=True)


def test_direct_generators_match_pipeline(tiny_config):
    tables = run_pipeline(tiny_config)
    pd.testing.assert_frame_equal(generate_users(tiny_config), tables["users"])
    pd.testing.assert_frame_equal(generate_strategies(tiny_config), tables["strategies"])


def test_external_rng_effective_seed_replays_run(tiny_config):
    tables = run_pipeline(tiny_config, rng=np.random.default_rng(1234))
    tiny_config["random"]["seed"] = int(tables["metadata"].iloc[0]["seed"])
    replay = run_pipeline(tiny_config)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], replay[name], check_exact=True)


def test_named_stream_is_independent_of_other_draws():
    a = make_rng(42, "strategies").random(8)
    make_rng(42, "users").random(10000)
    np.testing.assert_array_equal(a, make_rng(42, "strategies").random(8))
    assert not np.array_equal(a, make_rng(42, "users").random(8))
    with pytest.raises(ValueError):
        make_rng(42, "typo")


def test_output_path_does_not_change_generation_digest(tiny_config):
    before = generation_config_digest(tiny_config)
    tiny_config["output"]["dir"] = "artifacts/another-run"
    assert generation_config_digest(tiny_config) == before


def test_custom_compatibility_weights_are_used(tiny_config):
    tiny_config["compatibility"] = {"weights": {
        "risk": 1.0, "drawdown": 0.0, "horizon": 0.0, "asset": 0.0, "complexity": 0.0,
    }}
    pairs = run_pipeline(tiny_config)["compatibility"]
    np.testing.assert_array_equal(pairs["compatibility_score"], pairs["risk_match"])
