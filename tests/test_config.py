# tests/test_config.py

from copytrading_sim.config import load_config
from copytrading_sim.paths import CONFIGS_DIR


def test_load_small_config():
    config = load_config(CONFIGS_DIR / "small.yaml")

    assert "project" in config
    assert "random" in config
    assert "simulation" in config
    assert "output" in config


def test_small_config_has_positive_values():
    config = load_config(CONFIGS_DIR / "small.yaml")
    simulation = config["simulation"]

    assert simulation["n_users"] > 0
    assert simulation["n_strategies"] > 0
    assert simulation["slate_size"] > 0
    assert simulation["slate_size"] <= simulation["n_strategies"]