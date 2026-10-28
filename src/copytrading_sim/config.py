# src/copytrading_sim/config.py

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("project", "random", "simulation", "output")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate YAML config."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    validate_config(config)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required top-level config sections."""
    missing = [section for section in REQUIRED_SECTIONS if section not in config]

    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    simulation = config["simulation"]

    if simulation["n_users"] <= 0:
        raise ValueError("simulation.n_users must be positive")

    if simulation["n_strategies"] <= 0:
        raise ValueError("simulation.n_strategies must be positive")

    if simulation["slate_size"] <= 0:
        raise ValueError("simulation.slate_size must be positive")

    if simulation["slate_size"] > simulation["n_strategies"]:
        raise ValueError("simulation.slate_size cannot be greater than n_strategies")