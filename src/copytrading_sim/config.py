"""Load, merge and resolve the configuration used by a complete simulation run."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml

from copytrading_sim.simulator.compatibility import DEFAULT_COMPATIBILITY_WEIGHTS
from copytrading_sim.simulator.copy_decision import DEFAULT_COPY_DECISION_PARAMS
from copytrading_sim.simulator.exposure_policy import DEFAULT_EXPOSURE_POLICY_PARAMS

REQUIRED_SECTIONS = ("project", "random", "simulation", "output")
OPTIONAL_SECTIONS = ("compatibility", "exposure_policy", "copy_decision")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Config must be a non-empty YAML mapping: {path}")
    return value


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    config_path: str | Path,
    override_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Apply explicitly requested YAML overlays, then return the resolved config.

    No other files in configs/ are implicitly loaded. Later overlays win.
    """
    config = _read_yaml(config_path)
    for path in override_paths or []:
        config = _merge(config, _read_yaml(path))
    return resolve_config(config)


def _mapping(value: Any, path: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(map(str, unknown))}")
    return value


def _integer(value: Any, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, path: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        condition = "> 0" if positive else ">= 0"
        raise ValueError(f"{path} must be finite and {condition}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate inputs and materialise all configurable defaults without mutation.

    Generator archetype parameters remain code-defined in this iteration. Their
    provenance is the source-code digest recorded alongside each saved run.
    """
    _mapping(config, "config", set(REQUIRED_SECTIONS + OPTIONAL_SECTIONS))
    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    result = deepcopy(config)
    for name in REQUIRED_SECTIONS:
        if not isinstance(result[name], dict):
            raise ValueError(f"{name} must be a mapping")

    project = _mapping(result["project"], "project", {"name", "version"})
    for key in ("name", "version"):
        project[key] = _text(project.get(key), f"project.{key}")
    random = _mapping(result["random"], "random", {"seed"})
    random["seed"] = _integer(random.get("seed"), "random.seed", minimum=0)
    if random["seed"] >= 2**63:
        raise ValueError("random.seed must be smaller than 2**63 for portable int64 metadata")

    simulation_keys = {"n_users", "n_strategies", "exposures_per_user", "slate_size"}
    simulation = _mapping(result["simulation"], "simulation", simulation_keys)
    for key in sorted(simulation_keys):
        simulation[key] = _integer(simulation.get(key), f"simulation.{key}")
    if simulation["slate_size"] > simulation["n_strategies"]:
        raise ValueError("simulation.slate_size cannot be greater than n_strategies")

    output = _mapping(result["output"], "output", {"dir", "format"})
    output["dir"] = _text(output.get("dir"), "output.dir")
    output.setdefault("format", "csv")
    if output["format"] != "csv":
        raise ValueError("output.format must be 'csv' in iteration 1")

    compatibility = _mapping(result.get("compatibility", {}), "compatibility", {"weights"})
    provided = _mapping(
        compatibility.get("weights", {}),
        "compatibility.weights",
        set(DEFAULT_COMPATIBILITY_WEIGHTS),
    )
    weights = {**DEFAULT_COMPATIBILITY_WEIGHTS, **provided}
    weights = {key: _number(value, f"compatibility.weights.{key}")
               for key, value in weights.items()}
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("compatibility.weights must sum to 1.0")
    result["compatibility"] = {"weights": weights}

    for section, defaults in (
        ("exposure_policy", DEFAULT_EXPOSURE_POLICY_PARAMS),
        ("copy_decision", DEFAULT_COPY_DECISION_PARAMS),
    ):
        provided = _mapping(result.get(section, {}), section, set(defaults))
        params = {**defaults, **provided}
        for key, value in params.items():
            if key == "policy_name":
                params[key] = _text(value, f"{section}.{key}")
            else:
                params[key] = _number(value, f"{section}.{key}", positive=key == "temperature")
        result[section] = params

    exposure = result["exposure_policy"]
    total = sum(exposure[key] for key in (
        "compatibility_weight", "popularity_weight", "social_popularity_weight"
    ))
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("exposure_policy weights must sum to 1.0")
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Backward-compatible validation-only entry point."""
    resolve_config(config)


def generation_config_digest(config: dict[str, Any]) -> str:
    """Hash the resolved configuration, excluding only the output destination."""
    resolved = resolve_config(config)
    resolved.pop("output")
    payload = json.dumps(resolved, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
