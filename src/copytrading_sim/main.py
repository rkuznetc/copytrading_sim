# src/copytrading_sim/main.py

import argparse
from pathlib import Path

from copytrading_sim.config import load_config
from copytrading_sim.paths import ensure_dir, resolve_project_path
from copytrading_sim.simulator.strategy_generator import generate_strategies
from copytrading_sim.simulator.user_generator import generate_users
from copytrading_sim.simulator.compatibility import compute_compatibility
from copytrading_sim.simulator.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run copy-trading simulator skeleton."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/small.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     config_path = resolve_project_path(args.config)
#     config = load_config(config_path)

#     output_dir = resolve_project_path(config["output"]["dir"])
#     ensure_dir(output_dir)

#     simulation = config["simulation"]

#     print(f"Config: {config_path}")
#     print(f"Output dir: {output_dir}")
#     print(f"Seed: {config['random']['seed']}")
#     print(f"Users: {simulation['n_users']}")
#     print(f"Strategies: {simulation['n_strategies']}")
#     print(f"Exposures per user: {simulation['exposures_per_user']}")
#     print()

#     users = generate_users(config)
#     strategies = generate_strategies(config)
#     compatibility = compute_compatibility(users, strategies, top_k=5)

#     print(users.head())
#     print(strategies.head())
#     print(compatibility.head())
#     print(users.shape)
#     print(strategies.shape)
#     print(compatibility.shape)

#     print("Success!")


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    tables = run_pipeline(config)

    for name, table in tables.items():
        print(f"\n{name}: {table.shape}")
        print(table.head())


if __name__ == "__main__":
    main()