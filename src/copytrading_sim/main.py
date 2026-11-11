"""Command-line entry point: generate, validate and save one simulation run."""

import argparse
import sys

from copytrading_sim.config import load_config
from copytrading_sim.datasets.exporters import build_summary, save_run
from copytrading_sim.paths import resolve_project_path
from copytrading_sim.simulator.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save a reproducible simulation run.")
    parser.add_argument("--config", default="configs/small.yaml", help="Path to the main YAML config.")
    parser.add_argument("--override", action="append", default=[], metavar="YAML",
                        help="Explicit YAML overlay; repeatable, later files take precedence.")
    parser.add_argument("--output-dir", help="Override output.dir (relative to project root).")
    parser.add_argument("--run-id", help="Run directory name; must not already exist.")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="Also print the first N rows of each table (default: no preview).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.preview < 0:
            raise ValueError("--preview must be non-negative")
        config_path = resolve_project_path(args.config)
        overrides = [resolve_project_path(path) for path in args.override]
        config = load_config(config_path, override_paths=overrides)
        if args.output_dir:
            config["output"]["dir"] = args.output_dir
        tables = run_pipeline(config)
        run_dir = save_run(tables, config, run_id=args.run_id)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = build_summary(tables)
    print(f"Run saved: {run_dir}")
    print("Structural validation: passed (not a validation of financial realism)")
    for name, table in tables.items():
        print(f"{name}: {table.shape}")
        if args.preview:
            print(table.head(args.preview).to_string(index=False))
    print(f"Copy rate per slate: {summary['copy_rate_per_slate']:.4f}")
    print(f"Copy rate per exposed item: {summary['copy_rate_per_exposed_item']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
