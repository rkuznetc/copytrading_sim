"""CLI for iteration-2 policy/model comparison."""
import argparse
import sys
from copytrading_sim.experiments.config import load_experiment_config
from copytrading_sim.experiments.runner import run_experiment
from copytrading_sim.paths import resolve_project_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare recommendation policies and centralized MF.')
    parser.add_argument('--config', default='configs/experiments/quick.yaml')
    parser.add_argument('--run-id')
    parser.add_argument('--output-dir', help='Experiment root, relative to current working directory')
    parser.add_argument('--dataset', help='Existing saved run from iteration 1/2; no regeneration')
    args = parser.parse_args(argv)
    try:
        config = load_experiment_config(resolve_project_path(args.config))
        run_experiment(config, run_id=args.run_id, output_dir=args.output_dir, dataset=args.dataset)
    except (OSError, ValueError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
