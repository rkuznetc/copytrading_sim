from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from copytrading_sim.experiments.config import load_experiment_config
from copytrading_sim.experiments.runner import run_experiment
from copytrading_sim.models.mf import LogisticMF
from copytrading_sim.provenance import sha256_file


def config_file(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    config = {
        'simulation_config': str(repo/'configs/small.yaml'),
        'simulation_overrides': {'simulation': {'n_users': 6, 'n_strategies': 8,
                                                'exposures_per_user': 10, 'slate_size': 3}},
        'seeds': [42],
        'mf': {'factors': 3, 'epochs': 4, 'patience': 3, 'batch_size': 32},
        'evaluation': {'catalog_k': [1,3], 'choice_k': [1,2], 'response_steps': 1},
        'output': {'dir': str(tmp_path/'experiments')},
    }
    path = tmp_path/'experiment.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return path


def test_experiment_complete_reproducible_and_reloadable(tmp_path):
    config = load_experiment_config(config_file(tmp_path))
    first = run_experiment(config, run_id='first', progress=lambda _: None)
    second = run_experiment(config, run_id='second', progress=lambda _: None)
    for name in ('metrics_per_seed.csv', 'catalog_per_user.csv', 'heldout_positive_slates.csv',
                 'response_per_slate.csv', 'mf_training.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(first/name), pd.read_csv(second/name), check_exact=True)
    assert len(list((first/'figures').glob('*.png'))) == 7
    assert (first/'report.html').is_file()
    manifest = json.loads((first/'manifest.json').read_text())
    for name, checksum in manifest['files'].items():
        assert sha256_file(first/name) == checksum
    model = LogisticMF.load(first/'seed_42/mf.npz')
    assert model.score(0, model.strategy_ids).shape == (8,)
    oracle = pd.read_csv(first/'seed_42/compatibility_oracle/catalog_per_user.csv')
    assert oracle.oracle_regret_at_k.max() < 1e-12
    with pytest.raises(FileExistsError):
        run_experiment(config, run_id='first', progress=lambda _: None)
    # Reuse exact saved input rather than regenerating under method-specific settings.
    reused = run_experiment(config, run_id='reused', dataset=first/'seed_42/dataset', progress=lambda _: None)
    pd.testing.assert_frame_equal(pd.read_csv(first/'metrics_per_seed.csv'),
                                  pd.read_csv(reused/'metrics_per_seed.csv'), check_exact=True)


def test_experiment_cli_and_error_handling(tmp_path):
    path = config_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['methods'] = ['random', 'compatibility_oracle']
    path.write_text(yaml.safe_dump(raw))
    command = [sys.executable, '-m', 'copytrading_sim.experiments', '--config', str(path), '--run-id', 'cli']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Experiment saved:' in result.stdout
    duplicate = subprocess.run(command, capture_output=True, text=True)
    assert duplicate.returncode == 2 and 'ERROR:' in duplicate.stderr


def test_failed_experiment_cleans_own_directory(tmp_path):
    config = load_experiment_config(config_file(tmp_path))
    config['simulation']['simulation']['exposures_per_user'] = 1
    base = Path(config['output']['dir'])
    base.mkdir()
    (base/'keep.txt').write_text('keep')
    with pytest.raises(ValueError, match='Too few'):
        run_experiment(config, run_id='bad', progress=lambda _: None)
    assert not (base/'bad').exists()
    assert (base/'keep.txt').read_text() == 'keep'


@pytest.mark.parametrize('change', [
    {'methods': ['typo']}, {'seeds': [42,42]}, {'mf': {'epoch': 10}},
    {'evaluation': {'choice_k': [3]}}, {'seeds': [-1]}, {'extra': 1},
])
def test_experiment_config_rejects_mistakes(tmp_path, change):
    path = config_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(change)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_experiment_config(path)
