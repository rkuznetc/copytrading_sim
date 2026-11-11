"""Experiment configs are separate from generation configs; paths relative to YAML."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import yaml
from copytrading_sim.config import load_config, resolve_config, _merge
from copytrading_sim.models.mf import MFConfig

METHODS = ('random', 'popularity_public', 'popularity_train', 'compatibility_oracle',
           'mixed_logger', 'mf')


def _keys(value, allowed, section):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'Invalid or unknown keys in {section}')


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as stream:
        raw = yaml.safe_load(stream)
    allowed = {'simulation_config', 'simulation_overrides', 'seeds', 'methods',
               'split', 'mf', 'evaluation', 'output'}
    _keys(raw, allowed, 'experiment')
    if not isinstance(raw.get('simulation_config'), str):
        raise ValueError('simulation_config path required')
    source_path = (path.parent / raw['simulation_config']).resolve()
    sim = load_config(source_path)
    overrides = raw.get('simulation_overrides', {})
    if not isinstance(overrides, dict):
        raise ValueError('simulation_overrides must be a mapping')
    sim = resolve_config(_merge(sim, overrides))
    seeds = raw.get('seeds', [sim['random']['seed']])
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in seeds)):
        raise ValueError('seeds must be distinct nonnegative integers < 2**63')
    methods = raw.get('methods', list(METHODS))
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
            or any(m not in METHODS for m in methods)):
        raise ValueError(f'methods must be distinct choices from {METHODS}')
    split = {'train_fraction': .6, 'validation_fraction': .2, **raw.get('split', {})}
    _keys(split, {'train_fraction','validation_fraction'}, 'split')
    if (any(isinstance(v, bool) or not isinstance(v, (float,int)) or not math.isfinite(v)
            or v <= 0 for v in split.values()) or sum(split.values()) >= 1):
        raise ValueError('Split fractions must be positive and sum to less than one')
    mf = raw.get('mf', {})
    _keys(mf, MFConfig.__dataclass_fields__, 'mf')
    mf_config = MFConfig(**mf)
    evaluation = {'catalog_k': [1,3,5], 'choice_k': [1,3], 'response_steps': 5,
                  **raw.get('evaluation', {})}
    _keys(evaluation, {'catalog_k','choice_k','response_steps'}, 'evaluation')
    for name, upper in [('catalog_k', sim['simulation']['n_strategies']),
                        ('choice_k', sim['simulation']['slate_size'] - 1)]:
        values = evaluation[name]
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= upper for v in values)):
            raise ValueError(f'evaluation.{name}: unique integers in [1, {upper}] required')
        evaluation[name] = sorted(values)
    steps = evaluation['response_steps']
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('response_steps must be positive')
    output = raw.get('output', {'dir': 'artifacts/experiments'})
    _keys(output, {'dir'}, 'output')
    if not isinstance(output.get('dir'), str) or not output['dir'].strip():
        raise ValueError('output.dir must be a non-empty path')
    from dataclasses import asdict
    return {'simulation': sim, 'seeds': seeds, 'methods': methods, 'split': split,
            'mf': asdict(mf_config), 'evaluation': evaluation, 'output': output,
            'config_source': str(path)}
