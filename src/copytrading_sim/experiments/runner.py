"""One command: fixed histories -> learned MF -> three separate evaluations -> plots."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from copytrading_sim.baselines.oracle_proxy import OracleCompatibilityScorer
from copytrading_sim.baselines.popularity_policy import PopularityScorer
from copytrading_sim.baselines.random_policy import RandomPolicy
from copytrading_sim.config import resolve_config
from copytrading_sim.datasets.exporters import load_run, save_run
from copytrading_sim.datasets.splits import learning_view, temporal_split
from copytrading_sim.evaluation.metrics import (
    evaluate_catalog, evaluate_logged_choices, evaluate_response, macro_summary,
    CATALOG_METRICS, BEHAVIOUR_METRICS, RESPONSE_METRICS,
)
from copytrading_sim.models.mf import LogisticMF, MFConfig
from copytrading_sim.paths import resolve_project_path
from copytrading_sim.policies.base import ScorePolicy
from copytrading_sim.policies.mixed import MixedLoggingPolicy
from copytrading_sim.provenance import environment_provenance, source_provenance, sha256_file
from copytrading_sim.simulator.pipeline import run_pipeline


def _json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                          encoding='utf-8')


def _csv(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g', encoding='utf-8')


def _policy(name, tables, config, train, model):
    if name == 'random':
        return RandomPolicy()
    if name == 'popularity_public':
        return ScorePolicy(PopularityScorer.from_public_catalog(tables['strategies']), name)
    if name == 'popularity_train':
        return ScorePolicy(PopularityScorer.from_training(
            learning_view(train), tables['strategies'].strategy_id.to_numpy()), name)
    if name == 'compatibility_oracle':
        return ScorePolicy(OracleCompatibilityScorer(tables['compatibility']), name)
    if name == 'mixed_logger':
        policy = MixedLoggingPolicy(tables['users'], tables['strategies'],
                                    tables['compatibility'], config['exposure_policy'])
        policy.name = name
        return policy
    if name == 'mf' and model is not None:
        return ScorePolicy(model, name)
    raise ValueError(f'Method unavailable: {name}')


def _summary_rows(summary, protocol, seed, metrics):
    rows = []
    for r in summary.to_dict('records'):
        for metric in metrics:
            rows.append({'seed': seed, 'protocol': protocol, 'method': r['method'],
                         'k': int(r.get('k', 0)), 'metric': metric, 'value': r[metric],
                         'n_users': int(r['n_users'])})
    return rows


def run_experiment(config: dict, *, run_id: str | None = None,
                   output_dir: str | Path | None = None, dataset: str | Path | None = None,
                   progress=print) -> Path:
    """Generate or reuse fixed logs; never tune on test or regenerated method-specific logs.

    `dataset` accepts one iteration-1/2 saved run. Its seed must be the only
    configured seed. Its actual generation config is used without overrides.
    """
    config = deepcopy(config)
    existing = load_run(dataset) if dataset is not None else None
    if existing is not None:
        if config['seeds'] != [existing.config['random']['seed']]:
            raise ValueError('--dataset requires exactly its root seed in experiment.seeds')
        config['simulation'] = existing.config
    base = Path(output_dir).expanduser().resolve() if output_dir else resolve_project_path(config['output']['dir'])
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_id):
        raise ValueError('run_id must be a safe directory name')
    base.mkdir(parents=True, exist_ok=True)
    root = base / run_id
    root.mkdir(exist_ok=False)
    complete = False
    try:
        (root/'config.resolved.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        all_metrics, all_catalog, all_choices, all_responses, histories = [], [], [], [], []
        data_records, split_records = [], []
        for seed in config['seeds']:
            folder = root/f'seed_{seed}'
            folder.mkdir()
            generation = deepcopy(config['simulation'])
            generation['random']['seed'] = seed
            generation = resolve_config(generation)
            if existing is None:
                progress(f'[seed {seed}] generating one shared dataset')
                tables = run_pipeline(generation)
                data_dir = save_run(tables, generation, folder, run_id='dataset')
                data_manifest = json.loads((data_dir/'manifest.json').read_text())
            else:
                tables, data_manifest = existing.tables, existing.manifest
                # Copy the verified input dataset so this experiment is self-contained.
                data_dir = folder/'dataset'
                shutil.copytree(Path(dataset).resolve(), data_dir)
            split = temporal_split(tables['copy_decisions'], **config['split'])
            _csv(split.manifest, folder/'split_manifest.csv')
            counts = []
            for name in ('train','validation','test'):
                frame = getattr(split, name)
                positive = frame.groupby('slate_id')['copied'].sum()
                counts.append({'split': name, 'rows': len(frame), 'slates': len(positive),
                               'positive_slates': int(positive.gt(0).sum()),
                               'no_copy_slates': int(positive.eq(0).sum()),
                               'first_step': int(frame.timestamp_step.min()),
                               'last_step': int(frame.timestamp_step.max())})
            _json(folder/'split_summary.json', counts)
            split_records.append({'seed': seed, 'splits': counts})
            model = None
            if 'mf' in config['methods']:
                progress(f'[seed {seed}] training logistic MF on observed rows only')
                model = LogisticMF(tables['users'].user_id.to_numpy(),
                                   tables['strategies'].strategy_id.to_numpy(),
                                   MFConfig(**config['mf']), seed=seed)
                model.fit(learning_view(split.train), learning_view(split.validation))
                model.save(folder/'mf.npz')
                _csv(model.history, folder/'mf_training.csv')
                h = model.history.copy()
                h['seed'], h['best_epoch'] = seed, model.best_epoch
                histories.append(h)
                _json(folder/'mf_summary.json', {'best_epoch': model.best_epoch,
                    'epochs_executed': int(model.history.epoch.max()),
                    'selection_criterion': 'minimum validation binary cross-entropy, no oracle/test access',
                    'test_bce': model._loss(*model._encode(learning_view(split.test)))})
            for name in config['methods']:
                progress(f'[seed {seed}] evaluating {name}')
                method_dir = folder/name
                method_dir.mkdir()
                policy = _policy(name, tables, generation, split.train, model)
                catalog, rankings = evaluate_catalog(tables, policy, config['evaluation']['catalog_k'], seed)
                choice = evaluate_logged_choices(split.test, policy, config['evaluation']['choice_k'], seed)
                response, logs = evaluate_response(tables, generation, policy,
                                                  config['evaluation']['response_steps'], seed)
                _csv(catalog, method_dir/'catalog_per_user.csv')
                _csv(rankings, method_dir/'catalog_recommendations.csv')
                _csv(choice, method_dir/'heldout_positive_slates.csv')
                _csv(response, method_dir/'response_per_slate.csv')
                # Includes diagnostic truth. Learning input must pass through learning_view.
                _csv(logs['copy_decisions'], method_dir/'fresh_decisions.diagnostic.csv')
                for frame, store in ((catalog, all_catalog), (choice, all_choices), (response, all_responses)):
                    frame = frame.copy()
                    frame['seed'] = seed
                    store.append(frame)
                cat_summary = macro_summary(catalog, CATALOG_METRICS, ['method','k'])
                all_metrics += _summary_rows(cat_summary, 'catalog', seed, CATALOG_METRICS)
                for k in config['evaluation']['catalog_k']:
                    covered = rankings[rankings['rank'] <= k].strategy_id.nunique()
                    all_metrics.append({'seed': seed, 'protocol': 'catalog', 'method': name,
                                        'k': k, 'metric': 'catalog_coverage_at_k',
                                        'value': covered/len(tables['strategies']), 'n_users': len(tables['users'])})
                choice_summary = macro_summary(choice, BEHAVIOUR_METRICS, ['method','k'])
                if choice_summary.empty:
                    for k in config['evaluation']['choice_k']:
                        for metric in BEHAVIOUR_METRICS:
                            all_metrics.append({'seed': seed, 'protocol': 'heldout_choice',
                                                'method': name, 'k': k, 'metric': metric,
                                                'value': np.nan, 'n_users': 0})
                else:
                    all_metrics += _summary_rows(choice_summary, 'heldout_choice', seed, BEHAVIOUR_METRICS)
                response_summary = macro_summary(response, RESPONSE_METRICS, ['method'])
                all_metrics += _summary_rows(response_summary, 'fresh_response', seed, RESPONSE_METRICS)
            data_records.append({'seed': seed, 'path': str(data_dir.relative_to(root)),
                                 'dataset_sha256': data_manifest['dataset_sha256'],
                                 'generation_config_sha256': data_manifest['generation_config_sha256']})
        metrics = pd.DataFrame(all_metrics)
        aggregate = metrics.groupby(['protocol','method','k','metric'], as_index=False).agg(
            mean=('value','mean'), std=('value','std'), n_seeds=('value','count'))
        aggregate['std'] = aggregate['std'].fillna(0.)
        _csv(metrics, root/'metrics_per_seed.csv')
        _csv(aggregate, root/'metrics_aggregate.csv')
        catalog = pd.concat(all_catalog, ignore_index=True)
        _csv(catalog, root/'catalog_per_user.csv')
        _csv(pd.concat(all_choices, ignore_index=True), root/'heldout_positive_slates.csv')
        _csv(pd.concat(all_responses, ignore_index=True), root/'response_per_slate.csv')
        training = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
        if not training.empty:
            _csv(training, root/'mf_training.csv')
        from copytrading_sim.experiments.reporting import create_report
        create_report(root, config, aggregate, catalog, training, split_records)
        manifest = {'experiment_format_version': 1, 'protocol_version': 'fixed-history-v1',
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'source': source_provenance(), 'environment': environment_provenance(),
                    'datasets': data_records, 'seed_aggregation': 'macro over users, then mean/std over world seeds',
                    'notes': ['No federated training yet', 'No feedback from fresh evaluation into fitting',
                              'Compatibility oracle only bounds the compatibility objective',
                              'Synthetic proxy features; no financial outcome or external validation'],
                    'files': {p.relative_to(root).as_posix(): sha256_file(p)
                              for p in sorted(root.rglob('*')) if p.is_file()}}
        _json(root/'manifest.json', manifest)
        complete = True
        progress(f'Experiment saved: {root}')
        return root
    finally:
        if not complete:
            # Only delete this call's exclusive new directory, never an older run.
            shutil.rmtree(root, ignore_errors=True)
