"""Time-step split preserving whole recommendation requests and allowed columns."""
from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd

LEARNING_COLUMNS = ('user_id', 'strategy_id', 'slate_id', 'timestamp_step', 'position', 'copied')


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: pd.DataFrame


def learning_view(decisions: pd.DataFrame) -> pd.DataFrame:
    """Do not pass oracle compatibility, decision probabilities or types to a learner."""
    missing = set(LEARNING_COLUMNS) - set(decisions)
    if missing:
        raise ValueError(f'Missing learning columns: {sorted(missing)}')
    return decisions.loc[:, list(LEARNING_COLUMNS)].copy()


def temporal_split(decisions: pd.DataFrame, train_fraction: float = .6,
                   validation_fraction: float = .2) -> TemporalSplit:
    if not (math.isfinite(train_fraction) and math.isfinite(validation_fraction)
            and train_fraction > 0 and validation_fraction > 0
            and train_fraction + validation_fraction < 1):
        raise ValueError('Positive train/validation fractions must sum to less than one')
    learning_view(decisions)
    if decisions.empty or decisions.isna().any().any():
        raise ValueError('Expected non-empty complete decisions')
    groups = decisions.groupby('slate_id', sort=False)
    if (not groups['timestamp_step'].nunique().eq(1).all()
            or not groups['user_id'].nunique().eq(1).all()):
        raise ValueError('Each slate must belong to one user and one time step')
    steps = sorted(decisions['timestamp_step'].unique())
    n_train = int(len(steps) * train_fraction)
    n_val = int(len(steps) * validation_fraction)
    if min(n_train, n_val, len(steps) - n_train - n_val) < 1:
        raise ValueError('Too few time steps for non-empty train, validation and test')
    parts = {}
    step_groups = {'train': steps[:n_train], 'validation': steps[n_train:n_train+n_val],
                   'test': steps[n_train+n_val:]}
    manifests = []
    for label, subset in step_groups.items():
        part = decisions[decisions['timestamp_step'].isin(subset)].copy().reset_index(drop=True)
        parts[label] = part
        manifest = part[['slate_id', 'user_id', 'timestamp_step']].drop_duplicates()
        manifest['split'] = label
        manifests.append(manifest)
    return TemporalSplit(**parts, manifest=pd.concat(manifests, ignore_index=True))
