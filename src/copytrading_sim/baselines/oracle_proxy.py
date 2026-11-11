"""Privileged diagnostic reference. Not an ordinary trainable recommender."""
from __future__ import annotations
import numpy as np
import pandas as pd


class OracleCompatibilityScorer:
    def __init__(self, compatibility: pd.DataFrame):
        if compatibility.duplicated(['user_id', 'strategy_id']).any():
            raise ValueError('Duplicate user-strategy pair')
        self._values = {
            (int(u), int(i)): float(c)
            for u, i, c in compatibility[
                ['user_id', 'strategy_id', 'compatibility_score']
            ].itertuples(index=False, name=None)
        }
        if not self._values or not np.isfinite(list(self._values.values())).all():
            raise ValueError('Compatibility values must be non-empty and finite')

    def score(self, user_id: int, strategy_ids: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([self._values[(int(user_id), int(i))] for i in strategy_ids])
        except KeyError as exc:
            raise ValueError(f'Unknown oracle pair: {exc.args[0]}') from exc
