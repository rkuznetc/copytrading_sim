"""Two distinct popularity baselines: public prior and training-only counts."""
from __future__ import annotations
import numpy as np
import pandas as pd


class PopularityScorer:
    def __init__(self, values: dict[int, float]):
        self.values = {int(k): float(v) for k, v in values.items()}
        if not self.values or not np.isfinite(list(self.values.values())).all():
            raise ValueError('Popularity values must be non-empty and finite')

    @classmethod
    def from_public_catalog(cls, strategies: pd.DataFrame) -> 'PopularityScorer':
        if strategies['strategy_id'].duplicated().any():
            raise ValueError('Duplicate strategy IDs')
        return cls(dict(zip(strategies['strategy_id'], strategies['popularity_score'])))

    @classmethod
    def from_training(cls, train: pd.DataFrame, strategy_ids: np.ndarray) -> 'PopularityScorer':
        # All catalogue items included, including those with no training copies.
        counts = train.groupby('strategy_id')['copied'].sum()
        return cls({int(i): float(counts.get(i, 0)) for i in strategy_ids})

    def score(self, user_id: int, strategy_ids: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([self.values[int(i)] for i in strategy_ids], dtype=float)
        except KeyError as exc:
            raise ValueError(f'Unknown strategy: {exc.args[0]}') from exc
