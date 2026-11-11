"""Uniform sampling of an ordered slate without replacement."""
from dataclasses import dataclass
import numpy as np
from copytrading_sim.policies.base import Selection, validate_candidates


@dataclass
class RandomPolicy:
    name: str = 'random'

    def select(self, user_id: int, strategy_ids: np.ndarray, k: int,
               rng: np.random.Generator) -> Selection:
        ids = validate_candidates(strategy_ids, k)
        indices = rng.permutation(len(ids))[:k]
        return Selection(indices, np.zeros(k),
                         1.0 / (len(ids) - np.arange(k)), 'uniform_without_replacement')
