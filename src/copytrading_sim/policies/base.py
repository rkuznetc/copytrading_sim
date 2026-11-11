"""Model-independent interface: user + catalogue -> ordered slate.

Policies never receive simulator truth implicitly. A learned scorer only needs
IDs; the oracle is an explicitly privileged diagnostic implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Scorer(Protocol):
    def score(self, user_id: int, strategy_ids: np.ndarray) -> np.ndarray:
        """Return finite real scores in exactly the requested order."""
        ...


@dataclass(frozen=True)
class Selection:
    indices: np.ndarray
    scores: np.ndarray
    conditional_probabilities: np.ndarray
    selection_rule: str


class SlatePolicy(Protocol):
    name: str

    def select(self, user_id: int, strategy_ids: np.ndarray, k: int,
               rng: np.random.Generator) -> Selection:
        ...


def validate_candidates(strategy_ids: np.ndarray, k: int) -> np.ndarray:
    ids = np.asarray(strategy_ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
        raise ValueError('Candidate IDs must be a unique one-dimensional integer array')
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(ids):
        raise ValueError('slate size must be an integer in [1, number of candidates]')
    return ids


def select_scores(scores: np.ndarray, strategy_ids: np.ndarray, k: int,
                  rng: np.random.Generator, rule: str = 'top_k',
                  temperature: float = 0.25) -> Selection:
    """Deterministic top-k or exact sequential Plackett--Luce sampling.

    Probabilities are conditional on the previous positions, NOT item inclusion
    probabilities. For top-k the selected ordered slate has probability one.
    """
    ids = validate_candidates(strategy_ids, k)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError('Scorer must return one finite score per candidate')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if rule == 'top_k':
        indices = np.lexsort((ids, -scores))[:k]  # explicit, reproducible tie rule
        return Selection(indices, scores[indices], np.ones(k), rule)
    if rule != 'plackett_luce':
        raise ValueError(f'Unknown selection rule: {rule}')
    remaining = np.arange(len(ids))
    selected, conditional = [], []
    for _ in range(k):
        scaled = scores[remaining] / temperature
        scaled = scaled - np.max(scaled)
        masses = np.exp(scaled)
        probabilities = masses / masses.sum()
        local_index = int(rng.choice(len(remaining), p=probabilities))
        selected.append(int(remaining[local_index]))
        conditional.append(float(probabilities[local_index]))
        remaining = np.delete(remaining, local_index)
    indices = np.asarray(selected, dtype=np.int64)
    return Selection(indices, scores[indices], np.asarray(conditional), rule)


@dataclass
class ScorePolicy:
    scorer: Scorer
    name: str = 'model_top_k'
    selection_rule: str = 'top_k'
    temperature: float = 0.25

    def select(self, user_id: int, strategy_ids: np.ndarray, k: int,
               rng: np.random.Generator) -> Selection:
        return select_scores(self.scorer.score(user_id, strategy_ids), strategy_ids,
                             k, rng, self.selection_rule, self.temperature)
