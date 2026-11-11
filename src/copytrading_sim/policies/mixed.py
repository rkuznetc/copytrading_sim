"""Existing synthetic logging rule, now usable via the common policy API."""
from __future__ import annotations
import numpy as np
import pandas as pd
from copytrading_sim.baselines.oracle_proxy import OracleCompatibilityScorer
from copytrading_sim.baselines.popularity_policy import PopularityScorer
from copytrading_sim.policies.base import Selection, select_scores


class MixedLoggingPolicy:
    """Privileged collector using simulator compatibility; not a deployable model."""
    def __init__(self, users: pd.DataFrame, strategies: pd.DataFrame,
                 compatibility: pd.DataFrame, params: dict):
        self.name = params['policy_name']
        self.params = dict(params)
        self.oracle = OracleCompatibilityScorer(compatibility)
        self.popularity = PopularityScorer.from_public_catalog(strategies)
        self.social = dict(zip(users['user_id'], users['social_sensitivity']))

    def select(self, user_id: int, strategy_ids: np.ndarray, k: int,
               rng: np.random.Generator) -> Selection:
        p = self.params
        popularity = self.popularity.score(user_id, strategy_ids)
        scores = np.clip(
            p['compatibility_weight'] * self.oracle.score(user_id, strategy_ids)
            + p['popularity_weight'] * popularity
            + p['social_popularity_weight'] * self.social[user_id] * popularity
            + rng.normal(0, p['noise_std'], len(strategy_ids)), 0, 1,
        )
        selection = select_scores(scores, strategy_ids, k, rng,
                                  'plackett_luce', p['temperature'])
        return Selection(selection.indices, selection.scores,
                         selection.conditional_probabilities,
                         'plackett_luce_conditional_on_score_noise')
