"""Separate simulator suitability, held-out behaviour, and fresh simulated response.

None of these is a claim about financial returns or off-policy value estimation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from copytrading_sim.baselines.oracle_proxy import OracleCompatibilityScorer
from copytrading_sim.policies.base import SlatePolicy
from copytrading_sim.simulator.environment import SimulationEnvironment, request_rng

CATALOG_METRICS = ('compatibility_at_k', 'compatibility_ndcg_at_k',
                   'oracle_regret_at_k', 'absolute_risk_gap_at_k', 'popularity_at_k')
BEHAVIOUR_METRICS = ('hit_rate_at_k', 'ndcg_at_k')
RESPONSE_METRICS = ('expected_copy_rate', 'sampled_copy_rate',
                    'expected_compatibility_yield', 'shown_compatibility')


def _ks(values, maximum):
    if (not values or any(isinstance(k, bool) or not isinstance(k, (int, np.integer))
                         or k < 1 or k > maximum for k in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'K values must be unique integers between 1 and {maximum}')
    return sorted(values)


def linear_ndcg(relevance: np.ndarray, ideal_relevance: np.ndarray, k: int) -> float:
    """Linear nonnegative gain, no arbitrary exponential gain on [0,1] proxies."""
    r, ideal = np.asarray(relevance), np.asarray(ideal_relevance)
    discount = 1 / np.log2(np.arange(2, k + 2))
    dcg, idcg = np.dot(r[:k], discount), np.dot(np.sort(ideal)[::-1][:k], discount)
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_catalog(tables: dict[str, pd.DataFrame], policy: SlatePolicy,
                     ks: list[int], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the ENTIRE shared catalogue, allowing already seen/copied strategies.

    Returns per-user metrics and selected ranking rows. For stochastic policies
    this is one sampled ranking per user and seed, not exact expected performance.
    """
    users = tables['users'].sort_values('user_id')
    strategies = tables['strategies'].sort_values('strategy_id').set_index('strategy_id')
    ids = strategies.index.to_numpy(dtype=np.int64)
    ks = _ks(ks, len(ids))
    truth = OracleCompatibilityScorer(tables['compatibility'])
    rows, rankings = [], []
    risk = strategies['risk_score'].to_numpy(dtype=float)
    popularity = strategies['popularity_score'].to_numpy(dtype=float)
    for user in users.itertuples(index=False):
        choice = policy.select(int(user.user_id), ids, max(ks),
                               request_rng(seed, 201, int(user.user_id), 0))
        order = choice.indices
        relevance = truth.score(int(user.user_id), ids)
        for rank, idx in enumerate(order, 1):
            rankings.append({'method': policy.name, 'user_id': int(user.user_id),
                             'strategy_id': int(ids[idx]), 'rank': rank,
                             'compatibility': float(relevance[idx])})
        for k in ks:
            selected = order[:k]
            mean = float(np.mean(relevance[selected]))
            ideal_mean = float(np.sort(relevance)[-k:].mean())
            rows.append({'method': policy.name, 'user_id': int(user.user_id),
                         'user_type': str(user.user_type), 'k': k,
                         'compatibility_at_k': mean,
                         'compatibility_ndcg_at_k': linear_ndcg(relevance[order], relevance, k),
                         'oracle_regret_at_k': max(0., ideal_mean - mean),
                         'absolute_risk_gap_at_k': float(np.abs(risk[selected]-user.risk_tolerance).mean()),
                         'popularity_at_k': float(popularity[selected].mean())})
    return pd.DataFrame(rows), pd.DataFrame(rankings)


def evaluate_logged_choices(test: pd.DataFrame, policy: SlatePolicy, ks: list[int],
                            seed: int) -> pd.DataFrame:
    """Rank only the ACTUALLY SHOWN test slate, conditional on a copy occurring.

    Unshown items are not labeled negative. No-copy slates are excluded from these
    ranking metrics and their number is reported in experiment split statistics.
    This is prediction on held-out behaviour, NOT counterfactual policy value.
    """
    if test.empty:
        raise ValueError('Test data must not be empty')
    sizes = test.groupby('slate_id').size()
    ks = _ks(ks, int(sizes.min()) - 1)  # HR@slate_size is trivially one
    rows = []
    for slate_id, slate in test.groupby('slate_id', sort=True):
        positives = slate[slate['copied'].eq(1)]
        if len(positives) > 1:
            raise ValueError('This evaluator expects at most one copy per slate')
        if positives.empty:
            continue
        slate = slate.sort_values('strategy_id')
        ids = slate.strategy_id.to_numpy(dtype=np.int64)
        user, step = int(slate.user_id.iloc[0]), int(slate.timestamp_step.iloc[0])
        ranking = policy.select(user, ids, max(ks), request_rng(seed, 202, user, step))
        ranked_ids = ids[ranking.indices]
        chosen = int(positives.strategy_id.iloc[0])
        locations = np.flatnonzero(ranked_ids == chosen)
        rank = int(locations[0]+1) if len(locations) else len(ids)+1
        for k in ks:
            hit = float(rank <= k)
            rows.append({'method': policy.name, 'user_id': user, 'slate_id': slate_id,
                         'k': k, 'hit_rate_at_k': hit,
                         'ndcg_at_k': float(1/np.log2(rank+1)) if hit else 0.})
    return pd.DataFrame(rows, columns=['method', 'user_id', 'slate_id', 'k', *BEHAVIOUR_METRICS])


def evaluate_response(tables: dict[str, pd.DataFrame], config: dict, policy: SlatePolicy,
                      steps: int, seed: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Frozen model, fresh shows and responses in identical reset worlds per method.

    New interactions are NOT fed back into training. Per-request noise is paired
    across methods, but is not a claim of identical item-level latent shocks.
    Expected rates average the conditional probabilities over simulated noise.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('response steps must be a positive integer')
    env = SimulationEnvironment.from_tables(
        tables, config, seed=seed, start_step=config['simulation']['exposures_per_user'],
    )
    for step in range(steps):
        for user in sorted(tables['users'].user_id):
            env.step(int(user), policy, model_version=policy.name)
    histories = env.export_history()
    decisions = histories['copy_decisions']
    rows = []
    for slate_id, group in decisions.groupby('slate_id', sort=False):
        rows.append({'method': policy.name, 'user_id': int(group.user_id.iloc[0]),
                     'slate_id': slate_id, 'timestamp_step': int(group.timestamp_step.iloc[0]),
                     'expected_copy_rate': float(group.copy_probability.sum()),
                     'sampled_copy_rate': int(group.copied.sum()),
                     'expected_compatibility_yield': float(
                         (group.copy_probability * group.compatibility_score).sum()),
                     'shown_compatibility': float(group.compatibility_score.mean())})
    return pd.DataFrame(rows), histories


def macro_summary(frame: pd.DataFrame, metrics: tuple[str, ...],
                  groups: list[str]) -> pd.DataFrame:
    """Average per user first, then per method/K; never count rows as independent seeds."""
    if frame.empty:
        return pd.DataFrame(columns=[*groups, *metrics, 'n_users'])
    per_user = frame.groupby([*groups, 'user_id'], as_index=False)[list(metrics)].mean()
    result = per_user.groupby(groups, as_index=False)[list(metrics)].mean()
    counts = per_user.groupby(groups, as_index=False).size().rename(columns={'size': 'n_users'})
    return result.merge(counts, on=groups, validate='one_to_one')
