import numpy as np
import pandas as pd
import pytest
from copytrading_sim.evaluation.metrics import (
    linear_ndcg, evaluate_catalog, evaluate_logged_choices, evaluate_response,
    macro_summary, BEHAVIOUR_METRICS,
)
from copytrading_sim.baselines.random_policy import RandomPolicy
from copytrading_sim.baselines.oracle_proxy import OracleCompatibilityScorer
from copytrading_sim.baselines.popularity_policy import PopularityScorer
from copytrading_sim.policies.base import ScorePolicy
from copytrading_sim.simulator.pipeline import run_pipeline
from copytrading_sim.datasets.splits import temporal_split


def test_ndcg_known_values():
    assert linear_ndcg(np.array([1.,.5,0]), np.array([1.,.5,0]), 2) == 1
    assert linear_ndcg(np.array([0.,1]), np.array([1.,0]), 2) == pytest.approx(1/np.log2(3))
    assert linear_ndcg(np.zeros(3), np.zeros(3), 2) == 0


def test_oracle_has_zero_suitability_regret(tiny_config):
    tables = run_pipeline(tiny_config)
    oracle = ScorePolicy(OracleCompatibilityScorer(tables['compatibility']), 'oracle')
    metrics, ranking = evaluate_catalog(tables, oracle, [1, 2], seed=12)
    np.testing.assert_allclose(metrics.compatibility_ndcg_at_k, 1)
    np.testing.assert_allclose(metrics.oracle_regret_at_k, 0)
    assert len(ranking) == 8


def test_choices_ignore_no_copy_slates_and_reject_trivial_k():
    test = pd.DataFrame({'slate_id': ['a']*3+['b']*3,
                        'user_id': [1]*6, 'timestamp_step': [1]*3+[2]*3,
                        'strategy_id': [1, 2, 3]*2, 'copied': [0, 1, 0, 0, 0, 0]})
    policy = ScorePolicy(PopularityScorer({1: 0, 2: 3, 3: 1}), 'test')
    out = evaluate_logged_choices(test, policy, [1, 2], 42)
    assert len(out) == 2 and out.hit_rate_at_k.eq(1).all()
    with pytest.raises(ValueError):
        evaluate_logged_choices(test, policy, [3], 42)
    test['copied'] = 0
    assert evaluate_logged_choices(test, policy, [1], 42).empty


def test_fresh_response_does_not_mutate_original_logs(tiny_config):
    tables = run_pipeline(tiny_config)
    original = tables['copy_decisions'].copy(deep=True)
    metrics, history = evaluate_response(tables, tiny_config, RandomPolicy(), 2, seed=1)
    assert len(metrics) == 8
    assert history['exposures'].timestamp_step.min() == 3
    assert metrics.expected_copy_rate.between(0,1).all()
    assert metrics.expected_compatibility_yield.between(0,1).all()
    pd.testing.assert_frame_equal(original, tables['copy_decisions'])
    second, _ = evaluate_response(tables, tiny_config, RandomPolicy(), 2, seed=1)
    pd.testing.assert_frame_equal(metrics, second, check_exact=True)


def test_macro_not_event_weighted():
    f = pd.DataFrame({'method':['a']*4, 'k':[1]*4, 'user_id':[1,2,2,2],
                      'hit_rate_at_k':[1.,0.,0.,0.], 'ndcg_at_k':[1.,0.,0.,0.]})
    out = macro_summary(f, BEHAVIOUR_METRICS, ['method','k'])
    assert out.hit_rate_at_k.iloc[0] == .5
