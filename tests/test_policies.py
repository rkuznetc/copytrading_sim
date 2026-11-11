import numpy as np
import pandas as pd
import pytest

from copytrading_sim.baselines.random_policy import RandomPolicy
from copytrading_sim.baselines.popularity_policy import PopularityScorer
from copytrading_sim.baselines.oracle_proxy import OracleCompatibilityScorer
from copytrading_sim.policies.base import ScorePolicy, select_scores, Selection
from copytrading_sim.policies.mixed import MixedLoggingPolicy
from copytrading_sim.config import resolve_config
from copytrading_sim.simulator.pipeline import run_pipeline
from copytrading_sim.simulator.environment import SimulationEnvironment
from copytrading_sim.simulator.exposure_policy import (
    _build_candidate_table, _sample_slate_plackett_luce, sample_policy_slate,
)


def test_topk_ties_and_conditional_probabilities():
    s = select_scores(np.array([.4, .8, .8]), np.array([7, 9, 2]), 3,
                      np.random.default_rng(1))
    assert s.indices.tolist() == [2, 1, 0]
    assert s.conditional_probabilities.tolist() == [1., 1., 1.]


def test_uniform_and_plackett_luce():
    ids = np.array([3, 8, 10])
    selected = RandomPolicy().select(0, ids, 3, np.random.default_rng(1))
    assert len(set(selected.indices)) == 3
    np.testing.assert_allclose(selected.conditional_probabilities, [1/3, 1/2, 1])
    s = select_scores(np.zeros(3), ids, 3, np.random.default_rng(3), 'plackett_luce')
    np.testing.assert_allclose(s.conditional_probabilities, [1/3, 1/2, 1])


@pytest.mark.parametrize('scores,k,rule', [([float('nan'), 0], 1, 'top_k'),
    ([1, 2], 3, 'top_k'), ([1, 2], 0, 'top_k'), ([1, 2], 1, 'typo')])
def test_invalid_selection(scores, k, rule):
    with pytest.raises(ValueError):
        select_scores(np.array(scores), np.array([1, 2]), k, np.random.default_rng(), rule)


def test_popularity_training_only():
    train = pd.DataFrame({'strategy_id': [1, 1, 2], 'copied': [1, 0, 1]})
    scorer = PopularityScorer.from_training(train, np.array([0, 1, 2]))
    np.testing.assert_array_equal(scorer.score(99, np.array([2, 0, 1])), [1, 0, 1])
    with pytest.raises(ValueError):
        scorer.score(0, np.array([7]))


def test_pipeline_policy_override(tiny_config):
    tables = run_pipeline(tiny_config, policy=RandomPolicy())
    assert tables['exposures']['policy_name'].eq('random').all()
    assert tables['exposures']['selection_rule'].eq('uniform_without_replacement').all()


def test_model_policy_accepts_unbounded_logits(tiny_config):
    scorer = PopularityScorer({i: float(i * 10 - 20) for i in range(6)})
    t = run_pipeline(tiny_config, policy=ScorePolicy(scorer))
    assert t['exposures']['strategy_id'].tolist() == [5, 4] * 12


def test_mixed_matches_existing_scores_and_selection(tiny_config):
    t = run_pipeline(tiny_config)
    config = resolve_config(tiny_config)
    c = _build_candidate_table(t['users'], t['strategies'], t['compatibility'])
    c = c[c.user_id == 0].reset_index(drop=True)
    p = MixedLoggingPolicy(t['users'], t['strategies'], t['compatibility'],
                           config['exposure_policy'])
    old = _sample_slate_plackett_luce(c, 2, 's', 0, config['exposure_policy'],
                                    np.random.default_rng(14))
    new = sample_policy_slate(c, p, 2, 's', 0, np.random.default_rng(14))
    assert new.strategy_id.tolist() == old.strategy_id.tolist()
    np.testing.assert_allclose(new.position_probability, old.position_probability, atol=1e-15)


def test_environment_is_resettable_and_supports_oracle(tiny_config):
    tables = run_pipeline(tiny_config)
    policy = ScorePolicy(OracleCompatibilityScorer(tables['compatibility']), name='oracle')
    a = SimulationEnvironment.from_tables(tables, tiny_config, seed=99)
    b = SimulationEnvironment.from_tables(tables, tiny_config, seed=99)
    x, y = a.step(0, policy), b.step(0, policy)
    pd.testing.assert_frame_equal(x.copy_decisions, y.copy_decisions)
    assert x.copy_decisions.copied.sum() <= 1
    a.step(0, policy, model_version='round1')
    h = a.export_history()
    assert h['exposures'].exposure_id.is_unique
    assert h['exposures'].timestamp_step.max() == 1
    # Returned data cannot change internal history.
    x.exposures.loc[:, 'was_shown'] = 0
    assert a.export_history()['exposures'].was_shown.eq(1).all()


def test_bad_policy_rejected(tiny_config):
    class BadPolicy:
        name = 'broken'
        def select(self, user_id, strategy_ids, k, rng):
            return Selection(np.array([0, 0]), np.ones(2), np.ones(2), 'bad')
    env = SimulationEnvironment.from_tables(run_pipeline(tiny_config), tiny_config)
    with pytest.raises(ValueError, match='indices'):
        env.step(0, BadPolicy())
