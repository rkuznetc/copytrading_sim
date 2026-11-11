import numpy as np
import pandas as pd
import pytest
from copytrading_sim.models.mf import LogisticMF, MFConfig
from copytrading_sim.datasets.splits import learning_view, temporal_split
from copytrading_sim.simulator.pipeline import run_pipeline


def synthetic_preferences():
    rows = [{'user_id': u, 'strategy_id': i, 'copied': int(u % 2 == i % 2)}
            for _ in range(10) for u in [10, 11, 12, 13] for i in [20, 21, 22, 23]]
    return pd.DataFrame(rows)


def test_mf_gradient_finite_difference():
    model = LogisticMF(np.array([10, 11]), np.array([20, 21]),
                       MFConfig(factors=2, regularization=.03))
    u, i, y = model._encode(pd.DataFrame({'user_id': [10, 10, 11],
        'strategy_id': [20, 20, 21], 'copied': [1, 0, 1]}))
    _, grads = model._loss_and_gradients(u, i, y)
    for name, values in model.params.items():
        for index in np.ndindex(values.shape):
            saved = values[index]
            values[index] = saved + 1e-6
            right = model._loss(u, i, y, regularized=True)
            values[index] = saved - 1e-6
            left = model._loss(u, i, y, regularized=True)
            values[index] = saved
            assert grads[name][index] == pytest.approx((right-left)/2e-6, abs=1e-7)


def test_mf_learns_and_roundtrips(tmp_path):
    data = synthetic_preferences()
    config = MFConfig(factors=4, epochs=80, batch_size=64, patience=80)
    model = LogisticMF(np.array([10, 11, 12, 13]), np.array([20, 21, 22, 23]), config)
    model.fit(data, data)  # deliberate separable optimisation unit test, not evaluation
    assert model.history.iloc[-1].train_bce < .3
    assert model.score(10, np.array([20, 21]))[0] > model.score(10, np.array([20, 21]))[1]
    model.save(tmp_path/'mf.npz')
    loaded = LogisticMF.load(tmp_path/'mf.npz')
    np.testing.assert_array_equal(loaded.score(10, np.array([21, 20])), model.score(10, np.array([21, 20])))
    with pytest.raises(FileExistsError):
        model.save(tmp_path/'mf.npz')


def test_no_truth_in_learning_view_and_fit(tiny_config):
    t = run_pipeline(tiny_config)
    view = learning_view(t['copy_decisions'])
    assert 'compatibility_score' not in view and 'copy_probability' not in view
    model = LogisticMF(t['users'].user_id.to_numpy(), t['strategies'].strategy_id.to_numpy())
    with pytest.raises(ValueError, match='forbidden'):
        model.fit(t['copy_decisions'], view)


def test_split_preserves_slates(tiny_config):
    tiny_config['simulation']['exposures_per_user'] = 10
    t = run_pipeline(tiny_config)
    split = temporal_split(t['copy_decisions'])
    assert split.train.timestamp_step.max() < split.validation.timestamp_step.min()
    assert split.validation.timestamp_step.max() < split.test.timestamp_step.min()
    assert split.train.slate_id.nunique() == 4 * 6
    assert len(split.manifest) == 40
    assert not set(split.train.slate_id) & set(split.test.slate_id)


def test_too_short_split(tiny_config):
    with pytest.raises(ValueError, match='Too few'):
        temporal_split(run_pipeline(tiny_config)['copy_decisions'])


def test_mf_reproducibility_and_best_checkpoint():
    data = synthetic_preferences()
    cfg = MFConfig(epochs=6)
    a = LogisticMF(np.array([10,11,12,13]), np.array([20,21,22,23]), cfg, seed=1).fit(data, data)
    b = LogisticMF(np.array([10,11,12,13]), np.array([20,21,22,23]), cfg, seed=1).fit(data, data)
    pd.testing.assert_frame_equal(a.history, b.history, check_exact=True)
    assert a._loss(*a._encode(data)) == pytest.approx(a.history.iloc[a.best_epoch].validation_bce)


@pytest.mark.parametrize('kwargs', [{'epochs': 0}, {'learning_rate': 0}, {'factors': True},
                                   {'regularization': float('nan')}])
def test_bad_mf_config(kwargs):
    with pytest.raises(ValueError):
        MFConfig(**kwargs)
