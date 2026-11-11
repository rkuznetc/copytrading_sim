"""Logistic matrix factorisation on shown items, with NumPy Adam optimisation.

Objective: average binary cross-entropy of observed copied/not-copied labels +
L2 regularisation. This predicts a marginal response under the training exposure
process, not a slate-conditional choice probability. Unshown items are not
invented negatives. No simulator truth is accepted by the training interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from copytrading_sim.datasets.splits import LEARNING_COLUMNS


@dataclass(frozen=True)
class MFConfig:
    factors: int = 12
    epochs: int = 60
    batch_size: int = 512
    learning_rate: float = .015
    regularization: float = .0001
    patience: int = 10
    min_delta: float = .00001

    def __post_init__(self):
        for name in ('factors', 'epochs', 'batch_size', 'patience'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'mf.{name} must be a positive integer')
        for name in ('learning_rate', 'regularization', 'min_delta'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not np.isfinite(value) or value < 0
                    or (name == 'learning_rate' and value == 0)):
                raise ValueError(f'mf.{name} must be finite and nonnegative (learning_rate > 0)')


def _sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


class LogisticMF:
    """ID-only model with explicit local user and shared item parameter arrays.

    There is no federated trainer yet. The score(user_id, ids) interface can also
    be used by a future client-side federated model. Catalogue is known in advance
    in this warm-start experiment; unknown IDs raise instead of silently guessing.
    """
    def __init__(self, user_ids: np.ndarray, strategy_ids: np.ndarray,
                 config: MFConfig | None = None, seed: int = 42):
        self.config = config or MFConfig()
        self.user_ids = self._ids(user_ids, 'user')
        self.strategy_ids = self._ids(strategy_ids, 'strategy')
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError('Model seed must be a non-negative integer')
        self.seed = int(seed)
        self._users = {int(v): k for k, v in enumerate(self.user_ids)}
        self._items = {int(v): k for k, v in enumerate(self.strategy_ids)}
        rng = np.random.default_rng(self.seed)
        k = self.config.factors
        self.params = {
            'P': rng.normal(0, .05, (len(self.user_ids), k)),
            'Q': rng.normal(0, .05, (len(self.strategy_ids), k)),
            'bu': np.zeros(len(self.user_ids)),
            'bi': np.zeros(len(self.strategy_ids)),
            'b0': np.zeros(1),
        }
        self.history = pd.DataFrame()
        self.best_epoch = 0
        self.is_fitted = False

    @staticmethod
    def _ids(values, name):
        ids = np.asarray(values)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) == 0
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0)):
            raise ValueError(f'{name} IDs must be a non-empty unique nonnegative integer array')
        return ids.astype(np.int64, copy=True)

    def _encode(self, data: pd.DataFrame):
        if not set(data).issubset(LEARNING_COLUMNS):
            raise ValueError('Training data contains forbidden/non-observed columns; use learning_view')
        if not {'user_id', 'strategy_id', 'copied'}.issubset(data) or data.empty:
            raise ValueError('Non-empty observed user_id, strategy_id, copied data required')
        if data[['user_id', 'strategy_id', 'copied']].isna().any().any():
            raise ValueError('Training data contains missing values')
        if not data['copied'].isin([0, 1]).all():
            raise ValueError('copied must be binary')
        if data.user_id.dtype.kind not in 'iu' or data.strategy_id.dtype.kind not in 'iu':
            raise ValueError('Training IDs must be integers')
        try:
            u = np.asarray([self._users[int(v)] for v in data.user_id], dtype=np.int64)
            i = np.asarray([self._items[int(v)] for v in data.strategy_id], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f'Unknown training ID: {exc.args[0]}') from exc
        return u, i, data.copied.to_numpy(dtype=float)

    def _logits(self, u, i):
        p = self.params
        return np.sum(p['P'][u] * p['Q'][i], axis=1) + p['bu'][u] + p['bi'][i] + p['b0'][0]

    def _loss(self, u, i, y, regularized=False):
        logits = self._logits(u, i)
        loss = float(np.mean(np.logaddexp(0., logits) - y * logits))
        if regularized:
            loss += .5 * self.config.regularization * sum(
                float(np.sum(value**2)) for name, value in self.params.items() if name != 'b0')
        return loss

    def _loss_and_gradients(self, u, i, y):
        """Analytic batch gradients (tested against finite differences)."""
        error = (_sigmoid(self._logits(u, i)) - y) / len(y)
        p = self.params
        grads = {name: np.zeros_like(value) for name, value in p.items()}
        np.add.at(grads['P'], u, error[:, None] * p['Q'][i])
        np.add.at(grads['Q'], i, error[:, None] * p['P'][u])
        np.add.at(grads['bu'], u, error)
        np.add.at(grads['bi'], i, error)
        grads['b0'][0] = error.sum()
        for name in grads:
            if name != 'b0':
                grads[name] += self.config.regularization * p[name]
        return self._loss(u, i, y, regularized=True), grads

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> 'LogisticMF':
        if self.is_fitted:
            raise ValueError('Create a fresh model for a new fit; incremental training is not implemented')
        u, i, y = self._encode(train)
        vu, vi, vy = self._encode(validation)
        if np.unique(y).size != 2:
            raise ValueError('Training requires both copied and not-copied examples')
        rate = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.params['b0'][0] = np.log(rate / (1-rate))
        moments = {name: np.zeros_like(v) for name, v in self.params.items()}
        squares = {name: np.zeros_like(v) for name, v in self.params.items()}
        shuffle = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        history = []
        best_loss, wait, update = float('inf'), 0, 0
        best = {name: v.copy() for name, v in self.params.items()}
        for epoch in range(self.config.epochs + 1):
            if epoch:
                order = shuffle.permutation(len(y))
                for start in range(0, len(y), self.config.batch_size):
                    idx = order[start:start+self.config.batch_size]
                    _, grads = self._loss_and_gradients(u[idx], i[idx], y[idx])
                    update += 1
                    for name, grad in grads.items():
                        moments[name] = .9 * moments[name] + .1 * grad
                        squares[name] = .999 * squares[name] + .001 * grad**2
                        corrected_m = moments[name] / (1 - .9**update)
                        corrected_v = squares[name] / (1 - .999**update)
                        self.params[name] -= self.config.learning_rate * corrected_m / (np.sqrt(corrected_v)+1e-8)
            train_loss = self._loss(u, i, y)
            val_loss = self._loss(vu, vi, vy)
            if not np.isfinite([train_loss, val_loss]).all():
                raise ValueError('Non-finite MF loss; inspect learning rate and input data')
            history.append({'epoch': epoch, 'train_bce': train_loss,
                            'validation_bce': val_loss,
                            'train_objective': self._loss(u, i, y, regularized=True)})
            if val_loss < best_loss - self.config.min_delta:
                best_loss, wait, self.best_epoch = val_loss, 0, epoch
                best = {name: v.copy() for name, v in self.params.items()}
            else:
                wait += 1
            if wait >= self.config.patience:
                break
        self.params = best
        self.history = pd.DataFrame(history)
        self.is_fitted = True
        return self

    def score(self, user_id: int, strategy_ids: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('Model has not been fitted')
        ids = np.asarray(strategy_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu':
            raise ValueError('strategy_ids must be an integer vector')
        try:
            u = np.full(len(ids), self._users[int(user_id)], dtype=np.int64)
            i = np.asarray([self._items[int(v)] for v in ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f'Unknown model ID: {exc.args[0]}') from exc
        return self._logits(u, i)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise ValueError('Cannot save an unfitted model')
        metadata = json.dumps({'format_version': 1, 'config': asdict(self.config),
                               'seed': self.seed, 'best_epoch': self.best_epoch})
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, **self.params, user_ids=self.user_ids,
                                strategy_ids=self.strategy_ids, metadata=np.array(metadata))

    @classmethod
    def load(cls, path: str | Path) -> 'LogisticMF':
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['format_version'] != 1:
                raise ValueError('Unsupported MF checkpoint version')
            model = cls(data['user_ids'], data['strategy_ids'],
                        MFConfig(**metadata['config']), metadata['seed'])
            for name, default in model.params.items():
                values = np.asarray(data[name], dtype=float)
                if values.shape != default.shape or not np.isfinite(values).all():
                    raise ValueError(f'Invalid checkpoint parameter: {name}')
                model.params[name] = values.copy()
        model.best_epoch = metadata['best_epoch']
        model.is_fitted = True
        return model
