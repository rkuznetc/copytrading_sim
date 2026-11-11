"""A request-level facade over the existing static world and choice model.

Supports model-driven inference, not yet automatic retraining, market dynamics,
portfolio positions or an economic outcome. Oracle state stays inside the world.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np
import pandas as pd

from copytrading_sim.config import resolve_config
from copytrading_sim.policies.base import SlatePolicy
from copytrading_sim.simulator.copy_decision import simulate_copy_decisions, build_copy_events
from copytrading_sim.simulator.exposure_policy import _build_candidate_table, sample_policy_slate


def request_rng(seed: int, stream: int, user_id: int, step: int) -> np.random.Generator:
    """Named per-request streams: independent of method order and user scheduling."""
    values = (seed, stream, user_id, step)
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in values):
        raise ValueError('Random stream keys must be non-negative integers')
    sequence = np.random.SeedSequence([int(x) for x in values])
    return np.random.Generator(np.random.PCG64(sequence))


@dataclass
class StepResult:
    exposures: pd.DataFrame
    copy_decisions: pd.DataFrame


class SimulationEnvironment:
    """Serve arbitrary SlatePolicy instances against one fixed generated world.

    A fresh instance resets histories but keeps identical users/strategies. Call
    step() once per recommendation; export_history() returns only new events.
    Repeated copies across requests are allowed, as in the original prototype.
    """
    def __init__(self, users: pd.DataFrame, strategies: pd.DataFrame,
                 compatibility: pd.DataFrame, config: dict, *,
                 seed: int | None = None, start_step: int = 0):
        self.config = resolve_config(config)
        self.seed = self.config['random']['seed'] if seed is None else seed
        request_rng(self.seed, 0, 0, start_step)  # validates all keys
        if users['user_id'].duplicated().any() or strategies['strategy_id'].duplicated().any():
            raise ValueError('World IDs must be unique')
        if compatibility.duplicated(['user_id', 'strategy_id']).any():
            raise ValueError('Duplicate compatibility pairs')
        if (len(compatibility) != len(users) * len(strategies)
                or not compatibility['user_id'].isin(users['user_id']).all()
                or not compatibility['strategy_id'].isin(strategies['strategy_id']).all()):
            raise ValueError('Environment requires full catalogue compatibility')
        candidates = _build_candidate_table(users, strategies, compatibility)
        self._candidates = {
            int(u): group.sort_values('strategy_id').reset_index(drop=True)
            for u, group in candidates.groupby('user_id', sort=True)
        }
        if any(u < 0 for u in self._candidates):
            raise ValueError('User IDs must be non-negative')
        self._next_step = {u: int(start_step) for u in self._candidates}
        self._next_exposure_id = 0
        self._history: list[StepResult] = []

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame], config: dict, **kwargs):
        return cls(tables['users'], tables['strategies'], tables['compatibility'],
                   config, **kwargs)

    def step(self, user_id: int, policy: SlatePolicy, *,
             slate_size: int | None = None, model_version: str = 'unversioned') -> StepResult:
        """Show one slate, simulate its response and append to this instance's log.

        No call to a trainer occurs here. A future experiment controller may swap
        policy.scorer after a training round without replacing the environment.
        """
        if user_id not in self._candidates:
            raise ValueError(f'Unknown user: {user_id}')
        step = self._next_step[user_id]
        k = self.config['simulation']['slate_size'] if slate_size is None else slate_size
        slate = sample_policy_slate(
            self._candidates[user_id], policy, k, f'user_{user_id}_step_{step}',
            step, request_rng(self.seed, 100, int(user_id), step),
        )
        slate.insert(0, 'exposure_id', np.arange(self._next_exposure_id,
                                               self._next_exposure_id + len(slate)))
        slate['was_shown'] = 1
        slate['policy_name'] = policy.name
        slate['model_version'] = str(model_version)
        decisions = simulate_copy_decisions(
            slate, self.config, request_rng(self.seed, 101, int(user_id), step),
        )
        result = StepResult(slate, decisions)
        self._history.append(StepResult(slate.copy(deep=True), decisions.copy(deep=True)))
        self._next_step[user_id] += 1
        self._next_exposure_id += len(slate)
        return result

    def export_history(self) -> dict[str, pd.DataFrame]:
        if not self._history:
            raise ValueError('No interactions have been generated')
        exposures = pd.concat([s.exposures for s in self._history], ignore_index=True)
        decisions = pd.concat([s.copy_decisions for s in self._history], ignore_index=True)
        return {'exposures': exposures, 'copy_decisions': decisions,
                'copy_events': build_copy_events(decisions)}
