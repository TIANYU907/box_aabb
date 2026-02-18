import numpy as np

from planner.box_rrt import BoxRRT
from planner.models import PlannerConfig


class _DummyBatchChecker:
    def __init__(self, collisions: np.ndarray):
        self._collisions = collisions
        self.batch_calls = 0
        self.single_calls = 0
        self.last_configs = None

    def check_config_collision_batch(self, configs: np.ndarray) -> np.ndarray:
        self.batch_calls += 1
        self.last_configs = configs.copy()
        n = configs.shape[0]
        if len(self._collisions) >= n:
            return self._collisions[:n]
        out = np.ones(n, dtype=bool)
        out[:len(self._collisions)] = self._collisions
        return out

    def check_config_collision(self, _: np.ndarray) -> bool:
        self.single_calls += 1
        return True


class _DummySingleChecker:
    def __init__(self, first_free_idx: int):
        self.first_free_idx = first_free_idx
        self.calls = 0

    def check_config_collision(self, _: np.ndarray) -> bool:
        hit = self.calls != self.first_free_idx
        self.calls += 1
        return hit


def _make_planner(checker, goal_bias: float = 0.0) -> BoxRRT:
    planner = BoxRRT.__new__(BoxRRT)
    planner.config = PlannerConfig(goal_bias=goal_bias)
    planner._n_dims = 2
    planner.joint_limits = [(-1.0, 1.0), (-2.0, 2.0)]
    planner.collision_checker = checker
    return planner


def test_sample_seed_uses_batch_and_returns_first_free() -> None:
    collisions = np.ones(20, dtype=bool)
    collisions[3] = False
    checker = _DummyBatchChecker(collisions)
    planner = _make_planner(checker, goal_bias=0.2)

    rng = np.random.default_rng(123)
    q = planner._sample_seed(
        q_start=np.array([0.0, 0.0]),
        q_goal=np.array([0.3, -0.4]),
        rng=rng,
    )

    assert q is not None
    assert checker.batch_calls == 1
    assert checker.single_calls == 0
    assert np.all(q >= np.array([-1.0, -2.0]))
    assert np.all(q <= np.array([1.0, 2.0]))
    assert np.allclose(q, checker.last_configs[3])


def test_sample_seed_returns_none_when_all_colliding() -> None:
    checker = _DummyBatchChecker(np.ones(20, dtype=bool))
    planner = _make_planner(checker, goal_bias=0.1)

    rng = np.random.default_rng(7)
    q = planner._sample_seed(
        q_start=np.array([0.0, 0.0]),
        q_goal=np.array([0.0, 0.0]),
        rng=rng,
    )

    assert q is None
    assert checker.batch_calls == 1


def test_sample_seed_fallback_to_single_checker() -> None:
    checker = _DummySingleChecker(first_free_idx=4)
    planner = _make_planner(checker, goal_bias=0.0)

    rng = np.random.default_rng(11)
    q = planner._sample_seed(
        q_start=np.array([0.0, 0.0]),
        q_goal=np.array([0.0, 0.0]),
        rng=rng,
    )

    assert q is not None
    assert checker.calls == 5
