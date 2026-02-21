import numpy as np

from planner.sbf_planner import SBFPlanner
from planner.models import SBFConfig


class _DummyChecker:
    """第 first_free_idx 次调用返回 False（无碰撞），其余返回 True。"""
    def __init__(self, first_free_idx: int):
        self.first_free_idx = first_free_idx
        self.calls = 0

    def check_config_collision(self, _: np.ndarray) -> bool:
        hit = self.calls != self.first_free_idx
        self.calls += 1
        return hit


class _DummyAllCollide:
    """所有调用均返回 True（碰撞）。"""
    def __init__(self):
        self.calls = 0

    def check_config_collision(self, _: np.ndarray) -> bool:
        self.calls += 1
        return True


def _make_planner(checker, goal_bias: float = 0.0) -> SBFPlanner:
    planner = SBFPlanner.__new__(SBFPlanner)
    planner.config = SBFConfig(goal_bias=goal_bias)
    planner._n_dims = 2
    planner.joint_limits = [(-1.0, 1.0), (-2.0, 2.0)]
    planner.collision_checker = checker
    return planner


def test_sample_seed_returns_first_free() -> None:
    checker = _DummyChecker(first_free_idx=3)
    planner = _make_planner(checker, goal_bias=0.2)

    rng = np.random.default_rng(123)
    q = planner._sample_seed(
        q_start=np.array([0.0, 0.0]),
        q_goal=np.array([0.3, -0.4]),
        rng=rng,
    )

    assert q is not None
    # 前 3 次碰撞 + 第 4 次无碰撞 = 共 4 次调用
    assert checker.calls == 4
    assert np.all(q >= np.array([-1.0, -2.0]))
    assert np.all(q <= np.array([1.0, 2.0]))


def test_sample_seed_returns_none_when_all_colliding() -> None:
    checker = _DummyAllCollide()
    planner = _make_planner(checker, goal_bias=0.1)

    rng = np.random.default_rng(7)
    q = planner._sample_seed(
        q_start=np.array([0.0, 0.0]),
        q_goal=np.array([0.0, 0.0]),
        rng=rng,
    )

    assert q is None
    assert checker.calls == 20  # max_attempts


def test_sample_seed_early_stop() -> None:
    checker = _DummyChecker(first_free_idx=0)
    planner = _make_planner(checker, goal_bias=0.0)

    rng = np.random.default_rng(11)
    q = planner._sample_seed(
        q_start=np.array([0.0, 0.0]),
        q_goal=np.array([0.0, 0.0]),
        rng=rng,
    )

    assert q is not None
    assert checker.calls == 1  # 第一次就成功，早停
