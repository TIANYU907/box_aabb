import numpy as np

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from planner.path_smoother import PathSmoother


def test_path_smoother_resample_and_shortcut_run() -> None:
    robot = load_robot("panda")
    scene = Scene()
    checker = CollisionChecker(robot=robot, scene=scene)
    smoother = PathSmoother(checker)

    p0 = np.zeros(robot.n_joints)
    p1 = np.ones(robot.n_joints) * 0.1
    p2 = np.ones(robot.n_joints) * 0.2
    path = [p0, p1, p2]

    short = smoother.shortcut(path, max_iters=10)
    resampled = smoother.resample(short, resolution=0.05)

    assert len(short) >= 2
    assert len(resampled) >= 2


class _BatchChecker:
    def __init__(self, collisions: np.ndarray):
        self.collisions = collisions
        self.batch_calls = 0
        self.single_calls = 0

    def check_segment_collision(self, *_args, **_kwargs) -> bool:
        return False

    def check_config_collision_batch(self, configs: np.ndarray) -> np.ndarray:
        self.batch_calls += 1
        n = configs.shape[0]
        out = np.zeros(n, dtype=bool)
        m = min(n, len(self.collisions))
        out[:m] = self.collisions[:m]
        return out

    def check_config_collision(self, _q: np.ndarray) -> bool:
        self.single_calls += 1
        return False


class _SingleChecker:
    def __init__(self, collisions: np.ndarray):
        self.collisions = collisions
        self.calls = 0

    def check_segment_collision(self, *_args, **_kwargs) -> bool:
        return False

    def check_config_collision(self, _q: np.ndarray) -> bool:
        idx = self.calls
        self.calls += 1
        if idx < len(self.collisions):
            return bool(self.collisions[idx])
        return False


def test_smooth_moving_average_uses_batch_checker() -> None:
    checker = _BatchChecker(np.array([False, True, False], dtype=bool))
    smoother = PathSmoother(checker)

    path = [
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 0.0]),
        np.array([3.0, 1.0]),
        np.array([4.0, 0.0]),
    ]

    out = smoother.smooth_moving_average(path, window=3, n_iters=1)

    assert checker.batch_calls == 1
    assert checker.single_calls == 0
    assert len(out) == len(path)
    assert np.allclose(out[2], path[2])


def test_smooth_moving_average_fallback_single_checker() -> None:
    checker = _SingleChecker(np.array([False, True, False], dtype=bool))
    smoother = PathSmoother(checker)

    path = [
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 0.0]),
        np.array([3.0, 1.0]),
        np.array([4.0, 0.0]),
    ]

    out = smoother.smooth_moving_average(path, window=3, n_iters=1)

    assert checker.calls == 3
    assert len(out) == len(path)
    assert np.allclose(out[2], path[2])
