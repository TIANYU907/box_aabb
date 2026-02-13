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
