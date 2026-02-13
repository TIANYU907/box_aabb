import numpy as np

from aabb.robot import load_robot
from forest.collision import CollisionChecker
from forest.scene import Scene


def test_collision_checker_box_api_runs() -> None:
    robot = load_robot("panda")
    scene = Scene()
    scene.add_obstacle([10.0, 10.0, 10.0], [11.0, 11.0, 11.0], name="far")

    checker = CollisionChecker(robot=robot, scene=scene)
    intervals = [(-0.1, 0.1)] * robot.n_joints

    result = checker.check_box_collision(intervals)
    assert isinstance(result, bool)


def test_collision_checker_config_api_runs() -> None:
    robot = load_robot("panda")
    scene = Scene()
    checker = CollisionChecker(robot=robot, scene=scene)

    q = np.zeros(robot.n_joints, dtype=np.float64)
    assert checker.check_config_collision(q) is False
