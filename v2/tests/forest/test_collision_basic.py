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


def test_collision_checker_spatial_index_auto_enable() -> None:
    robot = load_robot("panda")
    scene = Scene()

    # 远离机器人工作空间，确保结果稳定为无碰撞
    for i in range(30):
        base = 100.0 + i * 2.0
        scene.add_obstacle([base, base, base], [base + 0.5, base + 0.5, base + 0.5])

    checker = CollisionChecker(robot=robot, scene=scene, spatial_index_threshold=20)
    q = np.zeros(robot.n_joints, dtype=np.float64)
    assert checker.check_config_collision(q) is False
    assert checker._spatial_index is not None


def test_collision_checker_spatial_index_refresh_on_scene_change() -> None:
    robot = load_robot("panda")
    scene = Scene()

    for i in range(25):
        base = 100.0 + i * 2.0
        scene.add_obstacle([base, base, base], [base + 0.5, base + 0.5, base + 0.5])

    checker = CollisionChecker(robot=robot, scene=scene, spatial_index_threshold=20)
    q = np.zeros(robot.n_joints, dtype=np.float64)

    assert checker.check_config_collision(q) is False
    first_sig = checker._spatial_index_sig

    # 新增一个近场障碍物，应触发索引刷新并导致碰撞
    scene.add_obstacle([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1], name="near")
    collide = checker.check_config_collision(q)

    assert checker._spatial_index_sig != first_sig
    assert isinstance(collide, bool)
