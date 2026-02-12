"""test/planner/conftest.py - 共享 fixtures"""
import math
import pytest
import numpy as np

from box_aabb.robot import Robot, load_robot
from planner.obstacles import Scene
from planner.collision import CollisionChecker
from planner.box_tree import BoxTreeManager
from planner.models import PlannerConfig


# ==================== 2DOF 平面机械臂 ====================

@pytest.fixture(scope='session')
def robot_2dof():
    """2DOF 平面机械臂 (link lengths: 1.0, 1.0)"""
    return load_robot('2dof_planar')


@pytest.fixture(scope='session')
def joint_limits_2dof():
    """2DOF 关节限制"""
    return [(-math.pi, math.pi), (-math.pi, math.pi)]


@pytest.fixture
def scene_2dof_simple():
    """简单的 2DOF 场景：一个障碍物在 x=1.5~2.0, y=-0.3~0.3
    
    2DOF 平面臂用 Modified DH: 第一个连杆 a=1.0 沿 x 轴到 (1,0,0)。
    障碍物放在 x=1.5~2.0，远离 elbow 但能挡住直伸手臂 q≈0 的末端。
    """
    scene = Scene()
    scene.add_obstacle(
        min_point=[1.5, -0.3], max_point=[2.0, 0.3],
        name="obstacle_1",
    )
    return scene


@pytest.fixture
def scene_2dof_multi():
    """多障碍物 2DOF 场景"""
    scene = Scene()
    scene.add_obstacle([1.0, -0.3], [1.5, 0.3], name="obs1")
    scene.add_obstacle([-1.5, 0.5], [-1.0, 1.0], name="obs2")
    scene.add_obstacle([0.0, 1.2], [0.5, 1.7], name="obs3")
    return scene


@pytest.fixture
def scene_2dof_empty():
    """空场景（无障碍物）"""
    return Scene()


@pytest.fixture
def checker_2dof(robot_2dof, scene_2dof_simple):
    return CollisionChecker(robot_2dof, scene_2dof_simple)


@pytest.fixture
def checker_2dof_empty(robot_2dof, scene_2dof_empty):
    return CollisionChecker(robot_2dof, scene_2dof_empty)

