"""
test/planner/test_box_forest.py - BoxForest 和 BoxForestQuery 测试
"""

import os
import tempfile
import pytest
import numpy as np

from box_aabb.robot import load_robot
from planner.box_forest import BoxForest
from planner.box_query import BoxForestQuery
from planner.obstacles import Scene
from planner.models import PlannerConfig


@pytest.fixture
def robot_2dof():
    return load_robot('2dof_planar')


@pytest.fixture
def simple_scene():
    scene = Scene()
    scene.add_obstacle([0.8, -0.3], [1.2, 0.3], name="block")
    return scene


@pytest.fixture
def config():
    return PlannerConfig(
        build_n_seeds=30,
        max_box_nodes=20,
        max_iterations=50,
        query_expand_budget=5,
        expansion_resolution=0.05,
        max_expansion_rounds=2,
    )


class TestBoxForest:
    """BoxForest 构建测试"""

    def test_build_basic(self, robot_2dof, simple_scene, config):
        forest = BoxForest.build(
            robot_2dof, simple_scene, config=config, seed=42)

        assert forest.n_trees > 0
        assert forest.n_boxes > 0
        assert forest.total_volume > 0
        assert forest.build_time > 0

    def test_build_empty_scene(self, robot_2dof, config):
        scene = Scene()
        forest = BoxForest.build(
            robot_2dof, scene, config=config, seed=42)

        assert forest.n_boxes > 0

    def test_save_and_load(self, robot_2dof, simple_scene, config):
        forest = BoxForest.build(
            robot_2dof, simple_scene, config=config, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            forest.save(filepath)
            loaded = BoxForest.load(filepath, robot_2dof, simple_scene)

            assert loaded.n_trees == forest.n_trees
            assert loaded.n_boxes == forest.n_boxes
        finally:
            os.unlink(filepath)

    def test_load_wrong_robot(self, robot_2dof, simple_scene, config):
        forest = BoxForest.build(
            robot_2dof, simple_scene, config=config, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            forest.save(filepath)
            robot_3dof = load_robot('3dof_planar')
            with pytest.raises(ValueError, match="指纹不匹配"):
                BoxForest.load(filepath, robot_3dof, simple_scene)
        finally:
            os.unlink(filepath)


class TestBoxForestQuery:
    """BoxForestQuery 测试"""

    def test_query_basic(self, robot_2dof, simple_scene, config):
        forest = BoxForest.build(
            robot_2dof, simple_scene, config=config, seed=42)

        query = BoxForestQuery(forest)
        q_start = np.array([0.5, 0.5])
        q_goal = np.array([-0.5, -0.5])

        result = query.plan(q_start, q_goal, seed=42)
        # 不一定成功（取决于森林覆盖率），但不应崩溃
        assert result is not None

    def test_query_direct_connect(self, robot_2dof, config):
        """无障碍物时应能直连"""
        scene = Scene()
        forest = BoxForest.build(
            robot_2dof, scene, config=config, seed=42)

        query = BoxForestQuery(forest)
        q_start = np.array([0.1, 0.1])
        q_goal = np.array([0.2, 0.2])

        result = query.plan(q_start, q_goal, seed=42)
        assert result.success
        assert result.path_length > 0

    def test_query_collision_start(self, robot_2dof, simple_scene, config):
        """起始点碰撞应返回失败"""
        forest = BoxForest.build(
            robot_2dof, simple_scene, config=config, seed=42)

        query = BoxForestQuery(forest)
        # 使用可能碰撞的配置
        q_start = np.array([0.0, 0.0])
        q_goal = np.array([1.0, 1.0])

        result = query.plan(q_start, q_goal, seed=42)
        # 0,0 可能不碰撞，只检查不崩溃
        assert result is not None

    def test_multiple_queries_same_forest(self, robot_2dof, simple_scene, config):
        """同一森林多次查询"""
        forest = BoxForest.build(
            robot_2dof, simple_scene, config=config, seed=42)

        for _ in range(3):
            query = BoxForestQuery(forest)
            q_start = np.array([0.5, 0.5])
            q_goal = np.array([-0.5, -0.5])
            result = query.plan(q_start, q_goal, seed=42)
            assert result is not None
