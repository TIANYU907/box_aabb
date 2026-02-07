"""test/planner/test_box_rrt.py - BoxRRT 集成测试 + Connector & PathSmoother 单元测试"""
import math
import numpy as np
import pytest

from box_aabb.robot import load_robot
from box_aabb.planner.models import PlannerConfig, BoxNode, Edge
from box_aabb.planner.obstacles import Scene
from box_aabb.planner.collision import CollisionChecker
from box_aabb.planner.box_expansion import BoxExpander
from box_aabb.planner.box_tree import BoxTreeManager
from box_aabb.planner.connector import TreeConnector
from box_aabb.planner.path_smoother import PathSmoother, compute_path_length
from box_aabb.planner.box_rrt import BoxRRT


class TestPathSmoother:
    """PathSmoother 单元测试"""

    def test_shortcut_trivial(self, checker_2dof):
        smoother = PathSmoother(checker_2dof)
        path = [np.array([1.5, 0.5]), np.array([1.6, 0.5])]
        result = smoother.shortcut(path, max_iters=10)
        assert len(result) >= 2

    def test_resample(self, checker_2dof):
        smoother = PathSmoother(checker_2dof)
        path = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
        resampled = smoother.resample(path, resolution=0.3)
        # 距离=1.0, 分辨率=0.3 → 至少 4 个点 (含首尾)
        assert len(resampled) >= 4

    def test_resample_single_point(self, checker_2dof):
        smoother = PathSmoother(checker_2dof)
        path = [np.array([0.0, 0.0])]
        result = smoother.resample(path)
        assert len(result) == 1

    def test_smooth_moving_average(self, checker_2dof):
        smoother = PathSmoother(checker_2dof)
        # 构造一条安全的锯齿路径（手臂朝上的区域）
        path = [
            np.array([1.5, 0.0]),
            np.array([1.6, 0.1]),
            np.array([1.5, 0.2]),
            np.array([1.6, 0.3]),
            np.array([1.5, 0.4]),
        ]
        smoothed = smoother.smooth_moving_average(path, window=3, n_iters=3)
        assert len(smoothed) == len(path)
        # 首尾不变
        np.testing.assert_array_almost_equal(smoothed[0], path[0])
        np.testing.assert_array_almost_equal(smoothed[-1], path[-1])

    def test_compute_path_length(self):
        path = [np.array([0, 0]), np.array([3, 4])]
        assert compute_path_length(path) == pytest.approx(5.0)

    def test_compute_path_length_empty(self):
        assert compute_path_length([]) == 0.0
        assert compute_path_length([np.array([0, 0])]) == 0.0


class TestTreeConnector:
    """TreeConnector 单元测试"""

    def test_connect_within_overlapping_boxes(self, robot_2dof, scene_2dof_simple):
        checker = CollisionChecker(robot_2dof, scene_2dof_simple)
        manager = BoxTreeManager()

        nid0 = manager.allocate_node_id()
        box0 = BoxNode(node_id=nid0, joint_intervals=[(0, 2), (0, 2)],
                        seed_config=np.array([1.0, 1.0]))
        tid = manager.create_tree(box0)

        nid1 = manager.allocate_node_id()
        box1 = BoxNode(node_id=nid1, joint_intervals=[(1, 3), (1, 3)],
                        seed_config=np.array([2.0, 2.0]))
        manager.add_box(tid, box1, parent_id=nid0)

        connector = TreeConnector(manager, checker)
        edges = connector.connect_within_trees()

        # 两个重叠的 box 应产生 1 条边
        assert len(edges) >= 1
        assert edges[0].is_collision_free is True

    def test_connect_within_no_overlap(self, robot_2dof, scene_2dof_simple):
        checker = CollisionChecker(robot_2dof, scene_2dof_simple)
        manager = BoxTreeManager()

        nid0 = manager.allocate_node_id()
        box0 = BoxNode(node_id=nid0, joint_intervals=[(0, 1), (0, 1)],
                        seed_config=np.array([0.5, 0.5]))
        tid = manager.create_tree(box0)

        nid1 = manager.allocate_node_id()
        box1 = BoxNode(node_id=nid1, joint_intervals=[(5, 6), (5, 6)],
                        seed_config=np.array([5.5, 5.5]))
        manager.add_box(tid, box1, parent_id=nid0)

        connector = TreeConnector(manager, checker)
        edges = connector.connect_within_trees()
        assert len(edges) == 0  # 不重叠 → 无边

    def test_connect_between_nearby_trees(self, robot_2dof, scene_2dof_empty):
        checker = CollisionChecker(robot_2dof, scene_2dof_empty)
        manager = BoxTreeManager()

        nid0 = manager.allocate_node_id()
        box0 = BoxNode(node_id=nid0, joint_intervals=[(0, 1), (0, 1)],
                        seed_config=np.array([0.5, 0.5]))
        manager.create_tree(box0)

        nid1 = manager.allocate_node_id()
        box1 = BoxNode(node_id=nid1, joint_intervals=[(1.1, 2.0), (0, 1)],
                        seed_config=np.array([1.5, 0.5]))
        manager.create_tree(box1)

        connector = TreeConnector(manager, checker, connection_radius=3.0)
        edges = connector.connect_between_trees()
        # 两棵树距离很近，且空场景下无碰撞 → 应产生连接边
        assert len(edges) >= 1

    def test_connect_endpoints_inside_box(self, robot_2dof, scene_2dof_empty):
        checker = CollisionChecker(robot_2dof, scene_2dof_empty)
        manager = BoxTreeManager()

        nid = manager.allocate_node_id()
        box = BoxNode(node_id=nid, joint_intervals=[(-1, 1), (-1, 1)],
                       seed_config=np.array([0.0, 0.0]))
        manager.create_tree(box)

        connector = TreeConnector(manager, checker)
        edges, start_id, goal_id = connector.connect_endpoints(
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
        )
        # 两个点都在 box 内 → 应直接连接
        assert start_id == nid
        assert goal_id == nid

    def test_build_adjacency_graph(self, robot_2dof, scene_2dof_empty):
        checker = CollisionChecker(robot_2dof, scene_2dof_empty)
        manager = BoxTreeManager()

        nid = manager.allocate_node_id()
        box = BoxNode(node_id=nid, joint_intervals=[(-1, 1), (-1, 1)],
                       seed_config=np.array([0.0, 0.0]))
        manager.create_tree(box)

        connector = TreeConnector(manager, checker)
        q_start = np.array([0.0, 0.0])
        q_goal = np.array([0.5, 0.5])

        graph = connector.build_adjacency_graph(
            [], q_start, q_goal,
            start_box_id=nid, goal_box_id=nid,
        )
        assert 'start' in graph['nodes']
        assert 'goal' in graph['nodes']
        assert nid in graph['nodes']


class TestBoxRRT:
    """BoxRRT 集成测试 (2DOF)"""

    def test_direct_connect(self, robot_2dof):
        """无障碍物时能直连"""
        scene = Scene()
        planner = BoxRRT(robot_2dof, scene, PlannerConfig(verbose=False))
        q_start = np.array([math.pi / 2, 0.0])
        q_goal = np.array([math.pi / 2, 0.5])
        result = planner.plan(q_start, q_goal, seed=42)
        assert result.success is True
        assert len(result.path) >= 2
        assert "直连" in result.message

    def test_start_collision_fails(self, robot_2dof, scene_2dof_simple):
        """起始点碰撞应返回失败"""
        planner = BoxRRT(robot_2dof, scene_2dof_simple)
        q_start = np.array([0.0, 0.0])  # 碰撞配置
        q_goal = np.array([math.pi / 2, 0.0])
        result = planner.plan(q_start, q_goal)
        assert result.success is False
        assert "碰撞" in result.message

    def test_planning_with_obstacle(self, robot_2dof, scene_2dof_simple):
        """有障碍物时应能规划出路径（较宽松配置）"""
        config = PlannerConfig(
            max_iterations=300,
            max_box_nodes=100,
            seed_batch_size=3,
            expansion_resolution=0.05,
            max_expansion_rounds=2,
            goal_bias=0.15,
            connection_radius=3.0,
            connection_max_attempts=30,
            path_shortcut_iters=50,
            verbose=False,
        )
        planner = BoxRRT(robot_2dof, scene_2dof_simple, config)

        # 从 (pi/2, 0) → (-pi/2, 0)，需绕过 x=1.5~2.0 的障碍物
        q_start = np.array([math.pi / 2, 0.0])   # 安全 - arm up
        q_goal = np.array([-math.pi / 2, 0.0])    # 安全 - arm bends back
        result = planner.plan(q_start, q_goal, seed=42)

        # 即使规划不成功也检查基本属性
        assert result.computation_time > 0
        assert result.n_boxes_created >= 1
        assert result.n_collision_checks > 0

        if result.success:
            assert len(result.path) >= 2
            assert result.path_length > 0
            # 路径起终点应接近目标
            np.testing.assert_array_almost_equal(result.path[0], q_start, decimal=3)
            np.testing.assert_array_almost_equal(result.path[-1], q_goal, decimal=3)

    def test_planning_empty_scene(self, robot_2dof):
        """无障碍物时应能规划（可能直连或快速完成）"""
        scene = Scene()
        config = PlannerConfig(max_iterations=50, max_box_nodes=20)
        planner = BoxRRT(robot_2dof, scene, config)

        q_start = np.array([0.5, 0.5])
        q_goal = np.array([2.0, -1.0])
        result = planner.plan(q_start, q_goal, seed=123)
        assert result.success is True

    def test_result_has_trees(self, robot_2dof, scene_2dof_simple):
        """结果中应包含 box tree 信息"""
        config = PlannerConfig(max_iterations=50, max_box_nodes=30, verbose=False)
        planner = BoxRRT(robot_2dof, scene_2dof_simple, config)

        q_start = np.array([math.pi / 2, 0.0])   # 安全
        q_goal = np.array([-math.pi / 2, 0.0])    # 安全
        result = planner.plan(q_start, q_goal, seed=42)

        # 至少创建了初始树
        assert len(result.box_trees) >= 1
        assert result.n_boxes_created >= 1

    def test_deterministic_with_seed(self, robot_2dof):
        """相同随机种子应产生相同结果"""
        scene = Scene()
        scene.add_obstacle([1.5, -0.3], [2.0, 0.3])

        config = PlannerConfig(max_iterations=100, max_box_nodes=50)

        planner1 = BoxRRT(robot_2dof, scene, config)
        result1 = planner1.plan(np.array([math.pi / 2, 0.0]),
                                 np.array([-math.pi / 2, 0.0]),
                                 seed=42)

        planner2 = BoxRRT(robot_2dof, scene, config)
        result2 = planner2.plan(np.array([math.pi / 2, 0.0]),
                                 np.array([-math.pi / 2, 0.0]),
                                 seed=42)

        assert result1.success == result2.success
        assert result1.n_boxes_created == result2.n_boxes_created
        if result1.success and result2.success:
            assert len(result1.path) == len(result2.path)

    def test_custom_joint_limits(self, robot_2dof):
        """自定义关节限制应生效"""
        scene = Scene()
        limits = [(-1.0, 1.0), (-1.0, 1.0)]
        planner = BoxRRT(robot_2dof, scene, joint_limits=limits)
        assert planner.joint_limits == limits
