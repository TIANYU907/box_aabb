"""
test/planner/test_panda_integration.py - 7DOF Panda 集成测试

测试 Box-RRT 规划器在 7DOF Panda 机器人上的完整工作流程。
Panda 有 7 个 revolute 关节，夹爪连杆通过 tool_frame 表示（无额外 C-space 维度）。
"""
import math
import pytest
import numpy as np

from box_aabb.robot import load_robot
from planner import (
    BoxRRT, Scene, PlannerConfig, PlannerResult,
    CollisionChecker, BoxTreeManager,
)
from planner.hier_aabb_tree import HierAABBTree
from planner.path_smoother import compute_path_length


# ==================== Panda Fixtures ====================

@pytest.fixture(scope='module')
def robot_panda():
    """7DOF Panda (7 DH params + tool_frame)"""
    return load_robot('panda')


@pytest.fixture(scope='module')
def joint_limits_panda(robot_panda):
    """Panda 的 7 个关节限制"""
    return robot_panda.joint_limits


@pytest.fixture
def scene_panda_simple():
    """单障碍物 Panda 场景: 箱子在工作空间前方"""
    scene = Scene()
    scene.add_obstacle(
        min_point=[0.3, -0.15, 0.3],
        max_point=[0.6, 0.15, 0.6],
        name="front_box",
    )
    return scene


@pytest.fixture
def scene_panda_multi():
    """多障碍物 Panda 场景"""
    scene = Scene()
    scene.add_obstacle([0.3, -0.15, 0.3], [0.5, 0.15, 0.5], name="box1")
    scene.add_obstacle([-0.3, 0.2, 0.4], [-0.1, 0.4, 0.7], name="box2")
    scene.add_obstacle([0.0, -0.4, 0.1], [0.2, -0.2, 0.3], name="box3")
    return scene


@pytest.fixture
def scene_panda_empty():
    """空场景"""
    return Scene()


@pytest.fixture
def checker_panda(robot_panda, scene_panda_simple):
    return CollisionChecker(robot_panda, scene_panda_simple)


@pytest.fixture
def checker_panda_empty(robot_panda, scene_panda_empty):
    return CollisionChecker(robot_panda, scene_panda_empty)


# ==================== 基础验证测试 ====================

class TestPandaRobotBasic:
    """Panda 机器人基础属性验证"""

    def test_panda_n_joints(self, robot_panda):
        assert robot_panda.n_joints == 7

    def test_panda_joint_limits_count(self, joint_limits_panda):
        assert len(joint_limits_panda) == 7

    def test_panda_has_tool_frame(self, robot_panda):
        """夹爪连杆通过 tool_frame 表示"""
        assert robot_panda.tool_frame is not None
        assert abs(robot_panda.tool_frame['d'] - 0.107) < 1e-6

    def test_panda_active_joints(self, joint_limits_panda):
        """所有 7 个关节有非零范围"""
        for i in range(7):
            lo, hi = joint_limits_panda[i]
            assert hi > lo, f"关节 {i} 限制无效: [{lo}, {hi}]"

    def test_panda_dh_params(self, robot_panda):
        """所有 a=0.0, 部分 d 非零"""
        for i, p in enumerate(robot_panda.dh_params):
            assert abs(p['a']) < 1e-10, f"关节 {i}: a 应为 0"

    def test_panda_fk_home(self, robot_panda):
        """零位 FK 不抛出异常"""
        q_home = np.zeros(7)
        positions = robot_panda.get_link_positions(q_home)
        assert len(positions) >= 2

    def test_panda_fk_with_tool_frame(self, robot_panda):
        """使用非零关节值不抛出异常，tool_frame 正确附加"""
        q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        positions = robot_panda.get_link_positions(q)
        assert len(positions) >= 2
        # tool_frame 应该产生比关节数多一个位置
        transforms = robot_panda.forward_kinematics(q.tolist(), return_all=True)
        assert len(transforms) == 7 + 1 + 1  # base + 7 joints + tool_frame


# ==================== 碰撞检测测试 ====================

class TestPandaCollision:
    """Panda 碰撞检测测试"""

    def test_collision_free_home(self, checker_panda):
        """零位通常无碰撞"""
        q = np.zeros(7)
        # 可能碰也可能不碰，取决于障碍物位置
        # 这里主要测试不抛异常
        result = checker_panda.check_config_collision(q)
        assert isinstance(result, bool)

    def test_collision_empty_scene(self, checker_panda_empty):
        """空场景下任何配置都无碰撞"""
        q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        assert not checker_panda_empty.check_config_collision(q)

    def test_collision_check_various_configs(self, checker_panda):
        """多种配置不抛异常"""
        configs = [
            np.zeros(7),
            np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3]),
            np.array([-1.0, 0.5, -0.5, -0.5, 1.0, 2.0, 1.0]),
        ]
        for q in configs:
            result = checker_panda.check_config_collision(q)
            assert isinstance(result, bool)

    def test_box_collision_check(self, checker_panda):
        """区间碰撞检测不抛异常"""
        intervals = [
            (-0.1, 0.1), (-0.6, -0.4), (-0.1, 0.1), (-2.1, -1.9),
            (-0.1, 0.1), (1.4, 1.6), (-0.1, 0.1),
        ]
        result = checker_panda.check_box_collision(intervals)
        assert isinstance(result, bool)

    def test_segment_collision_check(self, checker_panda):
        """线段碰撞检测不抛异常"""
        q1 = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        q2 = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3])
        result = checker_panda.check_segment_collision(q1, q2, 0.1)
        assert isinstance(result, bool)


# ==================== Box 拓展测试 ====================

class TestPandaBoxExpansion:
    """Panda box 拓展测试（使用 HierAABBTree）"""

    def test_expand_at_safe_config(self, robot_panda, scene_panda_empty):
        """空场景下能拓展出 box"""
        limits = robot_panda.joint_limits
        tree = HierAABBTree(robot_panda, limits)
        obstacles = scene_panda_empty.get_obstacles()
        q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        ivs = tree.find_free_box(q, obstacles)
        assert ivs is not None
        active_vol = 1.0
        for i in range(7):
            lo, hi = ivs[i]
            active_vol *= (hi - lo)
        assert active_vol > 0, "活跃关节的体积应 > 0"

    def test_expand_volume_positive(self, robot_panda, scene_panda_empty):
        """空场景下 box 体积应> 0（7D 无退化维度）"""
        limits = robot_panda.joint_limits
        tree = HierAABBTree(robot_panda, limits)
        obstacles = scene_panda_empty.get_obstacles()
        q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        ivs = tree.find_free_box(q, obstacles)
        assert ivs is not None
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        assert vol > 0, "7D box 体积应 > 0"

    def test_expand_with_obstacles(self, robot_panda, scene_panda_simple):
        """有障碍物时也能拓展"""
        limits = robot_panda.joint_limits
        tree = HierAABBTree(robot_panda, limits)
        obstacles = scene_panda_simple.get_obstacles()
        # 使用远离障碍物的 tucked 配置，避免 7D 区间过估计导致 find_free_box 失败
        q = np.array([0.0, -1.0, 0.0, -2.5, 0.0, 2.0, 0.0])
        ivs = tree.find_free_box(q, obstacles)
        assert ivs is not None
        expanded_dims = sum(
            1 for i in range(7)
            if ivs[i][1] - ivs[i][0] > 1e-10
        )
        assert expanded_dims > 0, "应有至少一个活跃维度被拓展"


# ==================== Box-RRT 集成测试 ====================

class TestPandaBoxRRT:
    """Panda Box-RRT 端到端规划测试"""

    def test_plan_empty_scene(self, robot_panda, scene_panda_empty):
        """空场景下路径规划（直连成功）"""
        config = PlannerConfig(
            max_iterations=50,
            max_box_nodes=30,
            verbose=False,
        )
        planner = BoxRRT(robot_panda, scene_panda_empty, config)
        q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        q_goal = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3])
        result = planner.plan(q_start, q_goal, seed=42)
        assert result.success
        assert len(result.path) >= 2
        assert result.path_length > 0

    def test_plan_start_collision_rejected(self, robot_panda, scene_panda_simple):
        """起始点在障碍物中时被拒绝"""
        config = PlannerConfig(max_iterations=10, verbose=False)
        planner = BoxRRT(robot_panda, scene_panda_simple, config)
        # 找一个在障碍物中的配置
        checker = CollisionChecker(robot_panda, scene_panda_simple)
        # 尝试找碰撞配置
        rng = np.random.default_rng(0)
        q_collision = None
        for _ in range(200):
            q = rng.uniform(-1, 1, size=7)
            if checker.check_config_collision(q):
                q_collision = q
                break
        if q_collision is not None:
            q_goal = np.array([1.5, -0.5, -1.0, -2.0, 0.5, 1.0, 0.5])
            result = planner.plan(q_collision, q_goal, seed=42)
            assert not result.success
            assert "碰撞" in result.message

    def test_plan_result_fields(self, robot_panda, scene_panda_empty):
        """PlannerResult 所有字段正确填充"""
        config = PlannerConfig(max_iterations=10, verbose=False)
        planner = BoxRRT(robot_panda, scene_panda_empty, config)
        q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        q_goal = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3])
        result = planner.plan(q_start, q_goal, seed=42)
        assert result.computation_time >= 0
        assert isinstance(result.message, str)
        assert result.n_collision_checks >= 0

    def test_plan_path_configs_valid(self, robot_panda, scene_panda_empty):
        """路径上每个配置都是 7D 且在关节限制内"""
        config = PlannerConfig(max_iterations=10, verbose=False)
        planner = BoxRRT(robot_panda, scene_panda_empty, config)
        q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        q_goal = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3])
        result = planner.plan(q_start, q_goal, seed=42)
        assert result.success
        limits = robot_panda.joint_limits
        for q in result.path:
            assert len(q) == 7
            for i, (lo, hi) in enumerate(limits):
                assert q[i] >= lo - 1e-6, f"q[{i}]={q[i]} < {lo}"
                assert q[i] <= hi + 1e-6, f"q[{i}]={q[i]} > {hi}"

    @pytest.mark.slow
    def test_plan_with_obstacles(self, robot_panda, scene_panda_simple):
        """有障碍物场景下尝试规划（不保证成功但不应崩溃）"""
        config = PlannerConfig(
            max_iterations=200,
            max_box_nodes=100,
            seed_batch_size=3,
            expansion_resolution=0.05,
            max_expansion_rounds=2,
            goal_bias=0.15,
            connection_radius=3.0,
            connection_max_attempts=40,
            path_shortcut_iters=50,
            verbose=False,
        )
        planner = BoxRRT(robot_panda, scene_panda_simple, config)
        # 选择两个远离障碍物的安全配置
        q_start = np.array([1.5, -0.5, -1.0, -2.0, 0.5, 1.0, 0.5])
        q_goal = np.array([-1.0, 0.5, 1.0, -1.0, -0.5, 2.5, -0.5])
        result = planner.plan(q_start, q_goal, seed=42)
        # 不要求一定成功, 但不应抛异常
        assert isinstance(result, PlannerResult)
        assert result.computation_time > 0
        assert result.n_collision_checks > 0

    @pytest.mark.slow
    def test_plan_multi_obstacle(self, robot_panda, scene_panda_multi):
        """多障碑物场景下不崩溃"""
        config = PlannerConfig(
            max_iterations=100,
            max_box_nodes=50,
            verbose=False,
        )
        planner = BoxRRT(robot_panda, scene_panda_multi, config)
        q_start = np.array([1.5, -0.5, -1.0, -2.0, 0.5, 1.0, 0.5])
        q_goal = np.array([-1.0, 0.5, 1.0, -1.0, -0.5, 2.5, -0.5])
        result = planner.plan(q_start, q_goal, seed=123)
        assert isinstance(result, PlannerResult)

    def test_plan_seed_reproducibility(self, robot_panda, scene_panda_empty):
        """相同 seed 产生相同结果"""
        config = PlannerConfig(max_iterations=30, max_box_nodes=20, verbose=False)
        planner1 = BoxRRT(robot_panda, scene_panda_empty, config)
        planner2 = BoxRRT(robot_panda, scene_panda_empty, config)
        q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        q_goal = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3])
        r1 = planner1.plan(q_start, q_goal, seed=99)
        r2 = planner2.plan(q_start, q_goal, seed=99)
        assert r1.success == r2.success
        if r1.success:
            assert abs(r1.path_length - r2.path_length) < 1e-10
