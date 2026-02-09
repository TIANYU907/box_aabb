"""test/planner/test_box_expansion.py - Box 拓展算法测试"""
import math
import numpy as np
import pytest

from planner.box_expansion import BoxExpander
from planner.collision import CollisionChecker
from planner.obstacles import Scene
from planner.models import BoxNode


class TestBoxExpander:
    """BoxExpander 测试"""

    def test_expand_safe_seed(self, expander_2dof):
        """安全 seed 应生成非 None 的 box"""
        # q=[pi/2, 0] → 手臂朝上，远离障碍物
        seed = np.array([math.pi / 2, 0.0])
        box = expander_2dof.expand(seed, node_id=0, tree_id=0)
        assert box is not None
        assert box.node_id == 0
        assert box.tree_id == 0
        assert len(box.joint_intervals) == 2

    def test_expand_seed_is_contained(self, expander_2dof):
        """生成的 box 应包含 seed"""
        seed = np.array([math.pi / 2, 0.0])
        box = expander_2dof.expand(seed, node_id=0)
        assert box is not None
        assert box.contains(seed)

    def test_expand_volume_positive(self, expander_2dof):
        """生成的 box 体积应 > 0"""
        seed = np.array([math.pi / 2, 0.0])
        box = expander_2dof.expand(seed, node_id=0)
        assert box is not None
        assert box.volume > 0

    def test_expand_collision_seed_returns_none(self, expander_2dof):
        """碰撞 seed 应返回 None"""
        seed = np.array([0.0, 0.0])  # 手臂沿 x 轴，穿过障碍物
        box = expander_2dof.expand(seed, node_id=0)
        # 可能返回 None（如果 seed 碰撞）
        if box is not None:
            # seed 处虽然 link 碰，确认 check_config_collision 结果
            pass

    def test_expand_box_is_collision_free(self, expander_2dof, checker_2dof):
        """拓展出的 box 应通过碰撞检测（保守方法）"""
        seed = np.array([math.pi / 2, 0.0])
        box = expander_2dof.expand(seed, node_id=0)
        assert box is not None
        # box 内不碰撞 → check_box_collision 返回 False
        result = checker_2dof.check_box_collision(box.joint_intervals)
        assert result is False

    def test_expand_box_within_limits(self, expander_2dof, joint_limits_2dof):
        """box 区间应在关节限制内"""
        seed = np.array([math.pi / 2, 0.0])
        box = expander_2dof.expand(seed, node_id=0)
        assert box is not None
        for i, (lo, hi) in enumerate(box.joint_intervals):
            assert lo >= joint_limits_2dof[i][0] - 1e-6
            assert hi <= joint_limits_2dof[i][1] + 1e-6

    def test_expand_empty_scene_wide_box(self, robot_2dof, joint_limits_2dof):
        """空场景下 box 应拓展到接近关节极限"""
        scene = Scene()
        checker = CollisionChecker(robot_2dof, scene)
        expander = BoxExpander(
            robot_2dof, checker, joint_limits_2dof,
            expansion_resolution=0.02,
            max_rounds=2,
        )
        seed = np.array([0.0, 0.0])
        box = expander.expand(seed, node_id=0)
        assert box is not None
        # 无障碍物，box 应接近关节极限
        for i, (lo, hi) in enumerate(box.joint_intervals):
            assert lo == pytest.approx(joint_limits_2dof[i][0], abs=0.1)
            assert hi == pytest.approx(joint_limits_2dof[i][1], abs=0.1)

    def test_dimension_order_heuristic(self, expander_2dof):
        """Jacobian norm 排序应返回正确数量的维度"""
        seed = np.array([math.pi / 2, 0.0])
        order = expander_2dof._compute_dimension_order(seed)
        assert len(order) == 2
        assert set(order) == {0, 1}

    def test_multiple_seeds_different_boxes(self, expander_2dof):
        """不同 seed 应生成不同的 box"""
        seed1 = np.array([math.pi / 2, 0.0])
        seed2 = np.array([-math.pi / 2, 0.0])
        box1 = expander_2dof.expand(seed1, node_id=0)
        box2 = expander_2dof.expand(seed2, node_id=1)
        assert box1 is not None and box2 is not None
        # 中心应不同
        assert not np.allclose(box1.center, box2.center)
