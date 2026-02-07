"""test/planner/test_collision.py - 碰撞检测测试"""
import math
import numpy as np
import pytest

from box_aabb.planner.collision import CollisionChecker, aabb_overlap
from box_aabb.planner.obstacles import Scene


class TestAabbOverlap:
    """AABB 重叠检测辅助函数测试"""

    def test_overlap(self):
        assert aabb_overlap(
            np.array([0, 0, 0]), np.array([2, 2, 2]),
            np.array([1, 1, 1]), np.array([3, 3, 3]),
        ) is True

    def test_no_overlap_separated(self):
        assert aabb_overlap(
            np.array([0, 0, 0]), np.array([1, 1, 1]),
            np.array([2, 2, 2]), np.array([3, 3, 3]),
        ) is False

    def test_touching_edge(self):
        """刚好接触（边界重合）应视为重叠"""
        assert aabb_overlap(
            np.array([0, 0, 0]), np.array([1, 1, 1]),
            np.array([1, 0, 0]), np.array([2, 1, 1]),
        ) is True

    def test_2d(self):
        assert aabb_overlap(
            np.array([0, 0]), np.array([1, 1]),
            np.array([0.5, 0.5]), np.array([1.5, 1.5]),
        ) is True

    def test_2d_no_overlap(self):
        assert aabb_overlap(
            np.array([0, 0]), np.array([1, 1]),
            np.array([2, 2]), np.array([3, 3]),
        ) is False


class TestCollisionChecker:
    """CollisionChecker 测试"""

    def test_config_no_collision_zero_position(self, checker_2dof):
        """q=[0,0] 手臂沿 x 轴直伸，末端在 (2,0)
        
        Modified DH: p0=(0,0,0), p1=(1,0,0), p2=(2,0,0)
        障碍物: x=[1.5,2.0], y=[-0.3,0.3]
        Link1 段 (0,0,0)→(1,0,0): AABB (0,0,0)~(1,0,0), x最大=1 < 1.5 → 不碰
        Link2 段 (1,0,0)→(2,0,0): AABB (1,0,0)~(2,0,0), x=[1,2]∩[1.5,2.0]=有交, y=0在[-0.3,0.3]内 → 碰
        """
        q = np.array([0.0, 0.0])
        result = checker_2dof.check_config_collision(q)
        assert result is True

    def test_config_no_collision_arm_up(self, checker_2dof):
        """q=[pi/2, 0] 手臂从 (1,0,0) 向上到 (1,1,0)
        
        Link2 AABB: (1,0,0)~(1,1,0), 障碍物 x=[1.5,2.0]
        x: max_link=1.0 < min_obs=1.5 → 不重叠
        """
        q = np.array([math.pi / 2, 0.0])
        result = checker_2dof.check_config_collision(q)
        assert result is False

    def test_config_collision_counter(self, checker_2dof):
        """碰撞计数器正确递增"""
        checker_2dof.reset_counter()
        checker_2dof.check_config_collision(np.array([0.0, 0.0]))
        checker_2dof.check_config_collision(np.array([0.0, 0.0]))
        assert checker_2dof.n_collision_checks == 2

    def test_empty_scene_no_collision(self, checker_2dof_empty):
        """空场景下任何配置都不碰撞"""
        q = np.array([0.0, 0.0])
        assert checker_2dof_empty.check_config_collision(q) is False

    def test_box_collision_small_safe_box(self, checker_2dof):
        """在安全区域的小 box 应该不碰撞
        
        q ≈ pi/2: 第二段 link 从 (1,0,0) 向上，远离 x=1.5~2.0 障碍物。
        小的 interval 在 pi/2 附近，区间 FK 也应判为安全。
        """
        intervals = [(math.pi / 2 - 0.1, math.pi / 2 + 0.1),
                      (-0.1, 0.1)]
        result = checker_2dof.check_box_collision(intervals)
        assert result is False

    def test_box_collision_covering_obstacle(self, checker_2dof):
        """覆盖碰撞区域的大 box 应该碰撞"""
        # q ≈ 0 → 手臂沿 x 轴，穿过障碍物
        intervals = [(-0.3, 0.3), (-0.3, 0.3)]
        result = checker_2dof.check_box_collision(intervals)
        assert result is True

    def test_box_collision_empty_scene(self, checker_2dof_empty):
        """空场景 box 碰撞检测始终 False"""
        intervals = [(-1.0, 1.0), (-1.0, 1.0)]
        assert checker_2dof_empty.check_box_collision(intervals) is False

    def test_segment_collision_safe(self, checker_2dof):
        """两个安全配置之间（手臂朝上区域）无碰撞"""
        q1 = np.array([math.pi / 2, -0.05])
        q2 = np.array([math.pi / 2, 0.05])
        assert checker_2dof.check_segment_collision(q1, q2) is False

    def test_segment_collision_through_obstacle(self, checker_2dof):
        """穿过碰撞区域的线段应检测碰撞"""
        q1 = np.array([0.0, 0.0])   # 碰撞 (末端在 (2,0) 穿过障碍物)
        q2 = np.array([0.1, 0.0])   # 也大概率碰撞
        assert checker_2dof.check_segment_collision(q1, q2) is True

    def test_segment_same_point(self, checker_2dof):
        """起点=终点的线段应返回该点的碰撞状态"""
        q = np.array([math.pi / 2, 0.0])  # 手臂朝上，安全
        assert checker_2dof.check_segment_collision(q, q) is False

    def test_config_in_limits(self, checker_2dof):
        """关节限制检查"""
        q_in = np.array([0.5, 0.5])
        q_out = np.array([4.0, 0.5])  # 超出 [-pi, pi]
        limits = [(-math.pi, math.pi), (-math.pi, math.pi)]
        assert checker_2dof.check_config_in_limits(q_in, limits) is True
        assert checker_2dof.check_config_in_limits(q_out, limits) is False

    def test_config_in_limits_default(self, robot_2dof, scene_2dof_empty):
        """使用 robot 自带的 joint_limits"""
        checker = CollisionChecker(robot_2dof, scene_2dof_empty)
        if robot_2dof.joint_limits:
            q_in = np.array([0.0, 0.0])
            assert checker.check_config_in_limits(q_in) is True

    def test_safety_margin(self, robot_2dof, scene_2dof_simple):
        """安全裕度应增大碰撞区域"""
        checker_no_margin = CollisionChecker(robot_2dof, scene_2dof_simple, safety_margin=0.0)
        checker_margin = CollisionChecker(robot_2dof, scene_2dof_simple, safety_margin=1.0)

        # q=[pi/2, 0] 安全配置: link2 AABB=([1,0,0],[1,1,0])，离障碍物 x=1.5 距离 0.5
        # 无裕度: 安全 (x_max=1.0 < 1.5)
        # 裕度=1.0: 障碍物扩展为 x=[0.5, 3.0] → x_max=1.0 >= 0.5 → 可能碰
        q = np.array([math.pi / 2, 0.0])
        result_no = checker_no_margin.check_config_collision(q)
        result_yes = checker_margin.check_config_collision(q)
        assert result_no is False
        assert result_yes is True
