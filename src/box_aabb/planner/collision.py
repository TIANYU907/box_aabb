"""
planner/collision.py - 碰撞检测模块

提供基于 AABB 的保守碰撞检测：
- 单点碰撞检测：FK → 逐 link AABB vs obstacle AABB
- Box (区间) 碰撞检测：区间 FK → 保守 AABB vs obstacle AABB
- 线段碰撞检测：等间隔采样逐点检查

保守性说明：
    box 碰撞检测使用区间算术得到的 link AABB 是**过估计**的。
    因此 ``check_box_collision`` 返回 True 时只表示"可能碰撞"，
    返回 False 时保证"一定无碰撞"。这确保了被判为安全的 box 确实安全。
"""

import logging
from typing import List, Tuple, Set, Optional

import numpy as np

from ..robot import Robot
from ..interval_fk import compute_interval_aabb
from .models import Obstacle
from .obstacles import Scene

logger = logging.getLogger(__name__)


def aabb_overlap(
    min1: np.ndarray, max1: np.ndarray,
    min2: np.ndarray, max2: np.ndarray,
) -> bool:
    """检测两个 AABB 是否重叠（分离轴测试）

    Args:
        min1, max1: 第一个 AABB 的最小/最大角点
        min2, max2: 第二个 AABB 的最小/最大角点

    Returns:
        True 表示重叠，False 表示分离
    """
    ndim = min(len(min1), len(min2))
    for i in range(ndim):
        if max1[i] < min2[i] - 1e-10 or max2[i] < min1[i] - 1e-10:
            return False
    return True


class CollisionChecker:
    """碰撞检测器

    封装机械臂与障碍物之间的碰撞检测逻辑。

    Args:
        robot: 机器人模型
        scene: 障碍物场景
        safety_margin: 安全裕度（可选，对 obstacle AABB 向外扩展）

    Example:
        >>> checker = CollisionChecker(robot, scene)
        >>> is_collide = checker.check_config_collision(q)
        >>> is_box_collide = checker.check_box_collision(intervals)
    """

    def __init__(
        self,
        robot: Robot,
        scene: Scene,
        safety_margin: float = 0.0,
        skip_base_link: bool = False,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.safety_margin = safety_margin
        self._n_collision_checks = 0

        # 预计算零长度连杆集合
        self._zero_length_links: Set[int] = robot.zero_length_links.copy()
        if skip_base_link:
            # 跳过第一个连杆（通常是基座，不参与碰撞检测）
            self._zero_length_links.add(1)

    @property
    def n_collision_checks(self) -> int:
        """累计碰撞检测调用次数"""
        return self._n_collision_checks

    def reset_counter(self) -> None:
        """重置碰撞检测计数器"""
        self._n_collision_checks = 0

    def check_config_collision(self, joint_values: np.ndarray) -> bool:
        """单配置碰撞检测

        对给定关节配置：
        1. 正向运动学得到各 link 端点位置
        2. 对每对相邻端点计算 link 段的 AABB
        3. 检测每个 link AABB 与每个 obstacle AABB 是否重叠

        Args:
            joint_values: 关节配置 (n_joints,)

        Returns:
            True = 存在碰撞, False = 无碰撞
        """
        self._n_collision_checks += 1
        obstacles = self.scene.get_obstacles()
        if not obstacles:
            return False

        positions = self.robot.get_link_positions(joint_values)
        margin = self.safety_margin

        # 逐连杆段检查（link i 的段是 positions[i-1] 到 positions[i]）
        for li in range(1, len(positions)):
            if li in self._zero_length_links:
                continue

            p_start = positions[li - 1]
            p_end = positions[li]

            # 构造该 link 段的 AABB
            link_min = np.minimum(p_start, p_end)
            link_max = np.maximum(p_start, p_end)

            for obs in obstacles:
                obs_min = obs.min_point - margin
                obs_max = obs.max_point + margin
                if aabb_overlap(link_min, link_max, obs_min, obs_max):
                    return True

        return False

    def check_box_collision(
        self,
        joint_intervals: List[Tuple[float, float]],
    ) -> bool:
        """Box (区间) 碰撞检测（保守方法）

        使用区间/仿射算术 FK 计算每个 link 的保守 AABB，
        然后与障碍物 AABB 做重叠检测。

        保守性：返回 False 保证该 box 内所有配置无碰撞。
                返回 True 表示可能碰撞（可能是过估计导致的误报）。

        Args:
            joint_intervals: 关节区间 [(lo_0, hi_0), ..., (lo_n, hi_n)]

        Returns:
            True = 可能碰撞, False = 一定无碰撞
        """
        self._n_collision_checks += 1
        obstacles = self.scene.get_obstacles()
        if not obstacles:
            return False

        # 调用区间 FK 获取保守 AABB
        link_aabbs, _ = compute_interval_aabb(
            robot=self.robot,
            intervals=joint_intervals,
            zero_length_links=self._zero_length_links,
            skip_zero_length=True,
            n_sub=1,
        )

        margin = self.safety_margin

        for la in link_aabbs:
            if la.is_zero_length:
                continue
            la_min = np.array(la.min_point)
            la_max = np.array(la.max_point)

            for obs in obstacles:
                obs_min = obs.min_point - margin
                obs_max = obs.max_point + margin
                if aabb_overlap(la_min, la_max, obs_min, obs_max):
                    return True

        return False

    def check_segment_collision(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        resolution: Optional[float] = None,
    ) -> bool:
        """线段碰撞检测

        在两个关节配置之间等间隔采样，逐点做碰撞检测。

        Args:
            q_start: 起始关节配置
            q_end: 终止关节配置
            resolution: 采样间隔（关节空间 L2 距离），默认 0.05 rad

        Returns:
            True = 存在碰撞点, False = 所有采样点无碰撞
        """
        if resolution is None:
            resolution = 0.05

        dist = float(np.linalg.norm(q_end - q_start))
        if dist < 1e-10:
            return self.check_config_collision(q_start)

        n_steps = max(2, int(np.ceil(dist / resolution)) + 1)

        for i in range(n_steps):
            t = i / (n_steps - 1)
            q = q_start + t * (q_end - q_start)
            if self.check_config_collision(q):
                return True

        return False

    def check_config_in_limits(
        self,
        joint_values: np.ndarray,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ) -> bool:
        """检查配置是否在关节限制内

        Args:
            joint_values: 关节配置
            joint_limits: 关节限制列表，默认使用 robot.joint_limits

        Returns:
            True = 在限制内, False = 超出限制
        """
        limits = joint_limits or self.robot.joint_limits
        if limits is None:
            return True
        for i, (lo, hi) in enumerate(limits):
            if i >= len(joint_values):
                break
            if joint_values[i] < lo - 1e-10 or joint_values[i] > hi + 1e-10:
                return False
        return True
