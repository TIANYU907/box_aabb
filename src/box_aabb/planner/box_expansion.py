"""
planner/box_expansion.py - 启发式 Box 拓展算法

从一个无碰撞 seed 配置出发，在关节空间中拓展出尽可能大的
无碰撞 box（超矩形区间）。

拓展策略：
1. 在 seed 处计算 Jacobian 范数，按从小到大排序维度
   （范数小 → 该关节变化对末端位置影响小 → 优先拓展）
2. 对排序后的每个维度执行二分搜索，分别向正/负方向找碰撞边界
3. 迭代优化：拓展完一轮后可再做一轮（因为先拓展的维度在后续拓展后可能有更大空间）
"""

import logging
from typing import List, Tuple, Optional

import numpy as np

from ..robot import Robot
from .collision import CollisionChecker
from .models import BoxNode

logger = logging.getLogger(__name__)


class BoxExpander:
    """Box 拓展器

    从 seed 配置出发，启发式地拓展无碰撞 box。

    Args:
        robot: 机器人模型
        collision_checker: 碰撞检测器
        joint_limits: 关节限制列表
        expansion_resolution: 二分搜索精度 (rad)
        max_rounds: 最大迭代轮数
        jacobian_delta: Jacobian 数值差分步长
        min_initial_half_width: 初始半宽（seed 两侧的最小初始区间）

    Example:
        >>> expander = BoxExpander(robot, checker, limits)
        >>> box = expander.expand(seed_config, node_id=0)
    """

    def __init__(
        self,
        robot: Robot,
        collision_checker: CollisionChecker,
        joint_limits: List[Tuple[float, float]],
        expansion_resolution: float = 0.01,
        max_rounds: int = 3,
        jacobian_delta: float = 0.01,
        min_initial_half_width: float = 0.001,
        use_sampling: bool = False,
        sampling_n: int = 80,
    ) -> None:
        self.robot = robot
        self.collision_checker = collision_checker
        self.joint_limits = joint_limits
        self.resolution = expansion_resolution
        self.max_rounds = max_rounds
        self.jacobian_delta = jacobian_delta
        self.min_initial_half_width = min_initial_half_width
        self._n_dims = len(joint_limits)
        self.use_sampling = use_sampling
        self.sampling_n = sampling_n
        self._rng: Optional[np.random.Generator] = None

    def expand(
        self,
        seed: np.ndarray,
        node_id: int = 0,
        tree_id: int = -1,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[BoxNode]:
        """从 seed 配置拓展无碰撞 box

        Args:
            seed: 无碰撞 seed 配置 (n_joints,)
            node_id: 分配给该 box 节点的 ID
            tree_id: 所属树 ID
            rng: 随机数生成器（采样模式使用）

        Returns:
            BoxNode 实例，或 None（若 seed 本身就碰撞）
        """
        self._rng = rng or np.random.default_rng()

        if self.collision_checker.check_config_collision(seed):
            logger.debug("seed 配置碰撞，跳过: %s", seed)
            return None

        # 初始区间：以 seed 为中心、极小宽度
        intervals = []
        for i in range(self._n_dims):
            lo = max(self.joint_limits[i][0], seed[i] - self.min_initial_half_width)
            hi = min(self.joint_limits[i][1], seed[i] + self.min_initial_half_width)
            intervals.append((lo, hi))

        # 确认初始 box 无碰撞
        if self._check_collision(intervals):
            # 过估计导致极小 box 也被判碰撞，回退到点
            logger.debug("初始极小 box 碰撞（过估计），使用点区间")
            intervals = [(seed[i], seed[i]) for i in range(self._n_dims)]

        # 计算每个维度的探索优先级
        dim_order = self._compute_dimension_order(seed)

        # 多轮迭代拓展
        prev_volume = self._volume(intervals)
        for round_idx in range(self.max_rounds):
            intervals = self._expand_one_round(seed, intervals, dim_order)
            new_volume = self._volume(intervals)

            if new_volume <= prev_volume * 1.001:
                # 体积不再明显增长，提前停止
                logger.debug("第 %d 轮拓展后体积未增长，停止", round_idx + 1)
                break
            prev_volume = new_volume
            logger.debug("第 %d 轮拓展: 体积 = %.6f", round_idx + 1, new_volume)

        box = BoxNode(
            node_id=node_id,
            joint_intervals=intervals,
            seed_config=seed.copy(),
            tree_id=tree_id,
        )
        return box

    def _compute_dimension_order(self, config: np.ndarray) -> List[int]:
        """计算拓展维度的优先级排序

        按 Jacobian 列向量范数从小到大排序。范数越小说明该关节变化
        对末端执行器位置的影响越小，可以更大胆地拓展。

        Args:
            config: 当前关节配置

        Returns:
            维度索引的排序列表（按优先级从高到低）
        """
        n_joints = self._n_dims
        delta = self.jacobian_delta

        # 基准末端位置
        base_pos = self.robot.get_link_positions(config)[-1]  # 末端

        jacobian_norms = []
        for i in range(n_joints):
            # 限制：固定关节（区间宽度为 0）直接给极大范数
            lo, hi = self.joint_limits[i]
            if abs(hi - lo) < 1e-10:
                jacobian_norms.append(float('inf'))
                continue

            q_plus = config.copy()
            q_plus[i] += delta
            pos_plus = self.robot.get_link_positions(q_plus)[-1]

            # 数值差分近似 ||∂p/∂qi||
            norm = float(np.linalg.norm(pos_plus - base_pos)) / delta
            jacobian_norms.append(norm)

        # 按范数从小到大排序
        order = sorted(range(n_joints), key=lambda i: jacobian_norms[i])

        if logger.isEnabledFor(logging.DEBUG):
            for i in order:
                logger.debug("  dim %d: Jacobian norm = %.4f", i, jacobian_norms[i])

        return order

    def _expand_one_round(
        self,
        seed: np.ndarray,
        intervals: List[Tuple[float, float]],
        dim_order: List[int],
    ) -> List[Tuple[float, float]]:
        """一轮拓展：按优先级逐维度向两侧二分搜索

        Args:
            seed: seed 配置
            intervals: 当前区间
            dim_order: 维度探索优先级

        Returns:
            拓展后的区间列表
        """
        intervals = list(intervals)  # copy

        for dim in dim_order:
            lo_limit, hi_limit = self.joint_limits[dim]
            current_lo, current_hi = intervals[dim]

            # 固定关节跳过
            if abs(hi_limit - lo_limit) < 1e-10:
                continue

            # 向正方向拓展
            new_hi = self._binary_search_boundary(
                intervals, dim, current_hi, hi_limit, direction=+1
            )

            # 更新区间用于后续负方向搜索
            intervals[dim] = (current_lo, new_hi)

            # 向负方向拓展
            new_lo = self._binary_search_boundary(
                intervals, dim, current_lo, lo_limit, direction=-1
            )
            intervals[dim] = (new_lo, new_hi)

        return intervals

    def _check_collision(
        self,
        test_intervals: List[Tuple[float, float]],
    ) -> bool:
        """box 碰撞检测（支持 hybrid 模式）

        当 use_sampling=True 时，先用区间 FK 检查，
        若区间 FK 判碰撞再用采样方式复核。
        采样无碰撞则覆盖为安全（概率性）。

        Returns:
            True = 碰撞, False = 安全
        """
        interval_result = self.collision_checker.check_box_collision(test_intervals)
        if not interval_result:
            # 区间 FK 说安全 → 一定安全
            return False
        if not self.use_sampling:
            # 不启用采样 → 直接信任区间 FK
            return True
        # 区间 FK 说碰撞但可能过估计 → 用采样复核
        sampling_result = self.collision_checker.check_box_collision_sampling(
            test_intervals, n_samples=self.sampling_n, rng=self._rng,
        )
        return sampling_result

    def _binary_search_boundary(
        self,
        intervals: List[Tuple[float, float]],
        dim: int,
        current_bound: float,
        limit: float,
        direction: int,
    ) -> float:
        """二分搜索某维度的碰撞边界

        Args:
            intervals: 当前所有维度的区间
            dim: 要拓展的维度索引
            current_bound: 当前已知安全的边界值
            limit: 该维度的关节极限
            direction: +1 向正方向拓展, -1 向负方向拓展

        Returns:
            新的安全边界值
        """
        safe = current_bound
        test = limit  # 尝试直接到极限

        # 先尝试一步到极限
        test_intervals = list(intervals)
        if direction > 0:
            test_intervals[dim] = (intervals[dim][0], test)
        else:
            test_intervals[dim] = (test, intervals[dim][1])

        if not self._check_collision(test_intervals):
            # 到极限都无碰撞
            return test

        # 二分搜索
        for _ in range(50):  # 防止死循环
            if abs(test - safe) < self.resolution:
                break

            mid = (safe + test) / 2.0
            test_intervals = list(intervals)
            if direction > 0:
                test_intervals[dim] = (intervals[dim][0], mid)
            else:
                test_intervals[dim] = (mid, intervals[dim][1])

            if self._check_collision(test_intervals):
                # mid 处碰撞，收缩
                test = mid
            else:
                # mid 处安全，拓展
                safe = mid

        return safe

    @staticmethod
    def _volume(intervals: List[Tuple[float, float]]) -> float:
        """计算区间体积（忽略固定关节的零宽度维度）"""
        vol = 1.0
        has_nonzero = False
        for lo, hi in intervals:
            w = hi - lo
            if w > 0:
                vol *= w
                has_nonzero = True
        return vol if has_nonzero else 0.0
