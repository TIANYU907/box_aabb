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
from typing import List, Tuple, Set, Optional, TYPE_CHECKING

import numpy as np

from aabb.robot import Robot
from aabb.interval_fk import compute_interval_aabb
from .models import Obstacle
from .scene import Scene

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
        spatial_index_threshold: int = 20,
        spatial_cell_size: float = 0.5,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.safety_margin = safety_margin
        self._n_collision_checks = 0
        self._spatial_index_threshold = int(spatial_index_threshold)
        self._spatial_cell_size = float(spatial_cell_size)
        self._spatial_index = None
        self._spatial_index_sig = None

        # 预计算零长度连杆集合
        self._zero_length_links: Set[int] = robot.zero_length_links.copy()
        if skip_base_link:
            # 跳过第一个连杆（通常是基座，不参与碰撞检测）
            self._zero_length_links.add(1)

    def _obstacle_signature(self, obstacles: List[Obstacle]) -> tuple:
        """用于判断空间索引是否需要刷新。"""
        return tuple(
            (id(obs), tuple(obs.min_point.tolist()), tuple(obs.max_point.tolist()))
            for obs in obstacles
        )

    def _ensure_spatial_index(self, obstacles: List[Obstacle]) -> None:
        """按需创建/刷新空间索引。"""
        if len(obstacles) <= self._spatial_index_threshold:
            self._spatial_index = None
            self._spatial_index_sig = None
            return

        sig = self._obstacle_signature(obstacles)
        if self._spatial_index is not None and self._spatial_index_sig == sig:
            return

        from .parallel_collision import SpatialIndex

        index = SpatialIndex(cell_size=self._spatial_cell_size)
        index.build(obstacles)
        self._spatial_index = index
        self._spatial_index_sig = sig

    def _candidate_obstacles(
        self,
        link_min: np.ndarray,
        link_max: np.ndarray,
        obstacles: List[Obstacle],
    ) -> List[Obstacle]:
        """返回与 link AABB 可能重叠的候选障碍物。"""
        self._ensure_spatial_index(obstacles)
        if self._spatial_index is None:
            return obstacles

        idxs = self._spatial_index.query(link_min, link_max)
        return [obstacles[i] for i in idxs]

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

            candidates = self._candidate_obstacles(link_min, link_max, obstacles)
            for obs in candidates:
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

            candidates = self._candidate_obstacles(la_min, la_max, obstacles)
            for obs in candidates:
                obs_min = obs.min_point - margin
                obs_max = obs.max_point + margin
                if aabb_overlap(la_min, la_max, obs_min, obs_max):
                    return True

        return False

    def check_box_collision_sampling(
        self,
        joint_intervals: List[Tuple[float, float]],
        n_samples: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        """基于采样的 box 碰撞检测

        在 box 内随机采样 n_samples 个配置，逐一做点碰撞检测。
        若任一采样点碰撞则返回 True，否则返回 False（概率性安全）。

        与 check_box_collision 的区别：
        - check_box_collision 使用区间 FK，保守但可能严重过估计
        - 本方法使用采样，适用于高自由度机器人的辅助验证

        Args:
            joint_intervals: 关节区间 [(lo_0, hi_0), ...]
            n_samples: 采样数量
            rng: 随机数生成器

        Returns:
            True = 存在采样点碰撞, False = 所有采样点无碰撞
        """
        if rng is None:
            rng = np.random.default_rng()

        obstacles = self.scene.get_obstacles()
        if not obstacles:
            return False

        # 先检查中心点
        center = np.array([(lo + hi) / 2.0 for lo, hi in joint_intervals])
        if self.check_config_collision(center):
            return True

        # 检查部分顶点（对高维不做全部 2^n 个顶点）
        n_dims = len(joint_intervals)
        n_vertex_samples = min(2 * n_dims, n_samples // 3)
        for _ in range(n_vertex_samples):
            q = np.array([
                rng.choice([lo, hi]) if hi > lo else lo
                for lo, hi in joint_intervals
            ])
            if self.check_config_collision(q):
                return True

        # 随机采样
        remaining = n_samples - n_vertex_samples - 1
        for _ in range(remaining):
            q = np.array([
                rng.uniform(lo, hi) if hi > lo else lo
                for lo, hi in joint_intervals
            ])
            if self.check_config_collision(q):
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

    # ── 批量碰撞检测（向量化） ──────────────────────────────

    def check_config_collision_batch(
        self,
        configs: np.ndarray,
    ) -> np.ndarray:
        """批量单配置碰撞检测（向量化 FK + AABB）

        对 N 个关节配置同时做碰撞检测，利用 NumPy 向量化避免
        Python for 循环。适用于 C-space 碰撞地图扫描。

        Args:
            configs: 关节配置矩阵 (N, n_joints)

        Returns:
            布尔数组 (N,)，True = 碰撞
        """
        obstacles = self.scene.get_obstacles()
        N = configs.shape[0]
        self._n_collision_checks += N
        if not obstacles:
            return np.zeros(N, dtype=bool)

        result = np.zeros(N, dtype=bool)
        margin = self.safety_margin
        self._ensure_spatial_index(obstacles)

        # 预提取障碍物 AABB 为数组 (M, ndim) 方便向量化
        obs_mins = np.array([obs.min_point - margin for obs in obstacles])
        obs_maxs = np.array([obs.max_point + margin for obs in obstacles])

        # 批量 FK：对每个配置计算所有 link 端点位置
        # 使用 DH 参数逐关节累乘变换矩阵（向量化 N 个配置）
        robot = self.robot
        n_joints = robot.n_joints

        # 初始化：N 个单位矩阵
        T = np.tile(np.eye(4), (N, 1, 1))  # (N, 4, 4)
        all_positions = [T[:, :3, 3].copy()]  # base position (N, 3)

        for i, param in enumerate(robot.dh_params):
            alpha = param['alpha']
            a = param['a']

            if param['type'] == 'revolute':
                d = param['d']
                theta = configs[:, i] + param['theta']  # (N,)
            else:
                d = param['d'] + configs[:, i]
                theta = np.full(N, param['theta'])

            ca, sa = np.cos(alpha), np.sin(alpha)
            ct = np.cos(theta)  # (N,)
            st = np.sin(theta)  # (N,)

            # 构造 N 个 DH 变换矩阵 (N, 4, 4)
            A = np.zeros((N, 4, 4))
            A[:, 0, 0] = ct
            A[:, 0, 1] = -st
            A[:, 0, 3] = a
            A[:, 1, 0] = st * ca
            A[:, 1, 1] = ct * ca
            A[:, 1, 2] = -sa
            if isinstance(d, np.ndarray):
                A[:, 1, 3] = -d * sa
            else:
                A[:, 1, 3] = -d * sa
            A[:, 2, 0] = st * sa
            A[:, 2, 1] = ct * sa
            A[:, 2, 2] = ca
            if isinstance(d, np.ndarray):
                A[:, 2, 3] = d * ca
            else:
                A[:, 2, 3] = d * ca
            A[:, 3, 3] = 1.0

            # T = T @ A  批量矩阵乘法
            T = np.einsum('nij,njk->nik', T, A)
            all_positions.append(T[:, :3, 3].copy())  # (N, 3)

        # tool_frame
        if robot.tool_frame is not None:
            tf = robot.tool_frame
            A_tool = robot.dh_transform(tf['alpha'], tf['a'], tf['d'], 0.0)
            T = np.einsum('nij,jk->nik', T, A_tool)
            all_positions.append(T[:, :3, 3].copy())

        # 逐连杆段检查碰撞
        n_links = len(all_positions)
        for li in range(1, n_links):
            if li in self._zero_length_links:
                continue

            p_start = all_positions[li - 1]  # (N, 3)
            p_end = all_positions[li]        # (N, 3)

            # 每个配置的 link AABB
            link_min = np.minimum(p_start, p_end)  # (N, 3)
            link_max = np.maximum(p_start, p_end)   # (N, 3)

            # 与每个障碍物做 AABB 重叠检测（可选空间索引筛选）
            if self._spatial_index is None:
                candidate_indices = range(len(obstacles))
            else:
                band_min = np.min(link_min, axis=0)
                band_max = np.max(link_max, axis=0)
                candidate_indices = self._spatial_index.query(band_min, band_max)

            for oi in candidate_indices:
                o_min = obs_mins[oi]
                o_max = obs_maxs[oi]

                # 分离轴测试：对所有 N 个配置向量化
                ndim = min(link_min.shape[1], len(o_min))
                separated = np.zeros(N, dtype=bool)
                for d in range(ndim):
                    separated |= (link_max[:, d] < o_min[d] - 1e-10)
                    separated |= (o_max[d] < link_min[:, d] - 1e-10)

                # 未分离 = 碰撞
                result |= ~separated

        return result
