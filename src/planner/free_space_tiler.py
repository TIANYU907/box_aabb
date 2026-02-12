"""
planner/free_space_tiler.py - 自由空间瓦片化器

给定 3D 障碍物，反选出尽可能大的无碰撞关节区间包络。

核心策略：
- 自适应单维度二分分割（每次只沿一个维度一分为二，避免 2^N 爆炸）
- 优先分割"最宽"或"最可能通过碰撞检测"的维度
- 碰撞自由的叶节点即为无碰撞区间瓦片

使用方式：
    tiler = FreeSpaceTiler(robot, scene, joint_limits)
    free_tiles = tiler.tile(max_depth=6, min_width=0.1)
    # free_tiles 是 [(intervals, link_aabbs), ...] 的列表
"""

import logging
from typing import List, Tuple, Optional

import numpy as np

from box_aabb.robot import Robot
from .collision import CollisionChecker
from .obstacles import Scene
from .models import BoxNode

logger = logging.getLogger(__name__)


class FreeSpaceTile:
    """一个无碰撞的 C-space 瓦片

    Attributes:
        intervals: 关节区间
        volume: 瓦片体积
        depth: 分割深度
    """

    def __init__(
        self,
        intervals: List[Tuple[float, float]],
        depth: int = 0,
    ) -> None:
        self.intervals = list(intervals)
        self.depth = depth
        self.volume = self._compute_volume()

    def _compute_volume(self) -> float:
        vol = 1.0
        has_nonzero = False
        for lo, hi in self.intervals:
            w = hi - lo
            if w > 0:
                vol *= w
                has_nonzero = True
        return vol if has_nonzero else 0.0

    @property
    def center(self) -> np.ndarray:
        return np.array([(lo + hi) / 2.0 for lo, hi in self.intervals])

    @property
    def widths(self) -> np.ndarray:
        return np.array([hi - lo for lo, hi in self.intervals])

    def to_box_node(self, node_id: int = 0) -> BoxNode:
        """转为 BoxNode"""
        return BoxNode(
            node_id=node_id,
            joint_intervals=self.intervals,
            seed_config=self.center,
        )


class FreeSpaceTiler:
    """自由空间瓦片化器

    通过递归自适应分割关节空间，识别所有无碰撞的区间瓦片。

    使用单维度二分分割策略，避免高维度下的 2^N 组合爆炸。
    每次选择最宽的维度进行分割。

    Args:
        robot: 机器人模型
        scene: 障碍物场景
        joint_limits: 关节限制
        min_width: 最小分割宽度（低于此宽度不再分割）
        max_depth: 最大递归深度
        use_sampling: 是否启用采样辅助碰撞检测
    """

    def __init__(
        self,
        robot: Robot,
        scene: Scene,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        min_width: float = 0.1,
        max_depth: int = 8,
        use_sampling: bool = False,
        sampling_n: int = 80,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.min_width = min_width
        self.max_depth = max_depth
        self.use_sampling = use_sampling
        self.sampling_n = sampling_n

        if joint_limits is None:
            if robot.joint_limits is not None:
                self.joint_limits = list(robot.joint_limits)
            else:
                self.joint_limits = [(-np.pi, np.pi)] * robot.n_joints
        else:
            self.joint_limits = list(joint_limits)

        self.collision_checker = CollisionChecker(robot, scene)
        self._rng = np.random.default_rng(42)

    def tile(
        self,
        initial_intervals: Optional[List[Tuple[float, float]]] = None,
        max_depth: Optional[int] = None,
        min_width: Optional[float] = None,
    ) -> List[FreeSpaceTile]:
        """执行自由空间瓦片化

        Args:
            initial_intervals: 初始区间（默认为关节限制全范围）
            max_depth: 最大递归深度（覆盖构造函数设置）
            min_width: 最小分割宽度

        Returns:
            无碰撞瓦片列表，按体积降序排列
        """
        if initial_intervals is None:
            initial_intervals = self.joint_limits
        if max_depth is None:
            max_depth = self.max_depth
        if min_width is None:
            min_width = self.min_width

        free_tiles: List[FreeSpaceTile] = []
        self._recursive_tile(initial_intervals, 0, max_depth, min_width,
                             free_tiles)

        # 按体积降序排列
        free_tiles.sort(key=lambda t: t.volume, reverse=True)

        logger.info(
            "FreeSpaceTiler: %d 个无碰撞瓦片, 总体积 %.4f",
            len(free_tiles), sum(t.volume for t in free_tiles))

        return free_tiles

    def _recursive_tile(
        self,
        intervals: List[Tuple[float, float]],
        depth: int,
        max_depth: int,
        min_width: float,
        free_tiles: List[FreeSpaceTile],
    ) -> None:
        """递归分割

        策略：
        1. 检查当前区间是否无碰撞
        2. 若无碰撞 → 加入结果
        3. 若碰撞且未达最大深度 → 选最宽维度二分分割
        4. 对两个子区间递归
        """
        # 检查碰撞
        is_collision = self._check_collision(intervals)

        if not is_collision:
            # 整个区间无碰撞！
            free_tiles.append(FreeSpaceTile(intervals, depth))
            return

        # 碰撞了，检查是否可以继续分割
        if depth >= max_depth:
            return  # 达到最大深度，放弃

        # 选择最宽的维度进行分割
        split_dim = self._choose_split_dimension(intervals, min_width)
        if split_dim is None:
            return  # 所有维度都太窄，放弃

        lo, hi = intervals[split_dim]
        mid = (lo + hi) / 2.0

        # 子区间 1: [lo, mid]
        child1 = list(intervals)
        child1[split_dim] = (lo, mid)
        self._recursive_tile(child1, depth + 1, max_depth, min_width,
                             free_tiles)

        # 子区间 2: [mid, hi]
        child2 = list(intervals)
        child2[split_dim] = (mid, hi)
        self._recursive_tile(child2, depth + 1, max_depth, min_width,
                             free_tiles)

    def _check_collision(
        self,
        intervals: List[Tuple[float, float]],
    ) -> bool:
        """碰撞检测（支持 hybrid 模式）"""
        result = self.collision_checker.check_box_collision(intervals)
        if not result:
            return False
        if not self.use_sampling:
            return True
        # 采样复核
        return self.collision_checker.check_box_collision_sampling(
            intervals, n_samples=self.sampling_n, rng=self._rng)

    def _choose_split_dimension(
        self,
        intervals: List[Tuple[float, float]],
        min_width: float,
    ) -> Optional[int]:
        """选择分割维度：最宽且 > min_width 的维度

        Returns:
            维度索引，或 None（所有维度都太窄）
        """
        best_dim = None
        best_width = 0.0

        for d, (lo, hi) in enumerate(intervals):
            w = hi - lo
            if w > min_width and w > best_width:
                best_width = w
                best_dim = d

        return best_dim

    def tiles_to_box_nodes(
        self,
        tiles: List[FreeSpaceTile],
    ) -> List[BoxNode]:
        """将瓦片转为 BoxNode 列表（可直接用于 BoxTree）"""
        nodes = []
        for i, tile in enumerate(tiles):
            nodes.append(tile.to_box_node(node_id=i))
        return nodes

    @property
    def n_collision_checks(self) -> int:
        return self.collision_checker.n_collision_checks
