"""
planner/parallel_collision.py - 并行碰撞检测优化

通过 concurrent.futures 线程/进程池加速批量碰撞检测。
主要优化场景：
- box_expansion: 二分搜索中的批量碰撞检测
- connector: 多对 box 之间的线段碰撞检测
- sampling: 批量 seed 点碰撞检测

设计说明:
    由于 NumPy 底层已释放 GIL (C 扩展)，使用 ThreadPoolExecutor
    可以获得真正的并行加速。对于纯 Python 计算密集型任务，
    ProcessPoolExecutor 能绕过 GIL 但有序列化开销。
    默认使用线程池，对大规模场景可切换为进程池。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Callable

import numpy as np

from .collision import CollisionChecker

logger = logging.getLogger(__name__)


class ParallelCollisionChecker:
    """并行碰撞检测包装器

    封装 CollisionChecker 提供批量并行碰撞检测接口。

    Args:
        checker: 基础碰撞检测器
        n_workers: 并行工作线程数 (None = CPU 核心数)
        use_processes: True 使用进程池, False 使用线程池
        batch_threshold: 批次大小低于此阈值时退化为串行 (避免调度开销)

    Example:
        >>> par_checker = ParallelCollisionChecker(checker, n_workers=4)
        >>> results = par_checker.batch_check_configs(configs)
    """

    def __init__(
        self,
        checker: CollisionChecker,
        n_workers: Optional[int] = None,
        use_processes: bool = False,
        batch_threshold: int = 8,
    ) -> None:
        self.checker = checker
        self.n_workers = n_workers
        self.use_processes = use_processes
        self.batch_threshold = batch_threshold

    def _get_executor(self):
        """获取执行器"""
        cls = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        return cls(max_workers=self.n_workers)

    def batch_check_configs(
        self,
        configs: List[np.ndarray],
    ) -> List[bool]:
        """批量点碰撞检测

        Args:
            configs: 关节配置列表

        Returns:
            碰撞结果列表 (True=碰撞)
        """
        if len(configs) <= self.batch_threshold:
            return [self.checker.check_config_collision(q) for q in configs]

        results = [False] * len(configs)
        with self._get_executor() as executor:
            futures = {
                executor.submit(self.checker.check_config_collision, q): i
                for i, q in enumerate(configs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("碰撞检测异常 (index %d): %s", idx, e)
                    results[idx] = True  # 保守处理: 出错视为碰撞
        return results

    def batch_check_boxes(
        self,
        box_intervals: List[List[Tuple[float, float]]],
    ) -> List[bool]:
        """批量 box (区间) 碰撞检测

        Args:
            box_intervals: 关节区间列表

        Returns:
            碰撞结果列表 (True=可能碰撞, False=一定无碰撞)
        """
        if len(box_intervals) <= self.batch_threshold:
            return [self.checker.check_box_collision(iv) for iv in box_intervals]

        results = [True] * len(box_intervals)
        with self._get_executor() as executor:
            futures = {
                executor.submit(self.checker.check_box_collision, iv): i
                for i, iv in enumerate(box_intervals)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("Box 碰撞检测异常 (index %d): %s", idx, e)
                    results[idx] = True
        return results

    def batch_check_segments(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray]],
        resolution: Optional[float] = None,
    ) -> List[bool]:
        """批量线段碰撞检测

        Args:
            segments: [(q_start, q_end), ...] 线段列表
            resolution: 采样分辨率

        Returns:
            碰撞结果列表
        """
        if len(segments) <= self.batch_threshold:
            return [
                self.checker.check_segment_collision(q_s, q_e, resolution)
                for q_s, q_e in segments
            ]

        results = [True] * len(segments)
        with self._get_executor() as executor:
            futures = {
                executor.submit(
                    self.checker.check_segment_collision,
                    q_s, q_e, resolution,
                ): i
                for i, (q_s, q_e) in enumerate(segments)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("线段碰撞检测异常 (index %d): %s", idx, e)
                    results[idx] = True
        return results

    def filter_collision_free(
        self,
        configs: List[np.ndarray],
    ) -> List[np.ndarray]:
        """过滤出无碰撞的配置

        Args:
            configs: 候选配置列表

        Returns:
            无碰撞的配置
        """
        results = self.batch_check_configs(configs)
        return [q for q, collide in zip(configs, results) if not collide]


class SpatialIndex:
    """空间索引 (简易 grid)

    将障碍物按空间网格索引，加速碰撞检测中的障碍物查询。
    对大量障碍物的场景 (>50) 特别有效。

    Args:
        cell_size: 网格单元尺寸
    """

    def __init__(self, cell_size: float = 0.5) -> None:
        self.cell_size = cell_size
        self._grid: dict = {}
        self._obstacles: list = []

    def build(self, obstacles: list) -> None:
        """构建空间索引

        Args:
            obstacles: Obstacle 列表
        """
        self._obstacles = list(obstacles)
        self._grid.clear()

        for i, obs in enumerate(obstacles):
            cells = self._get_cells(obs.min_point, obs.max_point)
            for cell in cells:
                self._grid.setdefault(cell, []).append(i)

    def query(
        self,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> List[int]:
        """查询与给定 AABB 可能重叠的障碍物索引

        Args:
            aabb_min: 查询 AABB 最小角点
            aabb_max: 查询 AABB 最大角点

        Returns:
            可能重叠的障碍物索引列表 (无重复)
        """
        cells = self._get_cells(aabb_min, aabb_max)
        result = set()
        for cell in cells:
            if cell in self._grid:
                result.update(self._grid[cell])
        return list(result)

    def _get_cells(
        self,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> List[tuple]:
        """计算 AABB 覆盖的网格单元"""
        ndim = min(3, len(aabb_min))
        lo = np.floor(aabb_min[:ndim] / self.cell_size).astype(int)
        hi = np.floor(aabb_max[:ndim] / self.cell_size).astype(int)

        cells = []
        if ndim == 2:
            for x in range(lo[0], hi[0] + 1):
                for y in range(lo[1], hi[1] + 1):
                    cells.append((x, y))
        else:
            for x in range(lo[0], hi[0] + 1):
                for y in range(lo[1], hi[1] + 1):
                    for z in range(lo[2], hi[2] + 1):
                        cells.append((x, y, z))
        return cells

    @property
    def n_cells(self) -> int:
        """索引中的非空网格单元数"""
        return len(self._grid)

    @property
    def n_obstacles(self) -> int:
        return len(self._obstacles)
