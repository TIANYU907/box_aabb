"""
planner/path_smoother.py - 路径后处理

提供路径平滑和优化功能：
1. Shortcut 优化：随机选两点尝试直连，若无碰撞则移除中间点
2. 等间距重采样
3. 简单的 B-spline / 线性插值平滑
"""

import logging
from typing import List, Optional

import numpy as np

from .collision import CollisionChecker

logger = logging.getLogger(__name__)


class PathSmoother:
    """路径后处理器

    Args:
        collision_checker: 碰撞检测器
        segment_resolution: 线段碰撞检测分辨率

    Example:
        >>> smoother = PathSmoother(checker)
        >>> smooth_path = smoother.shortcut(path, max_iters=200)
        >>> resampled = smoother.resample(smooth_path, resolution=0.1)
    """

    def __init__(
        self,
        collision_checker: CollisionChecker,
        segment_resolution: float = 0.05,
    ) -> None:
        self.collision_checker = collision_checker
        self.segment_resolution = segment_resolution

    def shortcut(
        self,
        path: List[np.ndarray],
        max_iters: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> List[np.ndarray]:
        """随机 shortcut 优化

        反复随机选两个非相邻路径点，若它们之间的直线段无碰撞，
        则移除中间所有点。

        Args:
            path: 原始路径点列表
            max_iters: 最大迭代次数
            rng: 随机数生成器

        Returns:
            优化后的路径
        """
        if len(path) <= 2:
            return list(path)

        if rng is None:
            rng = np.random.default_rng()

        path = list(path)  # copy
        improved = 0

        for _ in range(max_iters):
            if len(path) <= 2:
                break

            # 随机选两个索引 (i < j, j > i+1)
            i = rng.integers(0, len(path) - 2)
            j = rng.integers(i + 2, len(path))

            # 检查 path[i] → path[j] 直连是否无碰撞
            if not self.collision_checker.check_segment_collision(
                path[i], path[j], self.segment_resolution
            ):
                # 移除中间点
                path = path[:i + 1] + path[j:]
                improved += 1

        if improved > 0:
            logger.info("Shortcut 优化: 移除 %d 个中间段, 路径从 %d → %d 个点",
                         improved, improved + len(path), len(path))
        return path

    def resample(
        self,
        path: List[np.ndarray],
        resolution: float = 0.1,
    ) -> List[np.ndarray]:
        """等间距重采样

        在路径上以固定步长重新采样，使路径点间距均匀。

        Args:
            path: 输入路径
            resolution: 目标点间距（关节空间 L2）

        Returns:
            重采样后的路径
        """
        if len(path) <= 1:
            return list(path)

        resampled = [path[0].copy()]

        for i in range(1, len(path)):
            seg_vec = path[i] - path[i - 1]
            seg_len = float(np.linalg.norm(seg_vec))

            if seg_len < 1e-10:
                continue

            n_steps = max(1, int(np.ceil(seg_len / resolution)))
            for k in range(1, n_steps + 1):
                t = k / n_steps
                point = path[i - 1] + t * seg_vec
                resampled.append(point)

        return resampled

    def smooth_moving_average(
        self,
        path: List[np.ndarray],
        window: int = 3,
        n_iters: int = 5,
    ) -> List[np.ndarray]:
        """移动平均平滑

        保持首尾点不变，对中间点做加权平均平滑。
        每次平滑后验证碰撞，若有碰撞则回退。

        Args:
            path: 输入路径
            window: 平滑窗口大小
            n_iters: 迭代次数

        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return list(path)

        path = [p.copy() for p in path]
        half_w = window // 2

        for _ in range(n_iters):
            new_path = [path[0].copy()]
            changed = False

            for i in range(1, len(path) - 1):
                lo = max(0, i - half_w)
                hi = min(len(path), i + half_w + 1)
                avg = np.mean(path[lo:hi], axis=0)

                # 验证平滑后的点是否无碰撞
                if not self.collision_checker.check_config_collision(avg):
                    new_path.append(avg)
                    if not np.allclose(avg, path[i]):
                        changed = True
                else:
                    new_path.append(path[i].copy())

            new_path.append(path[-1].copy())

            if not changed:
                break
            path = new_path

        return path


def compute_path_length(path: List[np.ndarray]) -> float:
    """计算路径总长度 (L2)"""
    if len(path) < 2:
        return 0.0
    return sum(float(np.linalg.norm(path[i] - path[i - 1]))
               for i in range(1, len(path)))
