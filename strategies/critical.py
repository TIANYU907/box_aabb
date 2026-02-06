"""
strategies/critical.py - 关键点枚举策略

阶段 1: 关键枚举（梯度为 0 的点，精确极值无需优化）
阶段 2: 约束流形随机采样
阶段 3: L-BFGS-B 优化（当前最优 + 预筛选 top-2 种子）
"""

from typing import List, Tuple, Dict, Set

from .base import SamplingStrategy
from ..optimization import optimize_extremes


class CriticalStrategy(SamplingStrategy):
    """关键点枚举采样策略"""

    def _process_link(
        self,
        link_idx: int,
        relevant: Set[int],
        sorted_rel: List[int],
        rel_intervals: List[Tuple[float, float]],
        mid_q: List[float],
        intervals: List[Tuple[float, float]],
        seg_extremes: List[Dict],
        n_sub: int,
        **kwargs,
    ) -> int:
        total = 0

        # 阶段 1: 关键枚举（全部求值）
        pts, cpts = self.generate_critical_points(
            rel_intervals,
            coupled_pairs=self.robot.coupled_pairs,
            coupled_triples=self.robot.coupled_triples,
        )
        total += len(pts)
        full = self._expand_reduced(pts, sorted_rel, mid_q)
        self._evaluate_samples(link_idx, full, seg_extremes, n_sub)
        fc = self._expand_reduced(cpts, sorted_rel, mid_q) if cpts else []

        # 阶段 2: 约束流形随机采样
        mf = self.generate_manifold_random(
            rel_intervals,
            coupled_triples=self.robot.coupled_triples,
        )
        fm: List[List[float]] = []
        if mf:
            fm = self._expand_reduced(mf, sorted_rel, mid_q)
            total += len(mf)
            self._evaluate_samples(link_idx, fm, seg_extremes, n_sub)

        # 阶段 3: 合并优化
        opt_seeds = fc + fm
        optimize_extremes(
            self.robot, link_idx, sorted_rel, mid_q, intervals,
            seg_extremes, n_sub,
            extra_seeds=opt_seeds if opt_seeds else None,
            n_seeds=2)

        return total
