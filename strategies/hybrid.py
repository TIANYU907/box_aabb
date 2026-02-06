"""
strategies/hybrid.py - 混合采样策略

结合关键枚举 + 约束流形随机 + 随机补充 + L-BFGS-B 优化。
"""

from typing import List, Tuple, Dict, Set

from .base import SamplingStrategy
from ..optimization import optimize_extremes


class HybridStrategy(SamplingStrategy):
    """混合采样策略"""

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
        *,
        n_refine_samples: int = 200,
        critical_proximity_threshold: float = 0.05,
        **kwargs,
    ) -> int:
        total = 0

        # 阶段 1: 关键枚举
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

        # 阶段 3: 随机补充
        if n_refine_samples > 0:
            rr = self.random_avoiding_critical(
                rel_intervals, n_refine_samples, pts,
                critical_proximity_threshold)
            total += len(rr)
            fr = self._expand_reduced(rr, sorted_rel, mid_q)
            self._evaluate_samples(link_idx, fr, seg_extremes, n_sub)

        # 阶段 4: 合并优化
        opt_seeds = fc + fm
        optimize_extremes(
            self.robot, link_idx, sorted_rel, mid_q, intervals,
            seg_extremes, n_sub,
            extra_seeds=opt_seeds if opt_seeds else None,
            n_seeds=2)

        return total
