"""
strategies/random.py - 随机采样策略

大量随机采样（可配置避开关键点邻域）+ 边界组合 + L-BFGS-B 优化。
"""

import itertools
from typing import List, Tuple, Dict, Set

from .base import SamplingStrategy
from ..optimization import optimize_extremes


class RandomStrategy(SamplingStrategy):
    """随机采样策略"""

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
        n_samples: int = 10000,
        critical_proximity_threshold: float = 0.0,
        **kwargs,
    ) -> int:
        crit_pts = None
        if critical_proximity_threshold > 0:
            crit_pts, _ = self.generate_critical_points(
                rel_intervals,
                coupled_pairs=self.robot.coupled_pairs,
                coupled_triples=self.robot.coupled_triples,
            )

        samples = self.random_avoiding_critical(
            rel_intervals, n_samples,
            crit_pts or [], critical_proximity_threshold)

        # 加入边界组合
        for combo in itertools.product(
                *[(lo, hi) for lo, hi in rel_intervals]):
            samples.append(list(combo))

        total = len(samples)
        full = self._expand_reduced(samples, sorted_rel, mid_q)
        self._evaluate_samples(link_idx, full, seg_extremes, n_sub)
        optimize_extremes(
            self.robot, link_idx, sorted_rel, mid_q, intervals,
            seg_extremes, n_sub)

        return total
