"""
strategies/base.py - 采样策略基类

定义所有采样策略的公共接口和共享工具方法。
"""

import math
import itertools
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set, Optional

import numpy as np

from ..robot import Robot
from ..models import (
    LinkAABBInfo, BoundaryConfig, BOUNDARY_TYPES, DIM_MAP
)
from ..optimization import (
    optimize_extremes, _update_extremes, points_equal
)


class SamplingStrategy(ABC):
    """采样策略基类

    子类需实现 ``_process_link`` 方法，定义单个连杆的采样-求值-优化流程。
    基类提供：
    - 连杆遍历调度 (``execute``)
    - 极值跟踪 / AABB 构建工具
    - 关键点生成 / 流形采样等静态工具
    """

    def __init__(self, robot: Robot, robot_name: str = "Robot",
                 zero_length_links: Optional[Set[int]] = None,
                 link_relevant: Optional[Dict[int, Set[int]]] = None):
        self.robot = robot
        self.robot_name = robot_name
        self._zero_length_links = zero_length_links or set()
        self._link_relevant = link_relevant or {}

    # ------------------------------------------------------------------
    #  主调度
    # ------------------------------------------------------------------

    def execute(
        self,
        intervals: List[Tuple[float, float]],
        skip_zero_length: bool = True,
        n_sub: int = 1,
        **kwargs,
    ) -> Tuple[List[LinkAABBInfo], int]:
        """执行采样策略

        Args:
            intervals: 关节区间列表
            skip_zero_length: 是否跳过零长度连杆
            n_sub: 每根连杆等分段数
            **kwargs: 策略特定参数

        Returns:
            (link_aabbs, total_samples)
        """
        aabbs: List[LinkAABBInfo] = []
        total = 0

        for li in range(1, self.robot.n_joints + 1):
            prep = self._prepare_link(li, intervals, skip_zero_length)
            if prep is None:
                aabbs.extend(self._skip_or_fallback(li, intervals))
                continue
            rel, srel, rivl, mq = prep
            seg_ext = self._init_seg_extremes(n_sub)

            n_samples = self._process_link(
                li, rel, srel, rivl, mq, intervals, seg_ext, n_sub, **kwargs)
            total += n_samples

            aabbs.extend(
                self._build_link_aabbs(li, seg_ext, n_sub, intervals, rel))
        return aabbs, total

    @abstractmethod
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
        """处理单个连杆的采样-求值-优化流程

        Returns:
            该连杆上评估的采样点数
        """
        ...

    # ------------------------------------------------------------------
    #  连杆前期准备
    # ------------------------------------------------------------------

    def _prepare_link(
        self, link_idx: int,
        intervals: List[Tuple[float, float]],
        skip_zero_length: bool,
    ) -> Optional[Tuple[Set[int], List[int],
                         List[Tuple[float, float]], List[float]]]:
        """准备连杆计算所需信息，零长度或无关节影响时返回 None"""
        if skip_zero_length and link_idx in self._zero_length_links:
            return None
        relevant = self._link_relevant.get(link_idx, set())
        if not relevant:
            return None
        sorted_rel = sorted(relevant)
        rel_ivl = [(intervals[i] if i < len(intervals) else (0, 0))
                   for i in sorted_rel]
        mid_q = [(lo + hi) / 2 for lo, hi in intervals]
        return relevant, sorted_rel, rel_ivl, mid_q

    def _skip_or_fallback(
        self, link_idx: int,
        intervals: List[Tuple[float, float]],
    ) -> List[LinkAABBInfo]:
        """跳过的连杆返回单元素列表"""
        if link_idx in self._zero_length_links:
            return [self._make_zero_length_aabb(link_idx, intervals)]
        return [self._make_trivial_aabb(link_idx)]

    def _make_zero_length_aabb(
        self, link_idx: int,
        intervals: List[Tuple[float, float]],
    ) -> LinkAABBInfo:
        mid_q = [(lo + hi) / 2 for lo, hi in intervals]
        pos = self.robot.get_link_position(mid_q, link_idx)
        return LinkAABBInfo(link_idx, self._link_name(link_idx),
                            list(pos), list(pos), is_zero_length=True)

    def _make_trivial_aabb(self, link_idx: int) -> LinkAABBInfo:
        pos = self.robot.get_link_position(
            [0] * self.robot.n_joints, link_idx)
        return LinkAABBInfo(link_idx, self._link_name(link_idx),
                            list(pos), list(pos))

    # ------------------------------------------------------------------
    #  命名
    # ------------------------------------------------------------------

    @staticmethod
    def _link_name(link_idx: int) -> str:
        return f"Link {link_idx} (Joint {link_idx - 1})"

    @staticmethod
    def _seg_name(link_idx: int, seg_idx: int, n_sub: int) -> str:
        if n_sub <= 1:
            return f"Link {link_idx} (Joint {link_idx - 1})"
        return f"Link {link_idx} Seg {seg_idx}"

    # ------------------------------------------------------------------
    #  极值跟踪
    # ------------------------------------------------------------------

    @staticmethod
    def _init_extremes() -> Dict:
        return {bt: (float('inf') if 'min' in bt else float('-inf'), None)
                for bt in BOUNDARY_TYPES}

    def _init_seg_extremes(self, n_sub: int) -> List[Dict]:
        return [self._init_extremes() for _ in range(n_sub)]

    @staticmethod
    def _seg_t_values(n_sub: int) -> List[Tuple[float, float]]:
        return [(k / n_sub, (k + 1) / n_sub) for k in range(n_sub)]

    # ------------------------------------------------------------------
    #  采样点求值（核心：FK 缓存 + 线性插值）
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_reduced(
        reduced: List[List[float]],
        sorted_rel: List[int],
        mid_q: List[float],
    ) -> List[List[float]]:
        """将 reduced-space 采样点展开为 full-space"""
        full: List[List[float]] = []
        for rp in reduced:
            fp = list(mid_q)
            for i, j in enumerate(sorted_rel):
                if i < len(rp):
                    fp[j] = rp[i]
            full.append(fp)
        return full

    def _evaluate_samples(
        self,
        link_idx: int,
        samples: List[List[float]],
        seg_extremes: List[Dict],
        n_sub: int,
    ) -> None:
        """对采样点求值，更新每个段的极值

        FK 只调用两次（始端+末端），n 个段的端点位置通过线性插值 O(1) 获得。
        """
        if not samples:
            return

        t_vals = self._seg_t_values(n_sub)

        if hasattr(self.robot, "get_link_positions_batch"):
            sample_arr = np.asarray(samples, dtype=np.float64)
            pos_end_all = self.robot.get_link_positions_batch(sample_arr, link_idx)
            if link_idx > 1:
                pos_start_all = self.robot.get_link_positions_batch(sample_arr, link_idx - 1)
            else:
                pos_start_all = np.zeros_like(pos_end_all)

            for i in range(sample_arr.shape[0]):
                fp = sample_arr[i]
                pos_end = pos_end_all[i]
                pos_start = pos_start_all[i]
                for k, (t0, t1) in enumerate(t_vals):
                    p0 = (1.0 - t0) * pos_start + t0 * pos_end
                    p1 = (1.0 - t1) * pos_start + t1 * pos_end
                    _update_extremes(seg_extremes[k], p0, fp)
                    _update_extremes(seg_extremes[k], p1, fp)
            return

        for fp in samples:
            pos_end = self.robot.get_link_position(fp, link_idx)
            pos_start = (self.robot.get_link_position(fp, link_idx - 1)
                         if link_idx > 1 else np.zeros(3))
            for k, (t0, t1) in enumerate(t_vals):
                p0 = (1.0 - t0) * pos_start + t0 * pos_end
                p1 = (1.0 - t1) * pos_start + t1 * pos_end
                _update_extremes(seg_extremes[k], p0, fp)
                _update_extremes(seg_extremes[k], p1, fp)

    # ------------------------------------------------------------------
    #  AABB 构建
    # ------------------------------------------------------------------

    def _build_link_aabbs(
        self,
        link_idx: int,
        seg_extremes: List[Dict],
        n_sub: int,
        intervals: List[Tuple[float, float]],
        relevant: Set[int],
    ) -> List[LinkAABBInfo]:
        """从 n 个段极值构建 n 个 LinkAABBInfo"""
        return [self._build_segment_aabb(link_idx, k, n_sub, seg_extremes[k],
                                         intervals, relevant)
                for k in range(n_sub)]

    def _build_segment_aabb(
        self,
        link_idx: int, seg_idx: int, n_sub: int,
        seg_ext: Dict,
        intervals: List[Tuple[float, float]],
        relevant: Set[int],
    ) -> LinkAABBInfo:
        """从单个段的极值字典构建 LinkAABBInfo"""
        t0, t1 = seg_idx / n_sub, (seg_idx + 1) / n_sub

        final_min = [seg_ext[f'{a}_min'][0] for a in 'xyz']
        final_max = [seg_ext[f'{a}_max'][0] for a in 'xyz']

        bc: Dict[str, BoundaryConfig] = {}
        for bt in BOUNDARY_TYPES:
            val, pt = seg_ext[bt]
            if pt is None:
                continue
            bj = {i for i in relevant
                  if i < len(intervals) and i < len(pt)
                  and (abs(pt[i] - intervals[i][0]) < 1e-6
                       or abs(pt[i] - intervals[i][1]) < 1e-6)}
            cfg = BoundaryConfig(
                joint_values=np.array(pt, dtype=np.float64),
                boundary_value=val, boundary_type=bt,
                link_index=link_idx, relevant_joints=relevant,
                boundary_joints=bj)
            cfg.angle_constraints = detect_angle_constraints(
                pt,
                coupled_pairs=self.robot.coupled_pairs,
                coupled_triples=self.robot.coupled_triples,
            )
            cfg.is_aabb_vertex = sum(
                1 for _, (_, p2) in seg_ext.items()
                if p2 is not None and points_equal(pt, p2)) >= 2
            bc[bt] = cfg

        return LinkAABBInfo(
            link_index=link_idx,
            link_name=self._seg_name(link_idx, seg_idx, n_sub),
            min_point=final_min, max_point=final_max,
            boundary_configs=bc,
            segment_index=seg_idx, n_segments=n_sub,
            t_start=t0, t_end=t1)

    # ==================================================================
    #  关键点生成（策略 1-6）
    # ==================================================================

    @staticmethod
    def generate_critical_points(
        intervals: List[Tuple[float, float]],
        coupled_pairs: Optional[List[Tuple[int, ...]]] = None,
        coupled_triples: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """生成关键点（梯度为零的候选配置）

        Args:
            intervals: 各相关关节的区间
            coupled_pairs: 耦合关节对列表（默认使用 Panda 配置）
            coupled_triples: 耦合关节三元组列表（默认使用 Panda 配置）

        Returns:
            (all_points, constraint_points)
        """
        n = len(intervals)
        half_pi = math.pi / 2

        # 每个关节的关键值 = 边界 ∪ kπ/2 落在区间内的值
        key_values: List[List[float]] = []
        for lo, hi in intervals:
            vals = {lo, hi}
            for k in range(int(math.ceil(lo / half_pi)),
                           int(math.floor(hi / half_pi)) + 1):
                v = k * half_pi
                if lo <= v <= hi:
                    vals.add(v)
            key_values.append(sorted(vals))

        points: Set[Tuple[float, ...]] = set()
        cpoints: Set[Tuple[float, ...]] = set()
        defaults = [(lo + hi) / 2 for lo, hi in intervals]

        # 策略 1: 边界组合
        for combo in itertools.product(*[(lo, hi) for lo, hi in intervals]):
            points.add(combo)

        # 策略 2: 单关节关键值
        for i in range(n):
            for v in key_values[i]:
                pt = list(defaults)
                pt[i] = v
                points.add(tuple(pt))

        # 策略 3: 两关节和约束 (任意对)
        for i in range(n):
            for vi in key_values[i]:
                for j in range(i + 1, n):
                    lo_j, hi_j = intervals[j]
                    for k in range(-4, 5):
                        qj = k * half_pi - vi
                        if lo_j <= qj <= hi_j:
                            for bg in ([intervals[x][0] for x in range(n)],
                                       [intervals[x][1] for x in range(n)]):
                                bg[i], bg[j] = vi, qj
                                t = tuple(bg)
                                points.add(t)
                                cpoints.add(t)

        # 策略 4: 特定耦合关节对
        if coupled_pairs is None:
            coupled_pairs = []
        for pi_idx, pj_idx in coupled_pairs:
            if pi_idx >= n or pj_idx >= n:
                continue
            lo_j, hi_j = intervals[pj_idx]
            for vi in key_values[pi_idx]:
                for k in range(-4, 5):
                    qj = k * half_pi - vi
                    if not (lo_j <= qj <= hi_j):
                        continue
                    for bg_type in ('low', 'high', 'mid'):
                        if bg_type == 'low':
                            pt = [intervals[x][0] for x in range(n)]
                        elif bg_type == 'high':
                            pt = [intervals[x][1] for x in range(n)]
                        else:
                            pt = list(defaults)
                        pt[pi_idx], pt[pj_idx] = vi, qj
                        t = tuple(pt)
                        points.add(t)
                        cpoints.add(t)

        # 策略 5-6: 三关节组合
        if coupled_triples is None:
            coupled_triples = []
        for triple in coupled_triples:
            if all(idx < n for idx in triple):
                _add_triple_constraint(
                    points, cpoints, triple,
                    intervals, key_values, defaults, n, half_pi)

        return [list(p) for p in points], [list(p) for p in cpoints]

    # ==================================================================
    #  约束流形随机采样（策略 7）
    # ==================================================================

    @staticmethod
    def generate_manifold_random(
        intervals: List[Tuple[float, float]],
        n_per: int = 7,
        coupled_triples: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[List[float]]:
        """在约束流形上随机采样

        Args:
            intervals: 各相关关节的区间
            n_per: 每个有效约束值生成的样本数
            coupled_triples: 耦合关节三元组列表

        Returns:
            采样点列表
        """
        n = len(intervals)
        if n < 2:
            return []
        half_pi = math.pi / 2
        samples: List[List[float]] = []

        # 两关节约束流形
        for i in range(n):
            lo_i, hi_i = intervals[i]
            for j in range(i + 1, n):
                lo_j, hi_j = intervals[j]
                for k in range(-4, 5):
                    tgt = k * half_pi
                    qi_lo = max(lo_i, tgt - hi_j)
                    qi_hi = min(hi_i, tgt - lo_j)
                    if qi_lo > qi_hi:
                        continue
                    for _ in range(n_per):
                        qi = random.uniform(qi_lo, qi_hi)
                        pt = [random.uniform(lo, hi) for lo, hi in intervals]
                        pt[i], pt[j] = qi, tgt - qi
                        samples.append(pt)

        # 三关节约束流形
        if coupled_triples is None:
            coupled_triples = []
        for a, b, c in coupled_triples:
            if a >= n or b >= n or c >= n:
                continue
            lo_a, hi_a = intervals[a]
            lo_b, hi_b = intervals[b]
            lo_c, hi_c = intervals[c]
            for k in range(-5, 6):
                tgt = k * half_pi
                for _ in range(n_per):
                    qa = random.uniform(lo_a, hi_a)
                    qb = random.uniform(lo_b, hi_b)
                    qc = tgt - qa - qb
                    if lo_c <= qc <= hi_c:
                        pt = [random.uniform(lo, hi) for lo, hi in intervals]
                        pt[a], pt[b], pt[c] = qa, qb, qc
                        samples.append(pt)
        return samples

    # ==================================================================
    #  随机采样（避开关键点邻域）
    # ==================================================================

    @staticmethod
    def random_avoiding_critical(
        intervals: List[Tuple[float, float]],
        n_samples: int,
        critical: List[List[float]],
        threshold: float = 0.05,
    ) -> List[List[float]]:
        """生成随机采样点，避开关键点邻域

        Args:
            intervals: 各关节区间
            n_samples: 目标采样数
            critical: 需要避开的关键点列表
            threshold: 避开半径

        Returns:
            采样点列表
        """
        if threshold <= 0 or not critical:
            return [[random.uniform(lo, hi) for lo, hi in intervals]
                    for _ in range(n_samples)]

        crit_arr = np.array(critical)
        result: List[List[float]] = []
        max_att = n_samples * 3
        att = 0
        while len(result) < n_samples and att < max_att:
            s = [random.uniform(lo, hi) for lo, hi in intervals]
            if np.linalg.norm(crit_arr - np.array(s), axis=1).min() > threshold:
                result.append(s)
            att += 1
        return result


# ==================== 模块级辅助函数 ====================

def _add_triple_constraint(
    points: set, cpoints: set,
    triple: Tuple[int, int, int],
    intervals: List[Tuple[float, float]],
    key_values: List[List[float]],
    defaults: List[float], n: int, half_pi: float,
) -> None:
    """为三关节和约束添加关键点"""
    a, b, c = triple
    bg_joints = [j for j in range(n) if j not in triple]
    bg_opts = [[intervals[j][0], intervals[j][1]] for j in bg_joints]
    bg_combos = list(itertools.product(*bg_opts)) if bg_opts else [()]

    for i1, i2, i_s in [(a, b, c), (a, c, b), (b, c, a)]:
        for v1 in key_values[i1]:
            for v2 in key_values[i2]:
                for k in range(-5, 6):
                    vs = k * half_pi - v1 - v2
                    lo_s, hi_s = intervals[i_s]
                    if not (lo_s <= vs <= hi_s):
                        continue
                    for bg in bg_combos:
                        pt = list(defaults)
                        for bi, bj in enumerate(bg_joints):
                            pt[bj] = bg[bi]
                        pt[i1], pt[i2], pt[i_s] = v1, v2, vs
                        t = tuple(pt)
                        points.add(t)
                        cpoints.add(t)


def detect_angle_constraints(
    point: List[float],
    coupled_pairs: Optional[List[Tuple[int, ...]]] = None,
    coupled_triples: Optional[List[Tuple[int, int, int]]] = None,
) -> List[str]:
    """检测关节角度组合约束

    根据机器人的耦合关节配置，检测当前关节角是否满足约束条件。

    Args:
        point: 完整关节值列表
        coupled_pairs: 耦合关节对列表，如 [(0,2), (1,3)]
        coupled_triples: 耦合关节三元组列表，如 [(0,2,4), (1,3,5)]

    Returns:
        检测到的约束描述字符串列表
    """
    pi = math.pi
    cs: List[str] = []

    def _check(s: float, label: str, k_range: range) -> None:
        for k in k_range:
            if abs(s - k * pi) < 0.1:
                cs.append(f"{label}≈{k}π" if k != 0 else f"{label}≈0")
                return
            if abs(s - k * pi / 2) < 0.1 and k % 2 != 0:
                sign = "-" if k < 0 else ""
                cs.append(f"{label}≈{sign}π/2" if abs(k) == 1
                          else f"{label}≈{k}π/2")
                return

    # 耦合对约束
    for pair in (coupled_pairs or []):
        if all(idx < len(point) for idx in pair):
            s = sum(point[idx] for idx in pair)
            label = '+'.join(f'q{idx}' for idx in pair)
            _check(s, label, range(-2, 3))

    # 耦合三元组约束
    for triple in (coupled_triples or []):
        if all(idx < len(point) for idx in triple):
            s = sum(point[idx] for idx in triple)
            label = '+'.join(f'q{idx}' for idx in triple)
            _check(s, label, range(-3, 4))

    return cs
