"""
interval_fk.py - 区间正运动学

使用区间/仿射算术计算保守的 AABB 包围盒。
从 aabb_calculator.py 中提取。
"""

import math
import logging
from typing import List, Tuple, Set, Optional

from .robot import Robot
from .models import LinkAABBInfo
from .interval_math import (
    AffineForm, Interval, smart_sin, smart_cos, reset_affine_noise_counter
)

logger = logging.getLogger(__name__)


def compute_interval_aabb(
    robot: Robot,
    intervals: List[Tuple[float, float]],
    zero_length_links: Set[int],
    skip_zero_length: bool = True,
    n_sub: int = 1,
) -> Tuple[List[LinkAABBInfo], int]:
    """使用区间/仿射算术计算各连杆 AABB

    Args:
        robot: 机器人模型
        intervals: 关节区间列表
        zero_length_links: 零长度连杆集合
        skip_zero_length: 是否跳过零长度连杆
        n_sub: 等分段数

    Returns:
        (link_aabbs, 0)  — 区间法无采样点
    """
    reset_affine_noise_counter()

    # 将关节区间转换为 AffineForm
    q_afs = []
    for lo, hi in intervals:
        q_afs.append(float(lo) if lo == hi
                     else AffineForm.from_interval(lo, hi))

    # 区间正运动学
    T = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    transforms = [[row[:] for row in T]]

    for i, param in enumerate(robot.dh_params):
        alpha, a = param['alpha'], param['a']
        if param['type'] == 'revolute':
            d = param['d']
            theta = (q_afs[i] + param['theta']
                     if isinstance(q_afs[i], AffineForm)
                     else q_afs[i] + param['theta'])
        else:
            d = (q_afs[i] + param['d']
                 if isinstance(q_afs[i], AffineForm)
                 else q_afs[i] + param['d'])
            theta = param['theta']

        def _cos(x):
            return smart_cos(x) if isinstance(x, AffineForm) else math.cos(x)

        def _sin(x):
            return smart_sin(x) if isinstance(x, AffineForm) else math.sin(x)

        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = _cos(theta), _sin(theta)

        A = [
            [ct, st * -1.0, 0.0, a],
            [st * ca, ct * ca, -sa,
             d * -1.0 * sa if isinstance(d, AffineForm) else -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
        T = _mat_mul_interval(T, A)
        transforms.append([row[:] for row in T])

    # 构建 LinkAABBInfo
    link_aabbs: List[LinkAABBInfo] = []
    for li in range(1, len(transforms)):
        is_zl = skip_zero_length and li in zero_length_links
        T_end = transforms[li]
        T_start = transforms[li - 1] if li > 1 else transforms[0]

        start_bds = [_bounds(T_start[d][3]) for d in range(3)]
        end_bds = [_bounds(T_end[d][3]) for d in range(3)]

        if is_zl or n_sub <= 1:
            mins = [min(start_bds[d][0], end_bds[d][0]) for d in range(3)]
            maxs = [max(start_bds[d][1], end_bds[d][1]) for d in range(3)]
            link_aabbs.append(LinkAABBInfo(
                li, _link_name(li), mins, maxs, is_zero_length=is_zl))
        else:
            for k in range(n_sub):
                t0, t1 = k / n_sub, (k + 1) / n_sub
                seg_min, seg_max = [], []
                for d in range(3):
                    mn_s, mx_s = start_bds[d]
                    mn_e, mx_e = end_bds[d]
                    lo0 = (1 - t0) * mn_s + t0 * mn_e
                    hi0 = (1 - t0) * mx_s + t0 * mx_e
                    lo1 = (1 - t1) * mn_s + t1 * mn_e
                    hi1 = (1 - t1) * mx_s + t1 * mx_e
                    seg_min.append(min(lo0, lo1))
                    seg_max.append(max(hi0, hi1))
                link_aabbs.append(LinkAABBInfo(
                    li, _seg_name(li, k, n_sub), seg_min, seg_max,
                    segment_index=k, n_segments=n_sub,
                    t_start=t0, t_end=t1))

    return link_aabbs, 0


# ==================== 辅助函数 ====================

def _bounds(v) -> Tuple[float, float]:
    """从区间值或标量获取 (min, max)"""
    if hasattr(v, 'min') and hasattr(v, 'max'):
        return v.min, v.max
    return float(v), float(v)


def _mat_mul_interval(A, B):
    """区间/仿射 4×4 矩阵乘法"""
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            v = 0.0
            for k in range(4):
                v = v + A[i][k] * B[k][j]
            C[i][j] = v
    return C


def _link_name(link_idx: int) -> str:
    return f"Link {link_idx} (Joint {link_idx - 1})"


def _seg_name(link_idx: int, seg_idx: int, n_sub: int) -> str:
    if n_sub <= 1:
        return _link_name(link_idx)
    return f"Link {link_idx} Seg {seg_idx}"
