"""
optimization.py - L-BFGS-B 局部优化

从 aabb_calculator.py 中提取，负责对每个连杆的各段极值进行
scipy L-BFGS-B 局部优化。

策略：
  - 对整根连杆的始端/末端分别优化，然后将结果更新到所有段
  - 预筛选种子：top-1 (exploit) + 最远 1 个 (explore)
  - 减少 minimize 调用次数，保持多样性
"""

import logging
from typing import List, Dict, Set, Optional, Tuple

import numpy as np

from .robot import Robot
from .models import BOUNDARY_TYPES, DIM_MAP

logger = logging.getLogger(__name__)


def optimize_extremes(
    robot: Robot,
    link_idx: int,
    sorted_rel: List[int],
    mid_q: List[float],
    intervals: List[Tuple[float, float]],
    seg_extremes: List[Dict],
    n_sub: int,
    extra_seeds: Optional[List[List[float]]] = None,
    n_seeds: int = 2,
    skip_current_best: bool = False,
) -> None:
    """对每个段独立做 L-BFGS-B 优化

    对整根连杆的始端/末端分别优化（同之前），
    然后将优化结果更新到所有段。因为线性插值下，
    始端/末端极值确定后，各段端点极值也确定了。

    预筛选：对每个 eval_link 做一次批量 FK 拿到所有 extra_seeds
    的位置，然后选种子：top-1（exploit）+ 距离最远的 1 个（explore），
    在保持多样性的同时大幅减少 minimize 调用。

    Args:
        robot: 机器人模型
        link_idx: 目标连杆索引 (1-based)
        sorted_rel: 排序后的相关关节索引列表
        mid_q: 各关节区间中点
        intervals: 关节区间列表
        seg_extremes: 各段极值字典列表
        n_sub: 等分段数
        extra_seeds: 额外的优化种子点
        n_seeds: 每个边界方向选取的种子数
        skip_current_best: 是否跳过当前最优点作为种子
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        logger.debug("scipy 未安装，跳过局部优化")
        return

    bounds = [(intervals[j] if j < len(intervals) else (0, 0))
              for j in sorted_rel]

    for eval_link in [link_idx, link_idx - 1]:
        if eval_link < 1:
            continue

        # 批量预计算所有 extra_seeds 在此 eval_link 下的位置
        seed_pos: Optional[np.ndarray] = None
        seed_qr: Optional[np.ndarray] = None
        if extra_seeds:
            seed_pos = np.array([
                robot.get_link_position(sp, eval_link)
                for sp in extra_seeds])
            seed_qr = np.array([
                [sp[j] for j in sorted_rel]
                for sp in extra_seeds])

        for bt in BOUNDARY_TYPES:
            dim = DIM_MAP[bt[0]]
            is_min = bt.endswith('min')

            # 收集种子
            seeds: List[List[float]] = []
            if not skip_current_best:
                for seg_ext in seg_extremes:
                    best = seg_ext[bt][1]
                    if best is not None and not _points_equal_any(best, seeds):
                        seeds.append(best)

            # 预筛选：top-1 (exploit) + 最远 1 个 (explore)
            if seed_pos is not None and extra_seeds:
                scores = seed_pos[:, dim]
                order = np.argsort(scores) if is_min else np.argsort(-scores)

                # exploit: 目标值最好的种子
                best_idx = int(order[0])
                sp0 = extra_seeds[best_idx]
                if not _points_equal_any(sp0, seeds):
                    seeds.append(sp0)

                # explore: reduced-space 中离 exploit 种子最远的种子
                if n_seeds >= 2 and len(order) > 1:
                    ref = seed_qr[best_idx]
                    dists = np.linalg.norm(seed_qr - ref, axis=1)
                    dists[best_idx] = -1
                    far_idx = int(np.argmax(dists))
                    sp1 = extra_seeds[far_idx]
                    if not _points_equal_any(sp1, seeds):
                        seeds.append(sp1)

            if not seeds:
                continue

            def _obj(qr: np.ndarray, _d: int = dim, _l: int = eval_link,
                     _sr: List[int] = sorted_rel, _mq: List[float] = mid_q,
                     _im: bool = is_min) -> float:
                f = list(_mq)
                for ii, jj in enumerate(_sr):
                    f[jj] = qr[ii]
                v = float(robot.get_link_position(f, _l)[_d])
                return v if _im else -v

            for seed in seeds:
                x0 = [seed[j] for j in sorted_rel]
                try:
                    res = minimize(_obj, x0, method='L-BFGS-B',
                                  bounds=bounds,
                                  options={'maxiter': 30, 'ftol': 1e-8})
                    if res.success:
                        opt = list(mid_q)
                        for ii, jj in enumerate(sorted_rel):
                            opt[jj] = res.x[ii]
                        _update_segs_for_point(
                            robot, link_idx, opt, seg_extremes, n_sub)
                except Exception as e:
                    logger.debug("优化失败 link=%d bt=%s: %s",
                                 link_idx, bt, e)


# ==================== 辅助函数 ====================

def _update_segs_for_point(
    robot: Robot,
    link_idx: int,
    full_point: List[float],
    seg_extremes: List[Dict],
    n_sub: int,
) -> None:
    """为单个点更新所有段的极值"""
    pos_end = robot.get_link_position(full_point, link_idx)
    pos_start = (robot.get_link_position(full_point, link_idx - 1)
                 if link_idx > 1 else np.zeros(3))
    fp = list(full_point)
    for k in range(n_sub):
        t0, t1 = k / n_sub, (k + 1) / n_sub
        p0 = (1.0 - t0) * pos_start + t0 * pos_end
        p1 = (1.0 - t1) * pos_start + t1 * pos_end
        _update_extremes(seg_extremes[k], p0, fp)
        _update_extremes(seg_extremes[k], p1, fp)


def _update_extremes(extremes: Dict, pos: np.ndarray,
                     full_point: List[float]) -> None:
    """更新极值字典"""
    pt = list(full_point)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if x < extremes['x_min'][0]: extremes['x_min'] = (x, pt)
    if x > extremes['x_max'][0]: extremes['x_max'] = (x, pt)
    if y < extremes['y_min'][0]: extremes['y_min'] = (y, pt)
    if y > extremes['y_max'][0]: extremes['y_max'] = (y, pt)
    if z < extremes['z_min'][0]: extremes['z_min'] = (z, pt)
    if z > extremes['z_max'][0]: extremes['z_max'] = (z, pt)


def _points_equal_any(p: List[float], others: List[List[float]],
                      tol: float = 1e-6) -> bool:
    """检查点是否与列表中某个点近似相等"""
    return any(points_equal(p, o, tol) for o in others)


def points_equal(p1: List[float], p2: List[float],
                 tol: float = 1e-6) -> bool:
    """检查两个点是否近似相等"""
    if len(p1) != len(p2):
        return False
    return all(abs(a - b) < tol for a, b in zip(p1, p2))
