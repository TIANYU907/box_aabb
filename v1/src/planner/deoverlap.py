"""
planner/deoverlap.py - Box 去重叠与邻接检测

将一组可能有重叠的 axis-aligned hyperrectangle (BoxNode) 转化为
互不重叠（或微小重叠容忍）的碎片集，同时计算邻接关系。

核心算法：
- subtract_box: 超矩形减法 —— 沿重叠区域逐维切割，最多产生 2D 碎片
- deoverlap: 按生成顺序处理，先来先得，后来的被切分
- compute_adjacency: 向量化 O(N²·D) 全量邻接检测
- compute_adjacency_incremental: O(K·N·D) 增量邻接更新

设计决策：
- 当前 HierAABBTree 保证输出 box 不重叠，deoverlap 仅作为安全网保留
- 邻接条件：恰好一个维度面相接，其余维度投影有正面积重叠
"""

import logging
from typing import List, Tuple, Dict, Set, Optional

import numpy as np

from .models import BoxNode

logger = logging.getLogger(__name__)

# 类型别名：区间列表
Intervals = List[Tuple[float, float]]


def subtract_box(
    base_intervals: Intervals,
    cut_intervals: Intervals,
) -> List[Intervals]:
    """超矩形减法：base - cut

    从 base 中移除与 cut 的交集区域，返回剩余碎片列表。
    逐维度沿 cut 的边界切割 base，每维最多产生 2 个碎片
    （左侧和右侧），核心区域（与 cut 完全重叠的部分）被丢弃。

    算法：
    1. 对每个维度 d，检查 cut 是否在该维度"切入" base
    2. 若 cut.lo[d] > base.lo[d]：切出左碎片 [base.lo[d], cut.lo[d]]
    3. 若 cut.hi[d] < base.hi[d]：切出右碎片 [cut.hi[d], base.hi[d]]
    4. 将 base 在维度 d 收缩到 [max(base.lo, cut.lo), min(base.hi, cut.hi)]
    5. 继续处理下一维度

    Args:
        base_intervals: 被减的超矩形 [(lo_0, hi_0), ...]
        cut_intervals: 减去的超矩形

    Returns:
        碎片列表（每个碎片是 intervals 列表），可能为空（完全被切除）
    """
    n_dims = len(base_intervals)
    fragments: List[Intervals] = []

    # 首先检查是否有交集
    for d in range(n_dims):
        b_lo, b_hi = base_intervals[d]
        c_lo, c_hi = cut_intervals[d]
        if b_hi <= c_lo or c_hi <= b_lo:
            # 无交集，base 完整保留
            return [list(base_intervals)]

    # 逐维切割
    current = list(base_intervals)  # 逐步收缩的"中间区域"

    for d in range(n_dims):
        b_lo, b_hi = current[d]
        c_lo, c_hi = cut_intervals[d]

        # 左碎片：base 在维度 d 的 [b_lo, c_lo] 部分
        if c_lo > b_lo:
            frag = list(current)
            frag[d] = (b_lo, c_lo)
            fragments.append(frag)

        # 右碎片：base 在维度 d 的 [c_hi, b_hi] 部分
        if c_hi < b_hi:
            frag = list(current)
            frag[d] = (c_hi, b_hi)
            fragments.append(frag)

        # 收缩 current 到交集区域（继续处理后续维度）
        current[d] = (max(b_lo, c_lo), min(b_hi, c_hi))

    # current 最终是 base ∩ cut 的交集，被丢弃
    return fragments


def _interval_volume(intervals: Intervals) -> float:
    """计算超矩形体积"""
    vol = 1.0
    has_nonzero = False
    for lo, hi in intervals:
        w = hi - lo
        if w > 0:
            vol *= w
            has_nonzero = True
    return vol if has_nonzero else 0.0


def deoverlap(
    boxes: List[BoxNode],
    id_start: int = 0,
) -> List[BoxNode]:
    """将一组 box 去重叠，先来先得

    按 boxes 列表顺序处理：先加入的 box 保持完整，后来的 box 被
    已有 box 切分。

    注意：当前 HierAABBTree 已保证输出 box 不重叠，此函数仅作为
    安全网保留。

    Args:
        boxes: 输入 box 列表（按优先级排序，靠前优先）
        id_start: 新碎片 node_id 起始值

    Returns:
        去重叠后的 BoxNode 列表
    """
    if not boxes:
        return []

    committed: List[BoxNode] = []
    next_id = id_start

    for box in boxes:
        # 当前 box 的区间碎片（初始为整个 box）
        fragments: List[Intervals] = [list(box.joint_intervals)]

        # 对每个已提交的 box，切分当前碎片
        for committed_box in committed:
            new_fragments: List[Intervals] = []
            for frag in fragments:
                ovlp_vol = _overlap_volume_intervals(frag, list(committed_box.joint_intervals))
                if ovlp_vol <= 0:
                    # 无重叠，直接保留
                    new_fragments.append(frag)
                else:
                    # 需要切分
                    pieces = subtract_box(frag, list(committed_box.joint_intervals))
                    new_fragments.extend(pieces)
            fragments = new_fragments

        # 将存活碎片转为 BoxNode
        for frag in fragments:
            vol = _interval_volume(frag)
            if vol <= 0:
                continue
            new_box = BoxNode(
                node_id=next_id,
                joint_intervals=frag,
                seed_config=box.seed_config.copy(),
                parent_id=box.node_id,  # 追溯原始 box
                volume=vol,
                tree_id=box.tree_id,
            )
            committed.append(new_box)
            next_id += 1

    logger.info(
        "deoverlap: %d 输入 box → %d 无重叠碎片",
        len(boxes), len(committed),
    )
    return committed


def _overlap_volume_intervals(a: Intervals, b: Intervals) -> float:
    """计算两组区间的重叠体积"""
    vol = 1.0
    for (a_lo, a_hi), (b_lo, b_hi) in zip(a, b):
        lo = max(a_lo, b_lo)
        hi = min(a_hi, b_hi)
        if lo >= hi:
            return 0.0
        vol *= (hi - lo)
    return vol


def compute_adjacency(
    boxes: List[BoxNode],
    tol: float = 1e-8,
) -> Dict[int, Set[int]]:
    """计算所有 box 的邻接关系（向量化 O(N²·D)）

    邻接条件：
    - 恰好一个维度 d 满足面相接：|A.hi[d] - B.lo[d]| < tol
      或 |A.lo[d] - B.hi[d]| < tol
    - 其余所有维度的投影有正面积重叠：
      min(A.hi[k], B.hi[k]) - max(A.lo[k], B.lo[k]) > tol

    微小重叠也视为邻接（重叠部分等效于接触面有 ε 厚度）。

    Args:
        boxes: BoxNode 列表
        tol: 距离容差

    Returns:
        双向邻接表 {box_id: set of adjacent box_ids}
    """
    n = len(boxes)
    if n == 0:
        return {}

    n_dims = boxes[0].n_dims
    adj: Dict[int, Set[int]] = {b.node_id: set() for b in boxes}

    if n < 2:
        return adj

    # 构建 (N, D, 2) 数组
    intervals_arr = np.empty((n, n_dims, 2), dtype=np.float64)
    for i, box in enumerate(boxes):
        for d, (lo, hi) in enumerate(box.joint_intervals):
            intervals_arr[i, d, 0] = lo
            intervals_arr[i, d, 1] = hi

    # 向量化邻接检测：所有 (i,j) 对
    # lo: (N, D), hi: (N, D)
    lo = intervals_arr[:, :, 0]  # (N, D)
    hi = intervals_arr[:, :, 1]  # (N, D)

    for i in range(n):
        # 广播 i 与 [i+1:] 的比较
        remaining = n - i - 1
        if remaining <= 0:
            break

        # i 的区间广播
        i_lo = lo[i]  # (D,)
        i_hi = hi[i]  # (D,)
        j_lo = lo[i + 1:]  # (R, D)
        j_hi = hi[i + 1:]  # (R, D)

        # 每维投影重叠宽度：min(hi_i, hi_j) - max(lo_i, lo_j)
        overlap_width = np.minimum(i_hi, j_hi) - np.maximum(i_lo, j_lo)  # (R, D)

        # 每维的"面相接"判断：|A.hi[d] - B.lo[d]| < tol 或 |A.lo[d] - B.hi[d]| < tol
        # 关键：面相接时 overlap_width≈0，而其余维度 overlap_width > 0
        # 但微小重叠时 overlap_width 可能是个小正数

        # 方法：对每维判断是否"相接或微小重叠"
        # touching: -tol < overlap_width <= tol (面相接)
        # overlapping: overlap_width > tol (投影重叠)
        # separated: overlap_width < -tol (分离)

        separated = overlap_width < -tol  # (R, D)
        touching = (overlap_width >= -tol) & (overlap_width <= tol)  # (R, D)
        overlapping = overlap_width > tol  # (R, D)

        # 任何维度分离 → 不邻接
        any_separated = np.any(separated, axis=1)  # (R,)

        # 恰好一个维度 touching，其余都 overlapping（或 touching）
        n_touching = np.sum(touching, axis=1)  # (R,)
        n_overlapping = np.sum(overlapping, axis=1)  # (R,)

        # 邻接条件：无分离 && 恰好 1 个维度 touching && 其余 D-1 维度 overlapping
        is_adjacent = (~any_separated) & (n_touching >= 1) & (n_overlapping >= n_dims - 1)

        # 记录邻接
        idx_adj = np.where(is_adjacent)[0]
        for offset in idx_adj:
            j = i + 1 + offset
            adj[boxes[i].node_id].add(boxes[j].node_id)
            adj[boxes[j].node_id].add(boxes[i].node_id)

    n_edges = sum(len(v) for v in adj.values()) // 2
    logger.info("compute_adjacency: %d boxes, %d 条邻接边", n, n_edges)
    return adj


def compute_adjacency_incremental(
    new_boxes: List[BoxNode],
    all_boxes: List[BoxNode],
    existing_adj: Dict[int, Set[int]],
    tol: float = 1e-8,
) -> Dict[int, Set[int]]:
    """增量邻接更新：仅计算 new_boxes 与 all_boxes 之间的邻接

    O(K · N · D)，其中 K = len(new_boxes), N = len(all_boxes)。

    Args:
        new_boxes: 新增的 box 列表
        all_boxes: 全部 box 列表（含 new_boxes）
        existing_adj: 已有邻接表（会被原地修改）
        tol: 容差

    Returns:
        更新后的邻接表（同一对象，原地修改）
    """
    if not new_boxes or not all_boxes:
        return existing_adj

    n_dims = new_boxes[0].n_dims
    new_ids = {b.node_id for b in new_boxes}

    # 为新 box 初始化邻接
    for b in new_boxes:
        if b.node_id not in existing_adj:
            existing_adj[b.node_id] = set()

    # 构建新 box 数组 (K, D, 2)
    K = len(new_boxes)
    new_arr = np.empty((K, n_dims, 2), dtype=np.float64)
    for i, box in enumerate(new_boxes):
        for d, (lo, hi) in enumerate(box.joint_intervals):
            new_arr[i, d, 0] = lo
            new_arr[i, d, 1] = hi

    # 构建全部 box 数组 (N, D, 2)
    N = len(all_boxes)
    all_arr = np.empty((N, n_dims, 2), dtype=np.float64)
    id_map = []
    for i, box in enumerate(all_boxes):
        for d, (lo, hi) in enumerate(box.joint_intervals):
            all_arr[i, d, 0] = lo
            all_arr[i, d, 1] = hi
        id_map.append(box.node_id)

    new_lo = new_arr[:, :, 0]  # (K, D)
    new_hi = new_arr[:, :, 1]  # (K, D)
    all_lo = all_arr[:, :, 0]  # (N, D)
    all_hi = all_arr[:, :, 1]  # (N, D)

    n_new_edges = 0

    for ki in range(K):
        k_lo = new_lo[ki]  # (D,)
        k_hi = new_hi[ki]  # (D,)
        k_id = new_boxes[ki].node_id

        # 广播 vs 全部
        overlap_width = np.minimum(k_hi, all_hi) - np.maximum(k_lo, all_lo)  # (N, D)

        separated = overlap_width < -tol
        touching = (overlap_width >= -tol) & (overlap_width <= tol)
        overlapping = overlap_width > tol

        any_separated = np.any(separated, axis=1)
        n_touching = np.sum(touching, axis=1)
        n_overlapping = np.sum(overlapping, axis=1)

        is_adjacent = (~any_separated) & (n_touching >= 1) & (n_overlapping >= n_dims - 1)

        for j_idx in np.where(is_adjacent)[0]:
            j_id = id_map[j_idx]
            if j_id == k_id:
                continue  # 跳过自身
            if j_id not in existing_adj[k_id]:
                existing_adj[k_id].add(j_id)
                existing_adj.setdefault(j_id, set()).add(k_id)
                n_new_edges += 1

    logger.debug(
        "compute_adjacency_incremental: %d new boxes, %d new edges",
        K, n_new_edges,
    )
    return existing_adj


def shared_face(
    box_a: BoxNode,
    box_b: BoxNode,
    tol: float = 1e-8,
) -> Optional[Tuple[int, float, Intervals]]:
    """计算两个邻接 box 的共享面

    共享面定义：恰好一个维度 d 面相接（或微小重叠），
    其余维度的投影交集构成面的范围。

    Args:
        box_a, box_b: 两个相邻的 BoxNode
        tol: 容差

    Returns:
        (dim, face_value, face_intervals) 或 None
        - dim: 面相接的维度
        - face_value: 面在该维度的坐标
        - face_intervals: 其余维度的交集区间列表（长度 = n_dims）
          其中 face_intervals[dim] = (face_value, face_value)
    """
    n_dims = box_a.n_dims
    contact_dim = None
    contact_val = None

    for d in range(n_dims):
        a_lo, a_hi = box_a.joint_intervals[d]
        b_lo, b_hi = box_b.joint_intervals[d]
        overlap = min(a_hi, b_hi) - max(a_lo, b_lo)

        if overlap < -tol:
            return None  # 分离

        if abs(overlap) <= tol:
            # 面相接
            if contact_dim is not None:
                return None  # 多个维度都相接 → 不是面邻接
            contact_dim = d
            contact_val = (a_hi + b_lo) / 2.0 if a_hi <= b_lo + tol else (b_hi + a_lo) / 2.0

    if contact_dim is None:
        # 所有维度都重叠 → 微小重叠情况
        # 选择重叠最小的维度作为"虚拟接触面"
        min_overlap = float('inf')
        for d in range(n_dims):
            a_lo, a_hi = box_a.joint_intervals[d]
            b_lo, b_hi = box_b.joint_intervals[d]
            overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
            if 0 < overlap < min_overlap:
                min_overlap = overlap
                contact_dim = d
                contact_val = (max(a_lo, b_lo) + min(a_hi, b_hi)) / 2.0

    if contact_dim is None:
        return None

    # 构建面区间
    face_intervals: Intervals = []
    for d in range(n_dims):
        if d == contact_dim:
            face_intervals.append((contact_val, contact_val))
        else:
            a_lo, a_hi = box_a.joint_intervals[d]
            b_lo, b_hi = box_b.joint_intervals[d]
            f_lo = max(a_lo, b_lo)
            f_hi = min(a_hi, b_hi)
            face_intervals.append((f_lo, f_hi))

    return (contact_dim, contact_val, face_intervals)


def shared_face_center(
    box_a: BoxNode,
    box_b: BoxNode,
    tol: float = 1e-8,
) -> Optional[np.ndarray]:
    """计算共享面的中心点

    Args:
        box_a, box_b: 两个相邻的 BoxNode

    Returns:
        共享面中心坐标，或 None
    """
    face = shared_face(box_a, box_b, tol)
    if face is None:
        return None
    _, _, face_intervals = face
    return np.array([(lo + hi) / 2.0 for lo, hi in face_intervals])
