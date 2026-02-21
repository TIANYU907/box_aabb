"""
forest/deoverlap.py - 邻接检测与共享面计算

为一组 axis-aligned hyperrectangle (BoxNode) 计算邻接关系和共享面。

核心算法：
- compute_adjacency: 向量化 O(N²·D) 全量邻接检测
- compute_adjacency_incremental: O(K·N·D) 增量邻接更新
- shared_face / shared_face_center: 相邻 box 共享面提取

设计决策：
- 邻接条件：恰好一个维度面相接，其余维度投影有正面积重叠
- HierAABBTree 的占用跟踪保证 box 不重叠，无需 deoverlap 安全网
"""

import logging
from typing import List, Tuple, Dict, Set, Optional

import numpy as np

from .models import BoxNode

logger = logging.getLogger(__name__)

# 类型别名：区间列表
Intervals = List[Tuple[float, float]]



def compute_adjacency(
    boxes: List[BoxNode],
    tol: float = 1e-8,
    chunk_threshold: int = 300,
    chunk_size: int = 64,
    period: Optional[float] = None,
) -> Dict[int, Set[int]]:
    """计算所有 box 的邻接关系（全向量化 + 上三角分块）

    邻接条件：
    - 恰好一个维度 d 满足面相接：|A.hi[d] - B.lo[d]| < tol
      或 |A.lo[d] - B.hi[d]| < tol
    - 其余所有维度的投影有正面积重叠：
      min(A.hi[k], B.hi[k]) - max(A.lo[k], B.lo[k]) > tol

    微小重叠也视为邻接（重叠部分等效于接触面有 ε 厚度）。

    Args:
        boxes: BoxNode 列表
        tol: 距离容差
        chunk_threshold: N 不超过阈值时走全矩阵向量化
        chunk_size: 分块大小（仅在分块模式生效）

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

    lo = intervals_arr[:, :, 0]  # (N, D)
    hi = intervals_arr[:, :, 1]  # (N, D)
    ids_arr = np.array([b.node_id for b in boxes], dtype=np.int64)
    adj_mat = np.zeros((n, n), dtype=bool)

    if n <= chunk_threshold:
        # 小规模：一次性全矩阵 (N, N, D)
        overlap_width = np.minimum(hi[:, None, :], hi[None, :, :]) - \
            np.maximum(lo[:, None, :], lo[None, :, :])

        # 周期边界：取直接 / 左移 / 右移中的最大 overlap_width
        if period is not None:
            p = period
            ow_right = np.minimum(hi[:, None, :], hi[None, :, :] + p) - \
                np.maximum(lo[:, None, :], lo[None, :, :] + p)
            ow_left = np.minimum(hi[:, None, :], hi[None, :, :] - p) - \
                np.maximum(lo[:, None, :], lo[None, :, :] - p)
            overlap_width = np.maximum(overlap_width, np.maximum(ow_right, ow_left))

        separated = overlap_width < -tol
        touching = (overlap_width >= -tol) & (overlap_width <= tol)
        overlapping = overlap_width > tol

        any_separated = np.any(separated, axis=2)
        n_touching = np.sum(touching, axis=2)
        n_overlapping = np.sum(overlapping, axis=2)

        is_adjacent = (~any_separated) & (n_touching >= 1) & (n_overlapping >= n_dims - 1)
        tri = np.triu(np.ones((n, n), dtype=bool), k=1)
        adj_mat |= (is_adjacent & tri)
    else:
        # 大规模：仅计算上三角块，减少重复比较
        step = max(1, int(chunk_size))
        for i0 in range(0, n, step):
            i1 = min(n, i0 + step)
            lo_i = lo[i0:i1]  # (Bi, D)
            hi_i = hi[i0:i1]  # (Bi, D)

            for j0 in range(i0, n, step):
                j1 = min(n, j0 + step)
                lo_j = lo[j0:j1]  # (Bj, D)
                hi_j = hi[j0:j1]  # (Bj, D)

                overlap_width = np.minimum(hi_i[:, None, :], hi_j[None, :, :]) - \
                    np.maximum(lo_i[:, None, :], lo_j[None, :, :])

                # 周期边界
                if period is not None:
                    p = period
                    ow_right = np.minimum(hi_i[:, None, :], hi_j[None, :, :] + p) - \
                        np.maximum(lo_i[:, None, :], lo_j[None, :, :] + p)
                    ow_left = np.minimum(hi_i[:, None, :], hi_j[None, :, :] - p) - \
                        np.maximum(lo_i[:, None, :], lo_j[None, :, :] - p)
                    overlap_width = np.maximum(overlap_width, np.maximum(ow_right, ow_left))

                separated = overlap_width < -tol
                touching = (overlap_width >= -tol) & (overlap_width <= tol)
                overlapping = overlap_width > tol

                any_separated = np.any(separated, axis=2)
                n_touching = np.sum(touching, axis=2)
                n_overlapping = np.sum(overlapping, axis=2)

                block_adj = (~any_separated) & (n_touching >= 1) & (n_overlapping >= n_dims - 1)
                if i0 == j0:
                    block_adj = np.triu(block_adj, k=1)

                if np.any(block_adj):
                    adj_mat[i0:i1, j0:j1] |= block_adj

    adj_mat |= adj_mat.T
    for i in range(n):
        neighbors = np.flatnonzero(adj_mat[i])
        if neighbors.size:
            adj[int(ids_arr[i])] = set(ids_arr[neighbors].tolist())

    n_edges = int(np.sum(adj_mat) // 2)
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
