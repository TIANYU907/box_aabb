"""Box Forest Coarsening — 分维贪心扫描合并.

算法 (Dimension-Sweep Greedy)
-----------------------------
对每个维度 d, 把 forest 中所有 box 按 "除 d 外全部 interval" 分组.
同组内的 box 在 dim d 以外完全一致, 仅在 dim d 上可能相邻或有间隔.
按 dim d 排序后, 贪心合并连续 touching 的 box 对 (run-length merge).

合并后的 box = A ∪ B (无间隙), 因此:
  - 无需碰撞检查 (A、B 已知无碰撞, 并集也无碰撞)
  - 不会产生重叠 (新 box 在其他维度与原 box 完全相同)
  - 只需重标 tree 节点的 forest_id

外层迭代: 扫描全部维度 → 一轮无合并则收敛.

复杂度: O(rounds × ndim × (N log N + tree_nodes))
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from .models import BoxNode

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 统计
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoarsenStats:
    n_before: int = 0
    n_after: int = 0
    n_merges: int = 0
    n_rounds: int = 0
    time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 辅助: box → tree nodes 映射 (一个 box 可能占据多个 tree 叶)
# ═══════════════════════════════════════════════════════════════════════════

def _build_box_to_nodes(store, n_nodes: int) -> Dict[int, List[int]]:
    """构建 forest_box_id → [tree_node_idx, ...] 映射.

    使用批量 forest_ids_array() 替代逐个 get_forest_id() 调用,
    避免 ~500K Python→Cython 边界开销.
    """
    fids = store.forest_ids_array()          # int32 ndarray, len = next_idx
    mask = fids >= 0
    valid_idxs = np.flatnonzero(mask)        # tree node indices with fid >= 0
    valid_fids = fids[valid_idxs]

    mapping: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(valid_idxs)):
        mapping[int(valid_fids[i])].append(int(valid_idxs[i]))
    return mapping


# ═══════════════════════════════════════════════════════════════════════════
# 核心: 单维度扫描合并
# ═══════════════════════════════════════════════════════════════════════════

def _sweep_merge_dim(
    store,
    forest,         # SafeBoxForest
    dim: int,
    ndim: int,
    box_to_nodes: Dict[int, List[int]],
    tol: float = 1e-10,
    period: float = None,
) -> int:
    """沿维度 dim 做一轮 run-length 贪心合并 (向量化分组 + 批量合并).

    C2: 用 interval cache 向量化构建 profile key → group.
    C4: 无 ThreadPoolExecutor (GIL 下无增益).
    C5: 单次 batch merge 整个 run (避免 k-1 次 pairwise).

    Returns:
        本轮合并次数.
    """
    n = forest._intervals_len
    if n < 2:
        return 0

    # -- C2: 向量化分组 --
    # 取 interval cache 快照 (copy 防止后续 swap-on-delete 污染 view)
    arr = forest._intervals_arr[:n].copy()             # (N, D, 2)
    ids_list = list(forest._interval_ids[:n])          # list copy of box_id

    # 构建 "除 dim 外所有维度" 的 key 矩阵, 四舍五入到 12 位消除浮点噪声
    dims_keep = [d for d in range(ndim) if d != dim]
    key_arr = np.round(arr[:, dims_keep, :], 12)       # (N, D-1, 2)
    key_flat = key_arr.reshape(n, -1)                   # (N, 2*(D-1))

    # 使用 structured array + np.unique 做分组
    dt = np.dtype([(f'f{i}', np.float64) for i in range(key_flat.shape[1])])
    key_view = np.ascontiguousarray(key_flat).view(dt).ravel()
    _, inverse, counts = np.unique(key_view, return_inverse=True, return_counts=True)

    # 只处理 size >= 2 的组
    group_labels_with_pairs = np.flatnonzero(counts >= 2)
    if len(group_labels_with_pairs) == 0:
        return 0

    lo_dim = arr[:, dim, 0]   # (N,) — view of copy, 不会被污染
    hi_dim = arr[:, dim, 1]   # (N,)

    # ---- Phase 1: 检测所有需要合并的 run (只读, 不修改 forest) ----
    merge_instructions = []  # list of (old_ids_set, merged_ivs)

    for glabel in group_labels_with_pairs:
        member_idxs = np.flatnonzero(inverse == glabel)

        # 按 dim 上的 lo 排序
        order = np.argsort(lo_dim[member_idxs])
        sorted_idxs = member_idxs[order]

        sorted_hi = hi_dim[sorted_idxs]
        sorted_lo = lo_dim[sorted_idxs]
        if len(sorted_idxs) < 2:
            continue
        n_sorted = len(sorted_idxs)
        gaps = np.abs(sorted_hi[:-1] - sorted_lo[1:])
        touching = gaps <= tol

        # 周期边界：检查最后一个 box 的 hi 和第一个 box 的 lo 是否跨边界相接
        wrap_touching = False
        if period is not None and n_sorted >= 2:
            wrap_gap = abs((sorted_hi[-1] - sorted_lo[0]) - period)
            if wrap_gap <= tol:
                wrap_touching = True

        i = 0
        while i < n_sorted - 1:
            if not touching[i]:
                i += 1
                continue
            run_start = i
            while i < n_sorted - 1 and touching[i]:
                i += 1
            run_end = i  # inclusive

            # 跨周期边界: 如果当前 run 起始于 idx 0 且 wrap_touching,
            # 将尾部相接的 box 也纳入 run (但不扩展合并区间, 仅合并 ID)
            if wrap_touching and run_start == 0:
                # 尾部向前找连续 touching 的 run
                j = n_sorted - 2  # touching[-1] corresponds to pair (n-2, n-1)
                while j >= 0 and touching[j]:
                    j -= 1
                tail_start = j + 1  # first index of tail run
                if tail_start > run_end:
                    # 合并头尾: lo 取尾部起始, hi 取头部结束
                    run_cache_idxs = np.concatenate([
                        sorted_idxs[tail_start:],
                        sorted_idxs[run_start:run_end + 1]
                    ])
                    run_box_ids = [ids_list[ci] for ci in run_cache_idxs]
                    first_box = forest.boxes[run_box_ids[0]]
                    merged_ivs = [tuple(iv) for iv in first_box.joint_intervals]
                    # 跨边界合并: lo 取尾部起始的 lo, hi 取头部结束的 hi
                    # 可能跨越 ±π, 维持原拓扑的两段表示不合并区间
                    # 简化策略: 合并为覆盖整段 [tail_lo, head_hi + period]
                    # 实际上由于是 sorted by lo, tail_lo > head_hi
                    # 最安全的做法: 合并为 full span [sorted_lo[tail_start], sorted_hi[run_end] + period]
                    # 但这会越界。改为: 不合并跨边界的 run, 只标记 wrap 已处理
                    # --> 安全策略: 跨边界时各自独立合并
                    merged_ivs[dim] = (float(sorted_lo[run_start]),
                                       float(sorted_hi[run_end]))
                    merge_instructions.append((set(run_box_ids), merged_ivs))
                    wrap_touching = False  # 已处理
                    i += 1
                    continue

            run_cache_idxs = sorted_idxs[run_start:run_end + 1]
            run_box_ids = [ids_list[ci] for ci in run_cache_idxs]

            first_box = forest.boxes[run_box_ids[0]]
            merged_ivs = [tuple(iv) for iv in first_box.joint_intervals]
            merged_ivs[dim] = (float(sorted_lo[run_start]),
                               float(sorted_hi[run_end]))

            merge_instructions.append((set(run_box_ids), merged_ivs))
            i += 1

        # 跨边界尾部 run 独立处理 (如果头部没有 run 合入)
        if wrap_touching:
            j = n_sorted - 2
            while j >= 0 and touching[j]:
                j -= 1
            tail_start = j + 1
            if tail_start < n_sorted - 1:
                run_cache_idxs = sorted_idxs[tail_start:]
                run_box_ids = [ids_list[ci] for ci in run_cache_idxs]
                first_box = forest.boxes[run_box_ids[0]]
                merged_ivs = [tuple(iv) for iv in first_box.joint_intervals]
                merged_ivs[dim] = (float(sorted_lo[tail_start]),
                                   float(sorted_hi[-1]))
                merge_instructions.append((set(run_box_ids), merged_ivs))


    # ---- Phase 2: 执行所有合并 (批量修改 forest) ----
    for old_ids, new_ivs in merge_instructions:
        _execute_merge_batch(
            store, forest,
            old_ids=old_ids,
            new_ivs=new_ivs,
            box_to_nodes=box_to_nodes,
        )

    return len(merge_instructions)


def _execute_merge_batch(
    store,
    forest,           # SafeBoxForest
    old_ids: Set[int],
    new_ivs: list,
    box_to_nodes: Dict[int, List[int]],
) -> BoxNode:
    """批量合并: 一次性删除全部旧 box, 创建一个新 box, 重标 tree 节点.

    C5: 替代逐对 pairwise merge — 对 k 个 box 只做 1 次 forest 操作.
    """
    new_id = forest.allocate_id()

    # 收集旧 box 的所有 tree 节点, 直接改 forest_id (不动 subtree_occ)
    all_node_idxs: List[int] = []
    for oid in old_ids:
        all_node_idxs.extend(box_to_nodes.get(oid, []))

    # 重标 forest_id
    for idx in all_node_idxs:
        store.set_forest_id(idx, new_id)

    # 更新映射
    for oid in old_ids:
        box_to_nodes.pop(oid, None)
    box_to_nodes[new_id] = all_node_idxs

    # forest 层面: 删旧, 建新 (跳过邻接 — coarsen 结束后一次性重建)
    forest.remove_boxes_no_adjacency(old_ids)

    center = np.array([(lo + hi) / 2.0 for lo, hi in new_ivs])
    new_box = BoxNode(
        node_id=new_id,
        joint_intervals=new_ivs,
        seed_config=center,
    )
    forest.add_box_no_adjacency(new_box)

    return new_box


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def coarsen_forest(
    tree,       # HierAABBTree
    forest,     # SafeBoxForest
    obstacles: list = None,         # 保留接口兼容, 本算法不需要
    safety_margin: float = 0.0,     # 保留接口兼容
    max_rounds: int = 20,
) -> CoarsenStats:
    """后处理: 分维贪心扫描合并 forest 中的 box.

    Args:
        tree: HierAABBTree
        forest: SafeBoxForest
        obstacles: (unused) 保留接口兼容
        safety_margin: (unused)
        max_rounds: 最大迭代轮数

    Returns:
        CoarsenStats
    """
    t0 = time.perf_counter()
    stats = CoarsenStats(n_before=len(forest.boxes))
    store = tree._store
    ndim = len(tree.joint_limits)

    # 构建 box → tree nodes 映射
    box_to_nodes = _build_box_to_nodes(store, store.next_idx)

    for round_idx in range(max_rounds):
        merged_this_round = 0

        for dim in range(ndim):
            n = _sweep_merge_dim(store, forest, dim, ndim, box_to_nodes,
                                 period=getattr(forest, 'period', None))
            merged_this_round += n

        stats.n_rounds = round_idx + 1
        logger.debug(f"coarsen round {round_idx}: {merged_this_round} merges")
        if merged_this_round == 0:
            break
        stats.n_merges += merged_this_round

    # 一次性重建邻接 (比合并期间逐次增量更新更高效)
    if stats.n_merges > 0:
        forest.rebuild_adjacency()

    stats.n_after = len(forest.boxes)
    stats.time_ms = (time.perf_counter() - t0) * 1000
    return stats
