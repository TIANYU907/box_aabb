"""
forest/connectivity.py - 连通分量检测与岛间桥接

基于 Union-Find 的 overlap 连通分量检测，以及岛间线段桥接。
不依赖 tree 层级结构，直接在 Dict[int, BoxNode] 上操作。
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .models import BoxNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    """带路径压缩和按秩合并的并查集。"""

    __slots__ = ("_parent", "_rank")

    def __init__(self, keys):
        self._parent = {k: k for k in keys}
        self._rank = {k: 0 for k in keys}

    def find(self, x: int) -> int:
        r = x
        while self._parent[r] != r:
            r = self._parent[r]
        # 路径压缩
        while self._parent[x] != r:
            self._parent[x], x = r, self._parent[x]
        return r

    def union(self, x: int, y: int) -> bool:
        """合并 x, y 所在集合。返回 True 表示实际合并（原先不同集合）。"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return True

    def components(self) -> List[Set[int]]:
        """返回所有连通分量，按大小降序。"""
        groups: Dict[int, Set[int]] = {}
        for k in self._parent:
            r = self.find(k)
            groups.setdefault(r, set()).add(k)
        return sorted(groups.values(), key=len, reverse=True)

    def n_components(self) -> int:
        roots = {self.find(k) for k in self._parent}
        return len(roots)

    def same(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def add(self, key) -> None:
        """添加一个新的单独元素。"""
        self._parent[key] = key
        self._rank[key] = 0

    def remove_keys(self, keys) -> None:
        """从并查集中批量移除 keys（不维护 find 一致性，需之后重建 components）。"""
        for k in keys:
            self._parent.pop(k, None)
            self._rank.pop(k, None)


# ---------------------------------------------------------------------------
# Periodic (wrap-around) utilities   π ↔ −π
# ---------------------------------------------------------------------------

def _overlap_periodic_1d(
    lo1: float, hi1: float, lo2: float, hi2: float,
    period: float, eps: float = 1e-10,
) -> bool:
    """判断两个 1-D 区间在周期 *period* 的环上是否重叠。"""
    # 直接重叠
    if hi1 >= lo2 - eps and hi2 >= lo1 - eps:
        return True
    # 将 box2 右移一个周期
    if hi1 >= (lo2 + period) - eps and (hi2 + period) >= lo1 - eps:
        return True
    # 将 box2 左移一个周期
    if hi1 >= (lo2 - period) - eps and (hi2 - period) >= lo1 - eps:
        return True
    return False


def _overlap_periodic(
    box_a: BoxNode, box_b: BoxNode, period: Optional[float],
) -> bool:
    """多维周期 overlap 检测。period=None 退化为普通 overlap。"""
    if period is None:
        return box_a.overlap_with(box_b)
    for (lo1, hi1), (lo2, hi2) in zip(
        box_a.joint_intervals, box_b.joint_intervals,
    ):
        if not _overlap_periodic_1d(lo1, hi1, lo2, hi2, period):
            return False
    return True


def _wrapped_dist_1d(a: float, b: float, period: float) -> float:
    """环上两点的最短距离。"""
    d = abs(a - b) % period
    return min(d, period - d)


def _wrapped_center_dist(
    box_a: BoxNode, box_b: BoxNode, period: Optional[float],
) -> float:
    """两个 box 中心在环面上的欧氏距离。"""
    if period is None:
        return float(np.linalg.norm(box_a.center - box_b.center))
    d2 = 0.0
    for ca, cb in zip(box_a.center, box_b.center):
        d = _wrapped_dist_1d(float(ca), float(cb), period)
        d2 += d * d
    return float(np.sqrt(d2))


def _nearest_point_wrapped(
    box: BoxNode, config: np.ndarray, period: Optional[float],
) -> np.ndarray:
    """在 box 内找到距离 config 最近的点（考虑周期 wrap）。"""
    if period is None:
        return box.nearest_point_to(config)
    n = box.n_dims
    nearest = np.empty(n, dtype=np.float64)
    for i, (lo, hi) in enumerate(box.joint_intervals):
        c = config[i]
        best_dist = float('inf')
        best_val = lo
        for offset in (-period, 0.0, period):
            c_shifted = c + offset
            clipped = float(np.clip(c_shifted, lo, hi))
            dist = abs(c_shifted - clipped)
            if dist < best_dist:
                best_dist = dist
                best_val = clipped
        nearest[i] = best_val
    return nearest


def _check_segment_wrapped(
    q_a: np.ndarray,
    q_b: np.ndarray,
    collision_checker,
    resolution: float,
    period: float,
) -> bool:
    """沿环面最短路径做线段碰撞检测。返回 True = 存在碰撞。"""
    half = period / 2.0
    diff = ((q_b - q_a) + half) % period - half   # 每维最短有符号差
    dist = float(np.linalg.norm(diff))
    if dist < 1e-10:
        return collision_checker.check_config_collision(q_a)
    n_steps = max(2, int(np.ceil(dist / resolution)) + 1)
    for i in range(n_steps):
        t = i / (n_steps - 1)
        q = q_a + t * diff
        q = ((q + half) % period) - half          # 归一化到 [-π, π]
        if collision_checker.check_config_collision(q):
            return True
    return False


# ---------------------------------------------------------------------------
# Island detection
# ---------------------------------------------------------------------------

def find_islands(
    boxes: Dict[int, BoxNode],
    period: Optional[float] = None,
) -> Tuple[List[Set[int]], UnionFind]:
    """基于 overlap 检测连通分量（岛）。

    两个 box 在所有维度上都有重叠即算连通（传递闭包 = 岛）。
    当 *period* 不为 None 时，使用周期边界（π ↔ −π）。

    Args:
        boxes: {node_id: BoxNode}
        period: 关节空间周期（例如 2π），None 表示不 wrap

    Returns:
        (islands, uf) — 连通分量列表（按大小降序）和 UnionFind 实例
    """
    ids = list(boxes.keys())
    uf = UnionFind(ids)
    n = len(ids)

    if n < 2:
        return uf.components(), uf

    if period is None:
        # 向量化快速路径: NumPy 广播 O(N²) overlap
        ndim = next(iter(boxes.values())).n_dims
        lo = np.empty((n, ndim), dtype=np.float64)
        hi = np.empty((n, ndim), dtype=np.float64)
        for k, bid in enumerate(ids):
            ivs = boxes[bid].joint_intervals
            for d in range(ndim):
                lo[k, d] = ivs[d][0]
                hi[k, d] = ivs[d][1]

        eps = 1e-12
        overlap_ij = (hi[:, None, :] >= lo[None, :, :] - eps) & \
                     (hi[None, :, :] >= lo[:, None, :] - eps)
        overlap_all = np.all(overlap_ij, axis=2)
        ii, jj = np.where(np.triu(overlap_all, k=1))
        for idx in range(len(ii)):
            uf.union(ids[ii[idx]], ids[jj[idx]])
    else:
        # 周期边界: 保留 Python 逐对循环
        box_list = [(bid, boxes[bid]) for bid in ids]
        for i in range(n):
            bid_i, box_i = box_list[i]
            for j in range(i + 1, n):
                bid_j, box_j = box_list[j]
                if _overlap_periodic(box_i, box_j, period):
                    uf.union(bid_i, bid_j)

    islands = uf.components()
    return islands, uf


# ---------------------------------------------------------------------------
# Island bridging
# ---------------------------------------------------------------------------

def _check_segments_batch(
    segment_pairs: List[Tuple[np.ndarray, np.ndarray]],
    collision_checker,
    resolution: float,
) -> List[bool]:
    """批量线段碰撞检测 (向量化 FK + AABB).

    将多条线段的等距采样点合并成一个大矩阵,
    调用 collision_checker.check_config_collision_batch 一次性检测.

    Returns:
        collides[i] — 第 i 条线段是否碰撞
    """
    if not segment_pairs:
        return []

    # 生成所有采样点, 记录每条线段对应的下标范围
    all_configs: List[np.ndarray] = []
    seg_ranges: List[Tuple[int, int]] = []
    offset = 0
    for q_a, q_b in segment_pairs:
        diff = q_b - q_a
        dist = float(np.linalg.norm(diff))
        if dist < 1e-10:
            n_steps = 1
        else:
            n_steps = max(2, int(np.ceil(dist / resolution)) + 1)
        ts = np.linspace(0.0, 1.0, n_steps)
        configs = q_a[None, :] + ts[:, None] * diff[None, :]  # (n_steps, D)
        all_configs.append(configs)
        seg_ranges.append((offset, offset + n_steps))
        offset += n_steps

    big_configs = np.vstack(all_configs)  # (total, D)
    big_result = collision_checker.check_config_collision_batch(big_configs)

    collides: List[bool] = []
    for start, end in seg_ranges:
        collides.append(bool(np.any(big_result[start:end])))
    return collides


def _build_kdtree(boxes: Dict[int, BoxNode]):
    """构建 box centers 的 cKDTree, 返回 (kd, id_list, id_to_idx)."""
    id_list = list(boxes.keys())
    centers = np.array([boxes[bid].center for bid in id_list])
    kd = cKDTree(centers)
    id_to_idx = {bid: i for i, bid in enumerate(id_list)}
    return kd, id_list, id_to_idx, centers


def _find_closest_pairs_kdtree(
    island_a: Set[int],
    island_b: Set[int],
    boxes: Dict[int, BoxNode],
    k: int,
    kd: cKDTree,
    id_list: List[int],
    id_to_idx: Dict[int, int],
    centers: np.ndarray,
) -> List[Tuple[BoxNode, BoxNode, float]]:
    """用 KD-Tree 找两岛之间最近的 k 对 box (O(|A|·k·logN))."""
    b_set = island_b
    # A 侧 centers (跳过不在 id_to_idx 中的 ID, 可能因 bridge 内部变化)
    a_ids = [bid for bid in island_a if bid in id_to_idx]
    if not a_ids:
        return []
    a_idxs = [id_to_idx[bid] for bid in a_ids]
    a_centers = centers[a_idxs]  # (|A|, D)

    # 查询时多取一些近邻, 因为不是所有近邻都属于 island_b
    n_query = min(k * 5 + 20, len(id_list))
    dists_all, idxs_all = kd.query(a_centers, k=n_query)
    # dists_all: (|A|, n_query), idxs_all: (|A|, n_query)
    if dists_all.ndim == 1:
        dists_all = dists_all.reshape(-1, 1)
        idxs_all = idxs_all.reshape(-1, 1)

    pairs: List[Tuple[BoxNode, BoxNode, float]] = []
    for row, a_bid in enumerate(a_ids):
        for col in range(n_query):
            nn_idx = idxs_all[row, col]
            nn_bid = id_list[nn_idx]
            if nn_bid in b_set:
                pairs.append((boxes[a_bid], boxes[nn_bid], float(dists_all[row, col])))
                if len(pairs) >= k * 3:  # 收集足够多后提前退出
                    break
        if len(pairs) >= k * 3:
            break

    pairs.sort(key=lambda x: x[2])
    return pairs[:k]


def bridge_islands(
    boxes: Dict[int, BoxNode],
    collision_checker,
    segment_resolution: float = 0.03,
    max_pairs_per_island_pair: int = 10,
    max_rounds: int = 5,
    period: Optional[float] = None,
    hier_tree=None,
    obstacles: Optional[list] = None,
    forest=None,
    min_box_size: float = 0.001,
    n_bridge_seeds: int = 5,
    min_island_size: float = 0.005,
    # ---- 优化参数 (A/B/C/D) ----
    precomputed_uf: Optional[UnionFind] = None,
    precomputed_islands: Optional[List[Set[int]]] = None,
    target_pair: Optional[Tuple[int, int]] = None,
) -> Tuple[list, List[Set[int]], int, List[BoxNode], List[Set[int]]]:
    """检测岛并尝试用线段连接不连通的岛。

    优化亮点:
      A) 支持传入 precomputed_uf / precomputed_islands 避免重复 O(N²);
      B) 内部用 cKDTree 加速 closest pairs (O(|A|·k·logN) 替代 O(|A|·|B|));
      C) 传入 target_pair=(src, tgt) 时只桥接 s-t 相关岛, 连通即停;
      D) 用 check_config_collision_batch 批量碰撞检测.

    Args:
        boxes: {node_id: BoxNode}
        collision_checker: 碰撞检测器
        segment_resolution: 线段碰撞检测分辨率
        max_pairs_per_island_pair: 每对岛最多尝试的 box 对数
        max_rounds: 最大桥接轮数
        period: 关节空间周期（例如 2π），None 表示不 wrap
        hier_tree: HierAABBTree（可选），用于在 bridge 处展开 box
        obstacles: 碰撞环境障碍物列表（与 hier_tree 搭配使用）
        forest: SafeBoxForest（可选），用于分配 ID 和添加 bridge box
        min_box_size: box 几何平均边长下限
        n_bridge_seeds: 沿线段采样的 seed 数（用于 box 展开尝试）
        min_island_size: 岛总体积几何平均边长下限
        precomputed_uf: 预先算好的 UnionFind (优化 A)
        precomputed_islands: 预先算好的 islands (优化 A)
        target_pair: (src_box_id, tgt_box_id) 目标导向桥接 (优化 C)

    Returns:
        (bridge_edges, final_islands, n_islands_before, bridge_boxes, discarded_islands)
    """
    from planner.models import Edge, gmean_edge_length

    # ---- (A) 接受预计算的 uf / islands, 不重复 O(N²) ----
    if precomputed_uf is not None and precomputed_islands is not None:
        uf = precomputed_uf
        islands = precomputed_islands
    else:
        islands, uf = find_islands(boxes, period=period)
    n_islands_before = len(islands)

    bridge_edges: list = []
    bridge_boxes: List[BoxNode] = []
    discarded_islands: List[Set[int]] = []

    # ---- 舍弃几何平均体积太小的岛 ----
    if min_island_size > 0 and len(islands) > 1:
        ndim = next(iter(boxes.values())).n_dims if boxes else 1
        kept: List[Set[int]] = []
        remove_bids: List[int] = []
        for island in islands:
            total_vol = sum(boxes[bid].volume for bid in island)
            gm = gmean_edge_length(total_vol, ndim)
            if gm >= min_island_size:
                kept.append(island)
            else:
                discarded_islands.append(island)
                remove_bids.extend(island)
                logger.info(
                    "discard island (size=%d, total_vol=%.6f, gmean=%.4f < %.4f): %s",
                    len(island), total_vol, gm, min_island_size,
                    sorted(island)[:8],
                )
        if discarded_islands:
            for bid in remove_bids:
                boxes.pop(bid, None)
            # (A) 从 UF 中移除, 重建 components 即可
            uf.remove_keys(remove_bids)
            islands = uf.components()
            logger.info(
                "after discarding %d small islands (%d boxes): %d islands remain",
                len(discarded_islands), len(remove_bids), len(islands),
            )
        else:
            islands = kept

    if len(islands) <= 1:
        return bridge_edges, islands, n_islands_before, bridge_boxes, discarded_islands

    can_expand_box = (hier_tree is not None and obstacles is not None)
    next_edge_id = 0

    # ---- (C) 目标导向: 构建 box_id → island_index 映射 ----
    src_bid, tgt_bid = (target_pair if target_pair is not None
                        else (None, None))
    # 目标导向时允许更多轮次（需要 chain 多跳）
    effective_max_rounds = max_rounds * 4 if src_bid is not None else max_rounds

    for round_idx in range(effective_max_rounds):
        # 重新检测当前岛
        islands = uf.components()
        if len(islands) <= 1:
            break
        # (C) 目标导向早停: src-tgt 已连通
        if src_bid is not None and uf.same(src_bid, tgt_bid):
            break

        # ---- (B) 每轮重建 KD-Tree ----
        kd, id_list, id_to_idx, centers = _build_kdtree(boxes)

        # (C) 确定本轮要桥接的岛对
        if src_bid is not None:
            src_root = uf.find(src_bid)
            tgt_root = uf.find(tgt_bid)
            # 收集 src 岛和 tgt 岛
            src_island = None
            tgt_island = None
            island_by_root: Dict[int, Set[int]] = {}
            for isl in islands:
                rep = next(iter(isl))
                r = uf.find(rep)
                island_by_root[r] = isl
                if r == src_root:
                    src_island = isl
                elif r == tgt_root:
                    tgt_island = isl

            island_pairs_to_try: List[Tuple[Set[int], Set[int]]] = []

            # 1) 直接尝试 src ↔ tgt
            if src_island is not None and tgt_island is not None:
                island_pairs_to_try.append((src_island, tgt_island))

            # 2) 从 src 岛找离 tgt 最近的跳板岛（而非离 src 最近）
            #    从 tgt 岛找离 src 最近的跳板岛
            bid_to_island: Dict[int, Set[int]] = {}
            for isl in islands:
                for bid in isl:
                    bid_to_island[bid] = isl

            # tgt 方向的参考中心
            tgt_center = (centers[id_to_idx[tgt_bid]]
                          if tgt_bid in id_to_idx else None)
            src_center = (centers[id_to_idx[src_bid]]
                          if src_bid in id_to_idx else None)

            for anchor_island, anchor_root, target_center in [
                (src_island, src_root, tgt_center),
                (tgt_island, tgt_root, src_center),
            ]:
                if anchor_island is None or target_center is None:
                    continue
                # 在 anchor 岛中找到离 target_center 最近的 box
                anchor_ids = list(anchor_island)
                anchor_idxs = [id_to_idx[b] for b in anchor_ids if b in id_to_idx]
                if not anchor_idxs:
                    continue
                anchor_ctrs = centers[anchor_idxs]
                # 按距目标排序，取最近几个做出发点
                dists_to_tgt = np.linalg.norm(anchor_ctrs - target_center, axis=1)
                top_k = min(5, len(anchor_idxs))
                best_rows = np.argsort(dists_to_tgt)[:top_k]

                n_q = min(80, len(id_list))
                seen_roots: Set[int] = set()
                for row_i in best_rows:
                    pt = anchor_ctrs[row_i:row_i+1]
                    _, nn_row = kd.query(pt, k=n_q)
                    nn_row = nn_row.flatten()
                    for nn_idx in nn_row:
                        nn_bid = id_list[nn_idx]
                        nn_root = uf.find(nn_bid)
                        if nn_root != anchor_root and nn_root not in seen_roots:
                            other_island = bid_to_island.get(nn_bid)
                            if other_island is not None:
                                pair = (anchor_island, other_island)
                                if pair not in island_pairs_to_try:
                                    island_pairs_to_try.append(pair)
                                seen_roots.add(nn_root)
                            if len(seen_roots) >= 5:
                                break
                    if len(seen_roots) >= 5:
                        break
        else:
            # 非目标导向: 全部岛对
            island_pairs_to_try = []
            for i in range(len(islands)):
                for j in range(i + 1, len(islands)):
                    island_pairs_to_try.append((islands[i], islands[j]))

        merged_any = False

        # ── Phase A (可并行): 对所有岛对做只读探测 ──
        # 为每个岛对收集候选 segments + 碰撞检测，不修改任何共享状态
        def _probe_island_pair(island_a, island_b):
            """只读: 找最近 box 对 → 生成线段 → 碰撞检测.
            返回 (island_a, island_b, candidate_segments, candidate_pairs_info, collides_list)
            """
            pairs = _find_closest_pairs_kdtree(
                island_a, island_b, boxes,
                max_pairs_per_island_pair,
                kd, id_list, id_to_idx, centers,
            )
            segs = []
            info = []
            for box_a, box_b, dist in pairs:
                q_a = _nearest_point_wrapped(box_a, box_b.center, period)
                q_b = _nearest_point_wrapped(box_b, box_a.center, period)
                segs.append((q_a, q_b))
                info.append((box_a, box_b, dist))
            if not segs:
                return (island_a, island_b, segs, info, [])
            if period is not None:
                coll = [_check_segment_wrapped(q_a, q_b, collision_checker,
                                               segment_resolution, period)
                        for q_a, q_b in segs]
            else:
                coll = _check_segments_batch(segs, collision_checker,
                                             segment_resolution)
            return (island_a, island_b, segs, info, coll)

        # 筛选有效岛对 (UF-check 必须在主线程串行做)
        valid_pairs = []
        for island_a, island_b in island_pairs_to_try:
            rep_a = next(iter(island_a))
            rep_b = next(iter(island_b))
            if not uf.same(rep_a, rep_b):
                valid_pairs.append((island_a, island_b))

        # 并行探测 (collision check 内部是 NumPy 向量化, 释放 GIL)
        if len(valid_pairs) >= 4:
            with ThreadPoolExecutor(max_workers=min(4, len(valid_pairs))) as pool:
                probe_results = list(pool.map(
                    lambda p: _probe_island_pair(p[0], p[1]),
                    valid_pairs))
        else:
            probe_results = [_probe_island_pair(a, b) for a, b in valid_pairs]

        # ── Phase B (串行): 处理探测结果, 修改 UF / boxes / forest ──
        for island_a, island_b, candidate_segments, candidate_pairs_info, collides_list in probe_results:
            rep_a = next(iter(island_a))
            rep_b = next(iter(island_b))
            if uf.same(rep_a, rep_b):
                continue  # 前面的 bridge 可能已连通

            for idx, collides in enumerate(collides_list):
                if collides:
                    continue
                q_a, q_b = candidate_segments[idx]
                box_a, box_b, dist = candidate_pairs_info[idx]

                # ---- 成功连线，尝试用 box 取代线段 ----
                bridge_box = None
                if can_expand_box:
                    bridge_box = _try_expand_bridge_box(
                        q_a, q_b, box_a, box_b,
                        hier_tree, obstacles, forest, boxes,
                        period, min_box_size, n_bridge_seeds,
                    )

                if bridge_box is not None:
                    boxes[bridge_box.node_id] = bridge_box
                    uf.add(bridge_box.node_id)
                    uf.union(bridge_box.node_id, box_a.node_id)
                    uf.union(bridge_box.node_id, box_b.node_id)
                    bridge_boxes.append(bridge_box)
                    merged_any = True
                    logger.debug(
                        "bridge round %d: box %d ← new_box %d → box %d "
                        "(vol=%.6f)",
                        round_idx, box_a.node_id,
                        bridge_box.node_id, box_b.node_id,
                        bridge_box.volume,
                    )
                else:
                    edge = Edge(
                        edge_id=next_edge_id,
                        source_box_id=box_a.node_id,
                        target_box_id=box_b.node_id,
                        source_config=q_a,
                        target_config=q_b,
                        is_collision_free=True,
                    )
                    next_edge_id += 1
                    bridge_edges.append(edge)
                    uf.union(box_a.node_id, box_b.node_id)
                    merged_any = True
                    logger.debug(
                        "bridge round %d: box %d → box %d (segment, "
                        "dist=%.4f)",
                        round_idx, box_a.node_id, box_b.node_id, dist,
                    )

                break  # 成功一条即转下一对岛

            # (C) 目标导向早停: 每成功一条就检查 s-t
            if src_bid is not None and uf.same(src_bid, tgt_bid):
                merged_any = True
                break

        if not merged_any:
            break
        # (C) 目标导向早停
        if src_bid is not None and uf.same(src_bid, tgt_bid):
            break

    final_islands = uf.components()
    return bridge_edges, final_islands, n_islands_before, bridge_boxes, discarded_islands


# ---------------------------------------------------------------------------
# Bridge box expansion helper
# ---------------------------------------------------------------------------

def _try_expand_bridge_box(
    q_a: np.ndarray,
    q_b: np.ndarray,
    box_a: BoxNode,
    box_b: BoxNode,
    hier_tree,
    obstacles: list,
    forest,
    boxes: Dict[int, BoxNode],
    period: Optional[float],
    min_box_size: float,
    n_seeds: int,
) -> Optional[BoxNode]:
    """沿段 q_a → q_b 上采样 n_seeds 个种子点，尝试 find_free_box。

    若生成的 box 同时与 box_a 和 box_b 重叠（周期感知），则返回它；
    否则返回 None（不会留下副作用）。

    使用 dry-run（mark_occupied=False）先检测再正式标记，
    避免 clear_subtree_occupation 错误地清除其他 box 的占用。
    """
    from planner.models import gmean_edge_length

    if period is not None:
        half = period / 2.0
        diff = ((q_b - q_a) + half) % period - half
    else:
        diff = q_b - q_a

    n_dims = len(q_a)

    for k in range(n_seeds):
        t = (k + 1) / (n_seeds + 1)          # 均匀分布在 (0,1) 开区间
        q_seed = q_a + t * diff
        if period is not None:
            half_p = period / 2.0
            q_seed = ((q_seed + half_p) % period) - half_p  # 归一化

        # 已被占用则跳过
        if hier_tree.is_occupied(q_seed):
            continue

        # dry-run: 不标记占用, 只检测候选 box
        ffb = hier_tree.find_free_box(
            q_seed,
            obstacles,
            mark_occupied=False,
        )
        if ffb is None:
            continue

        ivs = ffb.intervals
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        if gmean_edge_length(vol, n_dims) < min_box_size:
            continue

        nid = forest.allocate_id() if forest is not None else 0
        new_box = BoxNode(
            node_id=nid,
            joint_intervals=ivs,
            seed_config=q_seed.copy(),
            volume=vol,
        )

        # 检查是否同时与两端 box 重叠
        overlap_a = _overlap_periodic(new_box, box_a, period)
        overlap_b = _overlap_periodic(new_box, box_b, period)
        if not (overlap_a and overlap_b):
            continue

        # 通过验证 → 正式标记占用并添加到 forest
        ffb2 = hier_tree.find_free_box(
            q_seed,
            obstacles,
            mark_occupied=True,
            forest_box_id=nid,
        )
        if ffb2 is not None:
            if forest is not None:
                if ffb2.absorbed_box_ids:
                    forest.remove_boxes(ffb2.absorbed_box_ids)
                forest.add_box_direct(new_box)
            return new_box

    return None


def _find_closest_pairs(
    island_a: Set[int],
    island_b: Set[int],
    boxes: Dict[int, BoxNode],
    k: int,
    period: Optional[float] = None,
) -> List[Tuple[BoxNode, BoxNode, float]]:
    """找两个岛之间距离最近的 k 对 box（按 center 距离排序，支持 wrap）。

    保留供 period!=None 时使用（KD-Tree 不支持周期距离）。
    """
    pairs: List[Tuple[BoxNode, BoxNode, float]] = []

    boxes_a = [boxes[bid] for bid in island_a]
    boxes_b = [boxes[bid] for bid in island_b]

    for ba in boxes_a:
        for bb in boxes_b:
            dist = _wrapped_center_dist(ba, bb, period)
            pairs.append((ba, bb, dist))

    pairs.sort(key=lambda x: x[2])
    return pairs[:k]
