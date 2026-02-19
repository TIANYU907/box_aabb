"""
forest/connectivity.py - 连通分量检测与岛间桥接

基于 Union-Find 的 overlap 连通分量检测，以及岛间线段桥接。
不依赖 tree 层级结构，直接在 Dict[int, BoxNode] 上操作。
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

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

    # O(N²·D) overlap 检测
    box_list = [(bid, boxes[bid]) for bid in ids]
    n = len(box_list)
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
) -> Tuple[list, List[Set[int]], int, List[BoxNode], List[Set[int]]]:
    """检测岛并尝试用线段连接不连通的岛。

    对每对不连通的岛，找最近 box 对，在 box 表面取点，
    验证线段无碰撞即建立 bridge edge。

    若提供了 *hier_tree* / *obstacles* / *forest*，会在成功连线后
    沿线段上多点尝试 ``find_free_box``；若新 box 同时与两端 box 重叠，
    则以 box 连接取代线段（更鲁棒）。

    当 *period* 不为 None 时，距离 / overlap / 线段插值均使用周期 wrap。

    当 *min_island_size* > 0 时，在桥接之前舍弃几何平均边长（岛内 box
    总体积的 gmean）低于该阈值的小岛（噪声岛），不参与桥接。

    Args:
        boxes: {node_id: BoxNode}
        collision_checker: 碰撞检测器
        segment_resolution: 线段碰撞检测分辨率
        max_pairs_per_island_pair: 每对岛最多尝试的 box 对数
        max_rounds: 最大桥接轮数
        period: 关节空间周期（例如 2π），None 表示不 wrap
        hier_tree: HierAABBTree（可选），用于在 bridge 处展开 box
        obstacles: 碰撞环境障碍物列表（与 hier_tree 搭配使用）
        forest: BoxForest（可选），用于分配 ID 和添加 bridge box
        min_box_size: box 几何平均边长下限
        n_bridge_seeds: 沿线段采样的 seed 数（用于 box 展开尝试）
        min_island_size: 岛总体积几何平均边长下限，低于此值的岛被舍弃

    Returns:
        (bridge_edges, final_islands, n_islands_before, bridge_boxes, discarded_islands)
    """
    from planner.models import Edge, gmean_edge_length

    islands, uf = find_islands(boxes, period=period)
    n_islands_before = len(islands)

    bridge_edges: list = []
    bridge_boxes: List[BoxNode] = []
    discarded_islands: List[Set[int]] = []

    # ---- 舍弃几何平均体积太小的岛 ----
    if min_island_size > 0 and len(islands) > 1:
        ndim = next(iter(boxes.values())).n_dims if boxes else 1
        kept: List[Set[int]] = []
        for island in islands:
            total_vol = sum(boxes[bid].volume for bid in island)
            gm = gmean_edge_length(total_vol, ndim)
            if gm >= min_island_size:
                kept.append(island)
            else:
                discarded_islands.append(island)
                logger.info(
                    "discard island (size=%d, total_vol=%.6f, gmean=%.4f < %.4f): %s",
                    len(island), total_vol, gm, min_island_size,
                    sorted(island)[:8],
                )
        if discarded_islands:
            # 从 boxes 字典和 UnionFind 中移除被舍弃的 box
            for island in discarded_islands:
                for bid in island:
                    boxes.pop(bid, None)
            # 用保留的 box 重新构建 UnionFind
            islands, uf = find_islands(boxes, period=period)
            logger.info(
                "after discarding %d small islands (%d boxes): %d islands remain",
                len(discarded_islands),
                sum(len(s) for s in discarded_islands),
                len(islands),
            )
        else:
            islands = kept

    if len(islands) <= 1:
        return bridge_edges, islands, n_islands_before, bridge_boxes, discarded_islands

    can_expand_box = (hier_tree is not None and obstacles is not None)
    next_edge_id = 0

    for round_idx in range(max_rounds):
        # 重新检测当前岛
        islands = uf.components()
        if len(islands) <= 1:
            break

        merged_any = False
        # 对每对岛尝试桥接
        for i in range(len(islands)):
            for j in range(i + 1, len(islands)):
                island_a = islands[i]
                island_b = islands[j]

                # 已经同一连通分量则跳过（可能上一轮 round 中合并了）
                rep_a = next(iter(island_a))
                rep_b = next(iter(island_b))
                if uf.same(rep_a, rep_b):
                    continue

                # 找最近 box 对
                pairs = _find_closest_pairs(
                    island_a, island_b, boxes,
                    max_pairs_per_island_pair, period=period,
                )

                for box_a, box_b, dist in pairs:
                    # 在 box 表面取连接点（考虑 wrap）
                    q_a = _nearest_point_wrapped(box_a, box_b.center, period)
                    q_b = _nearest_point_wrapped(box_b, box_a.center, period)

                    # 线段碰撞检测（考虑 wrap 最短路径）
                    if period is not None:
                        collides = _check_segment_wrapped(
                            q_a, q_b, collision_checker,
                            segment_resolution, period,
                        )
                    else:
                        collides = collision_checker.check_segment_collision(
                            q_a, q_b, segment_resolution,
                        )
                    if collides:
                        continue

                    # ---- 成功连线，尝试用 box 取代线段 ----
                    bridge_box = None
                    if can_expand_box:
                        bridge_box = _try_expand_bridge_box(
                            q_a, q_b, box_a, box_b,
                            hier_tree, obstacles, forest, boxes,
                            period, min_box_size, n_bridge_seeds,
                        )

                    if bridge_box is not None:
                        # box 连接成功 → 加入 boxes dict 和 UnionFind
                        boxes[bridge_box.node_id] = bridge_box
                        uf._parent[bridge_box.node_id] = bridge_box.node_id
                        uf._rank[bridge_box.node_id] = 0
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
                        # 回退到线段连接
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

        if not merged_any:
            break  # 本轮无进展，停止

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

        nid = forest.allocate_id() if forest is not None else 0

        # 第一步: 不标记占用，先看结果
        ffb = hier_tree.find_free_box(
            q_seed,
            obstacles,
            mark_occupied=False,
            forest_box_id=nid,
        )
        if ffb is None:
            continue

        ivs = ffb.intervals
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        if gmean_edge_length(vol, n_dims) < min_box_size:
            continue  # 太小，未标记占用所以无需回滚

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
            continue  # 未标记占用，直接跳过

        # 通过验证 → 正式标记占用并再次调用 find_free_box（mark=True）
        ffb2 = hier_tree.find_free_box(
            q_seed,
            obstacles,
            mark_occupied=True,
            forest_box_id=nid,
        )
        if ffb2 is None:
            continue  # 罕见竞态，跳过

        # 添加到 forest
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
    """找两个岛之间距离最近的 k 对 box（按 center 距离排序，支持 wrap）。"""
    pairs: List[Tuple[BoxNode, BoxNode, float]] = []

    boxes_a = [boxes[bid] for bid in island_a]
    boxes_b = [boxes[bid] for bid in island_b]

    for ba in boxes_a:
        for bb in boxes_b:
            dist = _wrapped_center_dist(ba, bb, period)
            pairs.append((ba, bb, dist))

    pairs.sort(key=lambda x: x[2])
    return pairs[:k]
