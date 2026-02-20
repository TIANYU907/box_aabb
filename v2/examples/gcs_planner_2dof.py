"""
gcs_planner_2dof.py — GCS (Graph of Convex Sets) 规划器 demo

在 2-DOF 场景中:
1. 构建 box forest（复用已有扩展逻辑）
2. 做岛检测 & 桥接，得到完整连通图
3. 用 cvxpy 求解 GCS 最短路径（凸松弛 → 路径提取）
4. 可视化结果

GCS 算法参考:
  Marcucci et al., "Shortest Paths in Graphs of Convex Sets", 2023
  https://arxiv.org/abs/2101.11565

核心思想:
  - 每个 box 是一个凸集 (vertex)
  - 相邻 box 对构成边 (edge)
  - 决策变量: 每条边两端点 (x_u^e, x_v^e) ∈ 各自 box，以及流变量 φ_e
  - 目标: min Σ_e φ_e · ‖x_u^e - x_v^e‖₂
  - 约束: 流守恒 (source→sink)，perspective 凸松弛
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
import cvxpy as cp

import sys, os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
from _bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.models import BoxNode
from forest.connectivity import find_islands, bridge_islands, _overlap_periodic
from planner.box_planner import BoxPlanner
from planner.models import PlannerConfig, gmean_edge_length
from common.output import make_output_dir
from forest.coarsen import coarsen_forest


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GCSConfig:
    seed: int = 20260219
    robot_name: str = "2dof_planar"
    n_obstacles: int = 8
    q_start: List[float] = field(
        default_factory=lambda: [0.8 * 3.141592653589793, 0.2])
    q_goal: List[float] = field(
        default_factory=lambda: [-0.7 * 3.141592653589793, -0.4])

    # forest
    max_consecutive_miss: int = 20
    min_box_size: float = 0.03    # 稍大一点减少 box 数量, 加速 GCS
    goal_bias: float = 0.15
    guided_sample_ratio: float = 0.6

    # island / bridge
    min_island_size: float = 0.0   # GCS 需要完整连通, 不舍弃小岛

    # GCS solver
    corridor_hops: int = 2  # 最短路径周围保留多少 BFS 跳

    # scene
    obs_cx_range: Tuple[float, float] = (-1.6, 1.6)
    obs_cy_range: Tuple[float, float] = (-1.6, 1.6)
    obs_w_range: Tuple[float, float] = (0.25, 0.65)
    obs_h_range: Tuple[float, float] = (0.25, 0.65)

    # coarsen
    coarsen_max_rounds: int = 20

    # viz
    collision_map_resolution: float = 0.03
    dpi: int = 140


# ═══════════════════════════════════════════════════════════════════════════
# Scene & Forest helpers  (与 visualize 脚本复用相同逻辑)
# ═══════════════════════════════════════════════════════════════════════════

def build_random_scene(robot, q_start, q_goal, rng, cfg: GCSConfig,
                       max_trials=200) -> Scene:
    for _ in range(max_trials):
        scene = Scene()
        for i in range(cfg.n_obstacles):
            cx = float(rng.uniform(*cfg.obs_cx_range))
            cy = float(rng.uniform(*cfg.obs_cy_range))
            w = float(rng.uniform(*cfg.obs_w_range))
            h = float(rng.uniform(*cfg.obs_h_range))
            scene.add_obstacle([cx - w * 0.5, cy - h * 0.5],
                               [cx + w * 0.5, cy + h * 0.5], name=f"obs_{i}")
        checker = CollisionChecker(robot=robot, scene=scene)
        if (not checker.check_config_collision(q_start)
                and not checker.check_config_collision(q_goal)
                and checker.check_segment_collision(q_start, q_goal, 0.03)):
            return scene
    raise RuntimeError("Failed to build random scene")


def make_planner_config(cfg: GCSConfig) -> PlannerConfig:
    return PlannerConfig(
        max_iterations=999999, max_box_nodes=999999,
        min_box_size=cfg.min_box_size, goal_bias=cfg.goal_bias,
        guided_sample_ratio=cfg.guided_sample_ratio,
        segment_collision_resolution=0.03,
        connection_radius=3.0, verbose=False, forest_path=None,
    )


def grow_forest(planner, q_start, q_goal, seed, max_miss=20):
    """生长 forest, 返回 (boxes_dict, forest_obj)."""
    rng = np.random.default_rng(seed)
    forest = planner._load_or_create_forest()
    forest.hier_tree = planner.hier_tree

    for qs in [q_start, q_goal]:
        if not planner.hier_tree.is_occupied(qs):
            nid = forest.allocate_id()
            ffb = planner.hier_tree.find_free_box(
                qs, planner.obstacles, mark_occupied=True, forest_box_id=nid)
            if ffb:
                vol = 1.0
                for lo, hi in ffb.intervals:
                    vol *= max(hi - lo, 0)
                forest.add_box_direct(BoxNode(
                    node_id=nid, joint_intervals=ffb.intervals,
                    seed_config=qs.copy(), volume=vol))

    consec = 0
    while consec < max_miss:
        q = planner._sample_seed(q_start, q_goal, rng)
        if q is None:
            consec += 1; continue
        if planner.hier_tree.is_occupied(q):
            consec += 1; continue
        nid = forest.allocate_id()
        ffb = planner.hier_tree.find_free_box(
            q, planner.obstacles, mark_occupied=True, forest_box_id=nid)
        if ffb is None:
            consec += 1; continue
        vol = 1.0
        for lo, hi in ffb.intervals:
            vol *= max(hi - lo, 0)
        if gmean_edge_length(vol, 2) < planner.config.min_box_size:
            consec += 1; continue
        box = BoxNode(node_id=nid, joint_intervals=ffb.intervals,
                      seed_config=q.copy(), volume=vol)
        if ffb.absorbed_box_ids:
            forest.remove_boxes(ffb.absorbed_box_ids)
        forest.add_box_direct(box)
        consec = 0

    boxes = {}
    for bid, b in forest.boxes.items():
        boxes[bid] = BoxNode(
            node_id=b.node_id,
            joint_intervals=[tuple(iv) for iv in b.joint_intervals],
            seed_config=b.seed_config.copy(), volume=b.volume)
    return boxes, forest


# ═══════════════════════════════════════════════════════════════════════════
# Build adjacency graph  (boxes + bridge edges → networkx-style adj list)
# ═══════════════════════════════════════════════════════════════════════════

def build_adjacency(boxes: Dict[int, BoxNode],
                    forest_adj: Dict[int, Set[int]],
                    bridge_edges: list,
                    bridge_boxes: list,
                    period: Optional[float] = None,
                    ) -> Dict[int, Set[int]]:
    """合并 forest 邻接 + bridge edge 得到完整无向邻接表。"""
    adj: Dict[int, Set[int]] = {bid: set() for bid in boxes}

    # forest 邻接 (box overlap)
    for bid, nbrs in forest_adj.items():
        if bid in adj:
            for nb in nbrs:
                if nb in adj:
                    adj[bid].add(nb)
                    adj[nb].add(bid)

    # bridge boxes — 它们已在 boxes 中, 通过 overlap 检测与已有 box 邻接
    for bb in bridge_boxes:
        if bb.node_id not in adj:
            adj[bb.node_id] = set()
        for bid, b in boxes.items():
            if bid == bb.node_id:
                continue
            if _overlap_periodic(bb, b, period):
                adj[bb.node_id].add(bid)
                adj[bid].add(bb.node_id)

    # bridge edges (segment) — 直接拿 source/target box 建边
    for e in bridge_edges:
        s, t = e.source_box_id, e.target_box_id
        if s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)

    return adj


# ═══════════════════════════════════════════════════════════════════════════
# Find source / target boxes
# ═══════════════════════════════════════════════════════════════════════════

def find_box_containing(q: np.ndarray, boxes: Dict[int, BoxNode]) -> Optional[int]:
    """返回包含 q 的 box id, 若无则返回距离最近的。"""
    best_id, best_d = None, float("inf")
    for bid, b in boxes.items():
        if b.contains(q):
            return bid
        d = b.distance_to_config(q)
        if d < best_d:
            best_d = d
            best_id = bid
    return best_id


# ═══════════════════════════════════════════════════════════════════════════
# Graph pruning — BFS 只保留 source→target 的连通分量
# ═══════════════════════════════════════════════════════════════════════════

def extract_connected_subgraph(
    adj: Dict[int, Set[int]],
    source_id: int,
    target_id: int,
) -> Optional[Set[int]]:
    """BFS 找 source 所在连通分量, 若 target 不在其中返回 None."""
    visited = set()
    queue = [source_id]
    visited.add(source_id)
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, set()):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    if target_id not in visited:
        return None
    return visited


def corridor_prune(
    adj: Dict[int, Set[int]],
    source_id: int,
    target_id: int,
    hops: int = 2,
) -> Optional[Set[int]]:
    """保留最短路径 ± hops 跳范围内的顶点 (corridor 子图).

    1. BFS source→target 得到最短路径上的 parent 树
    2. 回溯得到最短路径顶点集 P
    3. 从 P 中每个顶点做 BFS 扩展 hops 跳
    4. 合并得到 corridor 顶点集

    如果 source / target 不连通返回 None.
    """
    from collections import deque

    # step 1: BFS from source
    parent: Dict[int, Optional[int]] = {source_id: None}
    queue = deque([source_id])
    while queue:
        u = queue.popleft()
        if u == target_id:
            break
        for v in adj.get(u, set()):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if target_id not in parent:
        return None

    # step 2: backtrack to get shortest path vertices
    path_verts: Set[int] = set()
    cur = target_id
    while cur is not None:
        path_verts.add(cur)
        cur = parent[cur]

    # step 3: expand each path vertex by `hops` BFS jumps
    corridor: Set[int] = set(path_verts)
    for pv in path_verts:
        frontier = {pv}
        for _ in range(hops):
            next_frontier: Set[int] = set()
            for u in frontier:
                for v in adj.get(u, set()):
                    if v not in corridor:
                        next_frontier.add(v)
            corridor |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

    return corridor


# ═══════════════════════════════════════════════════════════════════════════
# GCS Shortest Path (convex relaxation via cvxpy)
# ═══════════════════════════════════════════════════════════════════════════

def solve_gcs(
    boxes: Dict[int, BoxNode],
    adj: Dict[int, Set[int]],
    source_id: int,
    target_id: int,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    ndim: int = 2,
    corridor_hops: int = 2,
) -> Tuple[bool, float, List[np.ndarray], List[int]]:
    """用 GCS 凸松弛求最短路径。

    模型 (Marcucci et al. 2023, Section III):
      对每条有向边 e=(u,v):
        - φ_e ∈ [0,1]  (flow, 松弛后连续)
        - x_u^e ∈ ℝ^d, x_v^e ∈ ℝ^d  (边两端在各自 box 内的点)
      约束:
        - 流守恒: Σ_{e out of v} φ_e - Σ_{e into v} φ_e = { 1 if source, -1 if target, 0 otherwise }
        - perspective box: φ_e·lo_u ≤ x_u^e ≤ φ_e·hi_u  (对所有维度)
        - 同理 x_v^e
        - 源点固定: Σ_{e out of s} x_s^e = q_start
        - 汇点固定: Σ_{e into t} x_t^e = q_goal
        - 连续性: 对每个中间顶点 v: Σ_{e into v} x_v^e = Σ_{e out of v} x_v^e
      目标:
        min Σ_e ‖x_u^e - x_v^e‖₂   (SOCP)

    Returns:
        (success, cost, waypoints, box_sequence)
    """
    # corridor pruning: 最短路径 ± corridor_hops 跳
    reachable = corridor_prune(adj, source_id, target_id, hops=corridor_hops)
    if reachable is None:
        print(f"    [GCS] source {source_id} and target {target_id} are disconnected")
        return False, float("inf"), [], []

    # 子图
    sub_adj: Dict[int, Set[int]] = {
        u: adj[u] & reachable for u in reachable
    }

    # 构建有向边列表 (双向)
    edges = []
    edge_set = set()
    for u, nbrs in sub_adj.items():
        for v in nbrs:
            if (u, v) not in edge_set:
                edges.append((u, v))
                edge_set.add((u, v))
            if (v, u) not in edge_set:
                edges.append((v, u))
                edge_set.add((v, u))

    ne = len(edges)
    if ne == 0:
        return False, float("inf"), [], []

    box_ids = sorted(reachable)
    nv = len(box_ids)
    print(f"    [GCS] subgraph: {nv} vertices, {ne} directed edges")
    id2idx = {bid: i for i, bid in enumerate(box_ids)}
    src_idx = id2idx[source_id]
    tgt_idx = id2idx[target_id]

    # ── 决策变量 ──
    phi = cp.Variable(ne, nonneg=True)  # flow
    xu = cp.Variable((ne, ndim))         # edge source point (perspective-scaled)
    xv = cp.Variable((ne, ndim))         # edge target point (perspective-scaled)

    constraints = []

    # ── 流约束: φ ≤ 1 ──
    constraints.append(phi <= 1.0)

    # ── 流守恒 ──
    # out_edges[v] = 流出 v 的边, in_edges[v] = 流入 v 的边
    out_edges = [[] for _ in range(nv)]
    in_edges = [[] for _ in range(nv)]
    for ei, (u, v) in enumerate(edges):
        out_edges[id2idx[u]].append(ei)
        in_edges[id2idx[v]].append(ei)

    for vi in range(nv):
        flow_out = cp.sum(phi[out_edges[vi]]) if out_edges[vi] else 0.0
        flow_in = cp.sum(phi[in_edges[vi]]) if in_edges[vi] else 0.0
        if vi == src_idx:
            constraints.append(flow_out - flow_in == 1.0)
        elif vi == tgt_idx:
            constraints.append(flow_out - flow_in == -1.0)
        else:
            constraints.append(flow_out - flow_in == 0.0)

    # ── perspective box 约束 ──
    # φ_e · lo_u ≤ x_u^e ≤ φ_e · hi_u  (对每条边的 u 端)
    for ei, (u, v) in enumerate(edges):
        box_u = boxes[u]
        box_v = boxes[v]
        for d in range(ndim):
            lo_u, hi_u = box_u.joint_intervals[d]
            lo_v, hi_v = box_v.joint_intervals[d]
            constraints.append(xu[ei, d] >= lo_u * phi[ei])
            constraints.append(xu[ei, d] <= hi_u * phi[ei])
            constraints.append(xv[ei, d] >= lo_v * phi[ei])
            constraints.append(xv[ei, d] <= hi_v * phi[ei])

    # ── 源点固定 ──
    if out_edges[src_idx]:
        constraints.append(
            cp.sum(xu[out_edges[src_idx], :], axis=0) == q_start
        )

    # ── 汇点固定 ──
    if in_edges[tgt_idx]:
        constraints.append(
            cp.sum(xv[in_edges[tgt_idx], :], axis=0) == q_goal
        )

    # ── 连续性: 对每个中间顶点 v, Σ_{e in} x_v^e = Σ_{e out} x_v^e ──
    for vi in range(nv):
        if vi == src_idx or vi == tgt_idx:
            continue
        if in_edges[vi] and out_edges[vi]:
            constraints.append(
                cp.sum(xv[in_edges[vi], :], axis=0) ==
                cp.sum(xu[out_edges[vi], :], axis=0)
            )

    # ── 目标: min Σ_e ‖x_u^e - x_v^e‖₂  (SOCP) ──
    cost_terms = []
    for ei in range(ne):
        cost_terms.append(cp.norm(xu[ei, :] - xv[ei, :], 2))
    objective = cp.Minimize(cp.sum(cost_terms))

    # ── 求解 + rounding (尝试多个 solver) ──
    prob = cp.Problem(objective, constraints)

    for solver_name, solver_kwargs in [
        ("CLARABEL", dict(solver=cp.CLARABEL, verbose=False)),
        ("SCS", dict(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-6)),
    ]:
        prob.solve(**solver_kwargs)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            print(f"    [GCS] {solver_name}: {prob.status}")
            continue
        print(f"    [GCS] {solver_name}: {prob.status}, obj={prob.value:.4f}")

        # ── rounding: 收集多种 rounding 候选, 取最优 ──
        rounding_args = (
            phi.value, xu.value, xv.value,
            edges, out_edges, id2idx,
            boxes, source_id, target_id, q_start, q_goal, ndim, nv,
        )
        candidates = []  # [(name, (ok, raw_cost, wps, bseq)), ...]

        for round_fn, round_name in [
            (_round_greedy, "greedy"),
            (_round_dfs, "DFS"),
        ]:
            result = round_fn(*rounding_args)
            if result is not None:
                candidates.append((round_name, result))

        # 随机 rounding: 20 次采样
        for trial in range(20):
            rng_trial = np.random.default_rng(trial * 7 + 13)
            result = _round_random_once(*rounding_args, rng=rng_trial)
            if result is not None:
                candidates.append((f"rand{trial}", result))

        if not candidates:
            print(f"    [GCS] {solver_name}: all rounding failed, trying next solver")
            continue

        # 按 raw cost 排序, 对 top-5 做 SOCP 精炼
        candidates.sort(key=lambda x: x[1][1])
        best_result = None
        best_cost = float("inf")
        best_name = ""

        for name, (ok, raw_cost, wps, bseq) in candidates[:5]:
            wps_r, refined_cost = _refine_path_in_boxes(
                wps, bseq, boxes, q_start, q_goal, ndim)
            if refined_cost < best_cost:
                best_cost = refined_cost
                best_result = (ok, refined_cost, wps_r, bseq)
                best_name = name

        if best_result is not None:
            print(f"    [GCS] best rounding: {best_name}, "
                  f"refined={best_cost:.4f} "
                  f"(from {len(candidates)} candidates)")
            return best_result
        print(f"    [GCS] {solver_name}: refinement failed, trying next solver")

    return False, float("inf"), [], []


def _refine_path_in_boxes(
    waypoints: List[np.ndarray],
    box_seq: List[int],
    boxes: Dict[int, BoxNode],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    ndim: int,
) -> Tuple[List[np.ndarray], float]:
    """SOCP 精炼: 在已有 box 序列内重新优化 waypoint 位置.

    solve:  min  Σ ‖w_{i+1} - w_i‖₂
            s.t. w_0 = q_start,  w_m = q_goal,  w_i ∈ B_i

    Returns:
        (refined_waypoints, refined_cost)
    """
    m = len(waypoints)
    if m <= 2:
        cost = float(np.linalg.norm(q_goal - q_start))
        return waypoints, cost

    n_free = m - 2  # 中间 waypoints (不含起终点)
    w = cp.Variable((n_free, ndim))

    constraints = []
    for i in range(n_free):
        box = boxes[box_seq[i + 1]]
        for d in range(ndim):
            lo, hi = box.joint_intervals[d]
            constraints.append(w[i, d] >= lo)
            constraints.append(w[i, d] <= hi)

    # 路径代价: q_start → w[0] → ... → w[-1] → q_goal
    segs = []
    segs.append(cp.norm(w[0, :] - q_start, 2))
    for i in range(n_free - 1):
        segs.append(cp.norm(w[i + 1, :] - w[i, :], 2))
    segs.append(cp.norm(q_goal - w[n_free - 1, :], 2))

    prob = cp.Problem(cp.Minimize(cp.sum(segs)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        # fallback 返回原始 waypoints
        cost = sum(float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
                   for i in range(m - 1))
        return waypoints, cost

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        refined = [q_start.copy()]
        for i in range(n_free):
            pt = w.value[i].copy()
            box = boxes[box_seq[i + 1]]
            for d in range(ndim):
                lo, hi = box.joint_intervals[d]
                pt[d] = np.clip(pt[d], lo, hi)
            refined.append(pt)
        refined.append(q_goal.copy())
        cost = sum(float(np.linalg.norm(refined[i+1] - refined[i]))
                   for i in range(len(refined) - 1))
        return refined, cost

    cost = sum(float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
               for i in range(m - 1))
    return waypoints, cost


def _deperspective_point(xv_val, phi_val, ei, boxes, v, ndim):
    """从 perspective 变量恢复实际坐标并 clip 到 box."""
    ph = max(phi_val[ei], 1e-10)
    pt = xv_val[ei] / ph
    box_v = boxes[v]
    for d in range(ndim):
        lo, hi = box_v.joint_intervals[d]
        pt[d] = np.clip(pt[d], lo, hi)
    return pt


def _round_greedy(
    phi_val, xu_val, xv_val,
    edges, out_edges, id2idx,
    boxes, source_id, target_id, q_start, q_goal, ndim, nv,
) -> Optional[Tuple[bool, float, List[np.ndarray], List[int]]]:
    """Greedy flow tracing rounding. 返回 None 表示失败。"""
    waypoints = [q_start.copy()]
    box_seq = [source_id]
    current = source_id

    visited_edges = set()
    max_steps = nv + 10

    for _ in range(max_steps):
        if current == target_id:
            break
        ci = id2idx[current]
        best_ei, best_phi = -1, -1e-9
        for ei in out_edges[ci]:
            if ei not in visited_edges and phi_val[ei] > best_phi:
                best_phi = phi_val[ei]
                best_ei = ei
        if best_ei < 0 or best_phi < 1e-8:
            break
        visited_edges.add(best_ei)
        u, v = edges[best_ei]
        pt_v = _deperspective_point(xv_val, phi_val, best_ei, boxes, v, ndim)
        waypoints.append(pt_v.copy())
        box_seq.append(v)
        current = v

    if current != target_id:
        return None

    waypoints[-1] = q_goal.copy()
    total_cost = sum(
        float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
        for i in range(len(waypoints) - 1)
    )
    return True, total_cost, waypoints, box_seq


def _round_dfs(
    phi_val, xu_val, xv_val,
    edges, out_edges, id2idx,
    boxes, source_id, target_id, q_start, q_goal, ndim, nv,
    phi_threshold: float = 1e-4,
) -> Optional[Tuple[bool, float, List[np.ndarray], List[int]]]:
    """DFS with backtracking rounding. 只走 phi > threshold 的边, 按 phi 降序尝试。"""
    # 预计算: 每个顶点的出边按 phi 降序排列
    sorted_out: List[List[int]] = [[] for _ in range(nv)]
    for vi in range(nv):
        eis = [ei for ei in out_edges[vi] if phi_val[ei] > phi_threshold]
        eis.sort(key=lambda ei: -phi_val[ei])
        sorted_out[vi] = eis

    # DFS stack: (current_vertex, edge_index_in_sorted_out, visited_verts)
    src_idx = id2idx[source_id]
    stack = [(source_id, 0, {source_id})]
    path_edges: List[int] = []

    while stack:
        cur, ei_pos, visited = stack[-1]
        ci = id2idx[cur]

        if cur == target_id:
            break

        # 尝试下一条出边
        found_next = False
        while ei_pos < len(sorted_out[ci]):
            ei = sorted_out[ci][ei_pos]
            ei_pos += 1
            stack[-1] = (cur, ei_pos, visited)
            u, v = edges[ei]
            if v not in visited:
                path_edges.append(ei)
                new_visited = visited | {v}
                stack.append((v, 0, new_visited))
                found_next = True
                break

        if not found_next:
            stack.pop()
            if path_edges:
                path_edges.pop()

    if not stack or stack[-1][0] != target_id:
        return None

    # 从 path_edges 构建 waypoints
    waypoints = [q_start.copy()]
    box_seq = [source_id]
    for ei in path_edges:
        u, v = edges[ei]
        pt_v = _deperspective_point(xv_val, phi_val, ei, boxes, v, ndim)
        waypoints.append(pt_v.copy())
        box_seq.append(v)

    waypoints[-1] = q_goal.copy()
    total_cost = sum(
        float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
        for i in range(len(waypoints) - 1)
    )
    return True, total_cost, waypoints, box_seq


def _round_random_once(
    phi_val, xu_val, xv_val,
    edges, out_edges, id2idx,
    boxes, source_id, target_id, q_start, q_goal, ndim, nv,
    rng: np.random.Generator,
    phi_threshold: float = 1e-4,
    noise_scale: float = 0.3,
) -> Optional[Tuple[bool, float, List[np.ndarray], List[int]]]:
    """Randomized DFS rounding: 对 phi 加噪声改变边排序, DFS 回溯找路径."""
    # 对每个顶点的出边按 (phi + noise) 降序排列
    sorted_out: List[List[int]] = [[] for _ in range(nv)]
    for vi in range(nv):
        eis = [ei for ei in out_edges[vi] if phi_val[ei] > phi_threshold]
        if not eis:
            sorted_out[vi] = []
            continue
        phi_arr = np.array([phi_val[ei] for ei in eis])
        noise = rng.uniform(0, noise_scale * phi_arr.max(), size=len(eis))
        order = np.argsort(-(phi_arr + noise))
        sorted_out[vi] = [eis[i] for i in order]

    # DFS with backtracking (same as _round_dfs but with perturbed order)
    stack = [(source_id, 0, {source_id})]
    path_edges: List[int] = []

    while stack:
        cur, ei_pos, visited = stack[-1]
        ci = id2idx[cur]
        if cur == target_id:
            break
        found_next = False
        while ei_pos < len(sorted_out[ci]):
            ei = sorted_out[ci][ei_pos]
            ei_pos += 1
            stack[-1] = (cur, ei_pos, visited)
            u, v = edges[ei]
            if v not in visited:
                path_edges.append(ei)
                stack.append((v, 0, visited | {v}))
                found_next = True
                break
        if not found_next:
            stack.pop()
            if path_edges:
                path_edges.pop()

    if not stack or stack[-1][0] != target_id:
        return None

    waypoints = [q_start.copy()]
    box_seq = [source_id]
    for ei in path_edges:
        u, v = edges[ei]
        pt_v = _deperspective_point(xv_val, phi_val, ei, boxes, v, ndim)
        waypoints.append(pt_v.copy())
        box_seq.append(v)

    waypoints[-1] = q_goal.copy()
    total_cost = sum(
        float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
        for i in range(len(waypoints) - 1)
    )
    return True, total_cost, waypoints, box_seq


# ═══════════════════════════════════════════════════════════════════════════
# Collision map  (复用)
# ═══════════════════════════════════════════════════════════════════════════

def scan_collision_map(robot, scene, joint_limits, resolution=0.03):
    checker = CollisionChecker(robot=robot, scene=scene)
    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]
    xs = np.arange(lo_x, hi_x, resolution)
    ys = np.arange(lo_y, hi_y, resolution)
    cmap = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for i, y in enumerate(ys):
        row = np.column_stack([xs, np.full(len(xs), y)])
        cmap[i, :] = checker.check_config_collision_batch(row).astype(np.float32)
    return cmap, [lo_x, hi_x, lo_y, hi_y]


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_gcs_result(
    boxes, adj, islands, bridge_edges, bridge_boxes,
    waypoints, box_seq, collision_map, extent,
    q_start, q_goal, title, cost,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(collision_map, origin="lower", extent=extent,
              cmap="Reds", alpha=0.30, aspect="auto")

    # 岛着色
    n_islands = len(islands)
    cmap_isl = plt.cm.tab20
    node_island = {}
    for idx, island in enumerate(islands):
        for bid in island:
            node_island[bid] = idx
    bridge_box_ids = {b.node_id for b in bridge_boxes}
    path_box_ids = set(box_seq)

    for bid, box in boxes.items():
        lo_x, hi_x = box.joint_intervals[0]
        lo_y, hi_y = box.joint_intervals[1]
        isl_idx = node_island.get(bid, 0)
        c = cmap_isl(isl_idx / max(n_islands, 1))

        if bid in path_box_ids:
            rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                              linewidth=1.5, edgecolor="blue",
                              facecolor=c, alpha=0.45, zorder=3)
        elif bid in bridge_box_ids:
            rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                              linewidth=1.5, edgecolor="lime",
                              facecolor=c, alpha=0.40, zorder=2)
        else:
            rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                              linewidth=0.4, edgecolor=c,
                              facecolor=c, alpha=0.22)
        ax.add_patch(rect)

    # bridge segment edges
    for e in bridge_edges:
        ax.plot([e.source_config[0], e.target_config[0]],
                [e.source_config[1], e.target_config[1]],
                color="lime", linewidth=1.5, alpha=0.7, zorder=4)

    # GCS path
    if waypoints and len(waypoints) >= 2:
        xs = [w[0] for w in waypoints]
        ys = [w[1] for w in waypoints]
        ax.plot(xs, ys, '-o', color='#0066ff', linewidth=2.5,
                markersize=5, markerfacecolor='white',
                markeredgecolor='#0066ff', markeredgewidth=1.2,
                zorder=8, label=f'GCS path (cost={cost:.3f})')

    # start / goal
    ax.plot(q_start[0], q_start[1], 'o', color='cyan', markersize=10,
            markeredgecolor='black', markeredgewidth=1.2, zorder=10,
            label='start')
    ax.plot(q_goal[0], q_goal[1], '*', color='yellow', markersize=14,
            markeredgecolor='black', markeredgewidth=1.0, zorder=10,
            label='goal')

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("q0 (rad)")
    ax.set_ylabel("q1 (rad)")
    n_boxes = len(boxes)
    ax.set_title(f"{title}  |  {n_boxes} boxes, {n_islands} islands, "
                 f"path cost={cost:.3f}", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    return fig


def plot_coarsen_comparison(
    boxes_before: Dict[int, BoxNode],
    boxes_after: Dict[int, BoxNode],
    collision_map, extent,
    q_start, q_goal,
    stats,
):
    """并排对比 coarsen 前后的 box 分布."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, bxs, label in [
        (ax_l, boxes_before, f"Before coarsen ({len(boxes_before)} boxes)"),
        (ax_r, boxes_after,  f"After coarsen ({len(boxes_after)} boxes)"),
    ]:
        ax.imshow(collision_map, origin="lower", extent=extent,
                  cmap="Reds", alpha=0.30, aspect="auto")

        # 按面积排序: 大 box 先画, 小 box 在上层
        sorted_boxes = sorted(bxs.values(), key=lambda b: b.volume, reverse=True)
        vol_max = max((b.volume for b in sorted_boxes), default=1.0)

        for box in sorted_boxes:
            lo_x, hi_x = box.joint_intervals[0]
            lo_y, hi_y = box.joint_intervals[1]
            # 面积越大 → 颜色越深
            frac = min(box.volume / vol_max, 1.0) if vol_max > 0 else 0.5
            c = plt.cm.viridis(0.2 + 0.6 * frac)
            rect = Rectangle(
                (lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                linewidth=0.5, edgecolor=(0, 0, 0, 0.5),
                facecolor=c, alpha=0.35,
            )
            ax.add_patch(rect)

        ax.plot(q_start[0], q_start[1], 'o', color='cyan', markersize=8,
                markeredgecolor='black', markeredgewidth=1.0, zorder=10)
        ax.plot(q_goal[0], q_goal[1], '*', color='yellow', markersize=12,
                markeredgecolor='black', markeredgewidth=0.8, zorder=10)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("q0 (rad)")
        ax.set_ylabel("q1 (rad)")
        ax.set_title(label, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Coarsen: {stats.n_before} → {stats.n_after} boxes  "
        f"({stats.n_merges} merges, {stats.time_ms:.0f} ms)",
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    return fig


def plot_path_comparison(
    result_no: dict,
    result_yes: dict,
    collision_map, extent,
    q_start, q_goal,
):
    """Side-by-side: GCS path without vs with coarsening."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 8))

    for ax, res, label in [
        (ax_l, result_no,  "Without Coarsen"),
        (ax_r, result_yes, "With Dim-Sweep Coarsen"),
    ]:
        boxes      = res['boxes']
        waypoints  = res['waypoints']
        cost       = res['cost']
        box_seq    = res['box_seq']
        islands    = res['islands']
        bridge_boxes  = res['bridge_boxes']
        bridge_edges  = res['bridge_edges']

        ax.imshow(collision_map, origin='lower', extent=extent,
                  cmap='Reds', alpha=0.30, aspect='auto')

        # island colouring
        n_islands = len(islands)
        cmap_isl = plt.cm.tab20
        node_island = {}
        for idx, island in enumerate(islands):
            for bid in island:
                node_island[bid] = idx
        bridge_box_ids = {b.node_id for b in bridge_boxes}
        path_box_ids   = set(box_seq)

        for bid, box in boxes.items():
            lo_x, hi_x = box.joint_intervals[0]
            lo_y, hi_y = box.joint_intervals[1]
            isl_idx = node_island.get(bid, 0)
            c = cmap_isl(isl_idx / max(n_islands, 1))

            if bid in path_box_ids:
                rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                                 linewidth=1.5, edgecolor="blue",
                                 facecolor=c, alpha=0.45, zorder=3)
            elif bid in bridge_box_ids:
                rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                                 linewidth=1.5, edgecolor="lime",
                                 facecolor=c, alpha=0.40, zorder=2)
            else:
                rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                                 linewidth=0.4, edgecolor=c,
                                 facecolor=c, alpha=0.22)
            ax.add_patch(rect)

        # bridge segments
        for e in bridge_edges:
            ax.plot([e.source_config[0], e.target_config[0]],
                    [e.source_config[1], e.target_config[1]],
                    color="lime", linewidth=1.5, alpha=0.7, zorder=4)

        # GCS path
        if waypoints and len(waypoints) >= 2:
            xs = [w[0] for w in waypoints]
            ys = [w[1] for w in waypoints]
            ax.plot(xs, ys, '-o', color='#0066ff', linewidth=2.5,
                    markersize=5, markerfacecolor='white',
                    markeredgecolor='#0066ff', markeredgewidth=1.2,
                    zorder=8)

        # start / goal
        ax.plot(q_start[0], q_start[1], 'o', color='cyan', markersize=10,
                markeredgecolor='black', markeredgewidth=1.2, zorder=10)
        ax.plot(q_goal[0], q_goal[1], '*', color='yellow', markersize=14,
                markeredgecolor='black', markeredgewidth=1.0, zorder=10)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("q0 (rad)")
        ax.set_ylabel("q1 (rad)")
        nb    = len(boxes)
        n_wp  = len(waypoints)
        gms   = res['gcs_ms']
        total = res['grow_ms'] + res['coarsen_ms'] + res['bridge_ms'] + gms
        ax.set_title(
            f"{label}\n{nb} boxes | cost={cost:.2f} | "
            f"{n_wp} wp | GCS {gms:.0f}ms | total {total:.0f}ms",
            fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    fig.suptitle("GCS Planning Path Comparison",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline helper
# ═══════════════════════════════════════════════════════════════════════════

def _run_gcs_pipeline(robot, scene, cfg, q_start, q_goal, period, ndim,
                      with_coarsen, label=""):
    """Run grow → (coarsen) → bridge → adjacency → GCS.  Returns result dict."""
    planner_cfg = make_planner_config(cfg)
    planner = BoxPlanner(robot=robot, scene=scene, config=planner_cfg)

    t0 = time.perf_counter()
    boxes, forest_obj = grow_forest(planner, q_start, q_goal, cfg.seed,
                                     cfg.max_consecutive_miss)
    grow_ms = (time.perf_counter() - t0) * 1000
    n_grown = len(forest_obj.boxes)

    # optional coarsen
    coarsen_stats = None
    boxes_before_coarsen = None
    if with_coarsen:
        boxes_before_coarsen = {
            bid: BoxNode(node_id=b.node_id,
                         joint_intervals=[tuple(iv) for iv in b.joint_intervals],
                         seed_config=b.seed_config.copy(), volume=b.volume)
            for bid, b in forest_obj.boxes.items()
        }
        coarsen_stats = coarsen_forest(
            tree=planner.hier_tree, forest=forest_obj,
            obstacles=planner.obstacles, safety_margin=0.0,
            max_rounds=cfg.coarsen_max_rounds,
        )
    coarsen_ms = coarsen_stats.time_ms if coarsen_stats else 0.0
    boxes = forest_obj.boxes

    # bridge
    t0 = time.perf_counter()
    bridge_result = bridge_islands(
        boxes=boxes,
        collision_checker=planner.collision_checker,
        segment_resolution=0.03,
        max_pairs_per_island_pair=10,
        max_rounds=5,
        period=period,
        hier_tree=planner.hier_tree,
        obstacles=planner.obstacles,
        forest=forest_obj,
        min_box_size=cfg.min_box_size,
        n_bridge_seeds=7,
        min_island_size=cfg.min_island_size,
    )
    bridge_edges, final_islands, n_before, bridge_boxes, discarded = bridge_result
    bridge_ms = (time.perf_counter() - t0) * 1000
    boxes = forest_obj.boxes  # includes bridge boxes

    # adjacency
    adj = build_adjacency(boxes, forest_obj.adjacency,
                          bridge_edges, bridge_boxes, period)

    # source / target
    src = find_box_containing(q_start, boxes)
    tgt = find_box_containing(q_goal, boxes)
    if src is None or tgt is None:
        print(f"  [{label}] ERROR: start or goal not in any box")
        return None

    # GCS — coarsen 后 box 更大, corridor 多扩 1 跳补偿
    n_now = len(boxes)
    if with_coarsen and n_now < n_grown:
        hops = cfg.corridor_hops + 1
    else:
        hops = cfg.corridor_hops
    t0 = time.perf_counter()
    success, cost, waypoints, box_seq = solve_gcs(
        boxes, adj, src, tgt, q_start, q_goal, ndim,
        corridor_hops=hops)
    gcs_ms = (time.perf_counter() - t0) * 1000

    if success:
        print(f"  [{label}] {len(boxes)} boxes, cost={cost:.4f}, "
              f"{len(waypoints)} wp, GCS {gcs_ms:.0f}ms")
    else:
        print(f"  [{label}] GCS FAILED ({gcs_ms:.0f}ms)")

    return dict(
        boxes=dict(boxes), success=success, cost=cost,
        waypoints=waypoints, box_seq=box_seq,
        islands=final_islands, n_before_islands=n_before,
        bridge_edges=bridge_edges, bridge_boxes=bridge_boxes,
        adj=adj, grow_ms=grow_ms, coarsen_ms=coarsen_ms,
        bridge_ms=bridge_ms, gcs_ms=gcs_ms,
        coarsen_stats=coarsen_stats,
        boxes_before_coarsen=boxes_before_coarsen,
        n_grown=n_grown,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = GCSConfig()
    rng = np.random.default_rng(cfg.seed)
    robot = load_robot(cfg.robot_name)
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    ndim = len(robot.joint_limits)

    jl = robot.joint_limits[0]
    period = float(jl[1] - jl[0])

    # 1) scene
    print("building scene ...")
    scene = build_random_scene(robot, q_start, q_goal, rng, cfg)

    # 2) collision map
    print("scanning collision map ...")
    cmap, extent = scan_collision_map(robot, scene, robot.joint_limits,
                                      cfg.collision_map_resolution)

    # 3) run two pipelines for comparison
    print("\n--- Pipeline A: no coarsen ---")
    result_no = _run_gcs_pipeline(
        robot, scene, cfg, q_start, q_goal, period, ndim,
        with_coarsen=False, label="no-coarsen")

    print("\n--- Pipeline B: dim-sweep coarsen ---")
    result_yes = _run_gcs_pipeline(
        robot, scene, cfg, q_start, q_goal, period, ndim,
        with_coarsen=True, label="coarsen")

    # 4) visualize
    import matplotlib.pyplot as plt
    out_dir = make_output_dir("visualizations", "gcs_planning_2dof")
    print(f"\noutput: {out_dir}")

    # (a) coarsen comparison (box distribution before / after)
    if result_yes and result_yes['coarsen_stats']:
        fig_cmp = plot_coarsen_comparison(
            result_yes['boxes_before_coarsen'], result_yes['boxes'],
            cmap, extent, q_start, q_goal, result_yes['coarsen_stats'],
        )
        p = out_dir / "coarsen_comparison.png"
        fig_cmp.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig_cmp)
        print(f"  saved: {p}")

    # (b) path comparison (no-coarsen vs coarsen, side-by-side)
    if result_no and result_yes:
        fig_pc = plot_path_comparison(
            result_no, result_yes, cmap, extent, q_start, q_goal,
        )
        p = out_dir / "path_comparison.png"
        fig_pc.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig_pc)
        print(f"  saved: {p}")

    # (c) individual GCS result plots
    for res, tag in [(result_no, "no_coarsen"), (result_yes, "coarsen")]:
        if res and res['success']:
            fig = plot_gcs_result(
                res['boxes'], res['adj'], res['islands'],
                res['bridge_edges'], res['bridge_boxes'],
                res['waypoints'], res['box_seq'], cmap, extent,
                q_start, q_goal,
                title=f"GCS ({tag})", cost=res['cost'],
            )
            p = out_dir / f"gcs_result_{tag}.png"
            fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved: {p}")

    # 5) summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for res, tag in [(result_no, "no-coarsen"), (result_yes, "coarsen")]:
        if res:
            total = res['grow_ms'] + res['coarsen_ms'] + res['bridge_ms'] + res['gcs_ms']
            ok = "OK" if res['success'] else "FAIL"
            print(f"\n  [{tag}]")
            print(f"    boxes      = {res['n_grown']} -> {len(res['boxes'])}")
            print(f"    islands    = {res['n_before_islands']} -> {len(res['islands'])}")
            print(f"    GCS        = {ok}")
            if res['success']:
                print(f"    cost       = {res['cost']:.4f}")
                print(f"    waypoints  = {len(res['waypoints'])}")
            print(f"    grow       = {res['grow_ms']:.1f} ms")
            print(f"    coarsen    = {res['coarsen_ms']:.1f} ms")
            print(f"    bridge     = {res['bridge_ms']:.1f} ms")
            print(f"    gcs        = {res['gcs_ms']:.1f} ms")
            print(f"    total      = {total:.1f} ms")

    if (result_no and result_yes
            and result_no['success'] and result_yes['success']):
        t_no  = result_no['grow_ms']  + result_no['bridge_ms']  + result_no['gcs_ms']
        t_yes = (result_yes['grow_ms'] + result_yes['coarsen_ms']
                 + result_yes['bridge_ms'] + result_yes['gcs_ms'])
        speedup = t_no / t_yes if t_yes > 0 else 0
        print(f"\n  speedup = {speedup:.2f}x  "
              f"({t_no:.0f}ms -> {t_yes:.0f}ms)")


if __name__ == "__main__":
    main()
