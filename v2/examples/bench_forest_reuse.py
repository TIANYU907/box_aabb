"""
bench_forest_reuse.py -- Forest incremental reuse benchmark

Same obstacle scene, different start/goal pairs:
  1. Build shared forest (grow + coarsen) once
  2. Compute adjacency + UnionFind once
  3. Each query:
     a) Expand box at q_start / q_goal via find_free_box
     b) Incrementally update adj + uf
     c) Bridge only if s-t disconnected (bridge boxes persist)
     d) Dijkstra + SOCP

Forest grows monotonically across queries; adjacency/bridge results
are reused by subsequent queries.

Usage:
    python -m v2.examples.bench_forest_reuse
    python -m v2.examples.bench_forest_reuse --seed 42 --queries 10 --obstacles 8
    python -m v2.examples.bench_forest_reuse --seed 42 --queries 5 --full-pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

import os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
from _bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.models import BoxNode
from forest.connectivity import bridge_islands, UnionFind
from forest.box_forest import BoxForest
from forest.coarsen import coarsen_forest
from planner.box_planner import BoxPlanner
from planner.models import PlannerConfig, gmean_edge_length
from common.output import make_output_dir

from v2.examples.panda_planner import (
    PandaGCSConfig,
    build_panda_scene,
    make_planner_config,
    grow_forest,
    _build_adjacency_and_islands,
    _solve_method_dijkstra,
    _save_plotly_html,
    plot_arm_scene_html,
)
from v2.examples.gcs_planner_2dof import (
    find_box_containing,
    _refine_path_in_boxes,
)


# =====================================================================
# Helper: generate random collision-free query pairs from the forest
# =====================================================================

def sample_query_pairs(
    boxes: Dict[int, BoxNode],
    collision_checker: CollisionChecker,
    n_queries: int,
    rng: np.random.Generator,
    min_config_dist: float = 1.0,
    max_attempts: int = 500,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample n_queries start/goal pairs from box forest."""
    box_list = list(boxes.values())
    if len(box_list) < 2:
        raise RuntimeError("Forest has <2 boxes, cannot sample pairs")

    ndim = box_list[0].n_dims
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for _ in range(max_attempts):
        if len(pairs) >= n_queries:
            break

        idx_s, idx_g = rng.choice(len(box_list), size=2, replace=False)
        box_s, box_g = box_list[idx_s], box_list[idx_g]

        q_s = np.array([rng.uniform(lo, hi) for lo, hi in box_s.joint_intervals])
        q_g = np.array([rng.uniform(lo, hi) for lo, hi in box_g.joint_intervals])

        dist = float(np.linalg.norm(q_g - q_s))
        if dist < min_config_dist:
            continue

        if collision_checker.check_config_collision(q_s):
            continue
        if collision_checker.check_config_collision(q_g):
            continue

        pairs.append((q_s, q_g))

    if len(pairs) < n_queries:
        print(f"  [WARN] only sampled {len(pairs)}/{n_queries} query pairs "
              f"(min_dist={min_config_dist:.1f})")
    return pairs


# =====================================================================
# Incremental adjacency helpers
# =====================================================================

def _incremental_adj_for_new_box(
    box: BoxNode,
    boxes: Dict[int, BoxNode],
    adj: Dict[int, Set[int]],
    uf: UnionFind,
):
    """Incrementally compute overlap adjacency for a new box against all
    existing boxes, and update adj + uf.

    Uses the same loose overlap condition as _build_adjacency_and_islands.
    """
    bid = box.node_id
    adj.setdefault(bid, set())
    uf.add(bid)

    ivs = box.joint_intervals
    ndim = len(ivs)
    eps = 1e-12
    lo_new = np.array([iv[0] for iv in ivs], dtype=np.float64)
    hi_new = np.array([iv[1] for iv in ivs], dtype=np.float64)

    for oid, other in boxes.items():
        if oid == bid:
            continue
        oivs = other.joint_intervals
        overlap = True
        for d in range(ndim):
            if hi_new[d] < oivs[d][0] - eps or oivs[d][1] < lo_new[d] - eps:
                overlap = False
                break
        if overlap:
            adj[bid].add(oid)
            adj.setdefault(oid, set()).add(bid)
            uf.union(bid, oid)


def _expand_seed_box(
    q: np.ndarray,
    planner: BoxPlanner,
    forest_obj: BoxForest,
    boxes: Dict[int, BoxNode],
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    min_box_size: float,
) -> Optional[int]:
    """Expand a box at q, add to forest + adj + uf.

    If q is already inside an existing box, return that box id.
    Returns box_id or None.
    """
    # Already covered?
    existing = find_box_containing(q, boxes)
    if existing is not None:
        return existing

    # Already occupied in hier_tree but not in boxes?
    if planner.hier_tree.is_occupied(q):
        return None

    nid = forest_obj.allocate_id()
    ffb = planner.hier_tree.find_free_box(
        q, planner.obstacles,
        mark_occupied=True,
        forest_box_id=nid,
        obs_packed=obs_packed,
    )
    if ffb is None:
        return None

    vol = 1.0
    ndim = len(ffb.intervals)
    for lo, hi in ffb.intervals:
        vol *= max(hi - lo, 0)
    if gmean_edge_length(vol, ndim) < min_box_size:
        return None

    box = BoxNode(
        node_id=nid,
        joint_intervals=ffb.intervals,
        seed_config=q.copy(),
        volume=vol,
    )

    # Handle absorbed boxes
    if ffb.absorbed_box_ids:
        for aid in ffb.absorbed_box_ids:
            boxes.pop(aid, None)
            forest_obj.boxes.pop(aid, None)
            if aid in adj:
                nbrs = adj.pop(aid, set())
                for nb in nbrs:
                    if nb in adj:
                        adj[nb].discard(aid)

    # Add to forest (includes interval cache update)
    forest_obj.add_box_direct(box)
    boxes[nid] = box
    # Incremental adj + uf update
    _incremental_adj_for_new_box(box, boxes, adj, uf)

    return nid


# =====================================================================
# Mode A: Full pipeline (from scratch)
# =====================================================================

def run_full_pipeline(
    robot, scene, cfg: PandaGCSConfig,
    q_start: np.ndarray, q_goal: np.ndarray, ndim: int,
) -> Dict:
    """Full pipeline from scratch: grow -> coarsen -> adj -> bridge -> Dijkstra."""
    t_total = time.perf_counter()

    planner_cfg = make_planner_config(cfg)
    planner = BoxPlanner(robot=robot, scene=scene, config=planner_cfg)

    t0 = time.perf_counter()
    boxes, forest_obj, grow_detail = grow_forest(
        planner, q_start, q_goal, cfg.seed,
        cfg.max_consecutive_miss, ndim,
        max_boxes=cfg.max_boxes)
    grow_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cs = coarsen_forest(
        tree=planner.hier_tree, forest=forest_obj,
        max_rounds=cfg.coarsen_max_rounds)
    coarsen_ms = (time.perf_counter() - t0) * 1000
    boxes = forest_obj.boxes

    solve_result = _run_solve_phase_full(
        boxes, forest_obj, planner, cfg, q_start, q_goal, ndim)

    total_ms = (time.perf_counter() - t_total) * 1000
    return dict(
        success=solve_result['success'],
        cost=solve_result.get('cost', float('inf')),
        n_waypoints=len(solve_result.get('waypoints', [])),
        n_boxes=len(boxes),
        grow_ms=grow_ms,
        coarsen_ms=coarsen_ms,
        adj_ms=solve_result['adj_ms'],
        bridge_ms=solve_result['bridge_ms'],
        plan_ms=solve_result['plan_ms'],
        total_ms=total_ms,
        waypoints=solve_result.get('waypoints', []),
    )


def _run_solve_phase_full(
    boxes: Dict[int, BoxNode],
    forest_obj: BoxForest,
    planner: BoxPlanner,
    cfg: PandaGCSConfig,
    q_start: np.ndarray, q_goal: np.ndarray, ndim: int,
) -> Dict:
    """Full pipeline solve: from-scratch adj -> bridge -> Dijkstra."""

    t0 = time.perf_counter()
    adj, uf, islands = _build_adjacency_and_islands(boxes)
    adj_ms = (time.perf_counter() - t0) * 1000

    src = find_box_containing(q_start, boxes)
    tgt = find_box_containing(q_goal, boxes)
    if src is None or tgt is None:
        return dict(success=False, adj_ms=adj_ms, bridge_ms=0, plan_ms=0)

    bridge_ms = 0.0
    if not uf.same(src, tgt):
        t0 = time.perf_counter()
        bridge_result = bridge_islands(
            boxes=boxes,
            collision_checker=planner.collision_checker,
            segment_resolution=0.03,
            max_pairs_per_island_pair=10,
            max_rounds=5,
            period=None,
            hier_tree=planner.hier_tree,
            obstacles=planner.obstacles,
            forest=forest_obj,
            min_box_size=cfg.min_box_size,
            n_bridge_seeds=7,
            min_island_size=cfg.min_island_size,
            precomputed_uf=uf,
            precomputed_islands=islands,
            target_pair=(src, tgt),
        )
        bridge_edges, final_islands, _, bridge_boxes, _ = bridge_result
        bridge_ms = (time.perf_counter() - t0) * 1000
        boxes = forest_obj.boxes

        for bb in bridge_boxes:
            if bb.node_id not in adj:
                neighbor_ids = forest_obj._adjacent_existing_ids_from_cache(
                    bb, tol=forest_obj.config.adjacency_tolerance)
                adj[bb.node_id] = set(neighbor_ids)
                for nb in neighbor_ids:
                    adj.setdefault(nb, set()).add(bb.node_id)
        for e in bridge_edges:
            s, t = e.source_box_id, e.target_box_id
            if s in adj and t in adj:
                adj[s].add(t)
                adj[t].add(s)
                uf.union(s, t)

        if src not in boxes:
            src = find_box_containing(q_start, boxes)
        if tgt not in boxes:
            tgt = find_box_containing(q_goal, boxes)

    if src is None or tgt is None:
        return dict(success=False, adj_ms=adj_ms, bridge_ms=bridge_ms, plan_ms=0)

    t0 = time.perf_counter()
    plan_result = _solve_method_dijkstra(
        boxes=boxes, adj=adj, src=src, tgt=tgt,
        q_start=q_start, q_goal=q_goal, ndim=ndim, label="Dijkstra")
    plan_ms = (time.perf_counter() - t0) * 1000

    return dict(
        success=plan_result.get('success', False),
        cost=plan_result.get('cost', float('inf')),
        waypoints=plan_result.get('waypoints', []),
        adj_ms=adj_ms,
        bridge_ms=bridge_ms,
        plan_ms=plan_ms,
    )


# =====================================================================
# Mode B: Incremental reuse query
# =====================================================================

def run_incremental_query(
    forest_obj: BoxForest,
    planner: BoxPlanner,
    cfg: PandaGCSConfig,
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    q_start: np.ndarray, q_goal: np.ndarray, ndim: int,
) -> Dict:
    """Incremental reuse: expand seeds -> incr adj/uf -> bridge -> Dijkstra.

    All new boxes / adj / uf persist, enriching subsequent queries.
    """
    t_total = time.perf_counter()
    boxes = forest_obj.boxes
    n_before = len(boxes)

    # (a) Expand q_start / q_goal
    t0 = time.perf_counter()
    src = _expand_seed_box(
        q_start, planner, forest_obj, boxes, adj, uf, obs_packed, cfg.min_box_size)
    tgt = _expand_seed_box(
        q_goal, planner, forest_obj, boxes, adj, uf, obs_packed, cfg.min_box_size)
    expand_ms = (time.perf_counter() - t0) * 1000
    n_expanded = len(boxes) - n_before

    if src is None or tgt is None:
        total_ms = (time.perf_counter() - t_total) * 1000
        return dict(success=False, expand_ms=expand_ms, n_expanded=n_expanded,
                    adj_ms=0, bridge_ms=0, plan_ms=0, total_ms=total_ms,
                    n_boxes=len(boxes), cost=float('inf'))

    # (b) Bridge (lazy: only if s-t disconnected)
    bridge_ms = 0.0
    n_bridge = 0
    if not uf.same(src, tgt):
        t0 = time.perf_counter()
        islands = uf.components()
        bridge_result = bridge_islands(
            boxes=boxes,
            collision_checker=planner.collision_checker,
            segment_resolution=0.03,
            max_pairs_per_island_pair=10,
            max_rounds=5,
            period=None,
            hier_tree=planner.hier_tree,
            obstacles=planner.obstacles,
            forest=forest_obj,
            min_box_size=cfg.min_box_size,
            n_bridge_seeds=7,
            min_island_size=cfg.min_island_size,
            precomputed_uf=uf,
            precomputed_islands=islands,
            target_pair=(src, tgt),
        )
        bridge_edges, final_islands, _, bridge_boxes, _ = bridge_result
        bridge_ms = (time.perf_counter() - t0) * 1000
        n_bridge = len(bridge_boxes)

        # Incremental adj+uf for bridge boxes (persistent)
        for bb in bridge_boxes:
            _incremental_adj_for_new_box(bb, boxes, adj, uf)
        for e in bridge_edges:
            s, t = e.source_box_id, e.target_box_id
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
            uf.union(s, t)

        # Refresh src/tgt in case boxes were absorbed
        if src not in boxes:
            src = find_box_containing(q_start, boxes)
        if tgt not in boxes:
            tgt = find_box_containing(q_goal, boxes)

    if src is None or tgt is None:
        total_ms = (time.perf_counter() - t_total) * 1000
        return dict(success=False, expand_ms=expand_ms, n_expanded=n_expanded,
                    adj_ms=0, bridge_ms=bridge_ms, n_bridge=n_bridge,
                    plan_ms=0, total_ms=total_ms, n_boxes=len(boxes),
                    cost=float('inf'))

    # (c) Dijkstra + SOCP
    t0 = time.perf_counter()
    plan_result = _solve_method_dijkstra(
        boxes=boxes, adj=adj, src=src, tgt=tgt,
        q_start=q_start, q_goal=q_goal, ndim=ndim, label="Dijkstra")
    plan_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return dict(
        success=plan_result.get('success', False),
        cost=plan_result.get('cost', float('inf')),
        n_waypoints=len(plan_result.get('waypoints', [])),
        waypoints=plan_result.get('waypoints', []),
        n_boxes=len(boxes),
        n_expanded=n_expanded,
        n_bridge=n_bridge,
        expand_ms=expand_ms,
        bridge_ms=bridge_ms,
        plan_ms=plan_ms,
        total_ms=total_ms,
    )


# =====================================================================
# Report
# =====================================================================

def print_report(
    seed: int, n_obs: int, n_queries: int,
    shared_grow_ms: float, shared_coarsen_ms: float,
    shared_adj_ms: float,
    n_shared_boxes: int,
    full_results: List[Dict],
    reuse_results: List[Dict],
    query_pairs: List[Tuple[np.ndarray, np.ndarray]],
):
    """Print comparison report."""
    W = 100
    print(f"\n{'=' * W}")
    print(f"  Forest Incremental Reuse Benchmark -- Panda 7-DOF")
    print(f"  seed={seed}, {n_obs} obstacles, {n_queries} queries, "
          f"{n_shared_boxes} initial boxes")
    print(f"{'=' * W}")

    # Shared build
    shared_total = shared_grow_ms + shared_coarsen_ms + shared_adj_ms
    print(f"\n  Shared build (once):")
    print(f"    grow      = {shared_grow_ms:8.1f} ms")
    print(f"    coarsen   = {shared_coarsen_ms:8.1f} ms")
    print(f"    adj+uf    = {shared_adj_ms:8.1f} ms")
    print(f"    total     = {shared_total:8.1f} ms")
    print(f"    boxes     = {n_shared_boxes}")

    # Per-query table
    print(f"\n  {chr(9472) * W}")
    hdr = (f"  {'Q#':>3s}  {'dist':>5s}  "
           f"| {'expd':>4s} {'brdg':>4s} {'#box':>5s}  "
           f"| {'expand':>6s} {'bridge':>6s} {'plan':>5s} {'TOTAL':>6s} {'cost':>7s}  "
           f"| {'Full total':>10s} {'cost':>7s}  "
           f"| {'Speed':>6s}")
    print(hdr)
    print(f"  {chr(9472) * W}")

    speedups = []
    for i, (rr, fr) in enumerate(zip(reuse_results, full_results)):
        q_s, q_g = query_pairs[i]
        dist = float(np.linalg.norm(q_g - q_s))

        r_cost = f"{rr['cost']:.3f}" if rr['success'] else "FAIL"
        f_cost = f"{fr['cost']:.3f}" if fr['success'] else "FAIL"

        r_total = rr['total_ms']
        f_total = fr['total_ms']
        speedup = f_total / r_total if r_total > 0 else float('inf')
        if rr['success'] and fr['success']:
            speedups.append(speedup)

        print(f"  {i:>3d}  {dist:5.1f}  "
              f"| {rr.get('n_expanded', 0):>4d} {rr.get('n_bridge', 0):>4d} {rr['n_boxes']:>5d}  "
              f"| {rr.get('expand_ms', 0):6.0f} {rr['bridge_ms']:6.0f} "
              f"{rr['plan_ms']:5.0f} {r_total:6.0f} {r_cost:>7s}  "
              f"| {f_total:10.0f} {f_cost:>7s}  "
              f"| {speedup:5.1f}x")

    print(f"  {chr(9472) * W}")

    # Summary
    reuse_totals = [r['total_ms'] for r in reuse_results if r['success']]
    full_totals = [r['total_ms'] for r in full_results if r['success']]
    reuse_ok = sum(1 for r in reuse_results if r['success'])
    full_ok = sum(1 for r in full_results if r['success'])

    print(f"\n  Summary:")
    print(f"    Full  pipeline: {full_ok}/{n_queries} success", end="")
    if full_totals:
        print(f", avg={np.mean(full_totals):.0f}ms, "
              f"med={np.median(full_totals):.0f}ms")
    else:
        print()

    print(f"    Reuse pipeline: {reuse_ok}/{n_queries} success", end="")
    if reuse_totals:
        print(f", avg={np.mean(reuse_totals):.0f}ms, "
              f"med={np.median(reuse_totals):.0f}ms")
    else:
        print()

    if speedups:
        print(f"    Speedup:  avg={np.mean(speedups):.1f}x, "
              f"med={np.median(speedups):.1f}x, "
              f"min={np.min(speedups):.1f}x, max={np.max(speedups):.1f}x")

    # Forest growth
    if reuse_results:
        final_n = reuse_results[-1]['n_boxes']
        print(f"\n  Forest growth: {n_shared_boxes} -> {final_n} boxes "
              f"(+{final_n - n_shared_boxes} from queries)")

    # Amortized
    if reuse_totals:
        build_once = shared_total
        avg_reuse = np.mean(reuse_totals)
        avg_full = np.mean(full_totals) if full_totals else 0
        amort_n = [2, 5, 10, 20, 50]
        print(f"\n  Amortized (build={build_once:.0f}ms, avg_query={avg_reuse:.0f}ms):")
        print(f"    {'N':>6s}  {'Amort/query':>12s}  {'vs Full':>8s}")
        for n in amort_n:
            amort = (build_once + n * avg_reuse) / n
            vs = avg_full / amort if amort > 0 and avg_full > 0 else 0
            print(f"    {n:>6d}  {amort:>12.0f} ms  {vs:>7.1f}x")

    print(f"\n{'=' * W}\n")


def save_report_json(
    seed: int, n_obs: int,
    shared_grow_ms: float, shared_coarsen_ms: float,
    shared_adj_ms: float,
    n_shared_boxes: int,
    full_results: List[Dict],
    reuse_results: List[Dict],
    query_pairs: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
):
    """Save JSON results."""
    data = {
        "seed": seed,
        "n_obstacles": n_obs,
        "n_shared_boxes": n_shared_boxes,
        "shared_grow_ms": shared_grow_ms,
        "shared_coarsen_ms": shared_coarsen_ms,
        "shared_adj_ms": shared_adj_ms,
        "queries": [],
    }
    for i, ((q_s, q_g), fr, rr) in enumerate(
            zip(query_pairs, full_results, reuse_results)):
        data["queries"].append({
            "q_start": q_s.tolist(),
            "q_goal": q_g.tolist(),
            "config_dist": float(np.linalg.norm(q_g - q_s)),
            "full": {k: v for k, v in fr.items() if k != 'waypoints'},
            "reuse": {k: v for k, v in rr.items() if k != 'waypoints'},
        })

    path = out_dir / "reuse_benchmark.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved -> {path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Forest incremental reuse benchmark: same obstacles, different queries")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (0 = timestamp)")
    parser.add_argument("--queries", type=int, default=5,
                        help="Number of start/goal query pairs")
    parser.add_argument("--obstacles", type=int, default=8,
                        help="Number of obstacles")
    parser.add_argument("--max-boxes", type=int, default=500,
                        help="Max boxes for initial forest growth")
    parser.add_argument("--min-dist", type=float, default=1.5,
                        help="Minimum config-space distance between start/goal")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Also run full pipeline per query (slower, for comparison)")
    parser.add_argument("--viz", action="store_true",
                        help="Generate visualization for each reuse query")
    args = parser.parse_args()

    t_experiment = time.perf_counter()

    seed = args.seed if args.seed != 0 else int(time.time()) % (2**31)
    rng = np.random.default_rng(seed)

    robot = load_robot("panda")
    ndim = robot.n_joints

    cfg = PandaGCSConfig()
    cfg.seed = seed
    cfg.n_obstacles = args.obstacles
    cfg.max_boxes = args.max_boxes

    q_start_default = np.array(cfg.q_start, dtype=np.float64)
    q_goal_default = np.array(cfg.q_goal, dtype=np.float64)

    print(f"{'=' * 70}")
    print(f"  Forest Incremental Reuse Benchmark -- Panda {ndim}-DOF")
    print(f"{'=' * 70}")
    print(f"  seed       = {seed}")
    print(f"  obstacles  = {args.obstacles}")
    print(f"  max_boxes  = {args.max_boxes}")
    print(f"  queries    = {args.queries}")
    print(f"  min_dist   = {args.min_dist:.1f} rad")
    print(f"  full_pipe  = {args.full_pipeline}")
    print()

    # == Phase 1: Build scene ==
    print("Building scene ...", flush=True)
    scene = build_panda_scene(rng, cfg, robot, q_start_default, q_goal_default)
    n_obs = scene.n_obstacles
    print(f"  {n_obs} obstacles")
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        sz = mx - mn
        print(f"    {obs.name}: center=({(mn[0]+mx[0])/2:.3f}, "
              f"{(mn[1]+mx[1])/2:.3f}, {(mn[2]+mx[2])/2:.3f})  "
              f"size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")

    # == Phase 2: Build shared forest + adjacency (once) ==
    print(f"\n{'=' * 60}")
    print(f"  Phase 1: Build shared forest + adjacency")
    print(f"{'=' * 60}")

    planner_cfg = make_planner_config(cfg)
    planner = BoxPlanner(robot=robot, scene=scene, config=planner_cfg)

    t0 = time.perf_counter()
    boxes, forest_obj, grow_detail = grow_forest(
        planner, q_start_default, q_goal_default, cfg.seed,
        cfg.max_consecutive_miss, ndim,
        max_boxes=cfg.max_boxes)
    shared_grow_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cs = coarsen_forest(
        tree=planner.hier_tree, forest=forest_obj,
        max_rounds=cfg.coarsen_max_rounds)
    shared_coarsen_ms = (time.perf_counter() - t0) * 1000

    # Adjacency + UnionFind (computed once, incrementally maintained)
    t0 = time.perf_counter()
    adj, uf, islands = _build_adjacency_and_islands(forest_obj.boxes)
    shared_adj_ms = (time.perf_counter() - t0) * 1000

    n_shared_boxes = len(forest_obj.boxes)
    n_edges = sum(len(v) for v in adj.values()) // 2
    n_islands = len(islands)

    print(f"\n  Shared forest: {n_shared_boxes} boxes, {n_edges} edges, "
          f"{n_islands} islands")
    print(f"    grow={shared_grow_ms:.0f}ms  coarsen={shared_coarsen_ms:.0f}ms  "
          f"adj={shared_adj_ms:.0f}ms")

    # Pre-pack obstacles for seed expansion
    obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)

    # == Phase 3: Sample query pairs ==
    print(f"\n{'=' * 60}")
    print(f"  Phase 2: Sample {args.queries} query pairs")
    print(f"{'=' * 60}")

    collision_checker = CollisionChecker(robot=robot, scene=scene)
    query_pairs = sample_query_pairs(
        forest_obj.boxes, collision_checker, args.queries, rng,
        min_config_dist=args.min_dist)
    n_queries = len(query_pairs)

    for i, (qs, qg) in enumerate(query_pairs):
        dist = float(np.linalg.norm(qg - qs))
        print(f"  Q{i}: dist={dist:.2f}")

    # == Phase 4: Incremental reuse queries ==
    print(f"\n{'=' * 60}")
    print(f"  Phase 3: Incremental reuse ({n_queries} queries)")
    print(f"{'=' * 60}")

    reuse_results: List[Dict] = []
    for i, (qs, qg) in enumerate(query_pairs):
        dist = float(np.linalg.norm(qg - qs))
        print(f"\n  -- Query {i} (dist={dist:.2f}) --")

        rr = run_incremental_query(
            forest_obj, planner, cfg, adj, uf, obs_packed,
            qs, qg, ndim)
        ok = "OK" if rr['success'] else "FAIL"
        print(f"  -> {ok}, cost={rr.get('cost', float('inf')):.4f}, "
              f"+{rr.get('n_expanded', 0)} expanded, +{rr.get('n_bridge', 0)} bridge, "
              f"total boxes={rr['n_boxes']}")
        print(f"    expand={rr.get('expand_ms', 0):.0f}ms  "
              f"bridge={rr['bridge_ms']:.0f}ms  "
              f"plan={rr['plan_ms']:.0f}ms  "
              f"total={rr['total_ms']:.0f}ms")
        reuse_results.append(rr)

    # == Phase 5: Full pipeline for comparison (optional) ==
    full_results: List[Dict] = []
    if args.full_pipeline:
        print(f"\n{'=' * 60}")
        print(f"  Phase 4: Full pipeline ({n_queries} queries) [from scratch]")
        print(f"{'=' * 60}")

        for i, (qs, qg) in enumerate(query_pairs):
            dist = float(np.linalg.norm(qg - qs))
            print(f"\n  -- Query {i} (dist={dist:.2f}) -- [FULL]")

            cfg_copy = PandaGCSConfig()
            cfg_copy.seed = int(rng.integers(0, 2**31))
            cfg_copy.n_obstacles = args.obstacles
            cfg_copy.max_boxes = args.max_boxes

            fr = run_full_pipeline(robot, scene, cfg_copy, qs, qg, ndim)
            ok = "OK" if fr['success'] else "FAIL"
            print(f"  -> {ok}, cost={fr.get('cost', float('inf')):.4f}, "
                  f"grow={fr['grow_ms']:.0f}ms  total={fr['total_ms']:.0f}ms")
            full_results.append(fr)
    else:
        # Placeholder: estimate full pipeline as shared build + reuse solve
        shared_build = shared_grow_ms + shared_coarsen_ms
        for rr in reuse_results:
            full_results.append(dict(
                success=rr['success'],
                cost=rr['cost'],
                n_waypoints=rr.get('n_waypoints', 0),
                n_boxes=rr['n_boxes'],
                grow_ms=shared_grow_ms,
                coarsen_ms=shared_coarsen_ms,
                adj_ms=shared_adj_ms,
                bridge_ms=rr['bridge_ms'],
                plan_ms=rr['plan_ms'],
                total_ms=shared_build + shared_adj_ms + rr['bridge_ms'] + rr['plan_ms'],
            ))

    # == Output ==
    out_dir = make_output_dir("benchmarks", "forest_reuse")

    if args.viz:
        print(f"\n  Generating visualizations ...", flush=True)
        for i, (rr, (qs, qg)) in enumerate(zip(reuse_results, query_pairs)):
            if rr['success'] and rr.get('waypoints'):
                fig = plot_arm_scene_html(
                    robot, scene, qs, qg, waypoints=rr['waypoints'],
                    title=f"Query {i} -- cost={rr['cost']:.3f}")
                p = out_dir / f"query_{i}_scene.html"
                _save_plotly_html(fig, p)
                print(f"    {p.name}")

    print_report(
        seed, n_obs, n_queries,
        shared_grow_ms, shared_coarsen_ms, shared_adj_ms,
        n_shared_boxes,
        full_results, reuse_results, query_pairs)

    save_report_json(
        seed, n_obs,
        shared_grow_ms, shared_coarsen_ms, shared_adj_ms,
        n_shared_boxes,
        full_results, reuse_results, query_pairs,
        out_dir)

    total_s = time.perf_counter() - t_experiment
    print(f"  Output: {out_dir}")
    print(f"  Total experiment time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
