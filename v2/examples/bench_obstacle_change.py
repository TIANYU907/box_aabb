"""
bench_obstacle_change.py -- Incremental forest reuse under obstacle changes

Three scenarios:
  A) Obstacle disappear  → freed space, regrow in freed region
  B) Obstacle appear     → invalidate + remove + regrow + repair adj/uf
  C) Obstacle move       → disappear(old) + appear(new)

Each scenario: incremental update vs full rebuild (from scratch).

Usage:
    python -m v2.examples.bench_obstacle_change
    python -m v2.examples.bench_obstacle_change --seed 42 --obstacles 8
    python -m v2.examples.bench_obstacle_change --seed 42 --regrow-budget 60
"""

from __future__ import annotations

import argparse
import copy
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
from forest.models import BoxNode, Obstacle
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
)
from v2.examples.gcs_planner_2dof import find_box_containing
from v2.examples.bench_forest_reuse import (
    _incremental_adj_for_new_box,
    _expand_seed_box,
    sample_query_pairs,
)


# =====================================================================
# Incremental regrow: expand seeds into freed hier_tree regions
# =====================================================================

def _regrow_in_freed_space(
    seeds: List[np.ndarray],
    planner: BoxPlanner,
    forest_obj: BoxForest,
    boxes: Dict[int, BoxNode],
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    min_box_size: float,
    budget: int = 60,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Grow new boxes in freed hier_tree regions.

    Seeds come from:
      1. Removed boxes' seed_config (highest priority)
      2. Random points sampled from removed boxes' joint_intervals
      3. sample_unoccupied_seed (naturally weighted toward freed regions
         because subtree_occ_vol was decreased by unoccupy)

    Returns number of boxes added.
    """
    n_added = 0
    attempted = set()

    # Phase 1: seed_configs from removed boxes
    for seed in seeds:
        if n_added >= budget:
            break
        key = tuple(np.round(seed, 8))
        if key in attempted:
            continue
        attempted.add(key)

        bid = _try_expand(seed, planner, forest_obj, boxes, adj, uf,
                          obs_packed, min_box_size)
        if bid is not None:
            n_added += 1

    # Phase 2: sample_unoccupied_seed (biased toward freed space)
    if rng is not None:
        miss = 0
        max_miss = 15
        while n_added < budget and miss < max_miss:
            try:
                q = planner.hier_tree.sample_unoccupied_seed(rng)
            except ValueError:
                break
            if q is None:
                miss += 1
                continue
            if planner.collision_checker.check_config_collision(q):
                miss += 1
                continue
            bid = _try_expand(q, planner, forest_obj, boxes, adj, uf,
                              obs_packed, min_box_size)
            if bid is not None:
                n_added += 1
                miss = 0
            else:
                miss += 1

    return n_added


def _try_expand(
    q: np.ndarray,
    planner: BoxPlanner,
    forest_obj: BoxForest,
    boxes: Dict[int, BoxNode],
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    min_box_size: float,
) -> Optional[int]:
    """Try to expand a box at q. Returns box_id or None."""
    # Already inside existing box?
    existing = find_box_containing(q, boxes)
    if existing is not None:
        return None  # don't count as added

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

    forest_obj.add_box_direct(box)
    boxes[nid] = box
    _incremental_adj_for_new_box(box, boxes, adj, uf)
    return nid


# =====================================================================
# Repair adj/uf after box removal
# =====================================================================

def _remove_boxes_from_adj_uf(
    removed_ids: Set[int],
    adj: Dict[int, Set[int]],
) -> None:
    """Remove box IDs from adjacency. UF will be rebuilt."""
    for bid in removed_ids:
        if bid in adj:
            nbrs = adj.pop(bid, set())
            for nb in nbrs:
                if nb in adj:
                    adj[nb].discard(bid)


def _rebuild_uf_from_adj(adj: Dict[int, Set[int]]) -> UnionFind:
    """Rebuild UnionFind from adjacency dict."""
    uf = UnionFind(list(adj.keys()))
    for bid, nbrs in adj.items():
        for nb in nbrs:
            uf.union(bid, nb)
    return uf


# =====================================================================
# Scenario A: Obstacle disappear (freed space—regrow only)
# =====================================================================

def run_obstacle_disappear(
    forest_obj: BoxForest,
    planner: BoxPlanner,
    cfg: PandaGCSConfig,
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    obstacle_to_remove: Obstacle,
    regrow_budget: int,
    rng: np.random.Generator,
    ndim: int,
) -> Dict:
    """Obstacle disappear: update scene, regrow in freed space."""
    t_total = time.perf_counter()
    n_before = len(forest_obj.boxes)

    # (1) Remove obstacle from scene
    t0 = time.perf_counter()
    planner.scene.remove_obstacle(obstacle_to_remove.name)
    # Rebuild collision checker with updated scene
    planner.collision_checker = CollisionChecker(
        robot=planner.robot, scene=planner.scene)
    # Re-pack obstacles
    new_obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)
    scene_update_ms = (time.perf_counter() - t0) * 1000

    # (2) Regrow: sample_unoccupied_seed is biased toward freed regions
    #     (No invalidation needed — existing boxes are all still valid)
    t0 = time.perf_counter()
    n_regrown = _regrow_in_freed_space(
        seeds=[],  # no removed boxes → rely on sample_unoccupied_seed
        planner=planner, forest_obj=forest_obj,
        boxes=forest_obj.boxes, adj=adj, uf=uf,
        obs_packed=new_obs_packed, min_box_size=cfg.min_box_size,
        budget=regrow_budget, rng=rng,
    )
    regrow_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return dict(
        scenario="disappear",
        obstacle=obstacle_to_remove.name,
        n_before=n_before,
        n_after=len(forest_obj.boxes),
        n_invalidated=0,
        n_removed=0,
        n_regrown=n_regrown,
        scene_update_ms=scene_update_ms,
        invalidate_ms=0,
        remove_ms=0,
        regrow_ms=regrow_ms,
        repair_ms=0,
        total_ms=total_ms,
        obs_packed=new_obs_packed,
    )


# =====================================================================
# Scenario B: Obstacle appear (invalidate + remove + regrow)
# =====================================================================

def run_obstacle_appear(
    forest_obj: BoxForest,
    planner: BoxPlanner,
    cfg: PandaGCSConfig,
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    new_obstacle: Obstacle,
    regrow_budget: int,
    rng: np.random.Generator,
    ndim: int,
) -> Dict:
    """Obstacle appear: invalidate, remove, update scene, regrow."""
    t_total = time.perf_counter()
    n_before = len(forest_obj.boxes)

    # (1) Invalidate: find boxes colliding with new obstacle
    t0 = time.perf_counter()
    colliding_ids = forest_obj.invalidate_against_obstacle(
        new_obstacle, planner.robot, safety_margin=0.0)
    invalidate_ms = (time.perf_counter() - t0) * 1000

    # (2) Remove invalidated boxes from forest + hier_tree
    t0 = time.perf_counter()
    removed_boxes = forest_obj.remove_invalidated(colliding_ids)
    remove_ms = (time.perf_counter() - t0) * 1000

    # (3) Remove from adj and rebuild UF
    t0 = time.perf_counter()
    _remove_boxes_from_adj_uf(colliding_ids, adj)
    uf = _rebuild_uf_from_adj(adj)
    repair_ms = (time.perf_counter() - t0) * 1000

    # (4) Update scene: add the new obstacle
    t0 = time.perf_counter()
    planner.scene.add_obstacle(
        new_obstacle.min_point, new_obstacle.max_point,
        name=new_obstacle.name)
    planner.collision_checker = CollisionChecker(
        robot=planner.robot, scene=planner.scene)
    new_obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)
    scene_update_ms = (time.perf_counter() - t0) * 1000

    # (5) Regrow using removed boxes' seeds
    seeds = [b.seed_config for b in removed_boxes if b.seed_config is not None]
    t0 = time.perf_counter()
    n_regrown = _regrow_in_freed_space(
        seeds=seeds,
        planner=planner, forest_obj=forest_obj,
        boxes=forest_obj.boxes, adj=adj, uf=uf,
        obs_packed=new_obs_packed, min_box_size=cfg.min_box_size,
        budget=regrow_budget, rng=rng,
    )
    regrow_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return dict(
        scenario="appear",
        obstacle=new_obstacle.name,
        n_before=n_before,
        n_after=len(forest_obj.boxes),
        n_invalidated=len(colliding_ids),
        n_removed=len(removed_boxes),
        n_regrown=n_regrown,
        scene_update_ms=scene_update_ms,
        invalidate_ms=invalidate_ms,
        remove_ms=remove_ms,
        regrow_ms=regrow_ms,
        repair_ms=repair_ms,
        total_ms=total_ms,
        obs_packed=new_obs_packed,
    )


# =====================================================================
# Scenario C: Obstacle move  = disappear(old) + appear(new)
# =====================================================================

def run_obstacle_move(
    forest_obj: BoxForest,
    planner: BoxPlanner,
    cfg: PandaGCSConfig,
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    old_obstacle: Obstacle,
    new_min: np.ndarray,
    new_max: np.ndarray,
    regrow_budget: int,
    rng: np.random.Generator,
    ndim: int,
) -> Dict:
    """Obstacle move: disappear(old) + appear(new position)."""
    t_total = time.perf_counter()
    n_before = len(forest_obj.boxes)

    new_obstacle = Obstacle(
        min_point=new_min, max_point=new_max,
        name=old_obstacle.name + "_moved",
    )

    # (1) Invalidate boxes against new obstacle position
    t0 = time.perf_counter()
    colliding_ids = forest_obj.invalidate_against_obstacle(
        new_obstacle, planner.robot, safety_margin=0.0)
    invalidate_ms = (time.perf_counter() - t0) * 1000

    # (2) Remove invalidated
    t0 = time.perf_counter()
    removed_boxes = forest_obj.remove_invalidated(colliding_ids)
    remove_ms = (time.perf_counter() - t0) * 1000

    # (3) Repair adj/uf
    t0 = time.perf_counter()
    _remove_boxes_from_adj_uf(colliding_ids, adj)
    uf = _rebuild_uf_from_adj(adj)
    repair_ms = (time.perf_counter() - t0) * 1000

    # (4) Update scene: remove old, add new
    t0 = time.perf_counter()
    planner.scene.remove_obstacle(old_obstacle.name)
    planner.scene.add_obstacle(
        new_min, new_max, name=old_obstacle.name)
    planner.collision_checker = CollisionChecker(
        robot=planner.robot, scene=planner.scene)
    new_obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)
    scene_update_ms = (time.perf_counter() - t0) * 1000

    # (5) Regrow using removed seeds + freed space from old obstacle
    seeds = [b.seed_config for b in removed_boxes if b.seed_config is not None]
    t0 = time.perf_counter()
    n_regrown = _regrow_in_freed_space(
        seeds=seeds,
        planner=planner, forest_obj=forest_obj,
        boxes=forest_obj.boxes, adj=adj, uf=uf,
        obs_packed=new_obs_packed, min_box_size=cfg.min_box_size,
        budget=regrow_budget, rng=rng,
    )
    regrow_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return dict(
        scenario="move",
        obstacle=old_obstacle.name,
        n_before=n_before,
        n_after=len(forest_obj.boxes),
        n_invalidated=len(colliding_ids),
        n_removed=len(removed_boxes),
        n_regrown=n_regrown,
        scene_update_ms=scene_update_ms,
        invalidate_ms=invalidate_ms,
        remove_ms=remove_ms,
        regrow_ms=regrow_ms,
        repair_ms=repair_ms,
        total_ms=total_ms,
        obs_packed=new_obs_packed,
    )


# =====================================================================
# Full rebuild baseline (for comparison)
# =====================================================================

def run_full_rebuild(
    robot, scene: Scene, cfg: PandaGCSConfig,
    q_start: np.ndarray, q_goal: np.ndarray, ndim: int,
) -> Dict:
    """Full pipeline from scratch: grow + coarsen + adj."""
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

    t0 = time.perf_counter()
    adj, uf, islands = _build_adjacency_and_islands(boxes)
    adj_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return dict(
        n_boxes=len(boxes),
        grow_ms=grow_ms,
        coarsen_ms=coarsen_ms,
        adj_ms=adj_ms,
        total_ms=total_ms,
    )


# =====================================================================
# Query (plan s→g) on given forest
# =====================================================================

def solve_query(
    forest_obj: BoxForest,
    planner: BoxPlanner,
    cfg: PandaGCSConfig,
    adj: Dict[int, Set[int]],
    uf: UnionFind,
    obs_packed,
    q_start: np.ndarray, q_goal: np.ndarray, ndim: int,
) -> Dict:
    """Solve a query on the current forest using Dijkstra."""
    t_total = time.perf_counter()
    boxes = forest_obj.boxes

    # Expand s/g if needed
    t0 = time.perf_counter()
    src = _expand_seed_box(
        q_start, planner, forest_obj, boxes, adj, uf, obs_packed, cfg.min_box_size)
    tgt = _expand_seed_box(
        q_goal, planner, forest_obj, boxes, adj, uf, obs_packed, cfg.min_box_size)
    expand_ms = (time.perf_counter() - t0) * 1000

    if src is None or tgt is None:
        total_ms = (time.perf_counter() - t_total) * 1000
        return dict(success=False, cost=float('inf'), expand_ms=expand_ms,
                    bridge_ms=0, plan_ms=0, total_ms=total_ms)

    # Bridge if needed
    bridge_ms = 0.0
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

        for bb in bridge_boxes:
            _incremental_adj_for_new_box(bb, boxes, adj, uf)
        for e in bridge_edges:
            s, t = e.source_box_id, e.target_box_id
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
            uf.union(s, t)

        if src not in boxes:
            src = find_box_containing(q_start, boxes)
        if tgt not in boxes:
            tgt = find_box_containing(q_goal, boxes)

    if src is None or tgt is None:
        total_ms = (time.perf_counter() - t_total) * 1000
        return dict(success=False, cost=float('inf'), expand_ms=expand_ms,
                    bridge_ms=bridge_ms, plan_ms=0, total_ms=total_ms)

    t0 = time.perf_counter()
    plan_result = _solve_method_dijkstra(
        boxes=boxes, adj=adj, src=src, tgt=tgt,
        q_start=q_start, q_goal=q_goal, ndim=ndim, label="Dijkstra")
    plan_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000
    return dict(
        success=plan_result.get('success', False),
        cost=plan_result.get('cost', float('inf')),
        expand_ms=expand_ms,
        bridge_ms=bridge_ms,
        plan_ms=plan_ms,
        total_ms=total_ms,
    )


# =====================================================================
# Report
# =====================================================================

def print_report(
    seed: int, n_obs: int,
    initial_build: Dict,
    scenario_results: List[Dict],
    query_before: Dict,
    query_after: Dict,
    full_rebuild: Dict,
):
    W = 100
    print(f"\n{'=' * W}")
    print(f"  Obstacle Change Incremental Reuse Benchmark -- Panda 7-DOF")
    print(f"  seed={seed}, {n_obs} obstacles")
    print(f"{'=' * W}")

    # Initial build
    print(f"\n  Initial build (once):")
    print(f"    grow     = {initial_build['grow_ms']:8.1f} ms")
    print(f"    coarsen  = {initial_build['coarsen_ms']:8.1f} ms")
    print(f"    adj+uf   = {initial_build['adj_ms']:8.1f} ms")
    total_init = initial_build['grow_ms'] + initial_build['coarsen_ms'] + initial_build['adj_ms']
    print(f"    total    = {total_init:8.1f} ms")
    print(f"    boxes    = {initial_build['n_boxes']}")

    # Scenario results
    print(f"\n  {'─' * W}")
    hdr = (f"  {'Scenario':<12s}  {'Obstacle':<12s}  "
           f"{'#before':>7s} {'#inval':>6s} {'#rmvd':>5s} {'#grow':>5s} {'#after':>7s}  "
           f"│ {'inval':>5s} {'remove':>6s} {'regrow':>6s} {'repair':>6s} {'scene':>5s} {'TOTAL':>6s}")
    print(hdr)
    print(f"  {'─' * W}")

    for sr in scenario_results:
        print(f"  {sr['scenario']:<12s}  {sr['obstacle']:<12s}  "
              f"{sr['n_before']:>7d} {sr['n_invalidated']:>6d} {sr['n_removed']:>5d} "
              f"{sr['n_regrown']:>5d} {sr['n_after']:>7d}  "
              f"│ {sr['invalidate_ms']:>5.0f} {sr['remove_ms']:>6.1f} "
              f"{sr['regrow_ms']:>6.0f} {sr['repair_ms']:>6.1f} "
              f"{sr['scene_update_ms']:>5.1f} {sr['total_ms']:>6.0f}")

    incr_total = sum(sr['total_ms'] for sr in scenario_results)
    print(f"  {'─' * W}")
    print(f"  {'INCR TOTAL':<12s}  {'':12s}  "
          f"{'':>7s} {'':>6s} {'':>5s} {'':>5s} {'':>7s}  "
          f"│ {'':>5s} {'':>6s} {'':>6s} {'':>6s} {'':>5s} {incr_total:>6.0f}")

    # Query results
    print(f"\n  Query (s→g) on forest:")
    print(f"    Before changes: success={query_before['success']}, "
          f"cost={query_before['cost']:.4f}, total={query_before['total_ms']:.0f}ms")
    print(f"    After  changes: success={query_after['success']}, "
          f"cost={query_after['cost']:.4f}, total={query_after['total_ms']:.0f}ms")

    # Full rebuild comparison
    print(f"\n  Full rebuild (from scratch after all changes):")
    print(f"    grow={full_rebuild['grow_ms']:.0f}ms  "
          f"coarsen={full_rebuild['coarsen_ms']:.0f}ms  "
          f"adj={full_rebuild['adj_ms']:.0f}ms  "
          f"total={full_rebuild['total_ms']:.0f}ms  "
          f"boxes={full_rebuild['n_boxes']}")

    speedup = full_rebuild['total_ms'] / incr_total if incr_total > 0 else float('inf')
    print(f"\n  Speedup: incremental={incr_total:.0f}ms vs rebuild={full_rebuild['total_ms']:.0f}ms "
          f"→ {speedup:.1f}×")
    print(f"\n{'=' * W}\n")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Obstacle-change incremental forest reuse benchmark")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (0 = timestamp)")
    parser.add_argument("--obstacles", type=int, default=8,
                        help="Number of obstacles")
    parser.add_argument("--max-boxes", type=int, default=500,
                        help="Max boxes for initial forest growth")
    parser.add_argument("--regrow-budget", type=int, default=60,
                        help="Max boxes to regrow after each obstacle change")
    parser.add_argument("--move-dist", type=float, default=0.08,
                        help="Distance to move obstacle (workspace meters)")
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

    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)

    W = 70
    print(f"{'=' * W}")
    print(f"  Obstacle Change Incremental Reuse Benchmark -- Panda {ndim}-DOF")
    print(f"{'=' * W}")
    print(f"  seed          = {seed}")
    print(f"  obstacles     = {args.obstacles}")
    print(f"  max_boxes     = {args.max_boxes}")
    print(f"  regrow_budget = {args.regrow_budget}")
    print(f"  move_dist     = {args.move_dist:.3f}")
    print()

    # ═════════════════════════════════════════════════════════════════
    # Phase 1: Build initial scene + forest
    # ═════════════════════════════════════════════════════════════════
    print("Phase 1: Build initial scene + forest ...", flush=True)
    scene = build_panda_scene(rng, cfg, robot, q_start, q_goal)
    n_obs = scene.n_obstacles
    obstacles_info = []
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        sz = mx - mn
        center = (mn + mx) / 2
        obstacles_info.append((obs.name, center, sz, mn.copy(), mx.copy()))
        print(f"    {obs.name}: center=({center[0]:.3f}, {center[1]:.3f}, "
              f"{center[2]:.3f})  size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")

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

    t0 = time.perf_counter()
    adj, uf, islands = _build_adjacency_and_islands(boxes)
    adj_ms = (time.perf_counter() - t0) * 1000

    obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)

    initial_build = dict(
        n_boxes=len(boxes), grow_ms=grow_ms,
        coarsen_ms=coarsen_ms, adj_ms=adj_ms,
    )
    print(f"\n  Forest: {len(boxes)} boxes, grow={grow_ms:.0f}ms, "
          f"coarsen={coarsen_ms:.0f}ms, adj={adj_ms:.0f}ms")

    # ═════════════════════════════════════════════════════════════════
    # Phase 2: Query BEFORE obstacle changes
    # ═════════════════════════════════════════════════════════════════
    print(f"\nPhase 2: Query before changes ...", flush=True)
    query_before = solve_query(
        forest_obj, planner, cfg, adj, uf, obs_packed,
        q_start, q_goal, ndim)
    print(f"  success={query_before['success']}, cost={query_before['cost']:.4f}, "
          f"total={query_before['total_ms']:.0f}ms")

    # ═════════════════════════════════════════════════════════════════
    # Phase 3: Apply obstacle changes (incremental)
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print(f"  Phase 3: Incremental obstacle changes")
    print(f"{'=' * W}")

    scenario_results = []

    # --- Scenario A: Remove one obstacle ---
    if n_obs >= 3:
        # Pick the obstacle with most box collisions for max impact
        obs_list = scene.get_obstacles()
        # Remove the last obstacle (index -1) to keep s/g valid
        obs_to_remove = obs_list[-1]
        print(f"\n  [A] Obstacle DISAPPEAR: '{obs_to_remove.name}'")

        result_a = run_obstacle_disappear(
            forest_obj, planner, cfg, adj, uf, obs_packed,
            obs_to_remove, args.regrow_budget, rng, ndim)
        obs_packed = result_a.pop('obs_packed')
        scenario_results.append(result_a)

        print(f"    → n_regrown={result_a['n_regrown']}, "
              f"boxes: {result_a['n_before']}→{result_a['n_after']}, "
              f"total={result_a['total_ms']:.0f}ms")

    # --- Scenario B: Add a new obstacle ---
    new_obs_center = np.array([
        rng.uniform(-0.4, 0.4),
        rng.uniform(-0.4, 0.4),
        rng.uniform(0.2, 0.7),
    ])
    new_obs_half = rng.uniform(0.04, 0.12, size=3)
    new_obstacle = Obstacle(
        min_point=new_obs_center - new_obs_half,
        max_point=new_obs_center + new_obs_half,
        name="obs_new",
    )
    print(f"\n  [B] Obstacle APPEAR: '{new_obstacle.name}' at "
          f"({new_obs_center[0]:.3f}, {new_obs_center[1]:.3f}, "
          f"{new_obs_center[2]:.3f})")

    result_b = run_obstacle_appear(
        forest_obj, planner, cfg, adj, uf, obs_packed,
        new_obstacle, args.regrow_budget, rng, ndim)
    obs_packed = result_b.pop('obs_packed')
    # Rebuild uf reference (may have been replaced inside)
    uf = _rebuild_uf_from_adj(adj)
    scenario_results.append(result_b)

    print(f"    → n_invalidated={result_b['n_invalidated']}, "
          f"n_removed={result_b['n_removed']}, n_regrown={result_b['n_regrown']}, "
          f"boxes: {result_b['n_before']}→{result_b['n_after']}, "
          f"total={result_b['total_ms']:.0f}ms")

    # --- Scenario C: Move one obstacle ---
    obs_list_now = scene.get_obstacles()
    if len(obs_list_now) >= 2:
        obs_to_move = obs_list_now[0]  # move the first obstacle
        old_center = (obs_to_move.min_point + obs_to_move.max_point) / 2
        old_half = (obs_to_move.max_point - obs_to_move.min_point) / 2

        # Random displacement
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        displacement = direction * args.move_dist
        new_center = old_center + displacement
        new_min = new_center - old_half
        new_max = new_center + old_half

        print(f"\n  [C] Obstacle MOVE: '{obs_to_move.name}' by "
              f"{args.move_dist:.3f}m → center=({new_center[0]:.3f}, "
              f"{new_center[1]:.3f}, {new_center[2]:.3f})")

        result_c = run_obstacle_move(
            forest_obj, planner, cfg, adj, uf, obs_packed,
            obs_to_move, new_min, new_max,
            args.regrow_budget, rng, ndim)
        obs_packed = result_c.pop('obs_packed')
        uf = _rebuild_uf_from_adj(adj)
        scenario_results.append(result_c)

        print(f"    → n_invalidated={result_c['n_invalidated']}, "
              f"n_removed={result_c['n_removed']}, n_regrown={result_c['n_regrown']}, "
              f"boxes: {result_c['n_before']}→{result_c['n_after']}, "
              f"total={result_c['total_ms']:.0f}ms")

    # ═════════════════════════════════════════════════════════════════
    # Phase 4: Query AFTER obstacle changes (on incrementally updated forest)
    # ═════════════════════════════════════════════════════════════════
    print(f"\nPhase 4: Query after changes ...", flush=True)
    query_after = solve_query(
        forest_obj, planner, cfg, adj, uf, obs_packed,
        q_start, q_goal, ndim)
    print(f"  success={query_after['success']}, cost={query_after['cost']:.4f}, "
          f"total={query_after['total_ms']:.0f}ms")

    # ═════════════════════════════════════════════════════════════════
    # Phase 5: Full rebuild for comparison
    # ═════════════════════════════════════════════════════════════════
    print(f"\nPhase 5: Full rebuild (from scratch) for comparison ...", flush=True)
    # Use the current (modified) scene
    cfg_rebuild = PandaGCSConfig()
    cfg_rebuild.seed = int(rng.integers(0, 2**31))
    cfg_rebuild.n_obstacles = args.obstacles
    cfg_rebuild.max_boxes = args.max_boxes
    full_rebuild = run_full_rebuild(
        robot, planner.scene, cfg_rebuild, q_start, q_goal, ndim)
    print(f"  boxes={full_rebuild['n_boxes']}, "
          f"grow={full_rebuild['grow_ms']:.0f}ms, "
          f"total={full_rebuild['total_ms']:.0f}ms")

    # ═════════════════════════════════════════════════════════════════
    # Report
    # ═════════════════════════════════════════════════════════════════
    print_report(
        seed, n_obs, initial_build,
        scenario_results, query_before, query_after, full_rebuild)

    # Save JSON
    out_dir = make_output_dir("benchmarks", "obstacle_change")
    data = {
        "seed": seed,
        "n_obstacles": n_obs,
        "regrow_budget": args.regrow_budget,
        "move_dist": args.move_dist,
        "initial_build": initial_build,
        "query_before": {k: v for k, v in query_before.items()
                         if k != 'waypoints'},
        "query_after": {k: v for k, v in query_after.items()
                        if k != 'waypoints'},
        "scenarios": scenario_results,
        "full_rebuild": full_rebuild,
    }
    json_path = out_dir / "obstacle_change_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved -> {json_path}")

    total_s = time.perf_counter() - t_experiment
    print(f"  Output: {out_dir}")
    print(f"  Total experiment time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
