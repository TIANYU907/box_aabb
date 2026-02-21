"""
_test_path_quality.py — 对比旧版 center-to-center Dijkstra 与新版
A* + boundary-aware weights + box-sequence shortcut 的路径质量.

用法:
    cd v3
    python _test_path_quality.py
"""
from __future__ import annotations

import heapq
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

_ROOT = Path(__file__).resolve().parent               # v3/
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aabb.robot import load_robot
from forest.scene import Scene
from forest.models import BoxNode
from planner.pipeline import (
    PandaGCSConfig,
    grow_and_prepare,
    run_method_with_bridge,
    _solve_method_dijkstra,
    _dijkstra_box_graph,
    _shortcut_box_sequence,
    _refine_path_in_boxes,
    _build_adjacency_and_islands,
    find_box_containing,
)


# ═══════════════════════════════════════════════════════════════════
# OLD Dijkstra (center-to-center) for baseline comparison
# ═══════════════════════════════════════════════════════════════════

def _old_dijkstra_box_graph(boxes, adj, src, tgt):
    """Reproduce the OLD center-to-center Dijkstra."""
    centers = {}
    for bid, box in boxes.items():
        centers[bid] = np.array(
            [(lo + hi) / 2 for lo, hi in box.joint_intervals])

    dist_map: Dict[int, float] = {bid: float("inf") for bid in boxes}
    prev_map: Dict[int, Optional[int]] = {bid: None for bid in boxes}
    dist_map[src] = 0.0
    heap = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist_map[u]:
            continue
        if u == tgt:
            break
        cu = centers[u]
        for v in adj.get(u, set()):
            w = float(np.linalg.norm(cu - centers[v]))
            nd = d + w
            if nd < dist_map[v]:
                dist_map[v] = nd
                prev_map[v] = u
                heapq.heappush(heap, (nd, v))

    if dist_map[tgt] == float("inf"):
        return None, float("inf")

    seq = []
    cur = tgt
    while cur is not None:
        seq.append(cur)
        cur = prev_map[cur]
    seq.reverse()
    return seq, dist_map[tgt]


def _old_solve_dijkstra(boxes, adj, src, tgt, q_start, q_goal, ndim):
    """OLD solver: center-to-center Dijkstra + SOCP refine (no shortcut)."""
    t0 = time.perf_counter()
    box_seq, raw_dist = _old_dijkstra_box_graph(boxes, adj, src, tgt)
    if box_seq is None:
        return None, float("inf"), 0, 0.0
    waypoints = [q_start.copy()]
    for bid in box_seq[1:-1]:
        box = boxes[bid]
        c = np.array([(lo + hi) / 2 for lo, hi in box.joint_intervals])
        waypoints.append(c)
    waypoints.append(q_goal.copy())
    refined_wps, refined_cost = _refine_path_in_boxes(
        waypoints, box_seq, boxes, q_start, q_goal, ndim)
    ms = (time.perf_counter() - t0) * 1000
    return refined_wps, refined_cost, len(box_seq), ms


def _new_solve_dijkstra(boxes, adj, src, tgt, q_start, q_goal, ndim):
    """NEW solver: A* + boundary weights + shortcut + SOCP refine."""
    t0 = time.perf_counter()
    box_seq, raw_dist = _dijkstra_box_graph(
        boxes, adj, src, tgt, q_goal=q_goal)
    if box_seq is None:
        return None, float("inf"), 0, 0, 0.0
    short_seq = _shortcut_box_sequence(box_seq, adj)
    waypoints = [q_start.copy()]
    for bid in short_seq[1:-1]:
        box = boxes[bid]
        c = np.array([(lo + hi) / 2 for lo, hi in box.joint_intervals])
        waypoints.append(c)
    waypoints.append(q_goal.copy())
    refined_wps, refined_cost = _refine_path_in_boxes(
        waypoints, short_seq, boxes, q_start, q_goal, ndim)
    ms = (time.perf_counter() - t0) * 1000
    return refined_wps, refined_cost, len(box_seq), len(short_seq), ms


# ═══════════════════════════════════════════════════════════════════
# Obstacle generation
# ═══════════════════════════════════════════════════════════════════

def _gen_obstacles(n_obs, seed):
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(n_obs):
        cx = float(rng.uniform(-0.6, 0.6))
        cy = float(rng.uniform(-0.6, 0.6))
        cz = float(rng.uniform(0.2, 0.8))
        h = float(rng.uniform(0.06, 0.15))
        obs.append(([cx - h, cy - h, cz - h], [cx + h, cy + h, cz + h]))
    return obs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  Path Quality Comparison: OLD center-to-center vs NEW A*/boundary")
    print("=" * 80)

    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])
    ndim = 7

    robot = load_robot("panda")

    scenarios = [
        (6, 42), (6, 100), (8, 42), (8, 100), (8, 7),
    ]

    print(f"\n{'Scenario':>14s}  {'OLD cost':>10s} {'OLD #box':>8s}  "
          f"{'NEW cost':>10s} {'NEW #box':>8s} {'NEW short':>9s}  "
          f"{'Improve':>8s}  {'OLD ms':>7s} {'NEW ms':>7s}")
    print("-" * 110)

    total_old_cost = 0.0
    total_new_cost = 0.0
    n_both_success = 0

    for n_obs, seed in scenarios:
        obs_data = _gen_obstacles(n_obs, seed)
        scene = Scene()
        for i, (mn, mx) in enumerate(obs_data):
            scene.add_obstacle(mn, mx, name=f"obs_{i}")

        cfg = PandaGCSConfig()
        cfg.seed = seed
        cfg.max_boxes = 200
        cfg.n_obstacles = n_obs

        # Grow forest (shared)
        prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)
        ct = prep.get('_cache_thread')
        if ct is not None:
            ct.join()

        boxes = prep['boxes']
        forest_obj = prep['forest_obj']

        # Build adjacency
        period = getattr(forest_obj, 'period', None)
        adj, uf, islands = _build_adjacency_and_islands(boxes, period=period)

        src = find_box_containing(q_start, boxes)
        tgt = find_box_containing(q_goal, boxes)

        if src is None or tgt is None:
            print(f"  [{n_obs}obs, s={seed}] SKIP: start/goal not in any box")
            continue

        if not uf.same(src, tgt):
            print(f"  [{n_obs}obs, s={seed}] SKIP: disconnected "
                  f"(would need bridge, skipped for pure comparison)")
            continue

        # OLD solve
        old_wps, old_cost, old_nbox, old_ms = _old_solve_dijkstra(
            boxes, adj, src, tgt, q_start, q_goal, ndim)

        # NEW solve
        new_wps, new_cost, new_nbox_raw, new_nbox_short, new_ms = \
            _new_solve_dijkstra(
                boxes, adj, src, tgt, q_start, q_goal, ndim)

        old_ok = old_wps is not None
        new_ok = new_wps is not None

        if old_ok and new_ok:
            improve = (1.0 - new_cost / old_cost) * 100 if old_cost > 0 else 0.0
            total_old_cost += old_cost
            total_new_cost += new_cost
            n_both_success += 1
        else:
            improve = float('nan')

        label = f"{n_obs}obs, s={seed}"
        print(f"  {label:>12s}  "
              f"{'FAIL' if not old_ok else f'{old_cost:10.4f}'} "
              f"{old_nbox:>8d}  "
              f"{'FAIL' if not new_ok else f'{new_cost:10.4f}'} "
              f"{new_nbox_raw:>8d} {new_nbox_short:>9d}  "
              f"{improve:>7.1f}%  "
              f"{old_ms:>6.0f}ms {new_ms:>6.0f}ms")

    print("-" * 110)
    if n_both_success > 0:
        avg_improve = (1.0 - total_new_cost / total_old_cost) * 100
        print(f"\n  Scenarios compared: {n_both_success}")
        print(f"  Total OLD cost:     {total_old_cost:.4f}")
        print(f"  Total NEW cost:     {total_new_cost:.4f}")
        print(f"  Average improvement: {avg_improve:.1f}%")
    else:
        print("\n  No comparable results (all failed or disconnected).")

    print()


if __name__ == "__main__":
    main()
