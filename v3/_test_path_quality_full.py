"""
_test_path_quality_full.py — 全流程对比: 使用 bridge + 更大 forest

用法:
    cd v3
    python _test_path_quality_full.py
"""
from __future__ import annotations

import heapq
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent
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
    bridge_islands,
)


# ═══════════════════════════════════════════════════════════════════
# OLD solver (center-to-center Dijkstra + no shortcut)
# ═══════════════════════════════════════════════════════════════════

def _old_dijkstra_box_graph(boxes, adj, src, tgt):
    centers = {}
    for bid, box in boxes.items():
        centers[bid] = np.array(
            [(lo + hi) / 2 for lo, hi in box.joint_intervals])
    dist_map = {bid: float("inf") for bid in boxes}
    prev_map = {bid: None for bid in boxes}
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


def _old_solve(boxes, adj, src, tgt, q_start, q_goal, ndim, label="OLD"):
    """OLD: center-to-center Dijkstra + SOCP, no shortcut."""
    t0 = time.perf_counter()
    box_seq, raw_dist = _old_dijkstra_box_graph(boxes, adj, src, tgt)
    if box_seq is None:
        ms = (time.perf_counter() - t0) * 1000
        return dict(method=label, success=False, cost=float('inf'),
                    waypoints=[], box_seq=[], plan_ms=ms)
    waypoints = [q_start.copy()]
    for bid in box_seq[1:-1]:
        box = boxes[bid]
        c = np.array([(lo + hi) / 2 for lo, hi in box.joint_intervals])
        waypoints.append(c)
    waypoints.append(q_goal.copy())
    refined_wps, refined_cost = _refine_path_in_boxes(
        waypoints, box_seq, boxes, q_start, q_goal, ndim)
    ms = (time.perf_counter() - t0) * 1000
    return dict(method=label, success=True, cost=refined_cost,
                waypoints=refined_wps, box_seq=box_seq, plan_ms=ms,
                n_box_raw=len(box_seq), n_box_short=len(box_seq))


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


def main():
    print("=" * 80)
    print("  Full Pipeline Comparison: OLD vs NEW Dijkstra (with bridge)")
    print("=" * 80)

    # Multiple start/goal pairs to capture diverse path topologies
    query_pairs = [
        ("pair_A",
         np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5]),
         np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])),
        ("pair_B",
         np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7]),
         np.array([-1.5, 0.8, -1.0, -1.0, -1.5, 2.5, -1.0])),
        ("pair_C",
         np.array([1.0, -0.8, 0.8, -2.2, 0.3, 1.0, 1.2]),
         np.array([-0.5, 0.5, -0.5, -1.5, -0.5, 2.0, -0.5])),
    ]

    ndim = 7
    robot = load_robot("panda")

    scenarios = [
        (4, 42), (4, 100), (6, 42), (6, 100), (6, 7),
        (8, 42), (8, 100), (8, 7),
    ]

    results = []

    for pair_name, q_start, q_goal in query_pairs:
        print(f"\n{'─' * 80}")
        print(f"  Query: {pair_name}")
        print(f"  start = {np.array2string(q_start, precision=2)}")
        print(f"  goal  = {np.array2string(q_goal, precision=2)}")
        print(f"{'─' * 80}")

        direct_dist = float(np.linalg.norm(q_goal - q_start))
        print(f"  Direct config-space distance: {direct_dist:.4f}\n")

        print(f"  {'Scenario':>14s}  {'OLD cost':>9s} {'OLD #b':>6s}  "
              f"{'NEW cost':>9s} {'NEW #b':>6s} {'short':>5s}  "
              f"{'Improve':>8s}  {'OLD ms':>7s} {'NEW ms':>7s}")
        print("  " + "-" * 95)

        for n_obs, seed in scenarios:
            obs_data = _gen_obstacles(n_obs, seed)
            scene = Scene()
            for i, (mn, mx) in enumerate(obs_data):
                scene.add_obstacle(mn, mx, name=f"obs_{i}")

            cfg = PandaGCSConfig()
            cfg.seed = seed
            cfg.max_boxes = 400
            cfg.n_obstacles = n_obs

            # grow forest
            prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)
            ct = prep.get('_cache_thread')
            if ct is not None:
                ct.join()

            # use full pipeline (with bridge) for NEW method
            t0 = time.perf_counter()
            raw_new = run_method_with_bridge(
                _solve_method_dijkstra, "NEW",
                prep, cfg, q_start, q_goal, ndim)
            new_ms = (time.perf_counter() - t0) * 1000

            new_ok = raw_new is not None and raw_new.get('success', False)
            new_cost = raw_new.get('cost', float('inf')) if new_ok else float('inf')
            new_nbox = len(raw_new.get('box_seq', [])) if new_ok else 0

            # for OLD: reuse the same adj/bridge setup
            if raw_new is not None and 'adj' in raw_new:
                adj = raw_new['adj']
                boxes = raw_new['boxes']
                src = find_box_containing(q_start, boxes)
                tgt = find_box_containing(q_goal, boxes)
                if src is not None and tgt is not None:
                    t0 = time.perf_counter()
                    old_result = _old_solve(
                        boxes, adj, src, tgt, q_start, q_goal, ndim)
                    old_ms_extra = (time.perf_counter() - t0) * 1000
                else:
                    old_result = dict(success=False, cost=float('inf'),
                                     box_seq=[], plan_ms=0)
                    old_ms_extra = 0
            else:
                old_result = dict(success=False, cost=float('inf'),
                                 box_seq=[], plan_ms=0)
                old_ms_extra = 0

            old_ok = old_result.get('success', False)
            old_cost = old_result.get('cost', float('inf'))
            old_nbox = old_result.get('n_box_raw', 0)

            both = old_ok and new_ok
            if both and old_cost > 0:
                improve = (1.0 - new_cost / old_cost) * 100
            else:
                improve = float('nan')

            label = f"{n_obs}obs, s={seed}"
            old_str = f"{old_cost:9.4f}" if old_ok else "     FAIL"
            new_str = f"{new_cost:9.4f}" if new_ok else "     FAIL"
            print(f"  {label:>12s}  "
                  f"{old_str} {old_nbox:>6d}  "
                  f"{new_str} {new_nbox:>6d} {new_nbox:>5d}  "
                  f"{improve:>7.1f}%  "
                  f"{old_result.get('plan_ms',0):>6.0f}ms {raw_new.get('plan_ms',0) if raw_new else 0:>6.0f}ms")

            if both:
                results.append({
                    "pair": pair_name, "n_obs": n_obs, "seed": seed,
                    "old_cost": old_cost, "new_cost": new_cost,
                    "improve_pct": improve,
                })

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if results:
        improvements = [r['improve_pct'] for r in results]
        n = len(results)
        mean_imp = np.mean(improvements)
        median_imp = np.median(improvements)
        min_imp = min(improvements)
        max_imp = max(improvements)
        n_positive = sum(1 for x in improvements if x > 0.01)
        total_old = sum(r['old_cost'] for r in results)
        total_new = sum(r['new_cost'] for r in results)
        overall = (1 - total_new / total_old) * 100 if total_old > 0 else 0

        print(f"  Scenarios compared : {n}")
        print(f"  Positive improve   : {n_positive}/{n} ({100*n_positive/n:.0f}%)")
        print(f"  Mean improvement   : {mean_imp:+.2f}%")
        print(f"  Median improvement : {median_imp:+.2f}%")
        print(f"  Min / Max          : {min_imp:+.2f}% / {max_imp:+.2f}%")
        print(f"  Total OLD cost     : {total_old:.4f}")
        print(f"  Total NEW cost     : {total_new:.4f}")
        print(f"  Overall improvement: {overall:+.2f}%")
    else:
        print("  No comparable results.")
    print()


if __name__ == "__main__":
    main()
