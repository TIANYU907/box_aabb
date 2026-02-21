"""
v2/scripts/bench_standalone.py — v2 独立基准测试 (子进程运行)

输出 JSON 到 stdout, 供 compare_v2_v3.py 读取.

用法:
    cd v2
    python scripts/bench_standalone.py [--quick] [--test e2e|incr]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Bootstrap v2 paths
_ROOT = Path(__file__).resolve().parents[1]  # v2/
_SRC = _ROOT / "src"
_WORKSPACE = _ROOT.parent  # parent of v2/ for 'v2.examples' imports
for p in (_ROOT, _SRC, _WORKSPACE):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _build_scene(obs_data):
    from forest.scene import Scene
    scene = Scene()
    for i, (mn, mx) in enumerate(obs_data):
        scene.add_obstacle(mn, mx, name=f"obs_{i}")
    return scene


def _gen_obstacles(n_obs, seed):
    rng = np.random.default_rng(seed)
    obs_data = []
    for _ in range(n_obs):
        cx = float(rng.uniform(-0.6, 0.6))
        cy = float(rng.uniform(-0.6, 0.6))
        cz = float(rng.uniform(0.2, 0.8))
        h = float(rng.uniform(0.06, 0.15))
        obs_data.append(([cx - h, cy - h, cz - h], [cx + h, cy + h, cz + h]))
    return obs_data


def bench_e2e(quick=False):
    """v2 端到端规划基准."""
    from aabb.robot import load_robot
    from examples.panda_planner import (
        PandaGCSConfig,
        grow_forest,
        _build_adjacency_and_islands,
        _solve_method_dijkstra,
        make_planner_config,
    )
    from planner.box_planner import BoxPlanner
    from forest.coarsen import coarsen_forest
    from examples.gcs_planner_2dof import find_box_containing

    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])
    ndim = 7

    obs_counts = [6, 8] if quick else [6, 8, 12]
    n_seeds = 2 if quick else 3
    max_boxes = 200 if quick else 400

    robot = load_robot("panda")
    results = []

    for n_obs in obs_counts:
        for s in range(n_seeds):
            seed = 42 + s
            obs_data = _gen_obstacles(n_obs, seed)
            scene = _build_scene(obs_data)

            cfg = PandaGCSConfig()
            cfg.seed = seed
            cfg.max_boxes = max_boxes
            cfg.n_obstacles = n_obs

            planner_cfg = make_planner_config(cfg)

            t_total = time.perf_counter()

            planner = BoxPlanner(robot=robot, scene=scene, config=planner_cfg)

            t0 = time.perf_counter()
            boxes, forest_obj, _ = grow_forest(
                planner, q_start, q_goal, cfg.seed,
                cfg.max_consecutive_miss, ndim,
                max_boxes=cfg.max_boxes)
            grow_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            coarsen_forest(tree=planner.hier_tree, forest=forest_obj,
                           max_rounds=cfg.coarsen_max_rounds)
            boxes = forest_obj.boxes
            coarsen_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            adj, uf, islands = _build_adjacency_and_islands(boxes)
            adj_ms = (time.perf_counter() - t0) * 1000

            src = find_box_containing(q_start, boxes)
            tgt = find_box_containing(q_goal, boxes)

            t0 = time.perf_counter()
            solve_result = {}
            if src is not None and tgt is not None:
                solve_result = _solve_method_dijkstra(
                    boxes=boxes, adj=adj, src=src, tgt=tgt,
                    q_start=q_start, q_goal=q_goal, ndim=ndim, label="Dijkstra")
            solve_ms = (time.perf_counter() - t0) * 1000

            total_ms = (time.perf_counter() - t_total) * 1000

            results.append({
                "n_obs": n_obs,
                "seed": seed,
                "success": bool(solve_result.get('success', False)),
                "cost": float(solve_result.get('cost', float('inf'))),
                "n_boxes": len(boxes),
                "grow_ms": grow_ms,
                "coarsen_ms": coarsen_ms,
                "adj_ms": adj_ms,
                "solve_ms": solve_ms,
                "total_ms": total_ms,
            })

    return results


def bench_incremental(quick=False):
    """v2 增量更新基准."""
    from aabb.robot import load_robot
    from forest.scene import Scene
    from forest.models import Obstacle
    from forest.collision import CollisionChecker
    from examples.panda_planner import (
        PandaGCSConfig,
        grow_forest,
        _build_adjacency_and_islands,
        make_planner_config,
    )
    from planner.box_planner import BoxPlanner
    from forest.coarsen import coarsen_forest

    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])
    ndim = 7
    n_obs = 8
    max_boxes = 300

    robot = load_robot("panda")
    n_seeds = 2 if quick else 3
    results = []

    for s in range(n_seeds):
        seed = 100 + s
        rng = np.random.default_rng(seed)
        obs_data = []
        for i in range(n_obs):
            cx = float(rng.uniform(-0.6, 0.6))
            cy = float(rng.uniform(-0.6, 0.6))
            cz = float(rng.uniform(0.2, 0.8))
            h = float(rng.uniform(0.06, 0.12))
            obs_data.append(([cx - h, cy - h, cz - h],
                             [cx + h, cy + h, cz + h]))

        scene = _build_scene(obs_data)

        cfg = PandaGCSConfig()
        cfg.seed = seed
        cfg.max_boxes = max_boxes

        # Build initial forest (including planner construction)
        t0 = time.perf_counter()
        planner_cfg = make_planner_config(cfg)
        planner = BoxPlanner(robot=robot, scene=scene, config=planner_cfg)
        boxes, forest_obj, _ = grow_forest(
            planner, q_start, q_goal, cfg.seed,
            cfg.max_consecutive_miss, ndim,
            max_boxes=cfg.max_boxes)
        coarsen_forest(tree=planner.hier_tree, forest=forest_obj,
                       max_rounds=cfg.coarsen_max_rounds)
        boxes = forest_obj.boxes
        adj, uf, _ = _build_adjacency_and_islands(boxes)
        build_ms = (time.perf_counter() - t0) * 1000

        n_init_boxes = len(boxes)

        # Incremental: add 1 obstacle
        new_obs = Obstacle(
            min_point=np.array([0.1, 0.1, 0.3]),
            max_point=np.array([0.25, 0.25, 0.55]),
            name='new_obs_test',
        )

        t0 = time.perf_counter()
        colliding = forest_obj.invalidate_against_obstacle(
            new_obs, robot, safety_margin=0.0)
        n_invalidated = len(colliding)
        removed = forest_obj.remove_invalidated(colliding)
        invalidate_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for bid in colliding:
            boxes.pop(bid, None)
            if bid in adj:
                nbrs = adj.pop(bid, set())
                for nb in nbrs:
                    if nb in adj:
                        adj[nb].discard(bid)
        scene.add_obstacle(
            list(new_obs.min_point), list(new_obs.max_point),
            name='new_obs_test')
        planner.collision_checker = CollisionChecker(
            robot=robot, scene=scene)
        planner.obstacles = scene.get_obstacles()
        update_ms = (time.perf_counter() - t0) * 1000
        incr_ms = invalidate_ms + update_ms

        # Full rebuild for comparison (include planner construction)
        full_scene = Scene()
        for i, (mn, mx) in enumerate(obs_data):
            full_scene.add_obstacle(mn, mx, name=f"obs_{i}")
        full_scene.add_obstacle([0.1, 0.1, 0.3], [0.25, 0.25, 0.55],
                                name='new_obs_test')

        cfg2 = PandaGCSConfig()
        cfg2.seed = seed + 1
        cfg2.max_boxes = max_boxes
        planner_cfg2 = make_planner_config(cfg2)

        t0 = time.perf_counter()
        planner2 = BoxPlanner(robot=robot, scene=full_scene, config=planner_cfg2)
        boxes2, forest2, _ = grow_forest(
            planner2, q_start, q_goal, cfg2.seed,
            cfg2.max_consecutive_miss, ndim,
            max_boxes=cfg2.max_boxes)
        coarsen_forest(tree=planner2.hier_tree, forest=forest2,
                       max_rounds=cfg2.coarsen_max_rounds)
        full_ms = (time.perf_counter() - t0) * 1000

        speedup = full_ms / incr_ms if incr_ms > 0 else float('inf')

        results.append({
            "seed": seed,
            "build_ms": build_ms,
            "n_init_boxes": n_init_boxes,
            "invalidate_ms": invalidate_ms,
            "update_ms": update_ms,
            "incr_ms": incr_ms,
            "n_invalidated": n_invalidated,
            "full_ms": full_ms,
            "speedup": speedup,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--test", choices=["e2e", "incr", "all"], default="all")
    args = parser.parse_args()

    output = {"version": "v2"}
    if args.test in ("e2e", "all"):
        output["e2e"] = bench_e2e(args.quick)
    if args.test in ("incr", "all"):
        output["incr"] = bench_incremental(args.quick)

    print(json.dumps(output, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))


if __name__ == "__main__":
    main()
