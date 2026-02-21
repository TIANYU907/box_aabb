"""
experiments/exp3_obstacle_change.py — 实验 3: 障碍物变化增量更新

★ 核心优势实验：展示 SBF 真增量更新 vs 全量重建

设计:
  场景 A — 障碍物消失: 移除 1/2/3 个 → incremental regrow vs 全量
  场景 B — 障碍物出现: 新增 1/2/3 个 → invalidate + regrow vs 全量
  场景 C — 障碍物移动: 1/2/3 个移位 → invalidate + regrow vs 全量

对照: RRTConnect / RRT* 每次全量重建

用法:
    python -m experiments.exp3_obstacle_change [--quick]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.runner import (ExperimentResults, SingleTrialResult,
                                 create_planner, load_scene_from_config)
from experiments.scenes import load_scenes, load_planners

from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.models import Obstacle, BoxNode
from forest.connectivity import UnionFind
from planner.pipeline import (
    PandaGCSConfig,
    make_planner_config,
    grow_and_prepare,
    run_method_with_bridge,
    _build_adjacency_and_islands,
    _solve_method_dijkstra,
    incremental_obstacle_update,
    find_box_containing,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"


def _build_base_scene(n_obs: int = 10, seed: int = 500):
    """构建带 n_obs 障碍物的基础场景配置."""
    return {
        "name": f"panda_{n_obs}obs_base",
        "robot": "panda",
        "random_obstacles": {
            "count": n_obs,
            "seed": seed,
            "x_range": [-0.8, 0.8],
            "y_range": [-0.8, 0.8],
            "z_range": [0.1, 0.9],
            "half_size_range": [0.05, 0.15],
        },
        "query_pairs": [
            {"start": [0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5],
             "goal":  [-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8]},
        ],
    }


def _solve_on_prep(prep, cfg, q_start, q_goal, ndim, timeout):
    """在已有 prep (forest) 上求解一次 Dijkstra."""
    t0 = time.perf_counter()
    raw = run_method_with_bridge(
        _solve_method_dijkstra, "Dijkstra",
        prep, cfg, q_start, q_goal, ndim)
    elapsed = time.perf_counter() - t0
    if raw is None or not raw.get('success', False):
        return dict(success=False, cost=float('inf'), plan_ms=elapsed * 1000)
    return dict(
        success=True,
        cost=raw.get('cost', float('inf')),
        plan_ms=elapsed * 1000,
        n_boxes=len(prep['boxes']),
    )


def run(quick: bool = False) -> Path:
    change_counts = [1, 2] if quick else [1, 2, 3]
    n_seeds = 2 if quick else 5
    timeout = 15.0 if quick else 30.0
    n_obs_base = 8
    regrow_budget = 40

    results = ExperimentResults(experiment_name="exp3_obstacle_change")

    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])
    ndim = 7

    change_types = ["remove", "add", "move"]

    print("=== Experiment 3: Obstacle Change (True Incremental) ===")
    print(f"  Change types: {change_types}")
    print(f"  Change counts: {change_counts}")
    print(f"  Seeds: {n_seeds}, base obstacles: {n_obs_base}")
    print(f"  Regrow budget: {regrow_budget}")
    print()

    for seed in range(n_seeds):
        rng_scene = np.random.default_rng(400 + seed)
        base_cfg = _build_base_scene(n_obs=n_obs_base, seed=400 + seed)
        robot, base_scene, _ = load_scene_from_config(base_cfg)

        # ── 构建初始 forest (一次) ──
        print(f"\n--- Seed {seed}: Building initial forest ---")
        cfg = PandaGCSConfig()
        cfg.seed = 400 + seed
        cfg.max_boxes = 300
        cfg.n_obstacles = n_obs_base

        t0 = time.perf_counter()
        prep = grow_and_prepare(robot, base_scene, cfg, q_start, q_goal, ndim)
        ct = prep.get('_cache_thread')
        if ct is not None:
            ct.join()
        initial_build_ms = (time.perf_counter() - t0) * 1000
        n_initial_boxes = len(prep['boxes'])
        print(f"  Initial forest: {n_initial_boxes} boxes, "
              f"{initial_build_ms:.0f}ms")

        # 求解一次验证 forest 可用
        init_result = _solve_on_prep(prep, cfg, q_start, q_goal, ndim, timeout)
        print(f"  Initial query: success={init_result.get('success')}, "
              f"cost={init_result.get('cost', float('inf')):.4f}")

        for change_type in change_types:
            for n_change in change_counts:
                scene_name = f"obs_{change_type}_{n_change}"
                print(f"\n  [{change_type} x{n_change}]", end=" ", flush=True)

                # ── 1) SBF 增量更新 ──
                # 深拷贝 prep 和 scene 以保持独立
                incr_scene = _deep_copy_scene(base_scene)
                incr_prep = _shallow_copy_prep(prep)

                change_rng = np.random.default_rng(seed * 100 + n_change)
                added_obs, removed_names = _compute_changes(
                    base_scene, change_type, n_change, change_rng)

                t0 = time.perf_counter()
                update_stats = incremental_obstacle_update(
                    prep=incr_prep,
                    scene=incr_scene,
                    added_obstacles=added_obs,
                    removed_obstacle_names=removed_names,
                    regrow_budget=regrow_budget,
                    rng=change_rng,
                )
                t_incr_update = (time.perf_counter() - t0) * 1000

                # 在增量更新后的 forest 上求解
                t0 = time.perf_counter()
                incr_query = _solve_on_prep(
                    incr_prep, cfg, q_start, q_goal, ndim, timeout)
                t_incr_solve = (time.perf_counter() - t0) * 1000
                t_incr_total = t_incr_update + t_incr_solve

                print(f"INCR: update={t_incr_update:.0f}ms "
                      f"(inval={update_stats['invalidate_ms']:.0f}ms, "
                      f"regrow={update_stats['regrow_ms']:.0f}ms), "
                      f"solve={t_incr_solve:.0f}ms, "
                      f"boxes={update_stats['n_after']}", end=" | ")

                results.add(SingleTrialResult(
                    scene_name=scene_name,
                    planner_name="SBF-Incremental",
                    seed=seed, trial=0,
                    result={
                        "change_type": change_type,
                        "n_changes": n_change,
                        "mode": "incremental",
                        "success": incr_query.get('success', False),
                        "cost": incr_query.get('cost', float('inf')),
                        "update_ms": t_incr_update,
                        "solve_ms": t_incr_solve,
                        "total_ms": t_incr_total,
                        "n_invalidated": update_stats['n_invalidated'],
                        "n_regrown": update_stats['n_regrown'],
                        "n_boxes_before": update_stats['n_before'],
                        "n_boxes_after": update_stats['n_after'],
                        "invalidate_ms": update_stats['invalidate_ms'],
                        "regrow_ms": update_stats['regrow_ms'],
                        "scene_ms": update_stats['scene_ms'],
                    },
                    wall_clock=t_incr_total / 1000,
                ))

                # ── 2) SBF 全量重建 ──
                modified_scene = _apply_changes_to_scene(
                    base_scene, change_type, n_change,
                    np.random.default_rng(seed * 100 + n_change))

                t0 = time.perf_counter()
                full_prep = grow_and_prepare(
                    robot, modified_scene, cfg, q_start, q_goal, ndim)
                fct = full_prep.get('_cache_thread')
                if fct is not None:
                    fct.join()
                t_full_build = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                full_query = _solve_on_prep(
                    full_prep, cfg, q_start, q_goal, ndim, timeout)
                t_full_solve = (time.perf_counter() - t0) * 1000
                t_full_total = t_full_build + t_full_solve

                print(f"FULL: build={t_full_build:.0f}ms, "
                      f"solve={t_full_solve:.0f}ms, "
                      f"boxes={len(full_prep['boxes'])}", end=" | ")

                speedup = t_full_total / t_incr_total if t_incr_total > 0 else float('inf')
                print(f"Speedup: {speedup:.1f}x")

                results.add(SingleTrialResult(
                    scene_name=scene_name,
                    planner_name="SBF-FullRebuild",
                    seed=seed, trial=0,
                    result={
                        "change_type": change_type,
                        "n_changes": n_change,
                        "mode": "full_rebuild",
                        "success": full_query.get('success', False),
                        "cost": full_query.get('cost', float('inf')),
                        "build_ms": t_full_build,
                        "solve_ms": t_full_solve,
                        "total_ms": t_full_total,
                        "n_boxes": len(full_prep['boxes']),
                    },
                    wall_clock=t_full_total / 1000,
                ))

                # ── 3) RRTConnect (全量重建基线) ──
                rrt_cfg = load_planners(["rrt_connect"])[0]
                rrt = create_planner(rrt_cfg)
                rrt_params = {k: v for k, v in rrt_cfg.items() if k != "type"}
                rrt_params["seed"] = seed
                rrt.setup(robot, modified_scene, rrt_params)
                t0 = time.perf_counter()
                r_rrt = rrt.plan(q_start, q_goal, timeout=timeout)
                t_rrt = time.perf_counter() - t0

                results.add(SingleTrialResult(
                    scene_name=scene_name,
                    planner_name="RRTConnect",
                    seed=seed, trial=0,
                    result={
                        **r_rrt.to_dict(),
                        "change_type": change_type,
                        "n_changes": n_change,
                        "mode": "full_rebuild",
                    },
                    wall_clock=t_rrt,
                ))

    out_path = OUTPUT_DIR / "exp3_obstacle_change.json"
    results.metadata = {
        "change_types": change_types,
        "change_counts": change_counts,
        "n_seeds": n_seeds,
        "n_obs_base": n_obs_base,
        "regrow_budget": regrow_budget,
    }
    results.save(out_path)

    print(f"\nResults saved to {out_path}")
    return out_path


# ─── 辅助函数 ─────────────────────────────────────────────

def _deep_copy_scene(scene: Scene) -> Scene:
    """深拷贝 Scene."""
    new_scene = Scene()
    for obs in scene.get_obstacles():
        new_scene.add_obstacle(
            list(obs.min_point), list(obs.max_point), name=obs.name)
    return new_scene


def _shallow_copy_prep(prep: dict) -> dict:
    """浅拷贝 prep, 深拷贝 boxes/adj/interval-cache 以支持独立修改."""
    import copy as _copy
    new_prep = dict(prep)
    new_prep['boxes'] = dict(prep['boxes'])
    fo = _copy.copy(prep['forest_obj'])
    fo.boxes = dict(prep['forest_obj'].boxes)
    fo.adjacency = {
        k: set(v) for k, v in prep['forest_obj'].adjacency.items()
    }
    # deep copy interval cache to avoid shared-state corruption
    fo._interval_ids = list(prep['forest_obj']._interval_ids)
    fo._interval_id_to_index = dict(prep['forest_obj']._interval_id_to_index)
    if hasattr(prep['forest_obj'], '_intervals_arr') and prep['forest_obj']._intervals_arr is not None:
        fo._intervals_arr = prep['forest_obj']._intervals_arr.copy()
    new_prep['forest_obj'] = fo
    return new_prep


def _compute_changes(base_scene, change_type, n_change, rng):
    """计算场景变化: 返回 (added_obstacles, removed_names)."""
    obstacles = base_scene.get_obstacles()
    added_obs = []
    removed_names = []

    if change_type == "remove":
        for obs in obstacles[-n_change:]:
            removed_names.append(obs.name)

    elif change_type == "add":
        for i in range(n_change):
            cx = float(rng.uniform(-0.6, 0.6))
            cy = float(rng.uniform(-0.6, 0.6))
            cz = float(rng.uniform(0.2, 0.8))
            h = float(rng.uniform(0.05, 0.12))
            added_obs.append({
                'min_point': [cx - h, cy - h, cz - h],
                'max_point': [cx + h, cy + h, cz + h],
                'name': f"new_obs_{i}",
            })

    elif change_type == "move":
        for obs in obstacles[-n_change:]:
            removed_names.append(obs.name)
            shift = rng.uniform(-0.2, 0.2, size=3)
            new_min = [float(obs.min_point[j] + shift[j]) for j in range(3)]
            new_max = [float(obs.max_point[j] + shift[j]) for j in range(3)]
            added_obs.append({
                'min_point': new_min,
                'max_point': new_max,
                'name': obs.name + "_moved",
            })

    return added_obs, removed_names


def _apply_changes_to_scene(base_scene, change_type, n_change, rng):
    """对 base_scene 应用变化, 返回新的 Scene (用于全量重建)."""
    obstacles = base_scene.get_obstacles()
    new_scene = Scene()

    if change_type == "remove":
        keep = obstacles[:max(0, len(obstacles) - n_change)]
        for obs in keep:
            new_scene.add_obstacle(obs.min_point, obs.max_point,
                                   name=obs.name)

    elif change_type == "add":
        for obs in obstacles:
            new_scene.add_obstacle(obs.min_point, obs.max_point,
                                   name=obs.name)
        for i in range(n_change):
            cx = float(rng.uniform(-0.6, 0.6))
            cy = float(rng.uniform(-0.6, 0.6))
            cz = float(rng.uniform(0.2, 0.8))
            h = float(rng.uniform(0.05, 0.12))
            new_scene.add_obstacle(
                [cx - h, cy - h, cz - h],
                [cx + h, cy + h, cz + h],
                name=f"new_obs_{i}")

    elif change_type == "move":
        for i, obs in enumerate(obstacles):
            if i >= len(obstacles) - n_change:
                shift = rng.uniform(-0.2, 0.2, size=3)
                new_min = [float(obs.min_point[j] + shift[j]) for j in range(3)]
                new_max = [float(obs.max_point[j] + shift[j]) for j in range(3)]
                new_scene.add_obstacle(new_min, new_max,
                                       name=obs.name + "_moved")
            else:
                new_scene.add_obstacle(obs.min_point, obs.max_point,
                                       name=obs.name)

    return new_scene


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
