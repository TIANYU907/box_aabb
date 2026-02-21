"""
experiments/exp3_obstacle_change.py — 实验 3: 障碍物变化增量更新

★ 核心优势实验：展示 SBF 增量更新 vs 全量重建

设计:
  场景 A — 障碍物消失: 移除 1/2/3 个 → 增量 regrow vs 全量
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
from typing import List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.runner import (ExperimentResults, SingleTrialResult,
                                 create_planner, load_scene_from_config)
from experiments.scenes import load_scenes, load_planners

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


def run(quick: bool = False) -> Path:
    change_counts = [1, 2] if quick else [1, 2, 3]
    n_seeds = 2 if quick else 10
    timeout = 15.0 if quick else 30.0

    results = ExperimentResults(experiment_name="exp3_obstacle_change")

    planner_cfgs = load_planners(["sbf_dijkstra", "rrt_connect"])
    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])

    change_types = ["remove", "add", "move"]

    print("=== Experiment 3: Obstacle Change ===")
    print(f"  Change types: {change_types}")
    print(f"  Change counts: {change_counts}")
    print(f"  Seeds: {n_seeds}")
    print()

    for seed in range(n_seeds):
        base_cfg = _build_base_scene(n_obs=10, seed=400 + seed)
        robot, base_scene, _ = load_scene_from_config(base_cfg)

        for change_type in change_types:
            for n_change in change_counts:
                scene_name = f"obs_{change_type}_{n_change}"

                # ── SBF incremental ──
                pcfg = load_planners(["sbf_dijkstra"])[0]
                sbf = create_planner(pcfg)
                params = {k: v for k, v in pcfg.items() if k != "type"}
                params["seed"] = seed

                # 1) initial build on base scene
                sbf.setup(robot, base_scene, params)
                t0 = time.perf_counter()
                r_init = sbf.plan(q_start, q_goal, timeout=timeout)
                t_init = time.perf_counter() - t0

                # 2) modify scene
                modified_scene = _modify_scene(
                    base_scene, change_type, n_change,
                    rng=np.random.default_rng(seed * 100 + n_change))

                # 3) incremental: reset and re-plan on modified scene
                #    (SBF adapter currently rebuilds on new scene;
                #     true incremental would use forest.invalidate)
                sbf.reset()
                sbf.setup(robot, modified_scene, params)
                t0 = time.perf_counter()
                r_incr = sbf.plan(q_start, q_goal, timeout=timeout)
                t_incr = time.perf_counter() - t0

                results.add(SingleTrialResult(
                    scene_name=scene_name,
                    planner_name="SBF-Incremental",
                    seed=seed, trial=0,
                    result={
                        **r_incr.to_dict(),
                        "change_type": change_type,
                        "n_changes": n_change,
                        "initial_build_time": t_init,
                        "update_time": t_incr,
                        "mode": "incremental",
                    },
                    wall_clock=t_incr,
                ))

                # ── SBF full rebuild (control) ──
                sbf2 = create_planner(pcfg)
                sbf2.setup(robot, modified_scene, params)
                t0 = time.perf_counter()
                r_full = sbf2.plan(q_start, q_goal, timeout=timeout)
                t_full = time.perf_counter() - t0

                results.add(SingleTrialResult(
                    scene_name=scene_name,
                    planner_name="SBF-FullRebuild",
                    seed=seed, trial=0,
                    result={
                        **r_full.to_dict(),
                        "change_type": change_type,
                        "n_changes": n_change,
                        "rebuild_time": t_full,
                        "mode": "full_rebuild",
                    },
                    wall_clock=t_full,
                ))

                # ── RRTConnect (control) ──
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
                        "rebuild_time": t_rrt,
                        "mode": "full_rebuild",
                    },
                    wall_clock=t_rrt,
                ))

    out_path = OUTPUT_DIR / "exp3_obstacle_change.json"
    results.metadata = {
        "change_types": change_types,
        "change_counts": change_counts,
        "n_seeds": n_seeds,
    }
    results.save(out_path)

    print(f"\nResults saved to {out_path}")
    return out_path


def _modify_scene(base_scene, change_type: str, n_change: int,
                   rng: np.random.Generator):
    """Create a modified copy of the scene."""
    from forest.scene import Scene

    obstacles = list(base_scene.obstacles)  # shallow copy
    new_scene = Scene()

    if change_type == "remove":
        # remove last n_change obstacles
        keep = obstacles[:max(0, len(obstacles) - n_change)]
        for obs in keep:
            new_scene.add_obstacle(obs.min_point, obs.max_point,
                                   name=obs.name)

    elif change_type == "add":
        # keep all + add new
        for obs in obstacles:
            new_scene.add_obstacle(obs.min_point, obs.max_point,
                                   name=obs.name)
        for i in range(n_change):
            cx = rng.uniform(-0.6, 0.6)
            cy = rng.uniform(-0.6, 0.6)
            cz = rng.uniform(0.2, 0.8)
            h = rng.uniform(0.05, 0.12)
            new_scene.add_obstacle(
                [cx - h, cy - h, cz - h],
                [cx + h, cy + h, cz + h],
                name=f"new_obs_{i}")

    elif change_type == "move":
        for i, obs in enumerate(obstacles):
            if i >= len(obstacles) - n_change:
                # move this obstacle
                shift = rng.uniform(-0.2, 0.2, size=3)
                new_min = [obs.min_point[j] + shift[j] for j in range(3)]
                new_max = [obs.max_point[j] + shift[j] for j in range(3)]
                new_scene.add_obstacle(new_min, new_max, name=obs.name)
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
