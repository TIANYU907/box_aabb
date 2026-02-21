"""
experiments/exp2_forest_reuse.py — 实验 2: Forest 跨查询复用

★ 核心优势实验：展示 SBF 首次 build 后跨查询 amortized cost

设计:
  方案 A: SBF — 首次完整 build, 后续 K-1 次复用 forest
  方案 B: RRTConnect — 每次从零构建
  方案 C: RRT* — 每次从零构建

K = [1, 5, 10, 20, 50, 100] (quick 模式下 [1, 3, 5])

指标:
  - 累积总耗时
  - 每查询平均耗时
  - 加速比 (SBF / RRTConnect)
  - 首查询 vs 后续查询耗时

用法:
    python -m experiments.exp2_forest_reuse [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.runner import (ExperimentRunner, ExperimentResults,
                                 SingleTrialResult, load_scene_from_config,
                                 create_planner)
from experiments.scenes import load_scenes, load_planners

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"


def _generate_query_pairs(robot, scene, n_pairs: int, seed: int):
    """生成 n_pairs 个 collision-free query pairs."""
    from forest.collision import CollisionChecker
    checker = CollisionChecker(robot=robot, scene=scene)
    rng = np.random.default_rng(seed)
    limits = np.array(robot.joint_limits)  # (ndim, 2)
    pairs = []
    while len(pairs) < n_pairs:
        q_start = rng.uniform(limits[:, 0], limits[:, 1])
        q_goal = rng.uniform(limits[:, 0], limits[:, 1])
        if (not checker.check_config_collision(q_start) and
                not checker.check_config_collision(q_goal)):
            pairs.append((q_start.copy(), q_goal.copy()))
    return pairs


def run(quick: bool = False) -> Path:
    K_values = [1, 3, 5] if quick else [1, 5, 10, 20, 50, 100]
    max_K = max(K_values)
    n_seeds = 2 if quick else 10
    timeout = 15.0 if quick else 30.0

    scene_cfg = load_scenes(["panda_8obs_reuse"])[0]
    robot, scene, _ = load_scene_from_config(scene_cfg)

    planner_cfgs = load_planners(["sbf_dijkstra", "rrt_connect", "rrt_star"])

    print(f"=== Experiment 2: Forest Reuse ===")
    print(f"  K values: {K_values}")
    print(f"  Seeds:    {n_seeds}")
    print(f"  Planners: {[p.get('algorithm', p.get('method', '?')) for p in planner_cfgs]}")
    print()

    results = ExperimentResults(experiment_name="exp2_forest_reuse")

    for seed in range(n_seeds):
        query_pairs = _generate_query_pairs(robot, scene, max_K, seed=seed * 1000)

        for pcfg in planner_cfgs:
            planner = create_planner(pcfg)
            params = {k: v for k, v in pcfg.items() if k != "type"}
            params["seed"] = seed
            planner.setup(robot, scene, params)

            cumulative_time = 0.0
            for qi in range(max_K):
                q_start, q_goal = query_pairs[qi]
                t0 = time.perf_counter()
                result = planner.plan(q_start, q_goal, timeout=timeout)
                wall = time.perf_counter() - t0
                cumulative_time += wall

                # record at each K checkpoint
                if (qi + 1) in K_values:
                    trial_result = SingleTrialResult(
                        scene_name=scene_cfg["name"],
                        planner_name=planner.name,
                        seed=seed,
                        trial=qi,
                        result={
                            **result.to_dict(),
                            "K": qi + 1,
                            "query_index": qi,
                            "cumulative_time": cumulative_time,
                            "avg_time_per_query": cumulative_time / (qi + 1),
                            "is_first_query": qi == 0,
                        },
                        wall_clock=wall,
                    )
                    results.add(trial_result)

            # reset for stateless planners is no-op;
            # for SBF reset clears forest
            if not planner.supports_reuse:
                pass  # already stateless

            logger.info("seed=%d planner=%s K=%d cum=%.3fs",
                        seed, planner.name, max_K, cumulative_time)

    out_path = OUTPUT_DIR / "exp2_forest_reuse.json"
    results.metadata = {
        "K_values": K_values,
        "n_seeds": n_seeds,
        "max_K": max_K,
    }
    results.save(out_path)

    # Summary
    print("\n=== Summary (at max K) ===")
    for pcfg in planner_cfgs:
        name = create_planner(pcfg).name
        matching = [t for t in results.trials
                    if t.planner_name == name
                    and t.result.get("K") == max_K]
        if matching:
            cum_times = [t.result["cumulative_time"] for t in matching]
            avg = np.mean(cum_times)
            print(f"  {name}: K={max_K} avg_cumulative={avg:.3f}s "
                  f"avg_per_query={avg/max_K:.4f}s")

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
