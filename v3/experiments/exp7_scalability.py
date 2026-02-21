"""
experiments/exp7_scalability.py — 实验 7: 可扩展性

维度:
  - 障碍物数量: 2, 4, 8, 12, 16, 20
  - 自由度: 2-DOF, 3-DOF, 7-DOF (Panda)

方法: SBF vs RRTConnect

用法:
    python -m experiments.exp7_scalability [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.runner import (ExperimentRunner, ExperimentResults,
                                 SingleTrialResult, create_planner,
                                 load_scene_from_config)
from experiments.scenes import load_planners

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"


def _make_scene_cfg(robot_name: str, n_obs: int, seed: int) -> dict:
    """Build scene config for given DOF and obstacle count."""
    if robot_name == "2dof_planar":
        q_start = [-1.0, 0.5]
        q_goal = [1.0, -0.5]
        obs_cfg = {
            "count": n_obs, "seed": seed,
            "x_range": [-0.8, 0.8], "y_range": [-0.8, 0.8],
            "z_range": [-0.5, 0.5], "half_size_range": [0.03, 0.08],
        }
    elif robot_name == "3dof_planar":
        q_start = [-1.0, 0.5, 0.3]
        q_goal = [1.0, -0.5, -0.3]
        obs_cfg = {
            "count": n_obs, "seed": seed,
            "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0],
            "z_range": [-0.5, 0.5], "half_size_range": [0.04, 0.10],
        }
    else:  # panda
        q_start = [0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5]
        q_goal = [-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8]
        obs_cfg = {
            "count": n_obs, "seed": seed,
            "x_range": [-0.8, 0.8], "y_range": [-0.8, 0.8],
            "z_range": [0.1, 0.9], "half_size_range": [0.05, 0.15],
        }

    return {
        "name": f"{robot_name}_{n_obs}obs",
        "robot": robot_name,
        "random_obstacles": obs_cfg,
        "query_pairs": [{"start": q_start, "goal": q_goal}],
    }


def run(quick: bool = False) -> Path:
    n_obs_list = [2, 4, 8] if quick else [2, 4, 8, 12, 16, 20]
    robot_names = ["2dof_planar", "panda"] if quick else [
        "2dof_planar", "3dof_planar", "panda"]
    n_seeds = 2 if quick else 10
    timeout = 10.0 if quick else 30.0

    planner_cfgs = load_planners(["sbf_dijkstra", "rrt_connect"])

    print("=== Experiment 7: Scalability ===")
    print(f"  Robots: {robot_names}")
    print(f"  Obstacle counts: {n_obs_list}")
    print(f"  Seeds: {n_seeds}")
    print()

    results = ExperimentResults(experiment_name="exp7_scalability")

    for robot_name in robot_names:
        for n_obs in n_obs_list:
            for seed in range(n_seeds):
                scene_cfg = _make_scene_cfg(robot_name, n_obs,
                                             seed=700 + seed)
                try:
                    robot, scene, qpairs = load_scene_from_config(scene_cfg)
                except Exception as e:
                    logger.warning("Scene build failed: %s", e)
                    continue

                q_start, q_goal = qpairs[0]

                for pcfg in planner_cfgs:
                    planner = create_planner(pcfg)
                    params = {k: v for k, v in pcfg.items() if k != "type"}
                    params["seed"] = seed
                    planner.setup(robot, scene, params)

                    t0 = time.perf_counter()
                    result = planner.plan(q_start, q_goal, timeout=timeout)
                    wall = time.perf_counter() - t0

                    results.add(SingleTrialResult(
                        scene_name=f"{robot_name}_{n_obs}obs",
                        planner_name=planner.name,
                        seed=seed, trial=0,
                        result={
                            **result.to_dict(),
                            "robot": robot_name,
                            "n_obstacles": n_obs,
                            "n_dof": robot.n_joints,
                        },
                        wall_clock=wall,
                    ))

            logger.info("%s %d obs done", robot_name, n_obs)

    out_path = OUTPUT_DIR / "exp7_scalability.json"
    results.metadata = {
        "robots": robot_names,
        "n_obs_list": n_obs_list,
        "n_seeds": n_seeds,
    }
    results.save(out_path)

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
