"""
experiments/exp6_config_sweep.py — 实验 6: 配置敏感性扫描

扫描 SBF 核心参数, 观察成功率与耗时变化:
- max_box_nodes
- min_box_size
- goal_bias
- boundary_expand on/off
- boundary_expand_max_failures
- boundary_expand_epsilon

用法:
    python -m experiments.exp6_config_sweep [--quick]
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
                                 SingleTrialResult, load_scene_from_config,
                                 create_planner)
from experiments.scenes import load_scenes

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"

# 参数扫描定义
SWEEP_PARAMS = {
    "max_boxes": {
        "values": [100, 200, 400, 800, 1600],
        "base": {"type": "SBF", "method": "dijkstra", "no_cache": True,
                 "boundary_expand": True},
    },
    "min_box_size": {
        "values": [0.01, 0.005, 0.001, 0.0005],
        "base": {"type": "SBF", "method": "dijkstra", "no_cache": True,
                 "boundary_expand": True, "max_boxes": 500},
    },
    "goal_bias": {
        "values": [0.0, 0.05, 0.1, 0.2, 0.3],
        "base": {"type": "SBF", "method": "dijkstra", "no_cache": True,
                 "boundary_expand": True, "max_boxes": 500},
    },
    "boundary_expand_max_failures": {
        "values": [3, 5, 8, 12],
        "base": {"type": "SBF", "method": "dijkstra", "no_cache": True,
                 "boundary_expand": True, "max_boxes": 500},
    },
    "boundary_expand_epsilon": {
        "values": [0.005, 0.01, 0.02, 0.05],
        "base": {"type": "SBF", "method": "dijkstra", "no_cache": True,
                 "boundary_expand": True, "max_boxes": 500},
    },
}


def run(quick: bool = False) -> Path:
    n_seeds = 2 if quick else 20
    timeout = 15.0 if quick else 30.0

    scene_cfg = load_scenes(["panda_8obs_open"])[0]
    robot, scene, query_pairs = load_scene_from_config(scene_cfg)
    q_start, q_goal = query_pairs[0]

    print("=== Experiment 6: Config Sweep ===")
    print(f"  Parameters: {list(SWEEP_PARAMS.keys())}")
    print(f"  Seeds: {n_seeds}")
    print()

    results = ExperimentResults(experiment_name="exp6_config_sweep")

    for param_name, sweep in SWEEP_PARAMS.items():
        values = sweep["values"]
        if quick:
            values = values[:3]  # reduce in quick mode

        for val in values:
            for seed in range(n_seeds):
                cfg = {**sweep["base"], param_name: val}
                planner = create_planner(cfg)
                params = {k: v for k, v in cfg.items() if k != "type"}
                params["seed"] = seed
                planner.setup(robot, scene, params)

                t0 = time.perf_counter()
                result = planner.plan(q_start, q_goal, timeout=timeout)
                wall = time.perf_counter() - t0

                results.add(SingleTrialResult(
                    scene_name="panda_8obs_open",
                    planner_name=f"SBF-{param_name}={val}",
                    seed=seed, trial=0,
                    result={
                        **result.to_dict(),
                        "sweep_param": param_name,
                        "sweep_value": val,
                    },
                    wall_clock=wall,
                ))

        logger.info("Swept %s: %s", param_name, values)

    out_path = OUTPUT_DIR / "exp6_config_sweep.json"
    results.metadata = {
        "sweep_params": {k: v["values"] for k, v in SWEEP_PARAMS.items()},
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
