"""
experiments/exp1_main_comparison.py — 实验 1: 主对比

SBF vs RRTConnect vs RRT* vs Informed-RRT* vs BiRRT*
(OMPL + IRIS-GCS 可选, 需要对应环境)

场景: Panda 7-DOF × {8, 15, 20} 障碍物
指标: 成功率, 首次解时间, 总规划时间, 路径长度, 碰撞检测次数
统计: N seeds × M trials

用法:
    python -m experiments.exp1_main_comparison [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ensure src/ on path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.runner import ExperimentRunner
from experiments.scenes import load_scenes, load_planners

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"


def build_config(quick: bool = False) -> dict:
    """构建实验 1 配置."""
    scenes = load_scenes(["panda_8obs_open", "panda_15obs_moderate",
                          "panda_20obs_dense"])

    # 核心 Python planners (always available)
    planner_names = ["sbf_dijkstra", "rrt_connect", "rrt_star",
                     "informed_rrt_star", "birrt_star"]
    planners = load_planners(planner_names)

    if quick:
        seeds = list(range(3))
        n_trials = 1
        timeout = 15.0
    else:
        seeds = list(range(50))
        n_trials = 3
        timeout = 30.0

    return {
        "name": "exp1_main_comparison",
        "scenes": scenes,
        "planners": planners,
        "seeds": seeds,
        "n_trials": n_trials,
        "timeout": timeout,
    }


def run(quick: bool = False) -> Path:
    config = build_config(quick=quick)
    runner = ExperimentRunner(config)

    total = (len(config["scenes"]) * len(config["planners"])
             * len(config["seeds"]) * config["n_trials"])
    print(f"=== Experiment 1: Main Comparison ===")
    print(f"  Scenes:   {len(config['scenes'])}")
    print(f"  Planners: {len(config['planners'])}")
    print(f"  Seeds:    {len(config['seeds'])}")
    print(f"  Trials:   {config['n_trials']}")
    print(f"  Total:    {total} runs")
    print()

    results = runner.run()

    out_path = OUTPUT_DIR / f"exp1_main_comparison.json"
    results.save(out_path)

    # Print summary
    stats = results.grouped_stats(
        group_keys=("scene", "planner"),
        value_key="planning_time")
    print("\n=== Summary ===")
    for key, s in sorted(stats.items()):
        n_success = sum(1 for t in results.trials
                        if str((t.scene_name, t.planner_name)) == key
                        and t.result.get("success"))
        n_total = s["n"]
        print(f"  {key}: success={n_success}/{n_total} "
              f"mean={s['mean']:.3f}s median={s['median']:.3f}s "
              f"p95={s['p95']:.3f}s")

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 seeds, 1 trial")
    args = parser.parse_args()
    run(quick=args.quick)
