"""
experiments/exp4_cache_warmstart.py — 实验 4: AABB 缓存热启动

★ 核心优势实验：HCACHE 持久化 + 增量 FK 加速效果

设计:
  A — 冷启动 vs 热启动:
      Run 1: 无 cache → 记录 FK 调用数、构建时间
      Run 2: 有 cache → 记录 FK 调用数、构建时间
  B — 跨场景 cache 复用:
      场景 1 构建 → 加载到场景 2 (不同障碍物)

用法:
    python -m experiments.exp4_cache_warmstart [--quick]
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

from experiments.runner import (ExperimentResults, SingleTrialResult,
                                 load_scene_from_config)
from experiments.scenes import load_scenes

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"


def run(quick: bool = False) -> Path:
    n_seeds = 2 if quick else 5
    timeout = 15.0 if quick else 30.0

    results = ExperimentResults(experiment_name="exp4_cache_warmstart")

    scene_cfg = load_scenes(["panda_8obs_open"])[0]
    robot, scene, query_pairs = load_scene_from_config(scene_cfg)

    q_start, q_goal = query_pairs[0]

    print("=== Experiment 4: Cache Warmstart ===")
    print(f"  Seeds: {n_seeds}")
    print()

    from planner.sbf_planner import SBFPlanner
    from planner.models import SBFConfig

    for seed in range(n_seeds):
        # ── Run 1: 冷启动 (no_cache=True → 不加载 HCACHE) ──
        cfg = SBFConfig(
            max_box_nodes=200,
            min_box_size=0.1,
            verbose=False,
        )
        planner_cold = SBFPlanner(robot=robot, scene=scene,
                                   config=cfg, no_cache=True)
        t0 = time.perf_counter()
        r1 = planner_cold.plan(q_start, q_goal, seed=seed)
        t_cold = time.perf_counter() - t0

        n_boxes_cold = len(r1.forest.boxes) if r1.forest else 0
        results.add(SingleTrialResult(
            scene_name="panda_8obs",
            planner_name="SBF-ColdStart",
            seed=seed, trial=0,
            result={
                "success": r1.success,
                "planning_time": t_cold,
                "mode": "cold",
                "n_boxes": n_boxes_cold,
            },
            wall_clock=t_cold,
        ))

        # ── Run 2: 热启动 (no_cache=False → 加载 auto-saved HCACHE) ──
        # HierAABBTree.auto_load() 自动从 .cache/ 加载已有的 FK 缓存
        # 这意味着 boundary_expand 中大量 FK 计算可以跳过
        planner_warm = SBFPlanner(robot=robot, scene=scene,
                                   config=cfg, no_cache=False)
        t0 = time.perf_counter()
        r2 = planner_warm.plan(q_start, q_goal, seed=seed)
        t_warm = time.perf_counter() - t0

        n_boxes_warm = len(r2.forest.boxes) if r2.forest else 0
        speedup = t_cold / t_warm if t_warm > 0 else 0
        results.add(SingleTrialResult(
            scene_name="panda_8obs",
            planner_name="SBF-WarmStart",
            seed=seed, trial=0,
            result={
                "success": r2.success,
                "planning_time": t_warm,
                "mode": "warm",
                "n_boxes": n_boxes_warm,
                "speedup": speedup,
            },
            wall_clock=t_warm,
        ))

        # ── Run 3: 跨场景 cache 复用 ──
        # 同一 robot 的 HCACHE 对不同场景仍有效 (AABB tree 不依赖场景)
        scene_cfg_b = load_scenes(["panda_15obs_moderate"])[0]
        _, scene_b, _ = load_scene_from_config(scene_cfg_b)

        planner_cross = SBFPlanner(robot=robot, scene=scene_b,
                                    config=cfg, no_cache=False)
        t0 = time.perf_counter()
        r3 = planner_cross.plan(q_start, q_goal, seed=seed)
        t_cross = time.perf_counter() - t0

        n_boxes_cross = len(r3.forest.boxes) if r3.forest else 0
        results.add(SingleTrialResult(
            scene_name="panda_15obs",
            planner_name="SBF-CrossSceneCache",
            seed=seed, trial=0,
            result={
                "success": r3.success,
                "planning_time": t_cross,
                "mode": "cross_scene_cache",
                "n_boxes": n_boxes_cross,
            },
            wall_clock=t_cross,
        ))

        logger.info("seed=%d cold=%.3fs warm=%.3fs cross=%.3fs",
                     seed, t_cold, t_warm, t_cross)

    out_path = OUTPUT_DIR / "exp4_cache_warmstart.json"
    results.metadata = {"n_seeds": n_seeds}
    results.save(out_path)

    # Summary
    print("\n=== Summary ===")
    for mode in ("SBF-ColdStart", "SBF-WarmStart", "SBF-CrossSceneCache"):
        matching = [t for t in results.trials if t.planner_name == mode]
        if matching:
            times = [t.result["planning_time"] for t in matching]
            print(f"  {mode}: mean={np.mean(times):.3f}s "
                  f"n={len(matching)}")

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
