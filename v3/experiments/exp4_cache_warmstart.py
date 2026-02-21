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
import tempfile
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

    for seed in range(n_seeds):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test.hcache"

            # ── Run 1: 冷启动 (no cache) ──
            from planner.sbf_planner import SBFPlanner
            from planner.models import SBFConfig

            cfg = SBFConfig(
                max_box_nodes=200,
                min_box_size=0.1,
                forest_path=None,
                verbose=False,
            )
            planner = SBFPlanner(robot=robot, scene=scene,
                                  config=cfg, no_cache=True)
            t0 = time.perf_counter()
            r1 = planner.plan(q_start, q_goal, seed=seed)
            t_cold = time.perf_counter() - t0

            results.add(SingleTrialResult(
                scene_name="panda_8obs",
                planner_name="SBF-ColdStart",
                seed=seed, trial=0,
                result={
                    "success": r1.success,
                    "planning_time": t_cold,
                    "mode": "cold",
                    "n_boxes": len(r1.forest.boxes) if r1.forest else 0,
                },
                wall_clock=t_cold,
            ))

            # Save cache
            if r1.forest and hasattr(r1.forest, '_tree'):
                try:
                    r1.forest._tree.save_binary(str(cache_path))
                    cache_saved = True
                except Exception:
                    cache_saved = False
            else:
                cache_saved = False

            # ── Run 2: 热启动 (with cache) ──
            if cache_saved:
                cfg2 = SBFConfig(
                    max_box_nodes=200,
                    min_box_size=0.1,
                    forest_path=str(cache_path),
                    verbose=False,
                )
                planner2 = SBFPlanner(robot=robot, scene=scene,
                                       config=cfg2, no_cache=False)
                t0 = time.perf_counter()
                r2 = planner2.plan(q_start, q_goal, seed=seed)
                t_warm = time.perf_counter() - t0

                results.add(SingleTrialResult(
                    scene_name="panda_8obs",
                    planner_name="SBF-WarmStart",
                    seed=seed, trial=0,
                    result={
                        "success": r2.success,
                        "planning_time": t_warm,
                        "mode": "warm",
                        "n_boxes": len(r2.forest.boxes) if r2.forest else 0,
                        "speedup": t_cold / t_warm if t_warm > 0 else 0,
                    },
                    wall_clock=t_warm,
                ))

            # ── Run 3: 跨场景 cache 复用 ──
            if cache_saved:
                scene_cfg_b = load_scenes(["panda_15obs_moderate"])[0]
                _, scene_b, _ = load_scene_from_config(scene_cfg_b)

                cfg3 = SBFConfig(
                    max_box_nodes=200,
                    min_box_size=0.1,
                    forest_path=str(cache_path),
                    verbose=False,
                )
                planner3 = SBFPlanner(robot=robot, scene=scene_b,
                                       config=cfg3, no_cache=False)
                t0 = time.perf_counter()
                r3 = planner3.plan(q_start, q_goal, seed=seed)
                t_cross = time.perf_counter() - t0

                results.add(SingleTrialResult(
                    scene_name="panda_15obs",
                    planner_name="SBF-CrossSceneCache",
                    seed=seed, trial=0,
                    result={
                        "success": r3.success,
                        "planning_time": t_cross,
                        "mode": "cross_scene_cache",
                        "n_boxes": len(r3.forest.boxes) if r3.forest else 0,
                    },
                    wall_clock=t_cross,
                ))

        logger.info("seed=%d cold=%.3fs warm=%.3fs",
                     seed, t_cold,
                     t_warm if cache_saved else float('nan'))

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
