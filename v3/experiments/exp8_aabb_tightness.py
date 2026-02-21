"""
experiments/exp8_aabb_tightness.py — 实验 8: AABB 紧致度

对比区间 FK 生成的 AABB 紧致度:
  - 关键配置采样 vs 随机配置采样
  - 不同 n_subdivisions

用法:
    python -m experiments.exp8_aabb_tightness [--quick]
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

from experiments.runner import ExperimentResults, SingleTrialResult

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"


def _measure_aabb_volume(robot, intervals):
    """Compute AABB volume for given joint intervals using interval FK."""
    from aabb.calculator import AABBCalculator
    calc = AABBCalculator(robot)
    aabb = calc.compute_aabb(intervals)
    if aabb is None:
        return 0.0
    vol = 1.0
    for lo, hi in aabb:
        vol *= (hi - lo)
    return vol


def _measure_numerical_volume(robot, intervals, n_samples: int = 1000,
                               rng=None):
    """Compute tight bounding box via random sampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    limits = np.array(intervals)
    samples = rng.uniform(limits[:, 0], limits[:, 1],
                          size=(n_samples, len(intervals)))

    all_positions = []
    for q in samples:
        positions = robot.get_link_positions(q)
        for pos in positions:
            all_positions.append(pos)

    all_positions = np.array(all_positions)
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    vol = np.prod(maxs - mins)
    return float(vol)


def run(quick: bool = False) -> Path:
    from aabb.robot import load_robot

    robot = load_robot("panda")
    n_joints = robot.n_joints
    limits = np.array(robot.joint_limits)

    # generate test intervals of varying sizes
    rng = np.random.default_rng(42)
    interval_sizes = [0.5, 1.0, 2.0] if quick else [0.2, 0.5, 1.0, 1.5, 2.0]
    n_samples_per_size = 5 if quick else 20

    print("=== Experiment 8: AABB Tightness ===")
    print(f"  Interval sizes: {interval_sizes}")
    print(f"  Samples per size: {n_samples_per_size}")
    print()

    results = ExperimentResults(experiment_name="exp8_aabb_tightness")

    for size in interval_sizes:
        for i in range(n_samples_per_size):
            # generate random interval of given size
            center = rng.uniform(limits[:, 0] + size / 2,
                                 limits[:, 1] - size / 2)
            intervals = [(float(center[j] - size / 2),
                          float(center[j] + size / 2))
                         for j in range(n_joints)]

            # clip to joint limits
            intervals = [(max(lo, float(limits[j, 0])),
                          min(hi, float(limits[j, 1])))
                         for j, (lo, hi) in enumerate(intervals)]

            # measure interval FK AABB volume
            t0 = time.perf_counter()
            vol_interval = _measure_aabb_volume(robot, intervals)
            t_interval = time.perf_counter() - t0

            # measure numerical tight volume
            n_numerical = 500 if quick else 2000
            t0 = time.perf_counter()
            vol_numerical = _measure_numerical_volume(
                robot, intervals, n_samples=n_numerical,
                rng=np.random.default_rng(i))
            t_numerical = time.perf_counter() - t0

            tightness = (vol_numerical / vol_interval
                         if vol_interval > 0 else 0.0)

            results.add(SingleTrialResult(
                scene_name=f"interval_size_{size}",
                planner_name="interval_fk",
                seed=i, trial=0,
                result={
                    "interval_size": size,
                    "vol_interval_fk": vol_interval,
                    "vol_numerical": vol_numerical,
                    "tightness_ratio": tightness,
                    "time_interval_fk": t_interval,
                    "time_numerical": t_numerical,
                },
                wall_clock=t_interval,
            ))

    out_path = OUTPUT_DIR / "exp8_aabb_tightness.json"
    results.metadata = {
        "interval_sizes": interval_sizes,
        "n_samples_per_size": n_samples_per_size,
    }
    results.save(out_path)

    # Summary
    print("\n=== Summary ===")
    for size in interval_sizes:
        matching = [t for t in results.trials
                    if t.result.get("interval_size") == size]
        if matching:
            ratios = [t.result["tightness_ratio"] for t in matching]
            print(f"  size={size}: mean_tightness={np.mean(ratios):.4f} "
                  f"(1.0=perfect, lower=looser)")

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
