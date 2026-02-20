"""
gcs_batch_test_2dof.py — 多场景批量测试 GCS 规划 (有/无 coarsen)

在多个随机 seed 下生成不同障碍物场景, 对比:
 - no-coarsen vs dim-sweep coarsen
 - 路径代价、求解时间、box 数量、成功率

用法:
    python -m v2.examples.gcs_batch_test_2dof
    python -m v2.examples.gcs_batch_test_2dof --n-scenes 20
"""

import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
from _bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from common.output import make_output_dir

from v2.examples.gcs_planner_2dof import (
    GCSConfig, build_random_scene, _run_gcs_pipeline,
    scan_collision_map, plot_path_comparison,
)


def run_one_scene(scene_idx: int, seed: int, cfg: GCSConfig):
    """对单个场景运行 no-coarsen / coarsen 两条 pipeline, 返回结果字典."""
    robot = load_robot(cfg.robot_name)
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    ndim = len(robot.joint_limits)
    jl = robot.joint_limits[0]
    period = float(jl[1] - jl[0])

    rng = np.random.default_rng(seed)
    try:
        scene = build_random_scene(robot, q_start, q_goal, rng, cfg)
    except RuntimeError:
        return None  # 场景生成失败

    cfg_scene = GCSConfig(
        seed=seed,
        robot_name=cfg.robot_name,
        n_obstacles=cfg.n_obstacles,
        q_start=cfg.q_start,
        q_goal=cfg.q_goal,
        max_consecutive_miss=cfg.max_consecutive_miss,
        min_box_size=cfg.min_box_size,
        corridor_hops=cfg.corridor_hops,
        coarsen_max_rounds=cfg.coarsen_max_rounds,
    )

    result_no = _run_gcs_pipeline(
        robot, scene, cfg_scene, q_start, q_goal, period, ndim,
        with_coarsen=False, label=f"S{scene_idx}-no")

    result_yes = _run_gcs_pipeline(
        robot, scene, cfg_scene, q_start, q_goal, period, ndim,
        with_coarsen=True, label=f"S{scene_idx}-crs")

    return dict(
        scene_idx=scene_idx, seed=seed,
        result_no=result_no, result_yes=result_yes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenes", type=int, default=10)
    parser.add_argument("--n-obstacles", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260220)
    args = parser.parse_args()

    cfg = GCSConfig(n_obstacles=args.n_obstacles)
    seeds = [args.base_seed + i * 37 for i in range(args.n_scenes)]

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Scene {i+1}/{args.n_scenes}  (seed={seed})")
        print(f"{'='*60}")
        r = run_one_scene(i, seed, cfg)
        if r is not None:
            all_results.append(r)

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    hdr = (f"  {'scene':>5s}  {'seed':>10s}  "
           f"{'cost_no':>8s}  {'cost_crs':>8s}  {'ratio':>6s}  "
           f"{'box_no':>6s}  {'box_crs':>7s}  "
           f"{'ms_no':>7s}  {'ms_crs':>7s}  {'speedup':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    n_both_ok = 0
    cost_ratios = []
    speedups = []

    for r in all_results:
        rn = r['result_no']
        ry = r['result_yes']
        if rn is None or ry is None:
            continue
        ok_no = rn['success']
        ok_yes = ry['success']
        c_no = rn['cost'] if ok_no else float('inf')
        c_yes = ry['cost'] if ok_yes else float('inf')
        t_no = rn['grow_ms'] + rn['bridge_ms'] + rn['gcs_ms']
        t_yes = ry['grow_ms'] + ry['coarsen_ms'] + ry['bridge_ms'] + ry['gcs_ms']
        b_no = len(rn['boxes'])
        b_yes = len(ry['boxes'])

        ratio = c_yes / c_no if (ok_no and ok_yes and c_no > 0) else float('nan')
        spd = t_no / t_yes if t_yes > 0 else float('nan')

        mark = ""
        if ok_no and ok_yes:
            n_both_ok += 1
            cost_ratios.append(ratio)
            speedups.append(spd)
        else:
            mark = " *FAIL*"

        print(f"  {r['scene_idx']:5d}  {r['seed']:10d}  "
              f"{c_no:8.2f}  {c_yes:8.2f}  {ratio:6.2f}  "
              f"{b_no:6d}  {b_yes:7d}  "
              f"{t_no:7.0f}  {t_yes:7.0f}  {spd:7.2f}x{mark}")

    print()
    print(f"  Scenes tested  : {len(all_results)}")
    print(f"  Both succeeded : {n_both_ok}")
    if cost_ratios:
        cr = np.array(cost_ratios)
        sp = np.array(speedups)
        print(f"  Cost ratio (crs/no)  : mean={cr.mean():.3f}  "
              f"median={np.median(cr):.3f}  min={cr.min():.3f}  max={cr.max():.3f}")
        print(f"  Speedup              : mean={sp.mean():.2f}x  "
              f"median={np.median(sp):.2f}x  min={sp.min():.2f}x  max={sp.max():.2f}x")
        print(f"  Cost < 1.0 (coarsen better): {(cr < 1.0).sum()}/{len(cr)}")

    # ── 画一个总体对比图 ──
    out_dir = make_output_dir("visualizations", "gcs_batch_2dof")
    print(f"\nOutput: {out_dir}")

    if n_both_ok >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # (a) cost scatter
        ax = axes[0]
        costs_no = [r['result_no']['cost'] for r in all_results
                    if r['result_no'] and r['result_yes']
                    and r['result_no']['success'] and r['result_yes']['success']]
        costs_yes = [r['result_yes']['cost'] for r in all_results
                     if r['result_no'] and r['result_yes']
                     and r['result_no']['success'] and r['result_yes']['success']]
        lo = min(min(costs_no), min(costs_yes)) * 0.9
        hi = max(max(costs_no), max(costs_yes)) * 1.1
        ax.scatter(costs_no, costs_yes, c='steelblue', edgecolor='k', s=60, zorder=5)
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y=x')
        ax.set_xlabel("Cost (no coarsen)")
        ax.set_ylabel("Cost (coarsen)")
        ax.set_title(f"Path Cost ({n_both_ok} scenes)")
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # (b) time scatter
        ax = axes[1]
        times_no = [r['result_no']['grow_ms'] + r['result_no']['bridge_ms'] + r['result_no']['gcs_ms']
                    for r in all_results
                    if r['result_no'] and r['result_yes']
                    and r['result_no']['success'] and r['result_yes']['success']]
        times_yes = [r['result_yes']['grow_ms'] + r['result_yes']['coarsen_ms'] +
                     r['result_yes']['bridge_ms'] + r['result_yes']['gcs_ms']
                     for r in all_results
                     if r['result_no'] and r['result_yes']
                     and r['result_no']['success'] and r['result_yes']['success']]
        lo = min(min(times_no), min(times_yes)) * 0.8
        hi = max(max(times_no), max(times_yes)) * 1.2
        ax.scatter(times_no, times_yes, c='coral', edgecolor='k', s=60, zorder=5)
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y=x')
        ax.set_xlabel("Time ms (no coarsen)")
        ax.set_ylabel("Time ms (coarsen)")
        ax.set_title(f"Total Time ({n_both_ok} scenes)")
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # (c) box count bar
        ax = axes[2]
        scene_ids = list(range(n_both_ok))
        box_no_list = [len(r['result_no']['boxes']) for r in all_results
                       if r['result_no'] and r['result_yes']
                       and r['result_no']['success'] and r['result_yes']['success']]
        box_yes_list = [len(r['result_yes']['boxes']) for r in all_results
                        if r['result_no'] and r['result_yes']
                        and r['result_no']['success'] and r['result_yes']['success']]
        w = 0.35
        ax.bar([x - w/2 for x in scene_ids], box_no_list, w,
               label='no coarsen', color='steelblue', alpha=0.7)
        ax.bar([x + w/2 for x in scene_ids], box_yes_list, w,
               label='coarsen', color='coral', alpha=0.7)
        ax.set_xlabel("Scene")
        ax.set_ylabel("Box count")
        ax.set_title("Box Count")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle("GCS 2-DOF Batch Comparison", fontsize=14, fontweight='bold')
        fig.tight_layout()
        p = out_dir / "batch_comparison.png"
        fig.savefig(p, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved: {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
