"""
compare_all_planners.py — 统一对比: Box-RRT Pipeline vs OMPL

在同一场景 (Panda 7-DOF, 同 seed) 下运行:
  A) Box-RRT Pipeline: grow forest → coarsen → bridge → Dijkstra+SOCP
  B) OMPL (C++ via WSL): RRT, RRTConnect, RRT*, InformedRRT*, BIT*

所有方法共享:
  - 同一 Panda 7-DOF 机器人
  - 同一障碍物场景 (seed 完全复现)
  - 同一碰撞检测后端 (CollisionChecker)

用法:
    python -m v2.examples.compare_all_planners
    python -m v2.examples.compare_all_planners --seed 42 --timeout 2
    python -m v2.examples.compare_all_planners --trials 5 --obstacles 15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows: force UTF-8 stdout to avoid GBK encoding errors for CJK / symbols
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

import os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
from _bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from common.output import make_output_dir

# ── Box-RRT Pipeline ──
from v2.examples.panda_planner import (
    PandaGCSConfig,
    build_panda_scene,
    make_planner_config,
    grow_and_prepare,
    run_method_with_bridge,
    _solve_method_dijkstra,
    _solve_method_gcs,
    plot_arm_scene_html,
    plot_arm_poses_html,
    create_animation_html,
    plot_joint_trajectory,
    _save_plotly_html,
    generate_report,
)
from planner.box_planner import BoxPlanner




# ═══════════════════════════════════════════════════════════════════════════
# Box-RRT Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_boxrrt_pipeline(robot, scene, cfg, q_start, q_goal, ndim,
                        parallel_grow: Optional[bool] = None):
    """运行 Box-RRT 完整 pipeline, 返回各方法结果.

    Pipeline:
      1. grow forest (shared)
      2. coarsen (shared)
      3. Dijkstra + SOCP refine

    Args:
        parallel_grow: 覆盖 cfg.parallel_grow (None=使用 cfg 默认值).

    Returns:
        dict with keys:
          - 'prep': shared forest info (grow_ms, coarsen_ms, etc.)
          - 'dijkstra': result dict or None
    """
    t0 = time.perf_counter()

    # ── 临时覆盖 parallel_grow ──
    orig_parallel = cfg.parallel_grow
    if parallel_grow is not None:
        cfg.parallel_grow = parallel_grow
    label = "parallel" if cfg.parallel_grow else "sequential"

    # ── grow + coarsen ──
    print("\n" + "=" * 60)
    print(f"  Box-RRT Pipeline: grow + coarsen ({label})")
    print("=" * 60)
    prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)

    # 等待 cache 线程完成
    cache_thread = prep.get('_cache_thread')
    if cache_thread is not None:
        cache_thread.join()
        cr = prep.get('_cache_result', {})
        prep['cache_ms'] = cr.get('ms', 0.0)

    grow_ms = prep['grow_ms']
    coarsen_ms = prep['coarsen_ms']

    # ── Dijkstra + SOCP ──
    print(f"\n  [Dijkstra + SOCP]")
    t_dij = time.perf_counter()
    result_dij = run_method_with_bridge(
        _solve_method_dijkstra, "Dijkstra", prep, cfg,
        q_start, q_goal, ndim)
    dij_total_ms = (time.perf_counter() - t_dij) * 1000

    if result_dij is None:
        result_dij = dict(success=False, cost=float('inf'), waypoints=[],
                          plan_ms=dij_total_ms, method='Dijkstra')

    if result_dij['success']:
        plan_ms = result_dij['plan_ms']
        bridge_ms = result_dij.get('bridge_ms', 0)
        adj_ms = result_dij.get('adj_ms', 0)
        # total = grow + coarsen + adj + bridge + plan
        result_dij['total_ms'] = grow_ms + coarsen_ms + adj_ms + bridge_ms + plan_ms
        print(f"    [OK] cost={result_dij['cost']:.4f}, "
              f"{len(result_dij['waypoints'])} wp, plan={plan_ms:.0f}ms, "
              f"total={result_dij['total_ms']:.0f}ms")
    else:
        result_dij['total_ms'] = grow_ms + coarsen_ms + dij_total_ms
        print(f"    [FAIL] ({dij_total_ms:.0f}ms)")

    # ── 恢复 cfg ──
    cfg.parallel_grow = orig_parallel

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        'prep': prep,
        'dijkstra': result_dij,
        'total_ms': total_ms,
    }


# ═══════════════════════════════════════════════════════════════════════════
# OMPL Family Runner (via WSL bridge)
# ═══════════════════════════════════════════════════════════════════════════

OMPL_ALGORITHMS = ["RRT", "RRTConnect", "RRTstar", "InformedRRTstar", "BITstar"]

def run_ompl_family(scene, q_start, q_goal,
                    algorithms: List[str] = None,
                    timeout: float = 1.0,
                    n_trials: int = 3,
                    seed: int = 42,
                    step_size: float = 0.5):
    """Run OMPL planners via WSL bridge subprocess.

    The bridge script (ompl_bridge.py) runs in WSL Ubuntu and uses the
    project's actual Robot + CollisionChecker for identical collision models.
    """
    import subprocess

    if algorithms is None:
        algorithms = OMPL_ALGORITHMS

    print(f"\n{'=' * 60}")
    print(f"  OMPL Family (via WSL): {algorithms}")
    print(f"  timeout={timeout:.1f}s, trials={n_trials}, seed={seed}")
    print(f"{'=' * 60}")

    # Serialize scene obstacles
    obs_list = []
    for obs in scene.get_obstacles():
        obs_list.append({
            "min_point": obs.min_point.tolist(),
            "max_point": obs.max_point.tolist(),
            "name": obs.name,
        })

    problem = {
        "q_start": q_start.tolist(),
        "q_goal": q_goal.tolist(),
        "obstacles": obs_list,
        "algorithms": algorithms,
        "timeout": timeout,
        "trials": n_trials,
        "seed": seed,
        "step_size": step_size,
    }
    json_input = json.dumps(problem)

    bridge_path = "/mnt/c/Users/TIAN/Documents/box_aabb/v2/examples/ompl_bridge.py"

    print("  Launching WSL bridge ...", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             f"python3 {bridge_path} 2>/dev/null"],
            input=json_input,
            capture_output=True,
            text=True,
            timeout=timeout * n_trials * len(algorithms) + 60,
        )
    except subprocess.TimeoutExpired:
        print("  [ERROR] WSL bridge timed out!")
        return {}

    wall_time = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  [ERROR] Bridge exit code {proc.returncode}")
        if proc.stderr:
            # Show last 5 lines of stderr
            lines = proc.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}")
        return {}

    # Parse JSON output
    try:
        raw_results = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parse: {e}")
        print(f"  stdout[:500]: {proc.stdout[:500]}")
        return {}

    # Reformat to match our rrt_results structure
    ompl_results = {}
    for algo, data in raw_results.items():
        if "error" in data:
            print(f"  [{algo}] ERROR: {data['error']}")
            continue

        s = data.get("summary", {})
        n_succ = s.get("n_success", 0)
        ompl_name = f"OMPL-{algo}"

        # Print summary line
        if n_succ > 0:
            fst_time = s.get('avg_first_solution_time', s['avg_plan_time_s'])
            fst_cost = s.get('avg_first_solution_cost', s['avg_path_length'])
            print(f"  [OMPL-{algo}]  "
                  f"{n_succ}/{s['n_trials']} ok  "
                  f"1st={fst_time:.3f}s (len={fst_cost:.3f})  "
                  f"plan={s['avg_plan_time_s']:.3f}s  "
                  f"len={s['avg_path_length']:.3f}  "
                  f"checks={s['avg_collision_checks']:.0f}")
        else:
            print(f"  [OMPL-{algo}]  0/{s.get('n_trials', n_trials)} ok  "
                  f"(all failed)")

        # Fallback: if bridge didn't provide avg_first_solution_time
        if "avg_first_solution_time" not in s:
            s["avg_first_solution_time"] = s.get("avg_plan_time_s", float("nan"))
        if "avg_n_nodes" not in s:
            s["avg_n_nodes"] = 0

        ompl_results[ompl_name] = {
            "summary": s,
            "trials": data.get("trials", []),
            "best_waypoints": data.get("best_waypoints", []),
        }

    print(f"  OMPL total wall time: {wall_time:.1f}s")
    return ompl_results


# ═══════════════════════════════════════════════════════════════════════════
# Unified Comparison Table
# ═══════════════════════════════════════════════════════════════════════════

def _boxrrt_row(boxrrt_results: Dict, method_label: str) -> Optional[Dict]:
    """从 boxrrt_results 提取一行对比数据."""
    dij = boxrrt_results.get('dijkstra')
    if not dij:
        return None
    ok = dij.get('success', False)
    total_s = dij['total_ms'] / 1000
    plan_s = dij['plan_ms'] / 1000
    n_boxes = len(dij.get('boxes', {}))
    if ok:
        cost = dij['cost']
        return {
            'method': method_label,
            'success': '1/1',
            'first_sol': f"{total_s:.3f}",
            'first_len': f"{cost:.3f}",
            'plan_s': f"{plan_s:.3f}",
            'total_s': f"{total_s:.3f}",
            'path_len': f"{cost:.3f}",
            'nodes': str(n_boxes),
        }
    else:
        return {
            'method': method_label,
            'success': '0/1',
            'first_sol': '—', 'first_len': '—',
            'plan_s': '—',
            'total_s': f"{total_s:.3f}",
            'path_len': '—',
            'nodes': str(n_boxes),
        }


def print_comparison(boxrrt_cache: Dict,
                     cfg, seed: int, n_obs: int, rrt_timeout: float,
                     n_trials: int, config_dist: float,
                     ompl_results: Optional[Dict] = None,
                     boxrrt_parallel: Optional[Dict] = None):
    """输出统一对比表."""

    print(f"\n{'=' * 100}")
    print(f"  UNIFIED COMPARISON — Panda 7-DOF")
    print(f"  seed={seed}, {n_obs} obstacles, config_dist={config_dist:.3f} rad")
    print(f"{'=' * 100}")

    # Header
    header = (f"  {'Method':<24s} {'Success':>8s} {'1st Sol(s)':>10s} {'1stLen':>10s} "
              f"{'Plan(s)':>10s} {'Total(s)':>10s} "
              f"{'PathLen':>10s} {'Nodes':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []

    # ── Box-RRT (sequential) ──
    row = _boxrrt_row(boxrrt_cache, 'BoxRRT(seq)')
    if row:
        rows.append(row)

    # ── Box-RRT (parallel) ──
    if boxrrt_parallel:
        row = _boxrrt_row(boxrrt_parallel, 'BoxRRT(par)')
        if row:
            rows.append(row)

    # ── OMPL Family ──
    if ompl_results:
        for algo, data in ompl_results.items():
            s = data['summary']
            rate = f"{s['n_success']}/{s['n_trials']}"
            if s['n_success'] > 0:
                fst = s.get('avg_first_solution_time', s.get('avg_plan_time_s', float('nan')))
                fst_s = f"{fst:.3f}" if not math.isnan(fst) else "—"
                fst_cost = s.get('avg_first_solution_cost', s.get('avg_path_length', float('nan')))
                fst_cost_s = f"{fst_cost:.3f}" if not math.isnan(fst_cost) else "—"
                rows.append({
                    'method': algo,
                    'success': rate,
                    'first_sol': fst_s,
                    'first_len': fst_cost_s,
                    'plan_s': f"{s['avg_plan_time_s']:.3f}",
                    'total_s': f"{s['avg_plan_time_s']:.3f}",
                    'path_len': f"{s['avg_path_length']:.3f}",
                    'nodes': f"{s.get('avg_n_nodes', 0):.0f}",
                })
            else:
                rows.append({
                    'method': algo,
                    'success': rate,
                    'first_sol': '—', 'first_len': '—',
                    'plan_s': '—',
                    'total_s': f"{rrt_timeout:.3f}",
                    'path_len': '—',
                    'nodes': '—',
                })

    for r in rows:
        print(f"  {r['method']:<24s} {r['success']:>8s} {r['first_sol']:>10s} {r['first_len']:>10s} "
              f"{r['plan_s']:>10s} {r['total_s']:>10s} "
              f"{r['path_len']:>10s} {r['nodes']:>8s}")
    print()

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Save Results
# ═══════════════════════════════════════════════════════════════════════════

def save_comparison(rows, boxrrt_results,
                    cfg, seed, config_dist, rrt_timeout, n_trials,
                    n_obs, out_dir: Path, total_s: float,
                    ompl_results: Optional[Dict] = None):
    """保存对比结果为 JSON + Markdown."""
    # ── JSON ──
    json_data = {
        "seed": seed,
        "config_dist": config_dist,
        "n_obstacles": n_obs,
        "rrt_timeout_s": rrt_timeout,
        "rrt_trials": n_trials,
        "box_rrt_config": {
            "max_boxes": cfg.max_boxes,
            "max_consecutive_miss": cfg.max_consecutive_miss,
            "min_box_size": cfg.min_box_size,
        },
        "comparison_table": rows,
        "total_time_s": total_s,
    }
    if ompl_results:
        json_data["ompl_details"] = {
            algo: {k: v for k, v in data.items() if k != "best_waypoints"}
            for algo, data in ompl_results.items()
        }
    json_path = out_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # ── Markdown ──
    md_lines = [
        f"# Unified Planner Comparison — Panda 7-DOF",
        f"",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Seed**: {seed}",
        f"- **Obstacles**: {n_obs}",
        f"- **Config distance**: {config_dist:.3f} rad",
        f"- **RRT timeout**: {rrt_timeout}s per trial",
        f"- **RRT trials**: {n_trials}",
        f"- **Box-RRT max_boxes**: {cfg.max_boxes}",
        f"",
        f"## Comparison Table",
        f"",
        f"| Method | Success | 1st Sol(s) | 1st Len | Plan(s) | Total(s) "
        f"| Path Len | Nodes |",
        f"|--------|---------|-----------|---------|---------|--------- "
        f"|----------|-------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['method']} | {r['success']} | {r['first_sol']} | {r['first_len']} | "
            f"{r['plan_s']} | {r['total_s']} | "
            f"{r['path_len']} | {r['nodes']} |"
        )

    # ── Box-RRT pipeline breakdown ──
    prep = boxrrt_results.get('prep', {})
    dij = boxrrt_results.get('dijkstra', {})
    md_lines.extend([
        f"",
        f"## Box-RRT Pipeline Breakdown",
        f"",
        f"| Stage | Time (ms) |",
        f"|-------|----------|",
        f"| Forest grow | {prep.get('grow_ms', 0):.0f} |",
        f"| Coarsen | {prep.get('coarsen_ms', 0):.0f} |",
        f"| Cache save (parallel) | {prep.get('cache_ms', 0):.0f} |",
    ])
    if dij:
        md_lines.append(f"| Dijkstra: adjacency | {dij.get('adj_ms', 0):.0f} |")
        md_lines.append(f"| Dijkstra: bridge | {dij.get('bridge_ms', 0):.0f} |")
        md_lines.append(f"| Dijkstra: plan | {dij.get('plan_ms', 0):.0f} |")

    # ── OMPL per-trial details ──
    if ompl_results:
        md_lines.extend([f"", f"## OMPL Per-Trial Details", f""])
        for algo, data in ompl_results.items():
            trials = data.get('trials', [])
            md_lines.append(f"### {algo}")
            md_lines.append(f"")
            md_lines.append(
                f"| Trial | OK | Plan(s) | Length | Raw Len | Checks |")
            md_lines.append(
                f"|-------|----|---------|--------|---------|--------|")
            for i, t in enumerate(trials):
                ok = "Y" if t.get("success") else "N"
                if t.get("success"):
                    md_lines.append(
                        f"| {i} | {ok} | {t['plan_time_s']:.4f} | "
                        f"{t['path_length']:.3f} | {t['raw_path_length']:.3f} | "
                        f"{t['n_collision_checks']} |")
                else:
                    md_lines.append(
                        f"| {i} | {ok} | {t.get('plan_time_s', 0):.4f} | "
                        f"— | — | {t.get('n_collision_checks', 0)} |")
            md_lines.append(f"")

    md_lines.append(f"---")
    md_lines.append(f"Total experiment time: {total_s:.1f}s")

    md_path = out_dir / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI + Main
# ═══════════════════════════════════════════════════════════════════════════

def run_one_scene(robot, cfg, args, seed, q_start, q_goal, ndim,
                  config_dist, out_dir):
    """运行单个场景的完整对比, 返回 rows list."""
    cfg.seed = seed
    rng = np.random.default_rng(seed)

    print(f"\n{'#' * 70}")
    print(f"  Scene: seed={seed}, {args.obstacles} obstacles")
    print(f"{'#' * 70}")

    # ── Build scene ──
    print("Building scene ...", flush=True)
    scene = build_panda_scene(rng, cfg, robot, q_start, q_goal)
    n_obs = scene.n_obstacles
    print(f"  {n_obs} obstacles")
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        sz = mx - mn
        print(f"    {obs.name}: center=({(mn[0]+mx[0])/2:.3f}, "
              f"{(mn[1]+mx[1])/2:.3f}, {(mn[2]+mx[2])/2:.3f})  "
              f"size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")

    # ══════════════════════════════════════════════════════════════════
    # Part A: Box-RRT Pipeline (sequential + parallel)
    # ══════════════════════════════════════════════════════════════════
    boxrrt_cache = run_boxrrt_pipeline(
        robot, scene, cfg, q_start, q_goal, ndim, parallel_grow=False)
    boxrrt_parallel = run_boxrrt_pipeline(
        robot, scene, cfg, q_start, q_goal, ndim, parallel_grow=True)

    # ══════════════════════════════════════════════════════════════════
    # Part B: OMPL Family (via WSL)
    # ══════════════════════════════════════════════════════════════════
    ompl_results = run_ompl_family(
        scene, q_start, q_goal,
        algorithms=args.ompl_algorithms,
        timeout=args.timeout,
        n_trials=args.trials,
        seed=seed,
        step_size=args.step_size,
    )

    # ══════════════════════════════════════════════════════════════════
    # Per-scene Table
    # ══════════════════════════════════════════════════════════════════
    rows = print_comparison(
        boxrrt_cache,
        cfg, seed, n_obs, args.timeout, args.trials, config_dist,
        ompl_results=ompl_results if ompl_results else None,
        boxrrt_parallel=boxrrt_parallel)

    # ── Save per-scene results ──
    scene_dir = out_dir / f"seed_{seed}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(rows, boxrrt_cache,
                    cfg, seed, config_dist, args.timeout, args.trials,
                    n_obs, scene_dir, 0.0,
                    ompl_results=ompl_results if ompl_results else None)

    # ── Viz ──
    best_boxrrt = None
    for br in (boxrrt_cache, boxrrt_parallel):
        for key in ('dijkstra',):
            r = br.get(key)
            if r and r.get('success'):
                if best_boxrrt is None or r['cost'] < best_boxrrt['cost']:
                    best_boxrrt = r

    best_rrt_name = None
    best_rrt_wps = None
    best_rrt_cost = float('inf')
    if ompl_results:
        for algo, data in ompl_results.items():
            s = data.get('summary', {})
            if s.get('n_success', 0) > 0:
                cost = s['avg_path_length']
                if cost < best_rrt_cost:
                    best_rrt_cost = cost
                    best_rrt_name = algo
                    best_rrt_wps = data.get('best_waypoints', [])

    if best_boxrrt and best_boxrrt['success']:
        wps = best_boxrrt['waypoints']
        method = best_boxrrt['method']
        fig = plot_arm_scene_html(
            robot, scene, q_start, q_goal, waypoints=wps,
            title=f"seed={seed} Box-RRT {method} — cost={best_boxrrt['cost']:.3f}")
        _save_plotly_html(fig, scene_dir / "boxrrt_arm_scene.html")

    if best_rrt_wps and len(best_rrt_wps) > 1:
        fig = plot_arm_scene_html(
            robot, scene, q_start, q_goal, waypoints=best_rrt_wps,
            title=f"seed={seed} {best_rrt_name} — cost={best_rrt_cost:.3f}")
        _save_plotly_html(fig, scene_dir / "ompl_arm_scene.html")

    return rows


def print_aggregate_summary(all_scene_rows: List[Tuple[int, List[Dict]]],
                            method_names: List[str]):
    """输出所有场景的汇总统计 (成功率 / 平均耗时 / 平均路径长度)."""
    from collections import defaultdict

    # 收集每个方法在各场景的数据
    method_stats = defaultdict(lambda: {
        'successes': 0, 'total_scenes': 0,
        'total_times': [], 'path_lens': [], 'first_sols': [],
    })

    for seed, rows in all_scene_rows:
        seen = set()
        for r in rows:
            m = r['method']
            seen.add(m)
            stats = method_stats[m]
            stats['total_scenes'] += 1
            suc_parts = r['success'].split('/')
            n_ok = int(suc_parts[0])
            stats['successes'] += (1 if n_ok > 0 else 0)
            if r['total_s'] != '—':
                try:
                    stats['total_times'].append(float(r['total_s']))
                except ValueError:
                    pass
            if r['path_len'] != '—':
                try:
                    stats['path_lens'].append(float(r['path_len']))
                except ValueError:
                    pass
            if r['first_sol'] != '—':
                try:
                    stats['first_sols'].append(float(r['first_sol']))
                except ValueError:
                    pass

    n_scenes = len(all_scene_rows)

    print(f"\n{'=' * 110}")
    print(f"  AGGREGATE SUMMARY — {n_scenes} scenes")
    print(f"{'=' * 110}")
    header = (f"  {'Method':<24s} {'Win/Total':>10s} {'Avg 1stSol(s)':>14s} "
              f"{'Avg Total(s)':>14s} {'Avg PathLen':>12s} {'Best PathLen':>12s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    # 使用 method_names 保持顺序
    ordered = []
    for m in method_names:
        if m in method_stats:
            ordered.append(m)
    for m in method_stats:
        if m not in ordered:
            ordered.append(m)

    for m in ordered:
        s = method_stats[m]
        n_total = s['total_scenes']
        n_ok = s['successes']
        rate = f"{n_ok}/{n_total}"

        avg_fst = f"{np.mean(s['first_sols']):.3f}" if s['first_sols'] else "—"
        avg_total = f"{np.mean(s['total_times']):.3f}" if s['total_times'] else "—"
        avg_plen = f"{np.mean(s['path_lens']):.3f}" if s['path_lens'] else "—"
        best_plen = f"{min(s['path_lens']):.3f}" if s['path_lens'] else "—"

        print(f"  {m:<24s} {rate:>10s} {avg_fst:>14s} "
              f"{avg_total:>14s} {avg_plen:>12s} {best_plen:>12s}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Unified planner comparison: Box-RRT pipeline vs OMPL (C++)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for single scene (0 = timestamp)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for multi-scene comparison")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="OMPL timeout per trial (seconds)")
    parser.add_argument("--trials", type=int, default=3,
                        help="OMPL trials per algorithm")
    parser.add_argument("--ompl-algorithms", nargs="+",
                        default=OMPL_ALGORITHMS,
                        help="OMPL algorithms to test")
    parser.add_argument("--obstacles", type=int, default=12,
                        help="Number of obstacles")
    parser.add_argument("--max-boxes", type=int, default=500,
                        help="Box-RRT max boxes")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="RRT step size (rad)")
    args = parser.parse_args()

    t_total = time.perf_counter()

    # 确定 seeds 列表
    if args.seeds:
        seeds = args.seeds
    elif args.seed != 0:
        seeds = [args.seed]
    else:
        seeds = [int(time.time()) % (2**31)]

    # ── Robot ──
    robot = load_robot("panda")
    ndim = robot.n_joints

    cfg = PandaGCSConfig()
    cfg.n_obstacles = args.obstacles
    cfg.max_boxes = args.max_boxes

    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    config_dist = float(np.linalg.norm(q_goal - q_start))

    print(f"{'=' * 70}")
    print(f"  Unified Planner Comparison — Panda {ndim}-DOF")
    print(f"{'=' * 70}")
    print(f"  seeds         = {seeds}")
    print(f"  q_start       = {np.array2string(q_start, precision=3)}")
    print(f"  q_goal        = {np.array2string(q_goal, precision=3)}")
    print(f"  config dist   = {config_dist:.3f} rad")
    print(f"  obstacles     = {args.obstacles}")
    print(f"  Box-RRT boxes = {args.max_boxes}")
    print(f"  OMPL timeout  = {args.timeout:.1f}s × {args.trials} trials")
    print(f"  OMPL algos    = {args.ompl_algorithms}")
    print()

    out_dir = make_output_dir("benchmarks", "planner_comparison")

    # ── 逐场景运行 ──
    all_scene_rows: List[Tuple[int, List[Dict]]] = []
    method_names_ordered: List[str] = []

    for i, seed in enumerate(seeds):
        print(f"\n  >>> Scene {i+1}/{len(seeds)}: seed={seed}")
        rows = run_one_scene(
            robot, cfg, args, seed, q_start, q_goal, ndim,
            config_dist, out_dir)
        all_scene_rows.append((seed, rows))
        # 记录方法名顺序 (以第一个场景为准)
        if not method_names_ordered:
            method_names_ordered = [r['method'] for r in rows]

    # ══════════════════════════════════════════════════════════════════
    # Aggregate Summary (多场景汇总)
    # ══════════════════════════════════════════════════════════════════
    if len(seeds) > 1:
        print_aggregate_summary(all_scene_rows, method_names_ordered)

    total_s = time.perf_counter() - t_total

    # ── Save aggregate JSON ──
    agg_json = {
        "seeds": seeds,
        "n_scenes": len(seeds),
        "config_dist": config_dist,
        "n_obstacles": args.obstacles,
        "rrt_timeout_s": args.timeout,
        "rrt_trials": args.trials,
        "max_boxes": args.max_boxes,
        "per_scene": [
            {"seed": seed, "rows": rows}
            for seed, rows in all_scene_rows
        ],
        "total_time_s": total_s,
    }
    agg_path = out_dir / "aggregate.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {agg_path}")

    print(f"\n  Output: {out_dir}")
    print(f"  Total experiment time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
