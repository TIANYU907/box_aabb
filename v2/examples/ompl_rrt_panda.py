"""
ompl_rrt_panda.py — OMPL RRT 系列算法对比 (Panda 7-DOF, 同场景)

复用 panda_planner.py 完全相同的:
  - Panda 7-DOF 机器人
  - 起/终点配置
  - 随机障碍物场景 (通过 seed 复现)
  - 碰撞检测 (box_aabb CollisionChecker)

测试的 OMPL 算法:
  1. RRT
  2. RRTConnect
  3. RRTstar (限时)
  4. InformedRRTstar (限时)
  5. BITstar (限时)
  6. PRM + A* query

用法:
    python -m v2.examples.ompl_rrt_panda
    python -m v2.examples.ompl_rrt_panda --seed 12345 --timeout 10
    python -m v2.examples.ompl_rrt_panda --trials 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys, os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
from _bootstrap import add_v2_paths
add_v2_paths()

# ── 严格依赖: OMPL ──
try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception as exc:
    raise ImportError(
        "Missing OMPL Python bindings. Install ompl before running.\n"
        "  conda install -c conda-forge ompl   (recommended)\n"
        "  or: pip install ompl"
    ) from exc

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from common.output import make_output_dir

# 复用 panda_planner 中的场景构建
from v2.examples.panda_planner import (
    PandaGCSConfig,
    build_panda_scene,
)


# ═══════════════════════════════════════════════════════════════
# OMPL adapter
# ═══════════════════════════════════════════════════════════════

def _make_ompl_space(robot) -> ob.RealVectorStateSpace:
    """创建 OMPL 关节空间 (RealVectorStateSpace)."""
    ndim = robot.n_joints
    space = ob.RealVectorStateSpace(ndim)
    bounds = ob.RealVectorBounds(ndim)
    for i, (lo, hi) in enumerate(robot.joint_limits):
        bounds.setLow(i, float(lo))
        bounds.setHigh(i, float(hi))
    space.setBounds(bounds)
    return space


def _make_validity_checker(si: ob.SpaceInformation,
                           checker: CollisionChecker,
                           ndim: int) -> ob.StateValidityChecker:
    """用 box_aabb CollisionChecker 作为 OMPL 碰撞检测后端."""

    class BoxAABBValidityChecker(ob.StateValidityChecker):
        def __init__(self, si_):
            super().__init__(si_)
            self._checker = checker
            self._ndim = ndim
            self._n_checks = 0

        def isValid(self, state):
            self._n_checks += 1
            q = np.array([state[i] for i in range(self._ndim)], dtype=np.float64)
            return not self._checker.check_config_collision(q)

    return BoxAABBValidityChecker(si)


def _state_to_numpy(state, ndim: int) -> np.ndarray:
    return np.array([state[i] for i in range(ndim)], dtype=np.float64)


def _set_state(state, q: np.ndarray):
    for i in range(len(q)):
        state[i] = float(q[i])


def _path_to_waypoints(path: og.PathGeometric, ndim: int) -> List[np.ndarray]:
    wps = []
    for i in range(path.getStateCount()):
        s = path.getState(i)
        wps.append(_state_to_numpy(s, ndim))
    return wps


def _path_length(waypoints: Sequence[np.ndarray]) -> float:
    if len(waypoints) < 2:
        return 0.0
    return float(sum(
        np.linalg.norm(waypoints[i] - waypoints[i - 1])
        for i in range(1, len(waypoints))
    ))


# ═══════════════════════════════════════════════════════════════
# OMPL planner runner
# ═══════════════════════════════════════════════════════════════

ALGORITHMS = {
    "RRT":              lambda si: og.RRT(si),
    "RRTConnect":       lambda si: og.RRTConnect(si),
    "RRTstar":          lambda si: og.RRTstar(si),
    "InformedRRTstar":  lambda si: og.InformedRRTstar(si),
    "BITstar":          lambda si: og.BITstar(si),
    "PRM":              lambda si: og.PRM(si),
}


def run_ompl_planner(
    algo_name: str,
    robot,
    checker: CollisionChecker,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    timeout: float = 30.0,
    simplify: bool = True,
) -> Dict:
    """
    运行单个 OMPL planner, 返回结果 dict.

    Returns:
        dict with keys: algorithm, success, plan_time_s, simplify_time_s,
        total_time_s, path_length, n_waypoints, n_validity_checks, waypoints
    """
    ndim = robot.n_joints
    space = _make_ompl_space(robot)
    si = ob.SpaceInformation(space)

    vc = _make_validity_checker(si, checker, ndim)
    si.setStateValidityChecker(vc)
    si.setStateValidityCheckingResolution(0.01)  # 1% of space extent
    si.setup()

    # start / goal states
    start = ob.State(space)
    goal = ob.State(space)
    _set_state(start(), q_start)
    _set_state(goal(), q_goal)

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    # 创建 planner
    if algo_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algo_name}. "
                         f"Available: {list(ALGORITHMS.keys())}")
    planner = ALGORITHMS[algo_name](si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    # ── solve ──
    t0 = time.perf_counter()
    solved = planner.solve(timeout)
    plan_time = time.perf_counter() - t0

    result = {
        "algorithm": algo_name,
        "success": False,
        "plan_time_s": plan_time,
        "simplify_time_s": 0.0,
        "total_time_s": plan_time,
        "path_length": float("nan"),
        "n_waypoints": 0,
        "n_validity_checks": vc._n_checks,
        "waypoints": [],
    }

    if not solved:
        return result

    result["success"] = True

    # 路径简化
    simplify_time = 0.0
    path = pdef.getSolutionPath()
    raw_wps = _path_to_waypoints(path, ndim)
    raw_length = _path_length(raw_wps)
    result["raw_path_length"] = raw_length
    result["raw_n_waypoints"] = len(raw_wps)

    if simplify:
        ps = og.PathSimplifier(si)
        t_simp = time.perf_counter()
        ps.simplifyMax(path)
        simplify_time = time.perf_counter() - t_simp
        path.interpolate()

    wps = _path_to_waypoints(path, ndim)
    length = _path_length(wps)

    result["simplify_time_s"] = simplify_time
    result["total_time_s"] = plan_time + simplify_time
    result["path_length"] = length
    result["n_waypoints"] = len(wps)
    result["n_validity_checks"] = vc._n_checks
    result["waypoints"] = wps

    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark driver
# ═══════════════════════════════════════════════════════════════

def run_benchmark(
    seed: int,
    algorithms: List[str],
    timeout: float = 30.0,
    n_trials: int = 1,
    cfg: Optional[PandaGCSConfig] = None,
) -> Dict:
    """
    运行完整 benchmark: 同场景, 多算法, 多 trial.

    Returns:
        dict with scene info + per-algorithm results
    """
    if cfg is None:
        cfg = PandaGCSConfig()
    cfg.seed = seed

    robot = load_robot("panda")
    ndim = robot.n_joints
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # ── 构建场景 ──
    scene = build_panda_scene(rng, cfg, robot, q_start, q_goal)
    checker = CollisionChecker(robot=robot, scene=scene)

    dist = float(np.linalg.norm(q_goal - q_start))
    n_obs = scene.n_obstacles
    print(f"\n{'=' * 65}")
    print(f"  OMPL RRT Benchmark — Panda {ndim}-DOF")
    print(f"{'=' * 65}")
    print(f"  seed        = {seed}")
    print(f"  q_start     = {np.array2string(q_start, precision=3)}")
    print(f"  q_goal      = {np.array2string(q_goal, precision=3)}")
    print(f"  config dist = {dist:.3f} rad")
    print(f"  obstacles   = {n_obs}")
    print(f"  timeout     = {timeout:.1f}s per solver")
    print(f"  trials      = {n_trials}")
    print()

    # 场景信息
    obs_info = []
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        obs_info.append({
            "name": obs.name,
            "min": mn.tolist(),
            "max": mx.tolist(),
        })

    # ── 逐算法测试 ──
    all_results = {}
    for algo in algorithms:
        print(f"  [{algo}] ", end="", flush=True)
        trial_results = []

        for trial in range(n_trials):
            r = run_ompl_planner(
                algo_name=algo,
                robot=robot,
                checker=checker,
                q_start=q_start,
                q_goal=q_goal,
                timeout=timeout,
                simplify=True,
            )
            trial_results.append(r)

            status = "OK" if r["success"] else "FAIL"
            sym = "." if r["success"] else "x"
            print(sym, end="", flush=True)

        # 汇总
        successes = [t for t in trial_results if t["success"]]
        n_succ = len(successes)

        if n_succ > 0:
            avg_plan = sum(t["plan_time_s"] for t in successes) / n_succ
            avg_total = sum(t["total_time_s"] for t in successes) / n_succ
            avg_length = sum(t["path_length"] for t in successes) / n_succ
            avg_checks = sum(t["n_validity_checks"] for t in successes) / n_succ
            min_length = min(t["path_length"] for t in successes)
            best_time = min(t["plan_time_s"] for t in successes)
        else:
            avg_plan = avg_total = avg_length = avg_checks = float("nan")
            min_length = best_time = float("nan")

        summary = {
            "n_trials": n_trials,
            "n_success": n_succ,
            "success_rate": n_succ / n_trials,
            "avg_plan_time_s": avg_plan,
            "best_plan_time_s": best_time,
            "avg_total_time_s": avg_total,
            "avg_path_length": avg_length,
            "min_path_length": min_length,
            "avg_validity_checks": avg_checks,
        }
        all_results[algo] = {
            "summary": summary,
            "trials": [
                {k: v for k, v in t.items() if k != "waypoints"}
                for t in trial_results
            ],
        }

        # 打印行
        if n_succ > 0:
            print(f"  {n_succ}/{n_trials} ok  "
                  f"plan={avg_plan:.3f}s  total={avg_total:.3f}s  "
                  f"len={avg_length:.3f}  checks={avg_checks:.0f}")
        else:
            print(f"  {n_succ}/{n_trials} ok  (all failed)")

    return {
        "seed": seed,
        "ndim": ndim,
        "n_obstacles": n_obs,
        "timeout_s": timeout,
        "n_trials": n_trials,
        "config_dist": dist,
        "q_start": q_start.tolist(),
        "q_goal": q_goal.tolist(),
        "obstacles": obs_info,
        "results": all_results,
    }


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

def print_summary_table(bench: Dict):
    """打印对比汇总表."""
    results = bench["results"]
    algos = list(results.keys())

    print(f"\n{'=' * 80}")
    print(f"  Summary — seed={bench['seed']}, "
          f"{bench['n_obstacles']} obstacles, "
          f"timeout={bench['timeout_s']}s, "
          f"{bench['n_trials']} trials")
    print(f"{'=' * 80}")

    header = f"{'Algorithm':<20} {'Success':>8} {'Plan(s)':>10} {'Total(s)':>10} {'Length':>10} {'Checks':>10}"
    print(header)
    print("-" * len(header))

    for algo in algos:
        s = results[algo]["summary"]
        rate = f"{s['n_success']}/{s['n_trials']}"
        if s["n_success"] > 0:
            plan = f"{s['avg_plan_time_s']:.4f}"
            total = f"{s['avg_total_time_s']:.4f}"
            length = f"{s['avg_path_length']:.3f}"
            checks = f"{s['avg_validity_checks']:.0f}"
        else:
            plan = total = length = checks = "—"
        print(f"{algo:<20} {rate:>8} {plan:>10} {total:>10} {length:>10} {checks:>10}")

    print()


def save_results(bench: Dict, out_dir: Path):
    """保存 JSON 和 Markdown 报告."""
    # JSON
    json_path = out_dir / "ompl_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # Markdown
    md_path = out_dir / "ompl_benchmark.md"
    results = bench["results"]
    algos = list(results.keys())

    lines = [
        f"# OMPL RRT Benchmark — Panda 7-DOF",
        f"",
        f"- **Seed**: {bench['seed']}",
        f"- **Obstacles**: {bench['n_obstacles']}",
        f"- **Timeout**: {bench['timeout_s']}s",
        f"- **Trials**: {bench['n_trials']}",
        f"- **Config distance**: {bench['config_dist']:.3f} rad",
        f"",
        f"## Results",
        f"",
        f"| Algorithm | Success | Avg Plan (s) | Avg Total (s) | Avg Length | Best Length | Avg Checks |",
        f"|-----------|---------|-------------|---------------|------------|-------------|------------|",
    ]
    for algo in algos:
        s = results[algo]["summary"]
        rate = f"{s['n_success']}/{s['n_trials']}"
        if s["n_success"] > 0:
            lines.append(
                f"| {algo} | {rate} | "
                f"{s['avg_plan_time_s']:.4f} | "
                f"{s['avg_total_time_s']:.4f} | "
                f"{s['avg_path_length']:.3f} | "
                f"{s['min_path_length']:.3f} | "
                f"{s['avg_validity_checks']:.0f} |"
            )
        else:
            lines.append(f"| {algo} | {rate} | — | — | — | — | — |")

    lines.extend([
        "",
        "## Per-trial Details",
        "",
    ])
    for algo in algos:
        trials = results[algo]["trials"]
        lines.append(f"### {algo}")
        lines.append("")
        lines.append(f"| Trial | Success | Plan (s) | Total (s) | Length | Checks |")
        lines.append(f"|-------|---------|----------|-----------|--------|--------|")
        for i, t in enumerate(trials):
            ok = "Yes" if t["success"] else "No"
            if t["success"]:
                lines.append(
                    f"| {i} | {ok} | {t['plan_time_s']:.4f} | "
                    f"{t['total_time_s']:.4f} | {t['path_length']:.3f} | "
                    f"{t['n_validity_checks']} |"
                )
            else:
                lines.append(
                    f"| {i} | {ok} | {t['plan_time_s']:.4f} | — | — | "
                    f"{t['n_validity_checks']} |"
                )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {md_path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OMPL RRT benchmark on Panda 7-DOF (same scene as gcs_planner_panda)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (0 = timestamp)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Solver timeout per algorithm (seconds)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per algorithm")
    parser.add_argument("--algorithms", nargs="+",
                        default=list(ALGORITHMS.keys()),
                        help=f"Algorithms to test. Available: {list(ALGORITHMS.keys())}")
    parser.add_argument("--obstacles", type=int, default=6,
                        help="Number of obstacles")
    args = parser.parse_args()

    seed = args.seed if args.seed != 0 else int(time.time()) % (2**31)

    cfg = PandaGCSConfig()
    cfg.n_obstacles = args.obstacles

    t0 = time.perf_counter()
    bench = run_benchmark(
        seed=seed,
        algorithms=args.algorithms,
        timeout=args.timeout,
        n_trials=args.trials,
        cfg=cfg,
    )
    total_s = time.perf_counter() - t0

    print_summary_table(bench)

    out_dir = make_output_dir("benchmarks", "ompl_rrt_panda")
    save_results(bench, out_dir)

    print(f"\nTotal benchmark time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
