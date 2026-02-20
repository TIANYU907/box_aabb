"""Benchmark: Box-RRT vs common RRT libraries vs Marcucci-GCS.

本脚本对比以下方法（统一场景/起终点/随机种子）：
1) Box-RRT（项目实现）
2) OMPL RRT
3) OMPL RRTConnect
4) OMPL RRTstar
5) Marcucci-GCS（Drake GraphOfConvexSets，基于 BoxForest 图优化）

重要：本基准不做降级。
- 缺少 OMPL 或 Drake(pydrake) 时，脚本会立即报错退出。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

# ===== 严格外部依赖（禁止降级） =====
try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception as exc:  # pragma: no cover - runtime dependency gate
    raise ImportError(
        "缺少外部库 ompl。请先安装 OMPL Python 绑定后再运行本 benchmark。"
    ) from exc

try:
    from pydrake.geometry.optimization import GraphOfConvexSets  # noqa: F401
except Exception as exc:  # pragma: no cover - runtime dependency gate
    raise ImportError(
        "缺少外部库 pydrake（Drake）。请先安装 Drake 后再运行本 benchmark。"
    ) from exc

from aabb.robot import load_robot
from common.output import make_output_dir
from forest.collision import CollisionChecker
from forest.scene import Scene
from planner.box_planner import BoxPlanner
from planner.gcs_optimizer import GCSOptimizer
from planner.models import PlannerConfig


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    q_start: np.ndarray
    q_goal: np.ndarray
    obstacles: Sequence[Tuple[Sequence[float], Sequence[float], str]]


def _path_length(path: Sequence[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))))


def _make_scene(case: BenchmarkCase) -> Scene:
    scene = Scene()
    for min_pt, max_pt, name in case.obstacles:
        scene.add_obstacle(min_pt, max_pt, name)
    return scene


def _make_cases() -> List[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="narrow_passage_2dof",
            q_start=np.array([-1.20, -0.20], dtype=np.float64),
            q_goal=np.array([1.20, 0.25], dtype=np.float64),
            obstacles=[
                ([0.35, -1.60], [0.65, -0.35], "wall_low"),
                ([0.35, 0.35], [0.65, 1.60], "wall_high"),
                ([-0.15, -0.15], [0.15, 0.15], "center_block"),
            ],
        ),
        BenchmarkCase(
            name="multi_obstacle_2dof",
            q_start=np.array([-1.35, -0.60], dtype=np.float64),
            q_goal=np.array([1.10, 0.65], dtype=np.float64),
            obstacles=[
                ([-0.45, -0.40], [-0.10, 0.90], "left_pillar"),
                ([0.20, -0.95], [0.55, 0.25], "mid_pillar"),
                ([0.85, -0.25], [1.20, 1.00], "right_pillar"),
            ],
        ),
    ]


def _make_box_rrt_config() -> PlannerConfig:
    return PlannerConfig(
        max_iterations=320,
        max_box_nodes=180,
        goal_bias=0.15,
        segment_collision_resolution=0.04,
        path_shortcut_iters=80,
        use_gcs=False,
        verbose=False,
    )


def _validate_path(path: Sequence[np.ndarray], checker: CollisionChecker, resolution: float) -> bool:
    if len(path) < 2:
        return False
    for i in range(1, len(path)):
        if checker.check_segment_collision(path[i - 1], path[i], resolution):
            return False
    return True


def _run_box_rrt(robot_name: str, case: BenchmarkCase, seed: int) -> Dict:
    robot = load_robot(robot_name)
    scene = _make_scene(case)
    planner = BoxPlanner(robot, scene, _make_box_rrt_config())

    t0 = time.perf_counter()
    result = planner.plan(case.q_start, case.q_goal, seed=seed)
    dt = time.perf_counter() - t0

    return {
        "success": bool(result.success),
        "time": float(dt),
        "path_length": float(result.path_length) if result.success else math.nan,
        "message": result.message,
    }


def _run_marcucci_gcs(robot_name: str, case: BenchmarkCase, seed: int) -> Dict:
    robot = load_robot(robot_name)
    scene = _make_scene(case)
    planner = BoxPlanner(robot, scene, _make_box_rrt_config())

    t0 = time.perf_counter()
    result = planner.plan(case.q_start, case.q_goal, seed=seed)
    if not result.success or result.forest is None:
        dt = time.perf_counter() - t0
        return {
            "success": False,
            "time": float(dt),
            "path_length": math.nan,
            "message": f"前置 BoxForest 构建失败: {result.message}",
        }

    forest = result.forest
    colliding = forest.validate_boxes(planner.collision_checker)
    valid_boxes = {bid: b for bid, b in forest.boxes.items() if bid not in colliding}
    valid_adj = {
        bid: (neighbors - colliding)
        for bid, neighbors in forest.adjacency.items()
        if bid not in colliding
    }

    adj_edges = planner.connector.build_adjacency_edges(valid_boxes, valid_adj)
    endpoint_edges, start_box_id, goal_box_id = planner.connector.connect_endpoints_to_forest(
        case.q_start,
        case.q_goal,
        valid_boxes,
    )
    if start_box_id is None or goal_box_id is None:
        dt = time.perf_counter() - t0
        return {
            "success": False,
            "time": float(dt),
            "path_length": math.nan,
            "message": "GCS 前置图构建失败：起终点无法连接到 forest",
        }

    graph = planner.connector.build_forest_graph(
        adj_edges,
        endpoint_edges,
        case.q_start,
        case.q_goal,
        start_box_id,
        goal_box_id,
        valid_boxes,
    )

    # 禁止 fallback：若 Drake 求解失败则直接失败
    gcs = GCSOptimizer(fallback=False, bezier_degree=3)
    path = gcs.optimize(graph, valid_boxes, case.q_start, case.q_goal)
    dt = time.perf_counter() - t0

    if path is None:
        return {
            "success": False,
            "time": float(dt),
            "path_length": math.nan,
            "message": "Drake GCS 求解失败（fallback 已禁用）",
        }

    if not _validate_path(path, planner.collision_checker, resolution=0.04):
        return {
            "success": False,
            "time": float(dt),
            "path_length": math.nan,
            "message": "Drake GCS 结果路径碰撞校验失败",
        }

    return {
        "success": True,
        "time": float(dt),
        "path_length": float(_path_length(path)),
        "message": "ok",
    }


def _run_ompl(
    robot_name: str,
    case: BenchmarkCase,
    seed: int,
    planner_ctor: Callable[[ob.SpaceInformation], ob.Planner],
    timeout_s: float,
) -> Dict:
    robot = load_robot(robot_name)
    scene = _make_scene(case)
    checker = CollisionChecker(robot=robot, scene=scene)

    n = robot.n_joints
    if not robot.joint_limits:
        return {
            "success": False,
            "time": 0.0,
            "path_length": math.nan,
            "message": "机器人缺少关节限制，无法创建 OMPL 边界",
        }

    space = ob.RealVectorStateSpace(n)
    bounds = ob.RealVectorBounds(n)
    for i, (lo, hi) in enumerate(robot.joint_limits):
        bounds.setLow(i, float(lo))
        bounds.setHigh(i, float(hi))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    def is_valid(state: ob.State) -> bool:
        q = np.array([float(state[i]) for i in range(n)], dtype=np.float64)
        return not checker.check_config_collision(q)

    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    si.setStateValidityCheckingResolution(0.01)

    pdef = ob.ProblemDefinition(si)
    start = ob.State(space)
    goal = ob.State(space)
    for i in range(n):
        start[i] = float(case.q_start[i])
        goal[i] = float(case.q_goal[i])
    pdef.setStartAndGoalStates(start, goal)

    planner = planner_ctor(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    ob.RNG.setSeed(int(seed))
    t0 = time.perf_counter()
    solved = planner.solve(float(timeout_s))
    dt = time.perf_counter() - t0

    if not solved:
        return {
            "success": False,
            "time": float(dt),
            "path_length": math.nan,
            "message": "OMPL 未在时限内求解",
        }

    sol = pdef.getSolutionPath()
    if sol is None:
        return {
            "success": False,
            "time": float(dt),
            "path_length": math.nan,
            "message": "OMPL 返回空路径",
        }

    path = []
    for i in range(sol.getStateCount()):
        st = sol.getState(i)
        path.append(np.array([float(st[d]) for d in range(n)], dtype=np.float64))

    if not _validate_path(path, checker, resolution=0.04):
        return {
            "success": False,
            "time": float(dt),
            "path_length": math.nan,
            "message": "OMPL 路径碰撞校验失败",
        }

    return {
        "success": True,
        "time": float(dt),
        "path_length": float(_path_length(path)),
        "message": "ok",
    }


def _summarize(records: List[Dict]) -> Dict:
    success_records = [r for r in records if r["success"]]
    times = [float(r["time"]) for r in records]
    succ_times = [float(r["time"]) for r in success_records]
    succ_lens = [float(r["path_length"]) for r in success_records if not math.isnan(r["path_length"])]

    return {
        "runs": len(records),
        "successes": len(success_records),
        "success_rate": (len(success_records) / len(records)) if records else 0.0,
        "time_mean_all": mean(times) if times else math.nan,
        "time_p50_all": median(times) if times else math.nan,
        "time_mean_success": mean(succ_times) if succ_times else math.nan,
        "path_len_mean_success": mean(succ_lens) if succ_lens else math.nan,
    }


def _to_markdown(summary: Dict) -> str:
    lines = [
        "# Planner Benchmark: Box-RRT vs OMPL-RRTs vs Marcucci-GCS",
        "",
        "说明：Marcucci 对比项为 Drake GraphOfConvexSets（fallback 禁用）。",
        "",
    ]

    for case_name, methods in summary.items():
        lines.append(f"## Case: {case_name}")
        lines.append("")
        lines.append("| method | runs | success | success_rate | mean_time(s) | mean_path_len |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method_name, stats in methods.items():
            lines.append(
                f"| {method_name} | {stats['runs']} | {stats['successes']} | "
                f"{stats['success_rate']:.3f} | {stats['time_mean_all']:.4f} | "
                f"{stats['path_len_mean_success']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Box-RRT/OMPL/Marcucci-GCS")
    parser.add_argument("--robot", default="2dof_planar", help="robot config name")
    parser.add_argument("--trials", type=int, default=8, help="trials per case/method")
    parser.add_argument("--ompl-timeout", type=float, default=1.5, help="OMPL solve timeout (s)")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    args = parser.parse_args()

    methods: Dict[str, Callable[[str, BenchmarkCase, int], Dict]] = {
        "box_rrt": _run_box_rrt,
        "ompl_rrt": lambda robot, case, seed: _run_ompl(robot, case, seed, og.RRT, args.ompl_timeout),
        "ompl_rrtconnect": lambda robot, case, seed: _run_ompl(robot, case, seed, og.RRTConnect, args.ompl_timeout),
        "ompl_rrtstar": lambda robot, case, seed: _run_ompl(robot, case, seed, og.RRTstar, args.ompl_timeout),
        "marcucci_gcs_drake": _run_marcucci_gcs,
    }

    cases = _make_cases()
    raw: Dict[str, Dict[str, List[Dict]]] = {}
    summary: Dict[str, Dict[str, Dict]] = {}

    for case in cases:
        raw[case.name] = {}
        summary[case.name] = {}
        for method_name, fn in methods.items():
            records: List[Dict] = []
            for i in range(args.trials):
                trial_seed = int(args.seed + i)
                records.append(fn(args.robot, case, trial_seed))
            raw[case.name][method_name] = records
            summary[case.name][method_name] = _summarize(records)

    out_dir = make_output_dir("benchmarks", "planner_rrt_vs_marcucci")
    raw_path = out_dir / "raw_results.json"
    summary_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"

    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_to_markdown(summary), encoding="utf-8")

    print(f"output_dir={out_dir}")
    print(f"raw={raw_path}")
    print(f"summary={summary_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
