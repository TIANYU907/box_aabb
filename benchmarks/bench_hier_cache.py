#!/usr/bin/env python
"""
benchmarks/bench_hier_cache.py - HierAABBTree 缓存效果基准测试

在随机障碍物环境下，关闭渲染，重复运行 N 次（第 1 次无缓存，后续使用缓存），
统计各项指标的均值/标准差。

输出：
  - 每次运行的关键指标
  - 汇总统计表

用法：
    python -m benchmarks.bench_hier_cache
    python -m benchmarks.bench_hier_cache --runs 10 --seed 42 --n-obs 5
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from box_aabb.robot import Robot, load_robot
from planner.models import BoxNode, PlannerConfig
from planner.obstacles import Scene
from planner.collision import CollisionChecker
from planner.box_forest import BoxForest
from planner.hier_aabb_tree import HierAABBTree

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("bench_cache")


# ─────────────────────────────────────────────────────────
#  随机场景生成
# ─────────────────────────────────────────────────────────

def random_scene_2d(robot: Robot, n_obs: int, rng: np.random.Generator) -> Scene:
    reach = sum(p['a'] for p in robot.dh_params)
    if robot.tool_frame:
        reach += robot.tool_frame.get('a', 0)
    reach = reach or 2.0
    scene = Scene()
    for i in range(n_obs):
        r = rng.uniform(0.3 * reach, 0.9 * reach)
        theta = rng.uniform(-math.pi, math.pi)
        cx, cy = r * math.cos(theta), r * math.sin(theta)
        hw = rng.uniform(0.05 * reach, 0.15 * reach)
        hh = rng.uniform(0.05 * reach, 0.15 * reach)
        scene.add_obstacle(
            min_point=[cx - hw, cy - hh],
            max_point=[cx + hw, cy + hh],
            name=f"obs_{i}",
        )
    return scene


# ─────────────────────────────────────────────────────────
#  覆盖率计算（栅格扫描，无渲染）
# ─────────────────────────────────────────────────────────

def compute_coverage(
    robot: Robot,
    scene: Scene,
    forest: BoxForest,
    joint_limits: List[Tuple[float, float]],
    resolution: float = 0.03,
) -> dict:
    """计算 C-space 覆盖率和自由空间覆盖率，不做任何绘图"""
    checker = CollisionChecker(robot, scene)
    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]
    xs = np.arange(lo_x, hi_x, resolution)
    ys = np.arange(lo_y, hi_y, resolution)
    n_rows, n_cols = len(ys), len(xs)

    collision_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    for i, y in enumerate(ys):
        row = np.column_stack([xs, np.full(n_cols, y)])
        collision_map[i, :] = checker.check_config_collision_batch(row).astype(
            np.float32
        )

    n_free = int(np.sum(collision_map == 0))

    box_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for box in forest.boxes.values():
        lo0, hi0 = box.joint_intervals[0]
        lo1, hi1 = box.joint_intervals[1]
        j0 = max(int((lo0 - lo_x) / resolution), 0)
        j1 = min(int(np.ceil((hi0 - lo_x) / resolution)), n_cols)
        i0 = max(int((lo1 - lo_y) / resolution), 0)
        i1 = min(int(np.ceil((hi1 - lo_y) / resolution)), n_rows)
        box_mask[i0:i1, j0:j1] = True

    total_cells = collision_map.size
    coverage = int(np.sum(box_mask)) / total_cells * 100
    free_coverage = int(np.sum(box_mask & (collision_map == 0))) / max(n_free, 1) * 100
    free_ratio = n_free / total_cells * 100

    return {
        "coverage": coverage,
        "free_coverage": free_coverage,
        "free_ratio": free_ratio,
    }


# ─────────────────────────────────────────────────────────
#  纯算法拓展（无渲染）
# ─────────────────────────────────────────────────────────

def grow_forest_no_render(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    max_boxes: int = 300,
    max_seeds: int = 2000,
    boundary_batch: int = 6,
    max_tree_stalls: int = 15,
    farthest_k: int = 12,
    rng_seed: int = 42,
    max_depth: int = 40,
    min_edge_length: float = 0.05,
    early_stop_window: int = 30,
    early_stop_min_vol: float = 1e-4,
) -> dict:
    """HierAABBTree box 拓展，无任何渲染，返回统计字典"""

    rng = np.random.default_rng(rng_seed)
    checker = CollisionChecker(robot, scene)
    obstacles = scene.get_obstacles()

    t_load_start = time.time()
    hier_tree = HierAABBTree.auto_load(robot, joint_limits)
    t_load = time.time() - t_load_start

    ht_stats_init = hier_tree.get_stats()
    cache_loaded_nodes = ht_stats_init["n_nodes"]
    cache_loaded_fk = ht_stats_init["n_fk_calls"]

    config = PlannerConfig(hard_overlap_reject=True, verbose=False)
    forest = BoxForest(robot.fingerprint(), joint_limits, config)

    t0 = time.time()

    # ── timers ──
    t_collision_check = 0.0
    t_find_free_box = 0.0
    t_deoverlap = 0.0
    t_sampling = 0.0
    n_seed_collision = 0
    n_seed_inside = 0
    n_find_none = 0
    n_find_tiny = 0
    n_boundary_attempts = 0
    n_farthest_attempts = 0
    recent_vols: List[float] = []
    step = 0

    def _sample_boundary(box_intervals, n, rng_):
        seeds = []
        ndim = len(box_intervals)
        for _ in range(n):
            dim = rng_.integers(0, ndim)
            side = rng_.integers(0, 2)
            q = np.array([rng_.uniform(lo, hi) for lo, hi in box_intervals])
            q[dim] = box_intervals[dim][side]
            offset = 0.01 if side == 1 else -0.01
            q[dim] = np.clip(
                q[dim] + offset,
                joint_limits[dim][0],
                joint_limits[dim][1],
            )
            seeds.append(q)
        return seeds

    def _try_add_box(seed_q, source=""):
        nonlocal t_collision_check, t_find_free_box, t_deoverlap
        nonlocal n_seed_collision, n_seed_inside, n_find_none, n_find_tiny

        tc0 = time.time()
        in_collision = checker.check_config_collision(seed_q)
        t_collision_check += time.time() - tc0
        if in_collision:
            n_seed_collision += 1
            return None

        # 用树的占用状态检查（O(depth)，代替 forest.find_containing 的 O(N)）
        tc0 = time.time()
        if hier_tree.is_occupied(seed_q):
            t_collision_check += time.time() - tc0
            n_seed_inside += 1
            return None
        t_collision_check += time.time() - tc0

        tf0 = time.time()
        ivs = hier_tree.find_free_box(
            seed_q, obstacles, max_depth=max_depth,
            min_edge_length=min_edge_length,
            mark_occupied=True,
        )
        t_find_free_box += time.time() - tf0
        if ivs is None:
            n_find_none += 1
            return None
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        if vol < 1e-6:
            n_find_tiny += 1
            return None

        nid = forest.allocate_id()
        box = BoxNode(
            node_id=nid,
            joint_intervals=ivs,
            seed_config=seed_q.copy(),
            volume=vol,
        )
        td0 = time.time()
        forest.add_box_direct(box)
        t_deoverlap += time.time() - td0

        recent_vols.append(vol)
        return [box]

    # ── main loop ──
    global_stalls = 0
    last_box_ivs = None
    stop_reason = "max_seeds"

    for seed_iter in range(max_seeds):
        if forest.n_boxes >= max_boxes:
            stop_reason = f"max_boxes={max_boxes}"
            break
        if global_stalls > max_tree_stalls * 3:
            stop_reason = f"global_stalls={global_stalls}"
            break

        if (
            early_stop_window > 0
            and len(recent_vols) >= early_stop_window
            and all(v < early_stop_min_vol for v in recent_vols[-early_stop_window:])
        ):
            stop_reason = f"early_stop(last {early_stop_window} vols < {early_stop_min_vol:.1e})"
            break

        added = None

        # DFS boundary
        if last_box_ivs is not None:
            ts0 = time.time()
            boundary_seeds = _sample_boundary(last_box_ivs, boundary_batch, rng)
            t_sampling += time.time() - ts0
            n_boundary_attempts += len(boundary_seeds)
            for bs in boundary_seeds:
                if forest.n_boxes >= max_boxes:
                    break
                added = _try_add_box(bs, source="boundary")
                if added:
                    step += 1
                    last_box_ivs = list(added[-1].joint_intervals)
                    global_stalls = 0
                    break
            else:
                last_box_ivs = None

        # farthest point
        if added is None:
            ts0 = time.time()
            candidates = []
            for _ in range(farthest_k):
                q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
                if checker.check_config_collision(q):
                    continue
                if hier_tree.is_occupied(q):
                    continue
                nearest = forest.find_nearest(q)
                dist = nearest.distance_to_config(q) if nearest else float("inf")
                candidates.append((q, dist))
            t_sampling += time.time() - ts0
            n_farthest_attempts += 1

            if not candidates:
                global_stalls += 1
                continue

            q_seed, _ = max(candidates, key=lambda x: x[1])
            added = _try_add_box(q_seed, source="farthest")

            if added:
                step += 1
                last_box_ivs = list(added[-1].joint_intervals)
                global_stalls = 0
            else:
                global_stalls += 1

    dt = time.time() - t0

    # save cache
    hier_tree.auto_save()

    ht_stats = hier_tree.get_stats()
    n_adj = sum(len(v) for v in forest.adjacency.values()) // 2
    degrees = [len(forest.adjacency.get(bid, set())) for bid in forest.boxes]
    vols = [b.volume for b in forest.boxes.values()]

    return {
        "n_boxes": forest.n_boxes,
        "total_volume": forest.total_volume,
        "n_adj": n_adj,
        "steps": step,
        "time_total": dt,
        "time_load_cache": t_load,
        "time_find_free_box": t_find_free_box,
        "time_collision_check": t_collision_check,
        "time_deoverlap": t_deoverlap,
        "time_sampling": t_sampling,
        "vol_mean": float(np.mean(vols)) if vols else 0.0,
        "vol_median": float(np.median(vols)) if vols else 0.0,
        "vol_min": float(np.min(vols)) if vols else 0.0,
        "vol_max": float(np.max(vols)) if vols else 0.0,
        "deg_mean": float(np.mean(degrees)) if degrees else 0.0,
        "deg_max": int(max(degrees)) if degrees else 0,
        "n_isolated": sum(1 for d in degrees if d == 0),
        "cache_nodes_init": cache_loaded_nodes,
        "cache_fk_init": cache_loaded_fk,
        "tree_nodes": ht_stats["n_nodes"],
        "tree_max_depth": ht_stats["max_depth"],
        "tree_fk_calls": ht_stats["n_fk_calls"],
        "n_seed_collision": n_seed_collision,
        "n_seed_inside": n_seed_inside,
        "n_find_none": n_find_none,
        "n_find_tiny": n_find_tiny,
        "n_boundary_attempts": n_boundary_attempts,
        "n_farthest_attempts": n_farthest_attempts,
        "stop_reason": stop_reason,
    }


# ─────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HierAABBTree 缓存效果基准测试（无渲染）"
    )
    parser.add_argument("--runs", type=int, default=10, help="重复次数 (默认: 10)")
    parser.add_argument("--seed", type=int, default=42, help="场景随机种子")
    parser.add_argument("--n-obs", type=int, default=5, help="障碍物数量")
    parser.add_argument("--max-boxes", type=int, default=300, help="最大 box 数")
    parser.add_argument("--max-seeds", type=int, default=2000, help="最大采样迭代")
    parser.add_argument("--max-depth", type=int, default=40, help="最大切分深度")
    parser.add_argument("--min-edge", type=float, default=0.01, help="最小分割边长")
    parser.add_argument("--boundary-batch", type=int, default=6, help="DFS 边界采样批量")
    parser.add_argument("--farthest-k", type=int, default=12, help="最远点采样候选数")
    parser.add_argument("--early-stop-window", type=int, default=30, help="早停窗口")
    parser.add_argument("--early-stop-min-vol", type=float, default=1e-4, help="早停阈值")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmarks/output") / f"cache_bench_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  HierAABBTree 缓存效果基准测试")
    logger.info("  runs=%d  seed=%d  n_obs=%d  max_boxes=%d  max_depth=%d",
                args.runs, args.seed, args.n_obs, args.max_boxes, args.max_depth)
    logger.info("  output: %s", output_dir)
    logger.info("=" * 65)

    # 1. 加载机器人
    robot = load_robot("2dof_planar")
    joint_limits = list(robot.joint_limits)
    logger.info("Robot: %s, %dDOF", robot.name, robot.n_joints)

    # 2. 生成场景（固定种子，所有 run 使用相同场景）
    rng = np.random.default_rng(args.seed)
    scene = random_scene_2d(robot, args.n_obs, rng)
    logger.info("Scene: %d obstacles", scene.n_obstacles)
    for obs in scene.get_obstacles():
        logger.info(
            "  [%s] (%.3f,%.3f)-(%.3f,%.3f)",
            obs.name,
            obs.min_point[0], obs.min_point[1],
            obs.max_point[0], obs.max_point[1],
        )

    # 3. 第一次运行前清除全局缓存
    cache_dir = HierAABBTree._global_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("已清除全局缓存: %s", cache_dir)
    else:
        logger.info("无全局缓存可清除")

    # 4. 运行 N 次
    all_results: List[dict] = []
    for run_idx in range(args.runs):
        logger.info("")
        logger.info("-" * 65)
        logger.info("  Run %d/%d  %s", run_idx + 1, args.runs,
                     "(无缓存)" if run_idx == 0 else "(使用缓存)")
        logger.info("-" * 65)

        t_run_start = time.time()

        run_seed = args.seed + run_idx  # 每次不同的拓展种子，场景固定
        result = grow_forest_no_render(
            robot=robot,
            scene=scene,
            joint_limits=joint_limits,
            max_boxes=args.max_boxes,
            max_seeds=args.max_seeds,
            boundary_batch=args.boundary_batch,
            farthest_k=args.farthest_k,
            rng_seed=run_seed,
            max_depth=args.max_depth,
            min_edge_length=args.min_edge,
            early_stop_window=args.early_stop_window,
            early_stop_min_vol=args.early_stop_min_vol,
        )

        result["run"] = run_idx + 1
        result["is_cold"] = run_idx == 0

        all_results.append(result)

        logger.info(
            "  → boxes=%d  vol=%.4f  adj=%d  time=%.2fs  "
            "find_free=%.2fs  cache_nodes=%d→%d  fk=%d→%d  stop=%s",
            result["n_boxes"], result["total_volume"], result["n_adj"],
            result["time_total"],
            result["time_find_free_box"],
            result["cache_nodes_init"], result["tree_nodes"],
            result["cache_fk_init"], result["tree_fk_calls"],
            result["stop_reason"],
        )

    # 5. 汇总报告
    report = generate_report(all_results, args)
    report_path = output_dir / "benchmark_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report)
    logger.info("报告已保存: %s", report_path)


def generate_report(results: List[dict], args) -> str:
    """生成汇总统计报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("  HierAABBTree 缓存效果基准测试报告")
    lines.append("=" * 80)
    lines.append(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  参数: runs={args.runs}  seed={args.seed}  n_obs={args.n_obs}  "
                 f"max_boxes={args.max_boxes}  max_depth={args.max_depth}")
    lines.append(f"        min_edge={args.min_edge}  boundary_batch={args.boundary_batch}  "
                 f"farthest_k={args.farthest_k}")
    lines.append(f"        early_stop_window={args.early_stop_window}  "
                 f"early_stop_min_vol={args.early_stop_min_vol}")
    lines.append("")

    # ── 逐次结果表 ──
    lines.append("-" * 80)
    lines.append("  逐次运行结果:")
    lines.append("-" * 80)
    hdr = (f"  {'run':>3s}  {'cache':>5s}  {'boxes':>5s}  {'vol':>7s}  "
           f"{'adj':>4s}  {'time':>6s}  {'find':>6s}  {'coll':>6s}  "
           f"{'deov':>6s}  {'samp':>6s}  {'nodes':>6s}  {'fk':>5s}  "
           f"{'stop':>20s}")
    lines.append(hdr)
    lines.append("  " + "-" * 76)

    for r in results:
        tag = "cold" if r["is_cold"] else "warm"
        lines.append(
            f"  {r['run']:3d}  {tag:>5s}  {r['n_boxes']:5d}  "
            f"{r['total_volume']:7.3f}  {r['n_adj']:4d}  "
            f"{r['time_total']:6.2f}  {r['time_find_free_box']:6.2f}  "
            f"{r['time_collision_check']:6.2f}  {r['time_deoverlap']:6.2f}  "
            f"{r['time_sampling']:6.2f}  {r['tree_nodes']:6d}  "
            f"{r['tree_fk_calls']:5d}  {r['stop_reason']:>20s}"
        )

    # ── 冷启动 vs 热缓存 对比 ──
    cold = [r for r in results if r["is_cold"]]
    warm = [r for r in results if not r["is_cold"]]

    lines.append("")
    lines.append("-" * 80)
    lines.append("  冷启动 vs 热缓存 统计汇总:")
    lines.append("-" * 80)

    def _stat_line(label, values, fmt=".2f"):
        arr = np.array(values)
        return (f"    {label:<24s}  "
                f"mean={np.mean(arr):{fmt}}  std={np.std(arr):{fmt}}  "
                f"min={np.min(arr):{fmt}}  max={np.max(arr):{fmt}}")

    if cold:
        lines.append("")
        lines.append(f"  冷启动 (run 1, 无缓存):")
        c = cold[0]
        lines.append(f"    总耗时               : {c['time_total']:.2f}s")
        lines.append(f"    find_free_box        : {c['time_find_free_box']:.2f}s")
        lines.append(f"    collision_check      : {c['time_collision_check']:.2f}s")
        lines.append(f"    deoverlap            : {c['time_deoverlap']:.2f}s")
        lines.append(f"    sampling             : {c['time_sampling']:.2f}s")
        lines.append(f"    boxes={c['n_boxes']}  vol={c['total_volume']:.4f}  adj={c['n_adj']}")
        lines.append(f"    tree: {c['cache_nodes_init']}→{c['tree_nodes']} nodes, "
                     f"{c['cache_fk_init']}→{c['tree_fk_calls']} FK calls")
        lines.append(f"    vol: mean={c['vol_mean']:.4f}  median={c['vol_median']:.4f}  "
                     f"min={c['vol_min']:.6f}  max={c['vol_max']:.4f}")
        lines.append(f"    deg: mean={c['deg_mean']:.2f}  max={c['deg_max']}  "
                     f"isolated={c['n_isolated']}")

    if warm:
        lines.append("")
        lines.append(f"  热缓存 (run 2-{len(results)}, 共 {len(warm)} 次):")
        lines.append(_stat_line("总耗时 (s)", [r["time_total"] for r in warm]))
        lines.append(_stat_line("find_free_box (s)", [r["time_find_free_box"] for r in warm]))
        lines.append(_stat_line("collision_check (s)", [r["time_collision_check"] for r in warm]))
        lines.append(_stat_line("deoverlap (s)", [r["time_deoverlap"] for r in warm]))
        lines.append(_stat_line("sampling (s)", [r["time_sampling"] for r in warm]))
        lines.append(_stat_line("boxes", [r["n_boxes"] for r in warm], fmt=".0f"))
        lines.append(_stat_line("total_volume", [r["total_volume"] for r in warm]))
        lines.append(_stat_line("adj_edges", [r["n_adj"] for r in warm], fmt=".0f"))
        lines.append(_stat_line("vol_mean", [r["vol_mean"] for r in warm]))
        lines.append(_stat_line("vol_min", [r["vol_min"] for r in warm], fmt=".6f"))
        lines.append(_stat_line("deg_mean", [r["deg_mean"] for r in warm]))
        lines.append(_stat_line("deg_max", [r["deg_max"] for r in warm], fmt=".0f"))
        lines.append(_stat_line("tree_nodes", [r["tree_nodes"] for r in warm], fmt=".0f"))
        lines.append(_stat_line("FK calls (final)", [r["tree_fk_calls"] for r in warm], fmt=".0f"))
        lines.append(_stat_line("cache_load (s)", [r["time_load_cache"] for r in warm]))

    # ── 加速比 ──
    if cold and warm:
        c = cold[0]
        w_mean_time = np.mean([r["time_total"] for r in warm])
        w_mean_ffb = np.mean([r["time_find_free_box"] for r in warm])
        lines.append("")
        lines.append("-" * 80)
        lines.append("  加速比:")
        lines.append("-" * 80)
        if w_mean_time > 0:
            lines.append(f"    总耗时加速      : {c['time_total']/w_mean_time:.2f}x  "
                         f"({c['time_total']:.2f}s → {w_mean_time:.2f}s)")
        if w_mean_ffb > 0:
            lines.append(f"    find_free_box   : {c['time_find_free_box']/w_mean_ffb:.2f}x  "
                         f"({c['time_find_free_box']:.2f}s → {w_mean_ffb:.2f}s)")
        new_fk_warm = np.mean(
            [r["tree_fk_calls"] - r["cache_fk_init"] for r in warm]
        )
        lines.append(f"    新增 FK calls   : cold={c['tree_fk_calls']}  "
                     f"warm_mean={new_fk_warm:.0f}")

    # ── 每次采样统计 ──
    lines.append("")
    lines.append("-" * 80)
    lines.append("  采样统计:")
    lines.append("-" * 80)
    for r in results:
        tag = "cold" if r["is_cold"] else "warm"
        lines.append(
            f"  run {r['run']:2d} ({tag}): "
            f"boundary={r['n_boundary_attempts']}  farthest={r['n_farthest_attempts']}  "
            f"coll_reject={r['n_seed_collision']}  inside={r['n_seed_inside']}  "
            f"none={r['n_find_none']}  tiny={r['n_find_tiny']}"
        )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
