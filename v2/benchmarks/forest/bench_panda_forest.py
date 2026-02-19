#!/usr/bin/env python
"""
benchmarks/forest/bench_panda_forest.py - Panda 7-DOF Box Forest 拓展测试 (v2)

在随机 3D 障碍物场景下，测试 Panda 机器人关节空间的 box forest 拓展。
记录详细的各环节耗时、运行次数、每个 box 的拓展日志。

用法：
    python -m v2.benchmarks.forest.bench_panda_forest
    python -m v2.benchmarks.forest.bench_panda_forest --n-obs 10 --max-boxes 200
    python -m v2.benchmarks.forest.bench_panda_forest --max-boxes 200 --max-seeds 3000 --max-depth 30
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import Robot, load_robot
from common.output import make_output_dir
from forest.models import BoxNode, PlannerConfig
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.box_forest import BoxForest
from forest.hier_aabb_tree import HierAABBTree

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
LOG_FMT_SIMPLE = "[%(asctime)s] %(message)s"

logging.basicConfig(level=logging.WARNING, format=LOG_FMT_SIMPLE, datefmt="%H:%M:%S")
logger = logging.getLogger("bench_panda")
console_logger = logging.getLogger("console")


def random_scene_3d(
    n_obs: int,
    rng: np.random.Generator,
    workspace_radius: float = 0.85,
    workspace_z_range: Tuple[float, float] = (0.0, 1.0),
    obs_size_range: Tuple[float, float] = (0.06, 0.20),
) -> Scene:
    """在 Panda 工作空间内随机生成 3D AABB 障碍物。"""
    scene = Scene()
    for i in range(n_obs):
        r = rng.uniform(0.25 * workspace_radius, 0.85 * workspace_radius)
        theta = rng.uniform(-math.pi, math.pi)
        cx = r * math.cos(theta)
        cy = r * math.sin(theta)
        cz = rng.uniform(workspace_z_range[0] + 0.1, workspace_z_range[1] - 0.1)

        hx = rng.uniform(obs_size_range[0], obs_size_range[1])
        hy = rng.uniform(obs_size_range[0], obs_size_range[1])
        hz = rng.uniform(obs_size_range[0], obs_size_range[1])

        scene.add_obstacle(
            min_point=[cx - hx, cy - hy, cz - hz],
            max_point=[cx + hx, cy + hy, cz + hz],
            name=f"obs_{i}",
        )
    return scene


class BoxLog:
    """记录每个 box 的拓展详情"""

    __slots__ = (
        "box_id", "step", "source", "seed_q",
        "volume", "intervals", "n_absorbed",
        "t_collision", "t_find_free", "t_deoverlap",
        "tree_nodes_after", "fk_calls_after",
    )

    def __init__(self):
        pass

    def header(self) -> str:
        return (
            f"{'step':>4s}  {'id':>4s}  {'source':>10s}  {'volume':>10s}  "
            f"{'absorbed':>8s}  {'t_coll':>7s}  {'t_ffb':>7s}  "
            f"{'t_deov':>7s}  {'nodes':>6s}  {'fk':>6s}"
        )

    def format(self) -> str:
        return (
            f"{self.step:4d}  {self.box_id:4d}  {self.source:>10s}  "
            f"{self.volume:10.6f}  {self.n_absorbed:8d}  "
            f"{self.t_collision:7.4f}  {self.t_find_free:7.4f}  "
            f"{self.t_deoverlap:7.4f}  {self.tree_nodes_after:6d}  "
            f"{self.fk_calls_after:6d}"
        )


def _normalized_size(vol: float, ndim: int) -> float:
    """vol^(1/d) — 几何平均边长，解耦维度影响"""
    if vol <= 0:
        return 0.0
    return vol ** (1.0 / ndim)


def grow_forest_panda(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    max_boxes: int = 200,
    max_seeds: int = 3000,
    boundary_batch: int = 6,
    max_tree_stalls: int = 20,
    farthest_k: int = 12,
    rng_seed: int = 42,
    max_depth: int = 30,
    min_edge_length: float = 0.05,
    early_stop_window: int = 20,
    early_stop_min_size: float = 0.01,
    use_cache: bool = True,
) -> dict:
    """Panda box forest 拓展，无渲染，返回详细统计 + 逐 box 日志。"""

    ndim = robot.n_joints
    rng = np.random.default_rng(rng_seed)
    checker = CollisionChecker(robot, scene)
    obstacles = scene.get_obstacles()

    t0 = time.time()

    t_load_start = time.time()
    if use_cache:
        hier_tree = HierAABBTree.auto_load(robot, joint_limits)
    else:
        hier_tree = HierAABBTree(robot, joint_limits)
    t_load = time.time() - t_load_start

    cache_loaded_nodes = hier_tree.n_nodes
    cache_loaded_fk = hier_tree.n_fk_calls

    config = PlannerConfig(adjacency_tolerance=1e-8)
    forest = BoxForest(robot.fingerprint(), joint_limits, config)
    forest.hier_tree = hier_tree

    t_collision_check = 0.0
    t_find_free_box = 0.0
    t_deoverlap = 0.0
    t_sampling = 0.0
    n_seed_collision = 0
    n_seed_inside = 0
    n_find_none = 0
    n_find_tiny = 0
    n_ffb_max_depth = 0
    n_ffb_min_edge = 0
    n_boundary_attempts = 0
    n_farthest_attempts = 0
    recent_sizes: List[float] = []
    step = 0
    box_logs: List[BoxLog] = []

    def _sample_boundary(box_intervals, n, rng_):
        seeds = []
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
        nonlocal n_seed_collision, n_seed_inside, n_find_none, n_find_tiny, step
        nonlocal n_ffb_max_depth, n_ffb_min_edge

        bl = BoxLog()
        bl.source = source

        tc0 = time.time()
        in_collision = checker.check_config_collision(seed_q)
        dt_coll = time.time() - tc0
        t_collision_check += dt_coll
        if in_collision:
            n_seed_collision += 1
            return None

        tc0 = time.time()
        if hier_tree.is_occupied(seed_q):
            dt_occ = time.time() - tc0
            t_collision_check += dt_occ
            n_seed_inside += 1
            return None
        t_collision_check += time.time() - tc0

        tf0 = time.time()
        nid = forest.allocate_id()
        ffb_result = hier_tree.find_free_box(
            seed_q, obstacles, max_depth=max_depth,
            min_edge_length=min_edge_length,
            mark_occupied=True, forest_box_id=nid,
        )
        dt_ffb = time.time() - tf0
        t_find_free_box += dt_ffb

        if ffb_result is None:
            n_find_none += 1
            reason = hier_tree._last_ffb_none_reason
            if reason == "max_depth":
                n_ffb_max_depth += 1
            elif reason == "min_edge":
                n_ffb_min_edge += 1
            return None
        ivs = ffb_result.intervals
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        nsize = _normalized_size(vol, ndim)
        if nsize < 1e-4:
            n_find_tiny += 1
            return None

        n_absorbed = len(ffb_result.absorbed_box_ids) if ffb_result.absorbed_box_ids else 0
        if ffb_result.absorbed_box_ids:
            forest.remove_boxes(ffb_result.absorbed_box_ids)

        box = BoxNode(
            node_id=nid,
            joint_intervals=ivs,
            seed_config=seed_q.copy(),
            volume=vol,
        )
        td0 = time.time()
        forest.add_box_direct(box)
        dt_deov = time.time() - td0
        t_deoverlap += dt_deov

        recent_sizes.append(nsize)
        step += 1

        bl.box_id = nid
        bl.step = step
        bl.seed_q = seed_q.copy()
        bl.volume = vol
        bl.intervals = ivs
        bl.n_absorbed = n_absorbed
        bl.t_collision = dt_coll
        bl.t_find_free = dt_ffb
        bl.t_deoverlap = dt_deov
        bl.tree_nodes_after = hier_tree.n_nodes
        bl.fk_calls_after = hier_tree.n_fk_calls
        box_logs.append(bl)

        log_fn = logger.info if (step == 1 or step % 10 == 0) else logger.debug
        log_fn(
            "  [Box %3d] step=%d src=%-10s nsize=%.4f vol=%.6f absorbed=%d "
            "t_ffb=%.3fs nodes=%d fk=%d",
            nid, step, source, nsize, vol, n_absorbed,
            dt_ffb, bl.tree_nodes_after, bl.fk_calls_after,
        )

        if step % 50 == 0 or n_absorbed > 5:
            console_logger.warning(
                f"  Box {nid:3d} ({step:3d}): nsize={nsize:.4f}  vol={vol:.6f}  "
                f"absorbed={n_absorbed}  [src={source}]"
            )

        return [box]

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
            and len(recent_sizes) >= early_stop_window
            and all(s < early_stop_min_size for s in recent_sizes[-early_stop_window:])
        ):
            stop_reason = (
                f"early_stop(last {early_stop_window} nsize < {early_stop_min_size:.4f})"
            )
            break

        added = None

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
                    last_box_ivs = list(added[-1].joint_intervals)
                    global_stalls = 0
                    break
            else:
                last_box_ivs = None

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
                last_box_ivs = list(added[-1].joint_intervals)
                global_stalls = 0
            else:
                global_stalls += 1

    t_save_start = time.time()
    if use_cache:
        hier_tree.auto_save()
    else:
        hier_tree.auto_merge_save()
    t_save = time.time() - t_save_start

    dt_total = time.time() - t0

    ht_stats = hier_tree.get_stats()
    n_adj = sum(len(v) for v in forest.adjacency.values()) // 2
    degrees = [len(forest.adjacency.get(bid, set())) for bid in forest.boxes]
    vols = [b.volume for b in forest.boxes.values()]
    nsizes = [_normalized_size(v, ndim) for v in vols]

    visited = set()
    components = []
    for bid in forest.boxes:
        if bid in visited:
            continue
        comp_size = 0
        queue = [bid]
        while queue:
            curr = queue.pop()
            if curr in visited:
                continue
            visited.add(curr)
            comp_size += 1
            for nb in forest.adjacency.get(curr, set()):
                if nb not in visited:
                    queue.append(nb)
        components.append(comp_size)
    n_components = len(components)
    largest_component = max(components) if components else 0

    return {
        "n_boxes": forest.n_boxes,
        "ndim": ndim,
        "total_volume": forest.total_volume,
        "n_adj": n_adj,
        "steps": step,
        "seed_iters": seed_iter + 1 if 'seed_iter' in dir() else 0,
        "time_total": dt_total,
        "time_load_cache": t_load,
        "time_save_cache": t_save,
        "time_find_free_box": t_find_free_box,
        "time_collision_check": t_collision_check,
        "time_deoverlap": t_deoverlap,
        "time_sampling": t_sampling,
        "nsize_mean": float(np.mean(nsizes)) if nsizes else 0.0,
        "nsize_median": float(np.median(nsizes)) if nsizes else 0.0,
        "nsize_min": float(np.min(nsizes)) if nsizes else 0.0,
        "nsize_max": float(np.max(nsizes)) if nsizes else 0.0,
        "vol_mean": float(np.mean(vols)) if vols else 0.0,
        "vol_median": float(np.median(vols)) if vols else 0.0,
        "vol_min": float(np.min(vols)) if vols else 0.0,
        "vol_max": float(np.max(vols)) if vols else 0.0,
        "deg_mean": float(np.mean(degrees)) if degrees else 0.0,
        "deg_max": int(max(degrees)) if degrees else 0,
        "n_isolated": sum(1 for d in degrees if d == 0),
        "n_components": n_components,
        "largest_component": largest_component,
        "cache_nodes_init": cache_loaded_nodes,
        "cache_fk_init": cache_loaded_fk,
        "tree_nodes": ht_stats["n_nodes"],
        "tree_max_depth": ht_stats["max_depth"],
        "tree_fk_calls": ht_stats["n_fk_calls"],
        "n_seed_collision": n_seed_collision,
        "n_seed_inside": n_seed_inside,
        "n_find_none": n_find_none,
        "n_ffb_max_depth": n_ffb_max_depth,
        "n_ffb_min_edge": n_ffb_min_edge,
        "n_find_tiny": n_find_tiny,
        "n_boundary_attempts": n_boundary_attempts,
        "n_farthest_attempts": n_farthest_attempts,
        "stop_reason": stop_reason,
        "box_logs": box_logs,
    }


def generate_report(result: dict, args) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("  Panda 7-DOF Box Forest 拓展详细报告 (v2)")
    lines.append("=" * 90)
    lines.append(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  机器人: Panda 7-DOF")
    lines.append(f"  参数: seed={args.seed}  n_obs={args.n_obs}  "
                 f"max_boxes={args.max_boxes}  max_seeds={args.max_seeds}")
    lines.append(f"        max_depth={args.max_depth}  min_edge={args.min_edge}  "
                 f"boundary_batch={args.boundary_batch}  farthest_k={args.farthest_k}")
    lines.append(f"        early_stop_window={args.early_stop_window}  "
                 f"early_stop_min_size={args.early_stop_min_size}")
    lines.append("")

    lines.append("-" * 90)
    lines.append("  总览")
    lines.append("-" * 90)
    lines.append(f"    最终 box 数量     : {result['n_boxes']}")
    lines.append(f"    总超体积          : {result['total_volume']:.6f}")
    lines.append(f"    邻接边数          : {result['n_adj']}")
    lines.append(f"    停止原因          : {result['stop_reason']}")
    lines.append(f"    成功步数          : {result['steps']}")
    lines.append(f"    采样迭代数        : {result['seed_iters']}")
    lines.append("")

    lines.append("-" * 90)
    lines.append("  耗时分解")
    lines.append("-" * 90)
    dt = result["time_total"]
    items = [
        ("缓存加载", result["time_load_cache"]),
        ("find_free_box", result["time_find_free_box"]),
        ("碰撞检测", result["time_collision_check"]),
        ("去重叠/邻接", result["time_deoverlap"]),
        ("采样", result["time_sampling"]),
    ]
    accounted = sum(v for _, v in items)
    items.append(("其他", max(dt - accounted, 0.0)))
    for label, v in items:
        pct = v / dt * 100 if dt > 0 else 0
        bar = "█" * int(pct / 2)
        lines.append(f"    {label:<16s}: {v:8.3f}s  ({pct:5.1f}%)  {bar}")
    lines.append(f"    {'总计':<16s}: {dt:8.3f}s")
    lines.append("")

    lines.append("-" * 90)
    lines.append("  采样统计")
    lines.append("-" * 90)
    lines.append(f"    DFS 边界采样次数    : {result['n_boundary_attempts']}")
    lines.append(f"    最远点采样轮次      : {result['n_farthest_attempts']}")
    lines.append(f"    种子碰撞拒绝        : {result['n_seed_collision']}")
    lines.append(f"    种子已占用拒绝      : {result['n_seed_inside']}")
    lines.append(f"    find_free_box=None  : {result['n_find_none']}")
    lines.append(f"    find_free_box 过小   : {result['n_find_tiny']}")
    lines.append("")

    ndim = result.get('ndim', 7)
    lines.append("-" * 90)
    lines.append(f"  Box 归一化尺寸 vol^(1/{ndim}) (几何平均边长, rad)")
    lines.append("-" * 90)
    lines.append(f"    均值      : {result['nsize_mean']:.6f}")
    lines.append(f"    中位数    : {result['nsize_median']:.6f}")
    lines.append(f"    最小值    : {result['nsize_min']:.6f}")
    lines.append(f"    最大值    : {result['nsize_max']:.6f}")
    lines.append(f"    (原始超体积 mean={result['vol_mean']:.6e}  max={result['vol_max']:.6e})")
    lines.append("")

    lines.append("-" * 90)
    lines.append("  邻接图统计")
    lines.append("-" * 90)
    lines.append(f"    平均度数  : {result['deg_mean']:.2f}")
    lines.append(f"    最大度数  : {result['deg_max']}")
    lines.append(f"    孤立节点  : {result['n_isolated']}")
    lines.append("")

    lines.append("-" * 90)
    lines.append("  HierAABBTree 统计")
    lines.append("-" * 90)
    lines.append(f"    初始节点 (缓存)  : {result['cache_nodes_init']}")
    lines.append(f"    最终节点          : {result['tree_nodes']}")
    lines.append(f"    最大深度          : {result['tree_max_depth']}")
    lines.append(f"    初始 FK calls    : {result['cache_fk_init']}")
    lines.append(f"    最终 FK calls    : {result['tree_fk_calls']}")
    lines.append(f"    新增 FK calls    : {result['tree_fk_calls'] - result['cache_fk_init']}")
    lines.append("")

    box_logs: List[BoxLog] = result.get("box_logs", [])
    if box_logs:
        lines.append("-" * 90)
        lines.append("  逐 Box 拓展日志")
        lines.append("-" * 90)
        lines.append("  " + box_logs[0].header())
        lines.append("  " + "-" * 86)
        for bl in box_logs:
            lines.append("  " + bl.format())

        lines.append("")
        lines.append("-" * 90)
        lines.append("  find_free_box 累积耗时 (前 10 / 最后 10)")
        lines.append("-" * 90)
        cum_ffb = 0.0
        show_indices = set()
        n_logs = len(box_logs)
        for i in range(min(10, n_logs)):
            show_indices.add(i)
        for i in range(max(0, n_logs - 10), n_logs):
            show_indices.add(i)

        for i, bl in enumerate(box_logs):
            cum_ffb += bl.t_find_free
            if i in show_indices:
                lines.append(
                    f"    box {bl.step:3d}: t_ffb={bl.t_find_free:.4f}s  "
                    f"cum={cum_ffb:.3f}s  nodes={bl.tree_nodes_after}"
                )
            elif i == min(10, n_logs):
                lines.append(f"    ... (省略 {n_logs - 20} 行) ..." if n_logs > 20 else "")

    if box_logs:
        lines.append("")
        lines.append("-" * 90)
        lines.append("  前 10 个 box 的各关节区间宽度 (rad)")
        lines.append("-" * 90)
        header = "  " + f"{'box':>4s}  " + "  ".join(f"{'q'+str(j):>8s}" for j in range(7))
        lines.append(header)
        lines.append("  " + "-" * 72)
        for bl in box_logs[:10]:
            widths = [hi - lo for lo, hi in bl.intervals]
            w_str = "  ".join(f"{w:8.4f}" for w in widths)
            lines.append(f"  {bl.box_id:4d}  {w_str}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panda 7-DOF Box Forest 拓展详细测试 (v2)"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认: 随机)")
    parser.add_argument("--n-obs", type=int, default=10, help="3D 障碍物数量 (默认: 10)")
    parser.add_argument("--max-boxes", type=int, default=200, help="最大 box 数 (默认: 200)")
    parser.add_argument("--max-seeds", type=int, default=3000,
                        help="最大采样迭代 (默认: 3000)")
    parser.add_argument("--max-depth", type=int, default=30, help="最大切分深度 (默认: 30)")
    parser.add_argument("--min-edge", type=float, default=0.05, help="最小分割边长 (默认: 0.05)")
    parser.add_argument("--boundary-batch", type=int, default=6,
                        help="DFS 边界采样批量 (默认: 6)")
    parser.add_argument("--farthest-k", type=int, default=12,
                        help="最远点采样候选数 (默认: 12)")
    parser.add_argument("--early-stop-window", type=int, default=20,
                        help="早停窗口 (默认: 20)")
    parser.add_argument("--early-stop-min-size", type=float, default=0.01,
                        help="早停阈值 vol^(1/d) (默认: 0.01 rad)")
    parser.add_argument("--no-cache", action="store_true",
                        help="不使用全局缓存（冷启动，不删除已有缓存）")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time()) % 100000

    output_dir = make_output_dir("benchmarks", "panda_forest")

    log_file = output_dir / "bench_panda_forest.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FMT, datefmt="%H:%M:%S"))

    for log_name in ["bench_panda", "planner", "forest", "aabb"]:
        log = logging.getLogger(log_name)
        log.setLevel(logging.INFO)
        log.propagate = False
        log.addHandler(file_handler)

    console_logger.warning("=" * 70)
    console_logger.warning("  Panda 7-DOF Box Forest 拓展测试 (v2)")
    console_logger.warning(
        f"  seed={args.seed}  n_obs={args.n_obs}  max_boxes={args.max_boxes}  "
        f"max_seeds={args.max_seeds}  max_depth={args.max_depth}"
    )
    console_logger.warning(f"  output: {output_dir}")
    console_logger.warning(f"  log: {log_file}")
    console_logger.warning("=" * 70)

    logger.info("=" * 70)
    logger.info("  Panda 7-DOF Box Forest 拓展测试 (v2)")
    logger.info("  seed=%d  n_obs=%d  max_boxes=%d  max_seeds=%d  max_depth=%d",
                args.seed, args.n_obs, args.max_boxes, args.max_seeds, args.max_depth)
    logger.info("  output: %s", output_dir)
    logger.info("=" * 70)

    robot = load_robot("panda")
    joint_limits = list(robot.joint_limits)
    console_logger.warning(f"Robot: {robot.name}, {robot.n_joints}DOF")
    logger.info("Robot: %s, %dDOF", robot.name, robot.n_joints)
    for j, (lo, hi) in enumerate(joint_limits):
        logger.info("  q%d: [%+.4f, %+.4f]  span=%.4f rad (%.1f°)",
                    j, lo, hi, hi - lo, math.degrees(hi - lo))

    rng = np.random.default_rng(args.seed)
    scene = random_scene_3d(args.n_obs, rng)
    console_logger.warning(f"Scene: {scene.n_obstacles} 3D obstacles")
    logger.info("Scene: %d 3D obstacles", scene.n_obstacles)
    for obs in scene.get_obstacles():
        logger.info(
            "  [%s] min=(%.3f,%.3f,%.3f) max=(%.3f,%.3f,%.3f) size=(%.3f,%.3f,%.3f)",
            obs.name,
            obs.min_point[0], obs.min_point[1], obs.min_point[2],
            obs.max_point[0], obs.max_point[1], obs.max_point[2],
            *(obs.max_point - obs.min_point),
        )

    use_cache = not args.no_cache
    cache_mode = "使用缓存" if use_cache else "冷启动(不使用缓存)"
    console_logger.warning(f"缓存模式: {cache_mode}")
    logger.info("缓存模式: %s", cache_mode)

    console_logger.warning("")
    console_logger.warning("开始 box forest 拓展...")
    logger.info("")
    logger.info("开始 box forest 拓展...")
    logger.info("-" * 70)

    result = grow_forest_panda(
        robot=robot,
        scene=scene,
        joint_limits=joint_limits,
        max_boxes=args.max_boxes,
        max_seeds=args.max_seeds,
        boundary_batch=args.boundary_batch,
        farthest_k=args.farthest_k,
        rng_seed=args.seed,
        max_depth=args.max_depth,
        min_edge_length=args.min_edge,
        early_stop_window=args.early_stop_window,
        early_stop_min_size=args.early_stop_min_size,
        use_cache=use_cache,
    )

    logger.info("-" * 70)
    logger.info("拓展完成: boxes=%d  vol=%.6f  time=%.2fs  stop=%s",
                result["n_boxes"], result["total_volume"],
                result["time_total"], result["stop_reason"])

    console_logger.warning("-" * 70)
    console_logger.warning(
        f"拓展完成: boxes={result['n_boxes']}  vol={result['total_volume']:.6f}  "
        f"time={result['time_total']:.2f}s  stop={result['stop_reason']}"
    )

    report = generate_report(result, args)
    report_path = output_dir / "panda_forest_report.txt"
    report_path.write_text(report, encoding="utf-8")

    console_logger.warning("")
    console_logger.warning("=" * 70)
    console_logger.warning("测试完成摘要")
    console_logger.warning("=" * 70)
    console_logger.warning(f"boxes: {result['n_boxes']}  total_volume: {result['total_volume']:.6f}")
    console_logger.warning(f"time_total: {result['time_total']:.2f}s  (ffb: {result['time_find_free_box']:.2f}s, coll: {result['time_collision_check']:.2f}s)")
    console_logger.warning(f"nsize: mean={result['nsize_mean']:.4f}  median={result['nsize_median']:.4f}  max={result['nsize_max']:.4f}")
    console_logger.warning(f"tree: nodes={result['tree_nodes']}  fk_calls={result['tree_fk_calls']}  cache_init={result['cache_nodes_init']}")
    console_logger.warning(f"报告: {report_path}")
    console_logger.warning(f"日志: {log_file}")
    console_logger.warning("=" * 70)

    print()
    print(report)
    logger.info("报告已保存: %s", report_path)


if __name__ == "__main__":
    main()
