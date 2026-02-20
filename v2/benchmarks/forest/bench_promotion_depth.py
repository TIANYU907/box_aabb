#!/usr/bin/env python
"""
benchmarks/forest/bench_promotion_depth.py
═══════════════════════════════════════════
Promotion Depth 实验对比

在 Panda 7-DOF 随机场景下，对比 promotion_depth = 0, 1, 2, 3 时：

  1) 单次 find_free_box 平均耗时
  2) 产出 box 的平均体积 (几何平均边长 gmean)
  3) promotion 成功率 (相对叶节点的上行层数)
  4) 有效 box 数 (非 None 且非微小)
  5) 整体 forest 覆盖总体积

用法:
    python -m v2.benchmarks.forest.bench_promotion_depth
    python -m v2.benchmarks.forest.bench_promotion_depth --seeds 42 100 --max-boxes 80 --depths 0 1 2 3
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── bootstrap v2 paths ──
from v2._bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import Robot, load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.hier_aabb_tree import HierAABBTree, FindFreeBoxResult


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

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
        theta = rng.uniform(-np.pi, np.pi)
        cx = r * math.cos(theta)
        cy = r * math.sin(theta)
        cz = rng.uniform(workspace_z_range[0] + 0.1, workspace_z_range[1] - 0.1)
        hx = rng.uniform(*obs_size_range)
        hy = rng.uniform(*obs_size_range)
        hz = rng.uniform(*obs_size_range)
        scene.add_obstacle(
            min_point=[cx - hx, cy - hy, cz - hz],
            max_point=[cx + hx, cy + hy, cz + hz],
            name=f"obs_{i}",
        )
    return scene


def box_volume(ivs: List[Tuple[float, float]]) -> float:
    v = 1.0
    for lo, hi in ivs:
        v *= max(hi - lo, 0.0)
    return v


def box_gmean(ivs: List[Tuple[float, float]]) -> float:
    n = len(ivs)
    v = box_volume(ivs)
    if v <= 0 or n == 0:
        return 0.0
    return v ** (1.0 / n)


# ═══════════════════════════════════════════════════════════════
#  单场景实验
# ═══════════════════════════════════════════════════════════════

@dataclass
class DepthResult:
    """单个 promotion_depth 在单场景下的实验结果。"""
    depth: int = 0
    n_seeds: int = 0
    n_boxes: int = 0            # 有效 (非 None 非微小)
    n_none: int = 0             # find_free_box 返回 None
    n_tiny: int = 0             # 体积过小
    total_volume: float = 0.0
    volumes: List[float] = field(default_factory=list)
    gmeans: List[float] = field(default_factory=list)
    ffb_times: List[float] = field(default_factory=list)
    promotion_levels: List[int] = field(default_factory=list)  # box_depth → result_depth

    @property
    def avg_ffb_time_ms(self) -> float:
        return (sum(self.ffb_times) / len(self.ffb_times) * 1000) if self.ffb_times else 0.0

    @property
    def med_ffb_time_ms(self) -> float:
        if not self.ffb_times:
            return 0.0
        s = sorted(self.ffb_times)
        return s[len(s) // 2] * 1000

    @property
    def avg_gmean(self) -> float:
        return (sum(self.gmeans) / len(self.gmeans)) if self.gmeans else 0.0

    @property
    def med_gmean(self) -> float:
        if not self.gmeans:
            return 0.0
        s = sorted(self.gmeans)
        return s[len(s) // 2]

    @property
    def avg_promotion_levels(self) -> float:
        return (sum(self.promotion_levels) / len(self.promotion_levels)
                if self.promotion_levels else 0.0)


def run_single_scene(
    robot: Robot,
    scene: Scene,
    seeds_q: np.ndarray,
    obstacles: list,
    depths: List[int],
    max_depth: int = 40,
    min_edge_length: float = 0.05,
    min_box_gmean: float = 1e-4,
) -> Dict[int, DepthResult]:
    """在同一场景、同一组 seed 上，对各 promotion_depth 运行 FFB。

    为公平对比，每个 depth 都使用独立的 HierAABBTree 副本，
    但使用完全相同的 seed 序列和障碍物。
    """
    results: Dict[int, DepthResult] = {}
    checker = CollisionChecker(robot=robot, scene=scene)

    # 预过滤: 只保留不在碰撞中的 seed
    valid_seeds = []
    for sq in seeds_q:
        if not checker.check_config_collision(sq):
            valid_seeds.append(sq)
    print(f"    有效 seed (非碰撞): {len(valid_seeds)} / {len(seeds_q)}")

    for d in depths:
        dr = DepthResult(depth=d)
        # 每个 depth 独立的树 (从空树开始)
        tree = HierAABBTree(robot=robot)
        obs_packed = tree._prepack_obstacles_c(obstacles)

        for sq in valid_seeds:
            dr.n_seeds += 1

            # 跳过已被前面 box 占用的 seed
            if tree.is_occupied(sq):
                continue

            t0 = time.perf_counter()
            ffb = tree.find_free_box(
                sq, obstacles,
                max_depth=max_depth,
                min_edge_length=min_edge_length,
                mark_occupied=True,
                obs_packed=obs_packed,
                promotion_depth=d,
            )
            dt = time.perf_counter() - t0
            dr.ffb_times.append(dt)

            if ffb is None:
                dr.n_none += 1
                continue

            vol = box_volume(ffb.intervals)
            gm = box_gmean(ffb.intervals)
            if gm < min_box_gmean:
                dr.n_tiny += 1
                continue

            dr.n_boxes += 1
            dr.total_volume += vol
            dr.volumes.append(vol)
            dr.gmeans.append(gm)

            # promotion level: 从 result_idx 到下行底部的深度差
            # 用树节点深度度量 (深度越大 = 更小的 box)
            result_depth = tree._store.get_depth(ffb.node_idx)
            dr.promotion_levels.append(result_depth)

        results[d] = dr

    return results


# ═══════════════════════════════════════════════════════════════
#  多场景聚合
# ═══════════════════════════════════════════════════════════════

@dataclass
class AggResult:
    """多场景聚合结果。"""
    depth: int = 0
    n_scenes: int = 0
    total_boxes: int = 0
    total_none: int = 0
    total_tiny: int = 0
    total_seeds: int = 0
    all_ffb_times: List[float] = field(default_factory=list)
    all_volumes: List[float] = field(default_factory=list)
    all_gmeans: List[float] = field(default_factory=list)
    all_promotion_levels: List[int] = field(default_factory=list)
    total_volume_per_scene: List[float] = field(default_factory=list)


def aggregate(
    all_results: List[Dict[int, DepthResult]],
    depths: List[int],
) -> Dict[int, AggResult]:
    """将多场景结果聚合。"""
    agg: Dict[int, AggResult] = {}
    for d in depths:
        a = AggResult(depth=d)
        for scene_results in all_results:
            dr = scene_results[d]
            a.n_scenes += 1
            a.total_boxes += dr.n_boxes
            a.total_none += dr.n_none
            a.total_tiny += dr.n_tiny
            a.total_seeds += dr.n_seeds
            a.all_ffb_times.extend(dr.ffb_times)
            a.all_volumes.extend(dr.volumes)
            a.all_gmeans.extend(dr.gmeans)
            a.all_promotion_levels.extend(dr.promotion_levels)
            a.total_volume_per_scene.append(dr.total_volume)
        agg[d] = a
    return agg


# ═══════════════════════════════════════════════════════════════
#  统计工具
# ═══════════════════════════════════════════════════════════════

def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _mean(data) -> float:
    return sum(data) / len(data) if data else 0.0


def _std(data) -> float:
    if len(data) < 2:
        return 0.0
    m = _mean(data)
    return (sum((x - m) ** 2 for x in data) / (len(data) - 1)) ** 0.5


# ═══════════════════════════════════════════════════════════════
#  报告
# ═══════════════════════════════════════════════════════════════

def format_report(
    agg: Dict[int, AggResult],
    depths: List[int],
    n_scenes: int,
    n_obs: int,
    max_boxes: int,
) -> str:
    """生成对比报告文本。"""
    lines: List[str] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("═" * 76)
    lines.append("  Promotion Depth 实验对比报告")
    lines.append(f"  {now_str}")
    lines.append("═" * 76)
    lines.append(f"  场景数: {n_scenes}   障碍物/场景: {n_obs}   每场景最大 seed: {max_boxes}")
    lines.append("")

    # ── 总览表 ──
    lines.append("┌─ 总览 ─────────────────────────────────────────────────────────────────┐")
    header = (
        f"  {'depth':>5s}  │ {'boxes':>6s}  {'none':>5s}  {'tiny':>5s}"
        f"  │ {'avg_ms':>7s}  {'med_ms':>7s}  {'p95_ms':>7s}"
        f"  │ {'avg_gm':>7s}  {'med_gm':>7s}  {'tot_vol':>9s}"
    )
    lines.append(header)
    lines.append("  " + "─" * 5 + "──┼─" + "─" * 20 +
                 "─┼─" + "─" * 25 + "─┼─" + "─" * 28)

    baseline_boxes = None
    baseline_vol = None
    baseline_ms = None

    for d in depths:
        a = agg[d]
        avg_ms = _mean(a.all_ffb_times) * 1000
        med_ms = _percentile(a.all_ffb_times, 50) * 1000
        p95_ms = _percentile(a.all_ffb_times, 95) * 1000
        avg_gm = _mean(a.all_gmeans)
        med_gm = _percentile(a.all_gmeans, 50)
        tot_vol = _mean(a.total_volume_per_scene) if a.total_volume_per_scene else 0.0

        if baseline_boxes is None:
            baseline_boxes = a.total_boxes
            baseline_vol = tot_vol
            baseline_ms = avg_ms

        line = (
            f"  {d:>5d}  │ {a.total_boxes:>6d}  {a.total_none:>5d}  {a.total_tiny:>5d}"
            f"  │ {avg_ms:>7.2f}  {med_ms:>7.2f}  {p95_ms:>7.2f}"
            f"  │ {avg_gm:>7.4f}  {med_gm:>7.4f}  {tot_vol:>9.4f}"
        )
        lines.append(line)

    lines.append("└────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── 相对 depth=0 的变化 ──
    lines.append("┌─ 相对 depth=0 变化 ────────────────────────────────────────────────────┐")
    lines.append(
        f"  {'depth':>5s}  │ {'boxes Δ%':>9s}  {'vol Δ%':>8s}  {'time Δ%':>9s}"
        f"  │ {'avg_promo':>9s}"
    )
    lines.append("  " + "─" * 5 + "──┼─" + "─" * 30 + "─┼─" + "─" * 10)

    for d in depths:
        a = agg[d]
        avg_ms = _mean(a.all_ffb_times) * 1000
        tot_vol = _mean(a.total_volume_per_scene) if a.total_volume_per_scene else 0.0
        avg_promo = _mean(a.all_promotion_levels) if a.all_promotion_levels else 0.0

        box_delta = ((a.total_boxes - baseline_boxes) / baseline_boxes * 100
                     if baseline_boxes else 0.0)
        vol_delta = ((tot_vol - baseline_vol) / baseline_vol * 100
                     if baseline_vol else 0.0)
        time_delta = ((avg_ms - baseline_ms) / baseline_ms * 100
                      if baseline_ms else 0.0)

        line = (
            f"  {d:>5d}  │ {box_delta:>+8.1f}%  {vol_delta:>+7.1f}%  {time_delta:>+8.1f}%"
            f"  │ {avg_promo:>9.2f}"
        )
        lines.append(line)

    lines.append("└────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── FFB 耗时分布 ──
    lines.append("┌─ FFB 耗时分布 (ms) ──────────────────────────────────────────────────┐")
    lines.append(
        f"  {'depth':>5s}  │ {'p5':>7s}  {'p25':>7s}  {'p50':>7s}"
        f"  {'p75':>7s}  {'p95':>7s}  {'p99':>7s}  {'max':>7s}"
    )
    lines.append("  " + "─" * 5 + "──┼─" + "─" * 56)

    for d in depths:
        a = agg[d]
        t = a.all_ffb_times
        if not t:
            continue
        p5 = _percentile(t, 5) * 1000
        p25 = _percentile(t, 25) * 1000
        p50 = _percentile(t, 50) * 1000
        p75 = _percentile(t, 75) * 1000
        p95 = _percentile(t, 95) * 1000
        p99 = _percentile(t, 99) * 1000
        mx = max(t) * 1000
        line = (
            f"  {d:>5d}  │ {p5:>7.2f}  {p25:>7.2f}  {p50:>7.2f}"
            f"  {p75:>7.2f}  {p95:>7.2f}  {p99:>7.2f}  {mx:>7.2f}"
        )
        lines.append(line)

    lines.append("└────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Box 体积分布 ──
    lines.append("┌─ Box 体积分布 (gmean edge length) ──────────────────────────────────┐")
    lines.append(
        f"  {'depth':>5s}  │ {'p5':>7s}  {'p25':>7s}  {'p50':>7s}"
        f"  {'p75':>7s}  {'p95':>7s}  {'max':>7s}"
    )
    lines.append("  " + "─" * 5 + "──┼─" + "─" * 48)

    for d in depths:
        a = agg[d]
        g = a.all_gmeans
        if not g:
            continue
        p5 = _percentile(g, 5)
        p25 = _percentile(g, 25)
        p50 = _percentile(g, 50)
        p75 = _percentile(g, 75)
        p95 = _percentile(g, 95)
        mx = max(g)
        line = (
            f"  {d:>5d}  │ {p5:>7.4f}  {p25:>7.4f}  {p50:>7.4f}"
            f"  {p75:>7.4f}  {p95:>7.4f}  {mx:>7.4f}"
        )
        lines.append(line)

    lines.append("└────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Promotion 深度分布 ──
    lines.append("┌─ 结果节点深度分布 (promotion 越低 = 上行越多 = box 越大) ────────────┐")
    lines.append(
        f"  {'depth':>5s}  │ {'min':>5s}  {'p25':>5s}  {'p50':>5s}"
        f"  {'p75':>5s}  {'max':>5s}  {'avg':>6s}"
    )
    lines.append("  " + "─" * 5 + "──┼─" + "─" * 36)

    for d in depths:
        a = agg[d]
        p = [float(x) for x in a.all_promotion_levels]
        if not p:
            continue
        mn = min(p)
        p25 = _percentile(p, 25)
        p50 = _percentile(p, 50)
        p75 = _percentile(p, 75)
        mx = max(p)
        avg = _mean(p)
        line = (
            f"  {d:>5d}  │ {mn:>5.0f}  {p25:>5.0f}  {p50:>5.0f}"
            f"  {p75:>5.0f}  {mx:>5.0f}  {avg:>6.1f}"
        )
        lines.append(line)

    lines.append("└────────────────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Promotion Depth 实验对比 (Panda 7-DOF)")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 100, 256],
        help="场景随机种子列表 (每个种子一个场景)")
    parser.add_argument(
        "--n-obs", type=int, default=8,
        help="每场景障碍物数 (default: 8)")
    parser.add_argument(
        "--max-boxes", type=int, default=120,
        help="每场景最大尝试 seed 数 (default: 120)")
    parser.add_argument(
        "--depths", nargs="+", type=int, default=[0, 1, 2, 3],
        help="要对比的 promotion_depth 值 (default: 0 1 2 3)")
    parser.add_argument(
        "--max-depth", type=int, default=40,
        help="FFB 最大下行深度 (default: 40)")
    parser.add_argument(
        "--min-edge", type=float, default=0.05,
        help="FFB 最小边长 (default: 0.05)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="报告输出文件路径")
    args = parser.parse_args()

    depths = sorted(args.depths)
    scene_seeds = args.seeds
    n_scenes = len(scene_seeds)

    print(f"╔═ Promotion Depth Benchmark ═══════════════════════════╗")
    print(f"║  场景数: {n_scenes}   障碍物: {args.n_obs}   最大 seed: {args.max_boxes}")
    print(f"║  对比 depth: {depths}")
    print(f"║  FFB max_depth={args.max_depth}  min_edge={args.min_edge}")
    print(f"╚═══════════════════════════════════════════════════════╝")
    print()

    robot = load_robot("panda")
    n_dims = robot.n_joints
    jl = robot.joint_limits

    all_scene_results: List[Dict[int, DepthResult]] = []

    for si, scene_seed in enumerate(scene_seeds):
        rng = np.random.default_rng(scene_seed)
        scene = random_scene_3d(args.n_obs, rng)
        obstacles = scene.get_obstacles()

        # 预生成一组公共 seed 点 (所有 depth 都用相同的 seed 序列)
        seed_rng = np.random.default_rng(scene_seed + 10000)
        seeds_q = np.empty((args.max_boxes, n_dims), dtype=np.float64)
        for i in range(args.max_boxes):
            for j in range(n_dims):
                seeds_q[i, j] = seed_rng.uniform(jl[j][0], jl[j][1])

        print(f"  ── 场景 {si+1}/{n_scenes}  (seed={scene_seed}, "
              f"{len(obstacles)} obstacles) ──")

        t_scene_start = time.perf_counter()
        scene_results = run_single_scene(
            robot=robot,
            scene=scene,
            seeds_q=seeds_q,
            obstacles=obstacles,
            depths=depths,
            max_depth=args.max_depth,
            min_edge_length=args.min_edge,
        )
        dt_scene = time.perf_counter() - t_scene_start

        for d in depths:
            dr = scene_results[d]
            print(f"    depth={d}: boxes={dr.n_boxes}  "
                  f"avg_ffb={dr.avg_ffb_time_ms:.2f}ms  "
                  f"avg_gm={dr.avg_gmean:.4f}  "
                  f"total_vol={dr.total_volume:.4f}")

        print(f"    场景总耗时: {dt_scene:.2f}s")
        print()

        all_scene_results.append(scene_results)

    # ── 聚合 & 报告 ──
    agg = aggregate(all_scene_results, depths)
    report = format_report(agg, depths, n_scenes, args.n_obs, args.max_boxes)

    print()
    print(report)

    # 保存报告
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(__file__).resolve().parent.parent.parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"promotion_depth_{ts}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
