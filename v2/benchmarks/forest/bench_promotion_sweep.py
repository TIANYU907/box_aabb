#!/usr/bin/env python
"""
benchmarks/forest/bench_promotion_sweep.py
═══════════════════════════════════════════
Promotion Depth 多维度参数扫描实验

在 Panda 7-DOF 场景下，系统扫描以下参数组合:
  - promotion_depth:  0, 1, 2, 3, 4
  - n_obstacles:      5, 10, 15, 20, 30
  - max_depth (FFB):  30, 40, 50
  - min_edge_length:  0.03, 0.05, 0.08

每个组合在多个随机场景上运行, 最后输出:
  1) 完整实验矩阵 (CSV 格式, 可导入 Excel/Pandas)
  2) 各参数维度的性能影响排名
  3) 最优配置推荐

注意: AABB 存储仍使用 union(children) → parent 维护,
      promotion_depth 仅影响上行阶段碰撞判定方式。

用法:
    python -m v2.benchmarks.forest.bench_promotion_sweep
    python -m v2.benchmarks.forest.bench_promotion_sweep --quick
    python -m v2.benchmarks.forest.bench_promotion_sweep --full
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
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


def _box_volume(ivs):
    v = 1.0
    for lo, hi in ivs:
        v *= max(hi - lo, 0.0)
    return v


def _box_gmean(ivs):
    n = len(ivs)
    v = _box_volume(ivs)
    return v ** (1.0 / n) if v > 0 and n > 0 else 0.0


def _mean(data):
    return sum(data) / len(data) if data else 0.0


def _percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ═══════════════════════════════════════════════════════════════
#  单次实验 (一个参数组合 × 一组 seeds)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    """一个参数组合在一个场景上的结果。"""
    # 参数
    promo_depth: int = 0
    n_obs: int = 0
    ffb_max_depth: int = 40
    min_edge: float = 0.05
    scene_seed: int = 0

    # 结果
    n_valid_seeds: int = 0
    n_boxes: int = 0
    n_none: int = 0
    total_volume: float = 0.0
    volumes: List[float] = field(default_factory=list)
    gmeans: List[float] = field(default_factory=list)
    ffb_times: List[float] = field(default_factory=list)
    node_depths: List[int] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def avg_ffb_ms(self):
        return _mean(self.ffb_times) * 1000 if self.ffb_times else 0.0

    @property
    def med_ffb_ms(self):
        return _percentile(self.ffb_times, 50) * 1000 if self.ffb_times else 0.0

    @property
    def p95_ffb_ms(self):
        return _percentile(self.ffb_times, 95) * 1000 if self.ffb_times else 0.0

    @property
    def avg_gmean(self):
        return _mean(self.gmeans)

    @property
    def med_gmean(self):
        return _percentile(self.gmeans, 50)

    @property
    def avg_node_depth(self):
        return _mean([float(x) for x in self.node_depths])


def run_one(
    robot: Robot,
    scene: Scene,
    valid_seeds: List[np.ndarray],
    obstacles: list,
    promo_depth: int,
    ffb_max_depth: int,
    min_edge: float,
    scene_seed: int,
    n_obs: int,
) -> RunResult:
    """运行一个参数组合。"""
    r = RunResult(
        promo_depth=promo_depth,
        n_obs=n_obs,
        ffb_max_depth=ffb_max_depth,
        min_edge=min_edge,
        scene_seed=scene_seed,
        n_valid_seeds=len(valid_seeds),
    )

    tree = HierAABBTree(robot=robot)
    obs_packed = tree._prepack_obstacles_c(obstacles)

    t_wall_0 = time.perf_counter()

    for sq in valid_seeds:
        if tree.is_occupied(sq):
            continue

        t0 = time.perf_counter()
        ffb = tree.find_free_box(
            sq, obstacles,
            max_depth=ffb_max_depth,
            min_edge_length=min_edge,
            mark_occupied=True,
            obs_packed=obs_packed,
            promotion_depth=promo_depth,
        )
        dt = time.perf_counter() - t0
        r.ffb_times.append(dt)

        if ffb is None:
            r.n_none += 1
            continue

        vol = _box_volume(ffb.intervals)
        gm = _box_gmean(ffb.intervals)
        if gm < 1e-4:
            continue

        r.n_boxes += 1
        r.total_volume += vol
        r.volumes.append(vol)
        r.gmeans.append(gm)
        r.node_depths.append(tree._store.get_depth(ffb.node_idx))

    r.wall_time = time.perf_counter() - t_wall_0
    return r


# ═══════════════════════════════════════════════════════════════
#  参数扫描引擎
# ═══════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    """实验参数网格。"""
    promo_depths: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_obs_list: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30])
    ffb_max_depths: List[int] = field(default_factory=lambda: [30, 40, 50])
    min_edges: List[float] = field(default_factory=lambda: [0.03, 0.05, 0.08])
    scene_seeds: List[int] = field(default_factory=lambda: [42, 100, 256])
    max_seeds_per_scene: int = 200


def get_quick_config() -> SweepConfig:
    """快速测试: 较少参数组合。"""
    return SweepConfig(
        promo_depths=[0, 1, 2, 3],
        n_obs_list=[8, 15, 25],
        ffb_max_depths=[40],
        min_edges=[0.05],
        scene_seeds=[42, 100],
        max_seeds_per_scene=150,
    )


def get_standard_config() -> SweepConfig:
    """标准实验: 覆盖主要参数组合。"""
    return SweepConfig(
        promo_depths=[0, 1, 2, 3, 4],
        n_obs_list=[5, 10, 15, 20, 30],
        ffb_max_depths=[30, 40],
        min_edges=[0.03, 0.05, 0.08],
        scene_seeds=[42, 100, 256],
        max_seeds_per_scene=200,
    )


def get_full_config() -> SweepConfig:
    """完整实验: 全参数网格。"""
    return SweepConfig(
        promo_depths=[0, 1, 2, 3, 4, 5],
        n_obs_list=[5, 8, 10, 15, 20, 25, 30],
        ffb_max_depths=[30, 40, 50],
        min_edges=[0.02, 0.03, 0.05, 0.08, 0.10],
        scene_seeds=[42, 100, 256, 777, 1234],
        max_seeds_per_scene=300,
    )


def run_sweep(cfg: SweepConfig) -> List[RunResult]:
    """执行参数扫描。"""
    robot = load_robot("panda")
    n_dims = robot.n_joints
    jl = robot.joint_limits

    # 计算总实验数
    combos = list(itertools.product(
        cfg.n_obs_list, cfg.ffb_max_depths, cfg.min_edges,
        cfg.scene_seeds, cfg.promo_depths,
    ))
    # 将 n_obs × scene_seed 视为独立场景
    scene_keys = set((n_obs, ss) for n_obs, _, _, ss, _ in combos)
    total_combos = len(combos)
    n_scene_keys = len(scene_keys)

    print(f"╔═ Promotion Depth Sweep ═════════════════════════════════════╗")
    print(f"║  Robot: Panda 7-DOF")
    print(f"║  promotion_depth: {cfg.promo_depths}")
    print(f"║  n_obstacles:     {cfg.n_obs_list}")
    print(f"║  ffb_max_depth:   {cfg.ffb_max_depths}")
    print(f"║  min_edge:        {cfg.min_edges}")
    print(f"║  scene_seeds:     {cfg.scene_seeds}")
    print(f"║  seeds/scene:     {cfg.max_seeds_per_scene}")
    print(f"║  独立场景数: {n_scene_keys}  总实验组合: {total_combos}")
    print(f"║  注: AABB 存储仍为 union(children) → parent")
    print(f"╚═════════════════════════════════════════════════════════════╝")
    print()

    # 预生成所有场景和 seed 序列 (缓存以避免重复生成)
    scene_cache: Dict[Tuple[int, int], Tuple[Scene, list, List[np.ndarray]]] = {}

    for n_obs, scene_seed in sorted(scene_keys):
        rng = np.random.default_rng(scene_seed)
        scene = random_scene_3d(n_obs, rng)
        obstacles = scene.get_obstacles()

        # 生成 seed 点
        seed_rng = np.random.default_rng(scene_seed + 10000)
        all_seeds_q = []
        for _ in range(cfg.max_seeds_per_scene):
            q = np.array([seed_rng.uniform(jl[j][0], jl[j][1])
                          for j in range(n_dims)], dtype=np.float64)
            all_seeds_q.append(q)

        # 预过滤碰撞 seed
        checker = CollisionChecker(robot=robot, scene=scene)
        valid_seeds = [sq for sq in all_seeds_q
                       if not checker.check_config_collision(sq)]
        scene_cache[(n_obs, scene_seed)] = (scene, obstacles, valid_seeds)

    # 执行所有组合
    all_results: List[RunResult] = []
    done = 0

    # 按 (n_obs, ffb_max_depth, min_edge, scene_seed) 分组, 内层循环 promo_depth
    # 这样同一场景下的不同 depth 在一起输出
    outer_combos = list(itertools.product(
        cfg.n_obs_list, cfg.ffb_max_depths, cfg.min_edges, cfg.scene_seeds,
    ))

    for n_obs, fmd, me, ss in outer_combos:
        scene, obstacles, valid_seeds = scene_cache[(n_obs, ss)]
        line_parts = []
        for pd in cfg.promo_depths:
            done += 1
            rr = run_one(
                robot=robot,
                scene=scene,
                valid_seeds=valid_seeds,
                obstacles=obstacles,
                promo_depth=pd,
                ffb_max_depth=fmd,
                min_edge=me,
                scene_seed=ss,
                n_obs=n_obs,
            )
            all_results.append(rr)
            line_parts.append(f"d{pd}={rr.n_boxes}box/{rr.avg_ffb_ms:.2f}ms")

        valid_n = len(valid_seeds)
        print(f"  [{done:4d}/{total_combos}] obs={n_obs:2d} fmd={fmd:2d} "
              f"me={me:.2f} ss={ss:3d} ({valid_n:3d}seeds)  "
              f"{' | '.join(line_parts)}")

    print()
    return all_results


# ═══════════════════════════════════════════════════════════════
#  分析引擎
# ═══════════════════════════════════════════════════════════════

@dataclass
class GroupStats:
    """某个分组条件下的聚合统计。"""
    key: str = ""
    label: str = ""
    n_runs: int = 0
    total_boxes: int = 0
    total_none: int = 0
    avg_ffb_ms: float = 0.0
    med_ffb_ms: float = 0.0
    p95_ffb_ms: float = 0.0
    avg_gmean: float = 0.0
    med_gmean: float = 0.0
    avg_total_vol: float = 0.0
    avg_node_depth: float = 0.0


def _group_by(results: List[RunResult], key_fn) -> Dict:
    """按 key_fn 分组并聚合。"""
    groups = defaultdict(list)
    for r in results:
        k = key_fn(r)
        groups[k].append(r)

    stats: Dict = {}
    for k, runs in sorted(groups.items()):
        gs = GroupStats(key=str(k), label=str(k), n_runs=len(runs))
        all_ffb = []
        all_gm = []
        all_vol = []
        all_nd = []
        for r in runs:
            gs.total_boxes += r.n_boxes
            gs.total_none += r.n_none
            all_ffb.extend(r.ffb_times)
            all_gm.extend(r.gmeans)
            all_vol.append(r.total_volume)
            all_nd.extend([float(x) for x in r.node_depths])

        gs.avg_ffb_ms = _mean(all_ffb) * 1000
        gs.med_ffb_ms = _percentile(all_ffb, 50) * 1000
        gs.p95_ffb_ms = _percentile(all_ffb, 95) * 1000
        gs.avg_gmean = _mean(all_gm)
        gs.med_gmean = _percentile(all_gm, 50)
        gs.avg_total_vol = _mean(all_vol) if all_vol else 0.0
        gs.avg_node_depth = _mean(all_nd) if all_nd else 0.0
        stats[k] = gs

    return stats


def _delta_pct(val, baseline):
    if baseline == 0:
        return 0.0
    return (val - baseline) / baseline * 100


# ═══════════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════════

def format_sweep_report(results: List[RunResult], cfg: SweepConfig) -> str:
    lines: List[str] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("═" * 80)
    lines.append("  Promotion Depth 多维度参数扫描报告")
    lines.append(f"  {now_str}  |  Robot: Panda 7-DOF")
    lines.append("  注: AABB 存储 = union(children), promotion_depth 仅影响上行碰撞判定")
    lines.append("═" * 80)
    lines.append("")

    # ═══════════════════════════════════════════════
    # 1. 按 promotion_depth 聚合 (核心对比)
    # ═══════════════════════════════════════════════
    lines.append("╔══ 1. 按 promotion_depth 聚合 (核心对比) ═══════════════════════════════╗")
    depth_stats = _group_by(results, lambda r: r.promo_depth)
    baseline = depth_stats.get(0)

    header = (
        f"  {'depth':>5s}  │ {'runs':>4s}  {'boxes':>6s}  {'none':>5s}"
        f"  │ {'avg_ms':>7s}  {'med_ms':>7s}  {'p95_ms':>7s}"
        f"  │ {'avg_gm':>7s}  {'avg_vol':>9s}"
        f"  │ {'vol_Δ%':>7s}  {'time_Δ%':>8s}"
    )
    lines.append(header)
    lines.append("  " + "─" * 70)

    for d in cfg.promo_depths:
        gs = depth_stats.get(d)
        if gs is None:
            continue
        vol_d = _delta_pct(gs.avg_total_vol, baseline.avg_total_vol) if baseline else 0
        time_d = _delta_pct(gs.avg_ffb_ms, baseline.avg_ffb_ms) if baseline else 0
        lines.append(
            f"  {d:>5d}  │ {gs.n_runs:>4d}  {gs.total_boxes:>6d}  {gs.total_none:>5d}"
            f"  │ {gs.avg_ffb_ms:>7.3f}  {gs.med_ffb_ms:>7.3f}  {gs.p95_ffb_ms:>7.3f}"
            f"  │ {gs.avg_gmean:>7.4f}  {gs.avg_total_vol:>9.4f}"
            f"  │ {vol_d:>+7.2f}%  {time_d:>+7.2f}%"
        )

    lines.append("╚════════════════════════════════════════════════════════════════════════╝")
    lines.append("")

    # ═══════════════════════════════════════════════
    # 2. 按 n_obstacles 聚合
    # ═══════════════════════════════════════════════
    lines.append("╔══ 2. 按障碍物数聚合 (不同密度下 promotion 收益) ══════════════════════╗")

    for n_obs in cfg.n_obs_list:
        sub = [r for r in results if r.n_obs == n_obs]
        if not sub:
            continue
        sub_depth = _group_by(sub, lambda r: r.promo_depth)
        sub_baseline = sub_depth.get(0)
        if sub_baseline is None:
            continue

        lines.append(f"  ── n_obs = {n_obs} ──")
        for d in cfg.promo_depths:
            gs = sub_depth.get(d)
            if gs is None:
                continue
            vol_d = _delta_pct(gs.avg_total_vol, sub_baseline.avg_total_vol)
            time_d = _delta_pct(gs.avg_ffb_ms, sub_baseline.avg_ffb_ms)
            lines.append(
                f"    depth={d}: boxes={gs.total_boxes:>5d}  "
                f"avg_ffb={gs.avg_ffb_ms:.3f}ms  avg_gm={gs.avg_gmean:.4f}  "
                f"avg_vol={gs.avg_total_vol:.4f}  "
                f"vol_Δ={vol_d:+.2f}%  time_Δ={time_d:+.2f}%"
            )
        lines.append("")

    lines.append("╚════════════════════════════════════════════════════════════════════════╝")
    lines.append("")

    # ═══════════════════════════════════════════════
    # 3. 按 ffb_max_depth 聚合
    # ═══════════════════════════════════════════════
    if len(cfg.ffb_max_depths) > 1:
        lines.append("╔══ 3. 按 FFB max_depth 聚合 ═════════════════════════════════════════╗")

        for fmd in cfg.ffb_max_depths:
            sub = [r for r in results if r.ffb_max_depth == fmd]
            if not sub:
                continue
            sub_depth = _group_by(sub, lambda r: r.promo_depth)
            sub_baseline = sub_depth.get(0)
            if sub_baseline is None:
                continue

            lines.append(f"  ── max_depth = {fmd} ──")
            for d in cfg.promo_depths:
                gs = sub_depth.get(d)
                if gs is None:
                    continue
                vol_d = _delta_pct(gs.avg_total_vol, sub_baseline.avg_total_vol)
                time_d = _delta_pct(gs.avg_ffb_ms, sub_baseline.avg_ffb_ms)
                lines.append(
                    f"    depth={d}: boxes={gs.total_boxes:>5d}  "
                    f"avg_ffb={gs.avg_ffb_ms:.3f}ms  avg_gm={gs.avg_gmean:.4f}  "
                    f"avg_vol={gs.avg_total_vol:.4f}  "
                    f"vol_Δ={vol_d:+.2f}%  time_Δ={time_d:+.2f}%"
                )
            lines.append("")

        lines.append("╚════════════════════════════════════════════════════════════════════════╝")
        lines.append("")

    # ═══════════════════════════════════════════════
    # 4. 按 min_edge 聚合
    # ═══════════════════════════════════════════════
    if len(cfg.min_edges) > 1:
        lines.append("╔══ 4. 按 min_edge 聚合 ══════════════════════════════════════════════╗")

        for me in cfg.min_edges:
            sub = [r for r in results if abs(r.min_edge - me) < 1e-6]
            if not sub:
                continue
            sub_depth = _group_by(sub, lambda r: r.promo_depth)
            sub_baseline = sub_depth.get(0)
            if sub_baseline is None:
                continue

            lines.append(f"  ── min_edge = {me} ──")
            for d in cfg.promo_depths:
                gs = sub_depth.get(d)
                if gs is None:
                    continue
                vol_d = _delta_pct(gs.avg_total_vol, sub_baseline.avg_total_vol)
                time_d = _delta_pct(gs.avg_ffb_ms, sub_baseline.avg_ffb_ms)
                lines.append(
                    f"    depth={d}: boxes={gs.total_boxes:>5d}  "
                    f"avg_ffb={gs.avg_ffb_ms:.3f}ms  avg_gm={gs.avg_gmean:.4f}  "
                    f"avg_vol={gs.avg_total_vol:.4f}  "
                    f"vol_Δ={vol_d:+.2f}%  time_Δ={time_d:+.2f}%"
                )
            lines.append("")

        lines.append("╚════════════════════════════════════════════════════════════════════════╝")
        lines.append("")

    # ═══════════════════════════════════════════════
    # 5. 交叉分析: (n_obs, promo_depth) 热力矩阵
    # ═══════════════════════════════════════════════
    lines.append("╔══ 5. 交叉分析: 体积提升 Δ% 矩阵 (n_obs × promo_depth) ══════════════╗")
    lines.append(f"  {'n_obs':>5s}  │ " +
                 "  ".join(f"  d={d:<3d}" for d in cfg.promo_depths))
    lines.append("  " + "─" * 5 + "──┼─" + "─" * (8 * len(cfg.promo_depths)))

    for n_obs in cfg.n_obs_list:
        sub = [r for r in results if r.n_obs == n_obs]
        sub_depth = _group_by(sub, lambda r: r.promo_depth)
        bl = sub_depth.get(0)
        if bl is None:
            continue
        parts = []
        for d in cfg.promo_depths:
            gs = sub_depth.get(d)
            if gs is None:
                parts.append(f"{'---':>7s}")
            else:
                dv = _delta_pct(gs.avg_total_vol, bl.avg_total_vol)
                parts.append(f"{dv:>+7.2f}%")
        lines.append(f"  {n_obs:>5d}  │ " + "  ".join(parts))

    lines.append("╚════════════════════════════════════════════════════════════════════════╝")
    lines.append("")

    # ═══════════════════════════════════════════════
    # 6. 交叉分析: 耗时变化 Δ% 矩阵
    # ═══════════════════════════════════════════════
    lines.append("╔══ 6. 交叉分析: 耗时变化 Δ% 矩阵 (n_obs × promo_depth) ══════════════╗")
    lines.append(f"  {'n_obs':>5s}  │ " +
                 "  ".join(f"  d={d:<3d}" for d in cfg.promo_depths))
    lines.append("  " + "─" * 5 + "──┼─" + "─" * (8 * len(cfg.promo_depths)))

    for n_obs in cfg.n_obs_list:
        sub = [r for r in results if r.n_obs == n_obs]
        sub_depth = _group_by(sub, lambda r: r.promo_depth)
        bl = sub_depth.get(0)
        if bl is None:
            continue
        parts = []
        for d in cfg.promo_depths:
            gs = sub_depth.get(d)
            if gs is None:
                parts.append(f"{'---':>7s}")
            else:
                dt = _delta_pct(gs.avg_ffb_ms, bl.avg_ffb_ms)
                parts.append(f"{dt:>+7.2f}%")
        lines.append(f"  {n_obs:>5d}  │ " + "  ".join(parts))

    lines.append("╚════════════════════════════════════════════════════════════════════════╝")
    lines.append("")

    # ═══════════════════════════════════════════════
    # 7. 推荐配置
    # ═══════════════════════════════════════════════
    lines.append("╔══ 7. 推荐配置 ═══════════════════════════════════════════════════════╗")

    # 找体积提升最大 / 效率最优的 depth
    best_vol_d = 0
    best_vol_pct = 0.0
    best_efficiency_d = 0
    best_efficiency_score = 0.0  # vol_gain / max(1, time_penalty)

    for d in cfg.promo_depths:
        if d == 0:
            continue
        gs = depth_stats.get(d)
        if gs is None or baseline is None:
            continue
        vol_d = _delta_pct(gs.avg_total_vol, baseline.avg_total_vol)
        time_d = _delta_pct(gs.avg_ffb_ms, baseline.avg_ffb_ms)

        if vol_d > best_vol_pct:
            best_vol_pct = vol_d
            best_vol_d = d

        # 效率分 = 体积提升 / max(1, 时间开销增长)
        eff = vol_d / max(1.0, max(0, time_d))
        if eff > best_efficiency_score:
            best_efficiency_score = eff
            best_efficiency_d = d

    lines.append(f"  最大体积提升:  promotion_depth = {best_vol_d}  "
                 f"(volume +{best_vol_pct:.2f}%)")

    if baseline:
        gs_eff = depth_stats.get(best_efficiency_d)
        if gs_eff:
            vol_d = _delta_pct(gs_eff.avg_total_vol, baseline.avg_total_vol)
            time_d = _delta_pct(gs_eff.avg_ffb_ms, baseline.avg_ffb_ms)
            lines.append(
                f"  最佳性价比:    promotion_depth = {best_efficiency_d}  "
                f"(volume +{vol_d:.2f}%, time {time_d:+.2f}%)"
            )

    # 按障碍物数给推荐
    lines.append("")
    lines.append("  按场景密度推荐:")
    for n_obs in cfg.n_obs_list:
        sub = [r for r in results if r.n_obs == n_obs]
        sub_depth = _group_by(sub, lambda r: r.promo_depth)
        bl = sub_depth.get(0)
        if bl is None:
            continue

        best_d = 0
        best_v = 0.0
        for d in cfg.promo_depths:
            if d == 0:
                continue
            gs = sub_depth.get(d)
            if gs is None:
                continue
            vd = _delta_pct(gs.avg_total_vol, bl.avg_total_vol)
            td = _delta_pct(gs.avg_ffb_ms, bl.avg_ffb_ms)
            # 考虑不产生过大时间开销的最大体积提升
            if vd > best_v and td < 20:
                best_v = vd
                best_d = d
        lines.append(f"    n_obs={n_obs:3d}: 推荐 depth={best_d}  (vol +{best_v:.2f}%)")

    lines.append("╚════════════════════════════════════════════════════════════════════════╝")
    lines.append("")

    return "\n".join(lines)


def generate_csv(results: List[RunResult]) -> str:
    """生成 CSV 格式的详细结果, 方便导入分析。"""
    sio = StringIO()
    writer = csv.writer(sio)
    writer.writerow([
        "promo_depth", "n_obs", "ffb_max_depth", "min_edge", "scene_seed",
        "n_valid_seeds", "n_boxes", "n_none", "total_volume",
        "avg_ffb_ms", "med_ffb_ms", "p95_ffb_ms",
        "avg_gmean", "med_gmean", "avg_node_depth",
        "wall_time_s",
    ])
    for r in results:
        writer.writerow([
            r.promo_depth, r.n_obs, r.ffb_max_depth, r.min_edge, r.scene_seed,
            r.n_valid_seeds, r.n_boxes, r.n_none,
            f"{r.total_volume:.6f}",
            f"{r.avg_ffb_ms:.4f}", f"{r.med_ffb_ms:.4f}", f"{r.p95_ffb_ms:.4f}",
            f"{r.avg_gmean:.6f}", f"{r.med_gmean:.6f}", f"{r.avg_node_depth:.2f}",
            f"{r.wall_time:.4f}",
        ])
    return sio.getvalue()


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Promotion Depth 多维度参数扫描 (Panda 7-DOF)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true",
                       help="快速模式 (少量参数组合)")
    group.add_argument("--full", action="store_true",
                       help="完整模式 (全参数网格)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        cfg = get_quick_config()
        mode = "quick"
    elif args.full:
        cfg = get_full_config()
        mode = "full"
    else:
        cfg = get_standard_config()
        mode = "standard"

    t0 = time.perf_counter()
    results = run_sweep(cfg)
    total_time = time.perf_counter() - t0

    report = format_sweep_report(results, cfg)
    csv_data = generate_csv(results)

    print(report)
    print(f"总实验耗时: {total_time:.1f}s")

    # save
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = out_dir / f"promotion_sweep_{mode}_{ts}.txt"
    report_path.write_text(report, encoding="utf-8")

    csv_path = out_dir / f"promotion_sweep_{mode}_{ts}.csv"
    csv_path.write_text(csv_data, encoding="utf-8")

    print(f"\n报告: {report_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
