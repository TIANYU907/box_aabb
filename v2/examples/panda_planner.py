"""
panda_planner.py — Panda 7-DOF 路径规划 (多方法对比)

在 7-DOF 关节空间中:
1. 构建 box forest (复用 BoxPlanner)
2. 可选 dim-sweep coarsen 减少 box 数
3. 三种规划方法对比:
   A) GCS (SOCP) — 凸松弛 + rounding + SOCP 精炼
   B) Dijkstra + SOCP refine — box 图最短路 + SOCP 精炼
   C) Visibility Graph — seed 配置可视图 + 贪心 shortcut

用法:
    python -m v2.examples.panda_planner
"""

import heapq
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import math

import matplotlib
matplotlib.use("Agg")

import numpy as np
import cvxpy as cp

import sys, os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
from _bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.models import BoxNode
from forest.connectivity import find_islands, bridge_islands, UnionFind
from forest.hier_aabb_tree import HierAABBTree, build_kd_partitions
from forest.box_forest import BoxForest
from planner.box_planner import BoxPlanner
from planner.models import PlannerConfig, gmean_edge_length
from common.output import make_output_dir
from forest.coarsen import coarsen_forest
from planner.dynamic_visualizer import resample_path

# 复用 2DOF 版中与维度无关的 GCS 核心函数
from v2.examples.gcs_planner_2dof import (
    build_adjacency,
    find_box_containing,
    extract_connected_subgraph,
    corridor_prune,
    solve_gcs,
    _refine_path_in_boxes,
)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PandaGCSConfig:
    seed: int = 0   # 0 → 运行时随机

    # Panda start/goal  — 远距离构型
    q_start: List[float] = field(
        default_factory=lambda: [0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal: List[float] = field(
        default_factory=lambda: [-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])

    # scene
    n_obstacles: int = 6
    workspace_radius: float = 0.85
    workspace_z_range: Tuple[float, float] = (0.0, 1.0)
    obs_size_range: Tuple[float, float] = (0.08, 0.25)

    # forest growth  (主终止条件: 连续 miss; 安全上限: max_boxes 防止 7D 爆炸)
    max_consecutive_miss: int = 20
    min_box_size: float = 0.1
    max_boxes: int = 500           # 7D 安全上限, 2D 不需要
    goal_bias: float = 0.10
    guided_sample_ratio: float = 0.6
    boundary_expand: bool = True   # 边缘扩张采样 (增加 box 连通性)
    boundary_expand_max_failures: int = 5
    boundary_expand_epsilon: float = 0.01
    parallel_grow: bool = False    # 启用分区并行生长
    n_partitions_depth: int = 3    # KD 分区深度 (2^depth 个分区, 默认8)
    parallel_workers: int = 4      # 并行进程数 (0=auto=cpu_count)

    # island / bridge
    min_island_size: float = 0.0

    # GCS solver
    corridor_hops: int = 2

    # coarsen
    coarsen_max_rounds: int = 20

    # viz
    dpi: int = 140


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

def build_panda_scene(rng: np.random.Generator, cfg: PandaGCSConfig,
                      robot=None, q_start=None, q_goal=None,
                      max_trials=100) -> Scene:
    """生成 3D 随机障碍物场景, 确保起终点无碰撞."""
    for _ in range(max_trials):
        scene = Scene()
        for i in range(cfg.n_obstacles):
            r = rng.uniform(0.25 * cfg.workspace_radius, 0.85 * cfg.workspace_radius)
            theta = rng.uniform(-math.pi, math.pi)
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            cz = rng.uniform(cfg.workspace_z_range[0] + 0.1,
                             cfg.workspace_z_range[1] - 0.1)
            hx = rng.uniform(*cfg.obs_size_range)
            hy = rng.uniform(*cfg.obs_size_range)
            hz = rng.uniform(*cfg.obs_size_range)
            scene.add_obstacle(
                min_point=[cx - hx, cy - hy, cz - hz],
                max_point=[cx + hx, cy + hy, cz + hz],
                name=f"obs_{i}",
            )
        if robot is not None:
            checker = CollisionChecker(robot=robot, scene=scene)
            if (checker.check_config_collision(q_start) or
                    checker.check_config_collision(q_goal)):
                continue
        return scene
    raise RuntimeError("Failed to build collision-free scene for Panda")


def make_planner_config(cfg: PandaGCSConfig) -> PlannerConfig:
    return PlannerConfig(
        max_iterations=999999, max_box_nodes=999999,
        min_box_size=cfg.min_box_size, goal_bias=cfg.goal_bias,
        guided_sample_ratio=cfg.guided_sample_ratio,
        segment_collision_resolution=0.05,
        connection_radius=1.5, verbose=False, forest_path=None,
        boundary_expand_enabled=cfg.boundary_expand,
        boundary_expand_max_failures=cfg.boundary_expand_max_failures,
        boundary_expand_epsilon=cfg.boundary_expand_epsilon,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parallel grow worker (top-level for pickle)
# ═══════════════════════════════════════════════════════════════════════════

def _grow_partition_worker(payload: Dict) -> Dict:
    """ProcessPool worker: 在单个 KD 分区内独立生长 boxes.

    每个 worker 创建独立的 HierAABBTree + CollisionChecker,
    运行批量采样→find_free_box 循环, 返回生成的 boxes 列表.
    """
    robot = payload["robot"]
    obstacles = payload["obstacles"]
    q_start = np.asarray(payload["q_start"], dtype=np.float64)
    q_goal = np.asarray(payload["q_goal"], dtype=np.float64)
    partition_intervals = payload["intervals"]  # List[(lo, hi)]
    max_boxes = int(payload["max_boxes"])
    max_miss = int(payload["max_miss"])
    min_box_size = float(payload["min_box_size"])
    goal_bias = float(payload.get("goal_bias", 0.1))
    seed_val = payload.get("seed")

    ndim = len(partition_intervals)
    rng = np.random.default_rng(seed_val)

    # 创建局部 HierAABBTree (仅覆盖分区区间)
    local_tree = HierAABBTree(robot, joint_limits=partition_intervals)
    obs_packed = local_tree._prepack_obstacles_c(obstacles)

    # FK 预热
    local_tree.warmup_fk_cache(max_depth=6)

    # 局部碰撞检测器
    local_scene = Scene()
    local_scene._obstacles = list(obstacles)
    local_checker = CollisionChecker(robot, local_scene)

    lows = np.array([lo for lo, _ in partition_intervals], dtype=np.float64)
    highs = np.array([hi for _, hi in partition_intervals], dtype=np.float64)
    batch_size = 32

    local_boxes: Dict[int, Dict] = {}
    next_id = 0
    consec = 0

    def _sample_batch_local():
        """仅 goal-biased + uniform (无 guided, 因无全局 hier_tree)."""
        rolls = rng.uniform(size=batch_size)
        configs = np.empty((batch_size, ndim), dtype=np.float64)
        goal_mask = rolls < goal_bias
        n_goal = int(goal_mask.sum())
        if n_goal > 0:
            noise = rng.normal(0.0, 0.3, size=(n_goal, ndim))
            configs[goal_mask] = np.clip(q_goal + noise, lows, highs)
        uni_mask = ~goal_mask
        n_uni = int(uni_mask.sum())
        if n_uni > 0:
            configs[uni_mask] = rng.uniform(lows, highs, size=(n_uni, ndim))
        collisions = local_checker.check_config_collision_batch(configs)
        return [configs[i] for i in range(batch_size) if not collisions[i]]

    seed_buffer: List[np.ndarray] = []

    # 先尝试扩展起点/终点 (如果在此分区内)
    for qs in [q_start, q_goal]:
        in_part = all(lo <= qs[d] <= hi for d, (lo, hi) in enumerate(partition_intervals))
        if in_part and not local_tree.is_occupied(qs):
            ffb = local_tree.find_free_box(
                qs, obstacles, mark_occupied=True, forest_box_id=next_id,
                obs_packed=obs_packed)
            if ffb is not None:
                vol = 1.0
                for lo, hi in ffb.intervals:
                    vol *= max(hi - lo, 0)
                if gmean_edge_length(vol, ndim) >= min_box_size:
                    local_boxes[next_id] = {
                        "joint_intervals": ffb.intervals,
                        "seed_config": qs.copy(),
                        "volume": vol,
                    }
                    next_id += 1

    while consec < max_miss:
        if len(local_boxes) >= max_boxes:
            break

        if not seed_buffer:
            seed_buffer = _sample_batch_local()
        if not seed_buffer:
            consec += 1
            continue
        q = seed_buffer.pop()

        if local_tree.is_occupied(q):
            consec += 1
            continue

        if not local_tree.can_expand(q, obs_packed=obs_packed):
            consec += 1
            continue

        ffb = local_tree.find_free_box(
            q, obstacles, mark_occupied=True, forest_box_id=next_id,
            obs_packed=obs_packed)
        if ffb is None:
            consec += 1
            continue

        vol = 1.0
        for lo, hi in ffb.intervals:
            vol *= max(hi - lo, 0)
        if gmean_edge_length(vol, ndim) < min_box_size:
            consec += 1
            continue

        if ffb.absorbed_box_ids:
            for aid in ffb.absorbed_box_ids:
                local_boxes.pop(int(aid), None)

        local_boxes[next_id] = {
            "joint_intervals": ffb.intervals,
            "seed_config": q.copy(),
            "volume": vol,
        }
        next_id += 1
        consec = 0

    return {
        "partition_id": int(payload.get("partition_id", 0)),
        "boxes": [local_boxes[k] for k in sorted(local_boxes.keys())],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Forest grow (same pattern as 2DOF but for 7-DOF)
# ═══════════════════════════════════════════════════════════════════════════

def grow_forest(planner, q_start, q_goal, seed, max_miss=20, ndim=7,
                max_boxes=1500, parallel_grow=False, n_partitions_depth=3,
                parallel_workers=4):
    """生长 forest.

    终止条件 (与 2DOF 一致, 加安全上限):
      - 主条件: 连续 max_miss 个 seed 未命中
      - 安全上限: 7D 空间太大, max_boxes 防止无限增长

    优化:
      A. 预计算 obs_packed，复用于所有 find_free_box / can_expand 调用
      B. FK 缓存预热 (warmup_fk_cache)
      C. shallow probe (can_expand) 减少无效 find_free_box
      E. 批量碰撞检测 (_sample_seeds_batch)
      F. 并行分区生长 (parallel_grow=True)

    Args:
      parallel_grow: 启用 ProcessPool 并行分区生长
      n_partitions_depth: KD 分区深度 (2^depth 个分区)
      parallel_workers: 进程池大小 (0=auto=cpu_count)

    返回 (boxes, forest, timing_detail)
      timing_detail: dict 各环节累积耗时 (ms)
    """
    rng = np.random.default_rng(seed)
    forest = planner._load_or_create_forest()
    forest.hier_tree = planner.hier_tree

    # ── 优化 B: FK 缓存预热 ──
    t_warmup_start = time.perf_counter()
    n_warmed = planner.hier_tree.warmup_fk_cache(max_depth=6)
    t_warmup = (time.perf_counter() - t_warmup_start) * 1000
    print(f"    [warmup] FK cache: {n_warmed} nodes ({t_warmup:.0f} ms)")

    # ── 优化 A: 预计算 obs_packed ──
    obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)

    # ═══════════════════════════════════════════════════════════════════
    # 优化 F: 并行分区生长 (ProcessPoolExecutor)
    # ═══════════════════════════════════════════════════════════════════
    if parallel_grow:
        t_par0 = time.perf_counter()
        partitions = build_kd_partitions(
            planner.joint_limits, n_partitions_depth)
        n_parts = len(partitions)
        per_part_boxes = max(10, max_boxes // n_parts)
        if parallel_workers > 0:
            workers = parallel_workers
        else:
            workers = max(1, (os.cpu_count() or 4))
        workers = min(workers, n_parts)  # 不需要超过分区数
        print(f"    [parallel grow] {n_parts} partitions, "
              f"{workers} workers, {per_part_boxes} boxes/part")

        payloads = []
        for pid, part_ivs in enumerate(partitions):
            payloads.append({
                "robot": planner.robot,
                "obstacles": planner.obstacles,
                "q_start": q_start,
                "q_goal": q_goal,
                "intervals": part_ivs,
                "partition_id": pid,
                "max_boxes": per_part_boxes,
                "max_miss": max_miss,
                "min_box_size": planner.config.min_box_size,
                "goal_bias": planner.config.goal_bias,
                "seed": int(rng.integers(0, 2**31 - 1)),
            })

        local_results: List[Dict] = []
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_grow_partition_worker, p) for p in payloads]
                for fut in as_completed(futs):
                    local_results.append(fut.result())
        except Exception as e:
            print(f"    [parallel grow] ProcessPool failed ({e}), "
                  f"fallback to serial")
            # 回退到串行: 在当前进程跑
            for p in payloads:
                local_results.append(_grow_partition_worker(p))

        # 合并
        partition_box_ids = forest.merge_partition_forests(local_results)
        t_par_ms = (time.perf_counter() - t_par0) * 1000

        n_total = sum(len(ids) for ids in partition_box_ids.values())
        print(f"    [parallel grow] merged {n_total} boxes, "
              f"{t_par_ms:.0f} ms total")

        timing = dict(
            warmup_ms=t_warmup,
            parallel_grow_ms=t_par_ms,
            n_partitions=n_parts,
            n_workers=workers,
        )
        boxes = {}
        for bid, b in forest.boxes.items():
            boxes[bid] = BoxNode(
                node_id=b.node_id,
                joint_intervals=[tuple(iv) for iv in b.joint_intervals],
                seed_config=b.seed_config.copy(), volume=b.volume)
        return boxes, forest, timing

    # ── 详细计时器 ──
    t_sample = 0.0      # _sample_seeds_batch (含碰撞检测)
    t_is_occ = 0.0      # is_occupied 查询
    t_probe = 0.0       # can_expand 浅层探测
    t_ffb = 0.0         # find_free_box (AABB 展开)
    t_add = 0.0         # add_box_direct + remove_boxes
    t_vol = 0.0         # 体积/min_box_size 判断
    n_sample_calls = 0
    n_is_occ_calls = 0
    n_probe_calls = 0
    n_probe_reject = 0
    n_ffb_calls = 0
    n_ffb_none = 0
    n_absorbed = 0

    # ── 优化 E: 批量采样预备 ──
    intervals = planner.joint_limits
    lows = np.array([lo for lo, _ in intervals], dtype=np.float64)
    highs = np.array([hi for _, hi in intervals], dtype=np.float64)
    goal_bias = planner.config.goal_bias
    guided_ratio = getattr(planner.config, 'guided_sample_ratio', 0.6)
    has_hier_tree = hasattr(planner, 'hier_tree') and planner.hier_tree is not None
    batch_size = 32  # 每批采样数

    def _sample_batch():
        """批量生成候选 seed，批量碰撞检测，返回无碰撞配置列表.

        向量化版本: goal-biased 和 uniform 分支用 NumPy 批量生成,
        guided 分支仍逐条调用 sample_unoccupied_seed.
        """
        rolls = rng.uniform(size=batch_size)
        configs = np.empty((batch_size, ndim), dtype=np.float64)

        # ── 分类索引 ──
        goal_mask = rolls < goal_bias
        if has_hier_tree:
            guided_mask = (~goal_mask) & (rolls < goal_bias + guided_ratio)
        else:
            guided_mask = np.zeros(batch_size, dtype=bool)
        uniform_mask = ~(goal_mask | guided_mask)

        # ── goal-biased: 批量生成 ──
        n_goal = int(goal_mask.sum())
        if n_goal > 0:
            noise = rng.normal(0.0, 0.3, size=(n_goal, ndim))
            configs[goal_mask] = np.clip(q_goal + noise, lows, highs)

        # ── uniform: 批量生成 ──
        n_uni = int(uniform_mask.sum())
        if n_uni > 0:
            configs[uniform_mask] = rng.uniform(lows, highs, size=(n_uni, ndim))

        # ── guided: 逐条 sample_unoccupied_seed ──
        guided_idxs = np.flatnonzero(guided_mask)
        for i in guided_idxs:
            try:
                q = planner.hier_tree.sample_unoccupied_seed(rng)
            except ValueError:
                q = None
            if q is None:
                q = rng.uniform(lows, highs)
            configs[i] = q

        collisions = planner.collision_checker.check_config_collision_batch(configs)
        return [configs[i] for i in range(batch_size) if not collisions[i]]

    # 预填充采样缓冲
    seed_buffer = []

    for qs in [q_start, q_goal]:
        if not planner.hier_tree.is_occupied(qs):
            nid = forest.allocate_id()
            ffb = planner.hier_tree.find_free_box(
                qs, planner.obstacles, mark_occupied=True, forest_box_id=nid,
                obs_packed=obs_packed)
            if ffb:
                vol = 1.0
                for lo, hi in ffb.intervals:
                    vol *= max(hi - lo, 0)
                forest.add_box_direct(BoxNode(
                    node_id=nid, joint_intervals=ffb.intervals,
                    seed_config=qs.copy(), volume=vol))

    # ── boundary expand state ──
    boundary_on = planner.config.boundary_expand_enabled
    boundary_max_fail = planner.config.boundary_expand_max_failures
    expand_target = None
    expand_fails = 0
    n_boundary_attempts = 0
    n_boundary_ok = 0
    t_boundary = 0.0

    consec = 0
    t0 = time.perf_counter()
    terminated_by = "miss"  # 记录终止原因
    while consec < max_miss:
        if forest.n_boxes >= max_boxes:
            terminated_by = "max_boxes"
            break

        # ── 采样模式: boundary expand vs 普通批量 ──
        use_boundary = boundary_on and expand_target is not None
        if use_boundary:
            _ts = time.perf_counter()
            q = planner._sample_boundary_seed(expand_target, rng)
            t_boundary += time.perf_counter() - _ts
            n_boundary_attempts += 1
            if q is None:
                expand_fails += 1
                if expand_fails >= boundary_max_fail:
                    expand_target = None
                continue
            if planner.hier_tree.is_occupied(q):
                expand_fails += 1
                if expand_fails >= boundary_max_fail:
                    expand_target = None
                continue
        else:
            # ── 优化 E: 批量采样 ──
            _ts = time.perf_counter()
            if not seed_buffer:
                seed_buffer = _sample_batch()
                n_sample_calls += 1
            if not seed_buffer:
                # 整批都碰撞 → 算一次 miss
                t_sample += time.perf_counter() - _ts
                consec += 1
                continue
            q = seed_buffer.pop()
            t_sample += time.perf_counter() - _ts

            # ── is_occupied ──
            _ts = time.perf_counter()
            occ = planner.hier_tree.is_occupied(q)
            t_is_occ += time.perf_counter() - _ts
            n_is_occ_calls += 1

            if occ:
                consec += 1
                continue

            # ── 优化 C: shallow probe ──
            _ts = time.perf_counter()
            can = planner.hier_tree.can_expand(q, obs_packed=obs_packed)
            t_probe += time.perf_counter() - _ts
            n_probe_calls += 1

            if not can:
                n_probe_reject += 1
                consec += 1
                continue

        # ── find_free_box (with cached obs_packed) ──
        nid = forest.allocate_id()
        _ts = time.perf_counter()
        ffb = planner.hier_tree.find_free_box(
            q, planner.obstacles, mark_occupied=True, forest_box_id=nid,
            obs_packed=obs_packed)
        t_ffb += time.perf_counter() - _ts
        n_ffb_calls += 1

        if ffb is None:
            n_ffb_none += 1
            consec += 1
            continue

        # ── volume check ──
        _ts = time.perf_counter()
        vol = 1.0
        for lo, hi in ffb.intervals:
            vol *= max(hi - lo, 0)
        too_small = gmean_edge_length(vol, ndim) < planner.config.min_box_size
        t_vol += time.perf_counter() - _ts

        if too_small:
            consec += 1
            continue

        # ── add box ──
        _ts = time.perf_counter()
        box = BoxNode(node_id=nid, joint_intervals=ffb.intervals,
                      seed_config=q.copy(), volume=vol)
        if ffb.absorbed_box_ids:
            n_absorbed += len(ffb.absorbed_box_ids)
            forest.remove_boxes(ffb.absorbed_box_ids)
        forest.add_box_direct(box)
        t_add += time.perf_counter() - _ts

        # ── trigger boundary expand ──
        if boundary_on:
            expand_target = box
            expand_fails = 0
            n_boundary_ok += 1

        consec = 0
        # progress
        if forest.n_boxes % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    [grow] {forest.n_boxes} boxes, {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"    [grow] terminated by {terminated_by}: "
          f"{forest.n_boxes} boxes, {elapsed:.1f}s")

    # ── 后处理: 若 s/t 孤立, 双向 "stepping stone" 铺路连通 ──
    # 沿 s/t↔主岛方向以小步长密集播种, 并从两端同时推进.
    if boundary_on:
        _ts_connect = time.perf_counter()
        n_gap_fill = 0
        _gap_rounds = 3

        def _box_center(bx):
            return np.array([(lo + hi) / 2
                             for lo, hi in bx.joint_intervals])

        def _find_st_bids(snap):
            sb = tb = None
            for bid, bx in snap.items():
                if sb is None and all(
                        bx.joint_intervals[d][0] - 1e-12 <= q_start[d]
                        <= bx.joint_intervals[d][1] + 1e-12
                        for d in range(ndim)):
                    sb = bid
                if tb is None and all(
                        bx.joint_intervals[d][0] - 1e-12 <= q_goal[d]
                        <= bx.joint_intervals[d][1] + 1e-12
                        for d in range(ndim)):
                    tb = bid
            return sb, tb

        def _grow_one_box(q):
            """尝试在 q 处生长 box, 返回 BoxNode 或 None.
            跳过 is_occupied 检查, 允许在已有 box 附近放置新 box.
            """
            if planner.collision_checker.check_config_collision(q):
                return None
            nid = forest.allocate_id()
            ffb = planner.hier_tree.find_free_box(
                q, planner.obstacles, mark_occupied=True,
                forest_box_id=nid, obs_packed=obs_packed)
            if ffb is None:
                return None
            vol = 1.0
            for lo, hi in ffb.intervals:
                vol *= max(hi - lo, 0)
            if gmean_edge_length(vol, ndim) < planner.config.min_box_size:
                return None
            box = BoxNode(node_id=nid, joint_intervals=ffb.intervals,
                          seed_config=q.copy(), volume=vol)
            if ffb.absorbed_box_ids:
                forest.remove_boxes(ffb.absorbed_box_ids)
            forest.add_box_direct(box)
            return box

        def _pave_toward(start_c, end_c, max_steps=300):
            """沿 start→end 方向以小步长密集播种."""
            direction = end_c - start_c
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                return 0
            direction /= dist
            pos = start_c.copy()
            n_placed = 0
            step_idx = 0
            while step_idx < max_steps:
                d_remain = np.linalg.norm(end_c - pos)
                if d_remain < 0.05:
                    break
                placed = False
                # 尝试在 pos 及其邻域放 box
                for attempt in range(15):
                    if attempt == 0:
                        q = pos.copy()
                    else:
                        q = pos + rng.normal(0, 0.05, size=ndim)
                    for d in range(ndim):
                        q[d] = np.clip(q[d],
                                       planner.joint_limits[d][0],
                                       planner.joint_limits[d][1])
                    box = _grow_one_box(q)
                    if box is not None:
                        n_placed += 1
                        # 自适应步长: box 最小边长的一半
                        min_edge = min(hi - lo
                                       for lo, hi in box.joint_intervals)
                        step_size = max(0.01, min(min_edge * 0.4, 0.1))
                        pos = pos + direction * step_size
                        placed = True
                        break
                if not placed:
                    # 跳过碰撞区域
                    pos = pos + direction * 0.03
                step_idx += 1
            return n_placed

        for _gr in range(_gap_rounds):
            _snap = {}
            for bid, b in forest.boxes.items():
                _snap[bid] = BoxNode(
                    node_id=b.node_id,
                    joint_intervals=[tuple(iv) for iv in b.joint_intervals],
                    seed_config=b.seed_config.copy(), volume=b.volume)
            _, _uf, _isls = _build_adjacency_and_islands(_snap)

            _src_bid, _tgt_bid = _find_st_bids(_snap)
            if _src_bid is None or _tgt_bid is None:
                break
            if _uf.same(_src_bid, _tgt_bid):
                break

            for _st_bid, _qs in [(_src_bid, q_start), (_tgt_bid, q_goal)]:
                _st_isl = set()
                for isl in _isls:
                    if _st_bid in isl:
                        _st_isl = isl
                        break
                _st_c = _box_center(_snap[_st_bid])

                # 找最近的非本岛 box
                _best_d = float('inf')
                _tgt_c = None
                for bid, bx in _snap.items():
                    if bid in _st_isl:
                        continue
                    mc = _box_center(bx)
                    dd = np.linalg.norm(mc - _st_c)
                    if dd < _best_d:
                        _best_d = dd
                        _tgt_c = mc
                if _tgt_c is None:
                    continue

                # 双向铺路: 从 s/t 向主岛 + 从主岛向 s/t
                n1 = _pave_toward(_st_c, _tgt_c, max_steps=200)
                n2 = _pave_toward(_tgt_c, _st_c, max_steps=200)
                n_gap_fill += n1 + n2

        gap_ms = (time.perf_counter() - _ts_connect) * 1000
        t_boundary += (time.perf_counter() - _ts_connect)
        if n_gap_fill > 0:
            print(f"    [gap fill] +{n_gap_fill} boxes in {_gr + 1} rounds, "
                  f"total={forest.n_boxes} ({gap_ms:.0f}ms)")
        elif _gr > 0:
            print(f"    [gap fill] s/t disconnected, 0 gap boxes ({gap_ms:.0f}ms)")


    # ── 详细计时报告 ──
    timing = dict(
        warmup_ms=t_warmup,
        sample_ms=t_sample * 1000,
        boundary_ms=t_boundary * 1000,
        is_occupied_ms=t_is_occ * 1000,
        probe_ms=t_probe * 1000,
        find_free_box_ms=t_ffb * 1000,
        volume_check_ms=t_vol * 1000,
        add_box_ms=t_add * 1000,
        overhead_ms=(elapsed - t_sample - t_boundary - t_is_occ - t_probe - t_ffb - t_vol - t_add) * 1000,
        n_sample_calls=n_sample_calls,
        n_is_occ_calls=n_is_occ_calls,
        n_probe_calls=n_probe_calls,
        n_probe_reject=n_probe_reject,
        n_ffb_calls=n_ffb_calls,
        n_ffb_none=n_ffb_none,
        n_absorbed=n_absorbed,
    )
    print(f"    [grow detail]")
    print(f"      warmup_fk       : {timing['warmup_ms']:8.1f} ms  "
          f"({n_warmed} nodes)")
    print(f"      sample_batch    : {timing['sample_ms']:8.1f} ms  "
          f"({n_sample_calls} batches)")
    print(f"      boundary_expand : {timing['boundary_ms']:8.1f} ms  "
          f"({n_boundary_ok}/{n_boundary_attempts} ok)")
    print(f"      is_occupied     : {timing['is_occupied_ms']:8.1f} ms  "
          f"({n_is_occ_calls} calls)")
    print(f"      can_expand      : {timing['probe_ms']:8.1f} ms  "
          f"({n_probe_calls} calls, {n_probe_reject} rejected)")
    print(f"      find_free_box   : {timing['find_free_box_ms']:8.1f} ms  "
          f"({n_ffb_calls} calls, {n_ffb_none} none)")
    print(f"      volume_check    : {timing['volume_check_ms']:8.1f} ms")
    print(f"      add_box         : {timing['add_box_ms']:8.1f} ms  "
          f"({n_absorbed} absorbed)")
    print(f"      overhead/other  : {timing['overhead_ms']:8.1f} ms")

    boxes = {}
    for bid, b in forest.boxes.items():
        boxes[bid] = BoxNode(
            node_id=b.node_id,
            joint_intervals=[tuple(iv) for iv in b.joint_intervals],
            seed_config=b.seed_config.copy(), volume=b.volume)
    return boxes, forest, timing


# ═══════════════════════════════════════════════════════════════════════════
# Shared infrastructure: adjacency + islands
# ═══════════════════════════════════════════════════════════════════════════

def _build_adjacency_and_islands(boxes):
    """O(N²) pairwise overlap → adjacency dict + UnionFind + islands.

    一次遍历同时得到邻接图和连通分量, 替代原先分别做的
    find_islands() + _build_full_adjacency().

    使用 NumPy 广播向量化: 将 N 个 box 的 lo/hi 拼成 (N,D) 矩阵,
    一次广播计算 (N,N) overlap 矩阵, 避免 Python 逐对循环.

    注意: 此处使用宽松的 overlap 条件 (所有维度区间有交集),
    与 compute_adjacency 的严格 face-touching 条件不同.
    Bridge 和 Dijkstra 需要宽松条件才能正确找路.
    """
    ids = list(boxes.keys())
    n = len(ids)
    adj: Dict[int, Set[int]] = {bid: set() for bid in ids}
    uf = UnionFind(ids)

    if n < 2:
        return adj, uf, uf.components()

    # 向量化: 提取所有 box 的 lo / hi 为 (N, D) 数组
    ndim = next(iter(boxes.values())).n_dims
    lo = np.empty((n, ndim), dtype=np.float64)
    hi = np.empty((n, ndim), dtype=np.float64)
    for k, bid in enumerate(ids):
        ivs = boxes[bid].joint_intervals
        for d in range(ndim):
            lo[k, d] = ivs[d][0]
            hi[k, d] = ivs[d][1]

    # 广播: overlap[i,j] = all_dims( hi[i,d] >= lo[j,d] - eps  AND  hi[j,d] >= lo[i,d] - eps )
    # 放宽 eps 使得几乎相邻的 box 也判为连通 (近触碰)
    eps = 1e-9
    # (N,1,D) vs (1,N,D) → (N,N,D)
    overlap_ij = (hi[:, None, :] >= lo[None, :, :] - eps) & \
                 (hi[None, :, :] >= lo[:, None, :] - eps)
    overlap_all = np.all(overlap_ij, axis=2)  # (N, N) bool

    # 只取上三角
    ii, jj = np.where(np.triu(overlap_all, k=1))
    for idx in range(len(ii)):
        bi, bj = ids[ii[idx]], ids[jj[idx]]
        adj[bi].add(bj)
        adj[bj].add(bi)
        uf.union(bi, bj)

    islands = uf.components()
    return adj, uf, islands


def _add_bridge_to_adj(adj, bridge_edges, uf=None):
    """把 bridge edges 加入已有邻接图."""
    for e in bridge_edges:
        s, t = e.source_box_id, e.target_box_id
        if s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)
            if uf is not None:
                uf.union(s, t)


# ═══════════════════════════════════════════════════════════════════════════
# Solver A: GCS SOCP (existing, 凸松弛 + rounding + refine)
# ═══════════════════════════════════════════════════════════════════════════

def _solve_method_gcs(boxes, adj, src, tgt, q_start, q_goal, ndim,
                      corridor_hops=2, label="GCS"):
    """GCS SOCP 求解 (走廊剪枝 + 凸松弛 + rounding + refine)."""
    t0 = time.perf_counter()
    success, cost, waypoints, box_seq = solve_gcs(
        boxes, adj, src, tgt, q_start, q_goal, ndim,
        corridor_hops=corridor_hops)
    ms = (time.perf_counter() - t0) * 1000
    return dict(method=label, success=success, cost=cost,
                waypoints=waypoints, box_seq=box_seq, plan_ms=ms)


# ═══════════════════════════════════════════════════════════════════════════
# Solver B: Dijkstra on box graph + SOCP refine
# ═══════════════════════════════════════════════════════════════════════════

def _dijkstra_box_graph(boxes, adj, src, tgt):
    """Dijkstra 最短路 (edge weight = center-to-center distance).

    Returns (box_sequence, dist) or (None, inf).
    """
    # center 缓存
    centers = {}
    for bid, box in boxes.items():
        centers[bid] = np.array([(lo + hi) / 2 for lo, hi in box.joint_intervals])

    dist_map: Dict[int, float] = {bid: float('inf') for bid in boxes}
    prev_map: Dict[int, Optional[int]] = {bid: None for bid in boxes}
    dist_map[src] = 0.0
    # (distance, box_id)
    heap = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist_map[u]:
            continue
        if u == tgt:
            break
        cu = centers[u]
        for v in adj.get(u, set()):
            w = float(np.linalg.norm(cu - centers[v]))
            nd = d + w
            if nd < dist_map[v]:
                dist_map[v] = nd
                prev_map[v] = u
                heapq.heappush(heap, (nd, v))

    if dist_map[tgt] == float('inf'):
        return None, float('inf')

    # 回溯路径
    seq = []
    cur = tgt
    while cur is not None:
        seq.append(cur)
        cur = prev_map[cur]
    seq.reverse()
    return seq, dist_map[tgt]


def _solve_method_dijkstra(boxes, adj, src, tgt, q_start, q_goal, ndim,
                           label="Dijkstra"):
    """Dijkstra on box graph → box sequence → SOCP refine."""
    t0 = time.perf_counter()

    box_seq, raw_dist = _dijkstra_box_graph(boxes, adj, src, tgt)
    if box_seq is None:
        ms = (time.perf_counter() - t0) * 1000
        print(f"    [{label}] Dijkstra: no path found")
        return dict(method=label, success=False, cost=float('inf'),
                    waypoints=[], box_seq=[], plan_ms=ms)

    # 初始 waypoints: start → box centers → goal
    waypoints = [q_start.copy()]
    for bid in box_seq[1:-1]:
        box = boxes[bid]
        c = np.array([(lo + hi) / 2 for lo, hi in box.joint_intervals])
        waypoints.append(c)
    waypoints.append(q_goal.copy())

    # SOCP refine
    refined_wps, refined_cost = _refine_path_in_boxes(
        waypoints, box_seq, boxes, q_start, q_goal, ndim)

    ms = (time.perf_counter() - t0) * 1000
    print(f"    [{label}] {len(box_seq)} boxes, raw_dist={raw_dist:.4f}, "
          f"refined={refined_cost:.4f}, {len(refined_wps)} wp ({ms:.0f}ms)")
    return dict(method=label, success=True, cost=refined_cost,
                waypoints=refined_wps, box_seq=box_seq, plan_ms=ms)


# ═══════════════════════════════════════════════════════════════════════════
# Solver C: Visibility Graph (seed configs + collision check)
# ═══════════════════════════════════════════════════════════════════════════

def _solve_method_visgraph(boxes, q_start, q_goal, collision_checker,
                           segment_resolution=0.05, label="VisGraph"):
    """可视图法: 以 box seed configs 为节点, 碰撞检测为边, Dijkstra 求解.

    不需要 bridge / adjacency — 直接在 free-space 中连接.
    """
    t0 = time.perf_counter()

    # 节点: start + goal + all box seeds
    node_ids = ['start', 'goal']  # 特殊 id
    node_configs = [q_start.copy(), q_goal.copy()]
    for bid, box in boxes.items():
        if box.seed_config is not None:
            node_ids.append(bid)
            node_configs.append(np.asarray(box.seed_config, dtype=np.float64))

    n_nodes = len(node_ids)
    print(f"    [{label}] {n_nodes} nodes, checking edges ...")

    # 构建 visibility graph: O(N²) segment collision checks
    # 使用 numpy 批量距离排序, 只检测最近的 k 个邻居减少碰撞检测次数
    configs_arr = np.array(node_configs)  # (n, ndim)
    k_nearest = min(50, n_nodes - 1)  # 最多检查最近 50 个邻居

    vis_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_nodes)}
    n_checks = 0
    n_edges = 0

    for i in range(n_nodes):
        qi = configs_arr[i]
        # 计算到所有其他节点的距离
        diffs = configs_arr - qi  # (n, ndim)
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = float('inf')  # 排除自身

        # 只检查最近的 k 个
        nearest_idxs = np.argpartition(dists, min(k_nearest, n_nodes - 2))[:k_nearest]
        nearest_idxs = nearest_idxs[np.argsort(dists[nearest_idxs])]

        for j_idx in nearest_idxs:
            j = int(j_idx)
            d = float(dists[j])
            if d == float('inf'):
                continue
            # 跳过已添加的反向边
            if any(nbr == i for nbr, _ in vis_adj[j]):
                # 已经有 j→i 边, 直接加 i→j
                vis_adj[i].append((j, d))
                n_edges += 1
                continue
            n_checks += 1
            if not collision_checker.check_segment_collision(
                    configs_arr[i], configs_arr[j], segment_resolution):
                vis_adj[i].append((j, d))
                vis_adj[j].append((i, d))
                n_edges += 2

    t_graph = (time.perf_counter() - t0) * 1000
    print(f"    [{label}] graph: {n_edges // 2} edges, "
          f"{n_checks} collision checks ({t_graph:.0f}ms)")

    # Dijkstra on vis_adj (index-based)
    dist_map = [float('inf')] * n_nodes
    prev_map = [-1] * n_nodes
    dist_map[0] = 0.0  # start = index 0
    heap = [(0.0, 0)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist_map[u]:
            continue
        if u == 1:  # goal = index 1
            break
        for v, w in vis_adj[u]:
            nd = d + w
            if nd < dist_map[v]:
                dist_map[v] = nd
                prev_map[v] = u
                heapq.heappush(heap, (nd, v))

    if dist_map[1] == float('inf'):
        ms = (time.perf_counter() - t0) * 1000
        print(f"    [{label}] no path found ({ms:.0f}ms)")
        return dict(method=label, success=False, cost=float('inf'),
                    waypoints=[], box_seq=[], plan_ms=ms)

    # 回溯
    path_idxs = []
    cur = 1
    while cur != -1:
        path_idxs.append(cur)
        cur = prev_map[cur]
    path_idxs.reverse()

    raw_waypoints = [node_configs[i].copy() for i in path_idxs]
    raw_cost = dist_map[1]

    # Greedy shortcut: 尝试跳过中间点
    shortcut_wps = _greedy_shortcut(raw_waypoints, collision_checker,
                                     segment_resolution)

    final_cost = sum(float(np.linalg.norm(
        shortcut_wps[i + 1] - shortcut_wps[i]))
        for i in range(len(shortcut_wps) - 1))

    ms = (time.perf_counter() - t0) * 1000
    print(f"    [{label}] raw={raw_cost:.4f} ({len(raw_waypoints)} wp) "
          f"→ shortcut={final_cost:.4f} ({len(shortcut_wps)} wp) ({ms:.0f}ms)")
    return dict(method=label, success=True, cost=final_cost,
                waypoints=shortcut_wps, box_seq=[], plan_ms=ms)


def _greedy_shortcut(waypoints, collision_checker, resolution):
    """贪心路径缩短: 依次尝试跳过中间点."""
    if len(waypoints) <= 2:
        return waypoints
    result = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        # 尝试跳到尽可能远的点
        farthest = i + 1
        for j in range(len(waypoints) - 1, i + 1, -1):
            if not collision_checker.check_segment_collision(
                    waypoints[i], waypoints[j], resolution):
                farthest = j
                break
        result.append(waypoints[farthest])
        i = farthest
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline: shared forest growth + per-method planning
# ═══════════════════════════════════════════════════════════════════════════

def grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim, no_cache=False):
    """共享前半段: grow → cache (异步) → coarsen.

    cache save 在后台线程执行, 与 coarsen 及后续规划并行.
    返回的 dict 包含 _cache_thread, 调用方应在不再需要 hier_tree
    写入时 join.

    Args:
        no_cache: 如果为 True, 不加载也不保存磁盘缓存.
    """
    planner_cfg = make_planner_config(cfg)
    planner = BoxPlanner(robot=robot, scene=scene, config=planner_cfg,
                         no_cache=no_cache)

    t0 = time.perf_counter()
    boxes, forest_obj, grow_detail = grow_forest(
        planner, q_start, q_goal, cfg.seed,
        cfg.max_consecutive_miss, ndim,
        max_boxes=cfg.max_boxes,
        parallel_grow=cfg.parallel_grow,
        n_partitions_depth=cfg.n_partitions_depth,
        parallel_workers=cfg.parallel_workers)
    grow_ms = (time.perf_counter() - t0) * 1000
    n_grown = len(forest_obj.boxes)
    n_nodes = planner.hier_tree.n_nodes

    # ── AABB cache: 后台线程保存 ──
    cache_result = {}  # 存放线程结果
    cache_thread = None

    if not no_cache:
        def _save_cache():
            _t0 = time.perf_counter()
            _path = planner.hier_tree.auto_save()
            _ms = (time.perf_counter() - _t0) * 1000
            cache_result['path'] = _path
            cache_result['ms'] = _ms

        cache_thread = threading.Thread(target=_save_cache, daemon=True)
        cache_thread.start()
        print(f"    [cache] saving {n_nodes} nodes in background thread ...")
    else:
        print(f"    [cache] skipped (no_cache mode), {n_nodes} nodes")

    # ── coarsen (与 cache save 并行) ──
    n_before_coarsen = len(forest_obj.boxes)
    coarsen_stats = coarsen_forest(
        tree=planner.hier_tree, forest=forest_obj,
        obstacles=planner.obstacles, safety_margin=0.0,
        max_rounds=cfg.coarsen_max_rounds,
    )
    n_after_coarsen = len(forest_obj.boxes)
    coarsen_ms = coarsen_stats.time_ms
    print(f"    [coarsen] {n_before_coarsen} -> {n_after_coarsen} boxes "
          f"({coarsen_stats.n_merges} merges in {coarsen_stats.n_rounds} rounds, "
          f"{coarsen_ms:.0f}ms)")

    boxes = forest_obj.boxes
    return dict(
        planner=planner, boxes=boxes, forest_obj=forest_obj,
        grow_ms=grow_ms, cache_ms=0.0, coarsen_ms=coarsen_ms,
        coarsen_stats=coarsen_stats, n_grown=n_grown,
        n_cache_nodes=n_nodes,
        grow_detail=grow_detail,
        _cache_thread=cache_thread,
        _cache_result=cache_result,
    )


def run_method_with_bridge(method_fn, method_name, prep, cfg, q_start, q_goal,
                           ndim, **method_kwargs):
    """运行需要 adjacency 的方法 (GCS / Dijkstra).

    Lazy bridge: 先看 s-t 是否已连通, 否则只桥接 s/t 所在的两个岛.
    """
    boxes = prep['boxes']
    planner = prep['planner']
    forest_obj = prep['forest_obj']

    # adjacency + islands (一次 O(N²) overlap)
    t0 = time.perf_counter()
    adj, uf, islands = _build_adjacency_and_islands(boxes)
    adj_ms = (time.perf_counter() - t0) * 1000
    n_edges = sum(len(v) for v in adj.values()) // 2
    print(f"    [adj] {len(adj)} vertices, {n_edges} edges ({adj_ms:.0f}ms)")

    src = find_box_containing(q_start, boxes)
    tgt = find_box_containing(q_goal, boxes)
    if src is None or tgt is None:
        print(f"    [{method_name}] ERROR: start or goal not in any box")
        return None

    n_before_islands = len(islands)
    bridge_ms = 0.0
    bridge_edges = []
    bridge_boxes_list = []

    if not uf.same(src, tgt):
        # 需要桥接 — 只桥接 s-t 所在岛 (优化 A/C: 传入 precomputed + target_pair)
        print(f"    [{method_name}] s-t disconnected ({n_before_islands} islands), bridging ...")
        t0 = time.perf_counter()
        bridge_result = bridge_islands(
            boxes=boxes,
            collision_checker=planner.collision_checker,
            segment_resolution=0.03,
            max_pairs_per_island_pair=10,
            max_rounds=5,
            period=None,
            hier_tree=planner.hier_tree,
            obstacles=planner.obstacles,
            forest=forest_obj,
            min_box_size=cfg.min_box_size,
            n_bridge_seeds=7,
            min_island_size=cfg.min_island_size,
            precomputed_uf=uf,
            precomputed_islands=islands,
            target_pair=(src, tgt),
        )
        bridge_edges, final_islands, _, bridge_boxes_res, discarded = bridge_result
        bridge_ms = (time.perf_counter() - t0) * 1000
        bridge_boxes_list = bridge_boxes_res
        boxes = forest_obj.boxes
        # 更新 adj with bridge edges + new bridge boxes
        # B3: 使用 forest 的向量化 interval cache 查邻接, 替代 Python 逐对 overlap
        for bb in bridge_boxes_list:
            if bb.node_id not in adj:
                neighbor_ids = forest_obj._adjacent_existing_ids_from_cache(
                    bb, tol=forest_obj.config.adjacency_tolerance)
                adj[bb.node_id] = set(neighbor_ids)
                for nb in neighbor_ids:
                    adj.setdefault(nb, set()).add(bb.node_id)
        _add_bridge_to_adj(adj, bridge_edges, uf)
        n_after_islands = len(uf.components())
        print(f"    [bridge] islands: {n_before_islands} -> {n_after_islands}  "
              f"({len(bridge_edges)} edges, {len(bridge_boxes_list)} bridge-boxes, "
              f"{bridge_ms:.0f} ms)")
        # 更新 src/tgt in case bridge added new boxes
        if src not in boxes:
            src = find_box_containing(q_start, boxes)
        if tgt not in boxes:
            tgt = find_box_containing(q_goal, boxes)
    else:
        print(f"    [{method_name}] s-t already connected! Skipping bridge.")
        n_after_islands = n_before_islands

    if src is None or tgt is None:
        print(f"    [{method_name}] ERROR: start or goal not in any box after bridge")
        return None

    # 运行规划方法
    plan_result = method_fn(
        boxes=boxes, adj=adj, src=src, tgt=tgt,
        q_start=q_start, q_goal=q_goal, ndim=ndim,
        label=method_name, **method_kwargs)

    # 补充共享时间信息
    plan_result.update(
        boxes=dict(boxes),
        adj=adj,
        n_before_islands=n_before_islands,
        n_after_islands=n_after_islands,
        bridge_edges=bridge_edges,
        bridge_boxes=bridge_boxes_list,
        adj_ms=adj_ms,
        bridge_ms=bridge_ms,
        grow_ms=prep['grow_ms'],
        cache_ms=prep['cache_ms'],
        coarsen_ms=prep['coarsen_ms'],
        coarsen_stats=prep['coarsen_stats'],
        n_grown=prep['n_grown'],
        n_cache_nodes=prep['n_cache_nodes'],
    )
    return plan_result


def run_method_visgraph(prep, cfg, q_start, q_goal, collision_checker, ndim):
    """运行 Visibility Graph 方法 (不需要 bridge / adjacency)."""
    boxes = prep['boxes']
    plan_result = _solve_method_visgraph(
        boxes, q_start, q_goal, collision_checker,
        segment_resolution=0.05, label="VisGraph")
    plan_result.update(
        boxes=dict(boxes),
        adj={},
        n_before_islands=0,
        n_after_islands=0,
        bridge_edges=[],
        bridge_boxes=[],
        adj_ms=0.0,
        bridge_ms=0.0,
        grow_ms=prep['grow_ms'],
        cache_ms=prep['cache_ms'],
        coarsen_ms=prep['coarsen_ms'],
        coarsen_stats=prep['coarsen_stats'],
        n_grown=prep['n_grown'],
        n_cache_nodes=prep['n_cache_nodes'],
    )
    return plan_result


# ═══════════════════════════════════════════════════════════════════════════
# Summary + Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_joint_trajectory(waypoints, q_start, q_goal, label="", joint_names=None):
    """画 7 条 joint 曲线 → 交互式 plotly HTML."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not waypoints:
        return None
    wps = np.array(waypoints)
    ndim = wps.shape[1]
    fig = make_subplots(
        rows=ndim, cols=1, shared_xaxes=True,
        subplot_titles=[joint_names[d] if joint_names else f"q{d}"
                        for d in range(ndim)],
        vertical_spacing=0.02,
    )
    xs = list(range(len(wps)))
    for d in range(ndim):
        row = d + 1
        fig.add_trace(go.Scatter(
            x=xs, y=wps[:, d].tolist(), mode='lines+markers',
            marker=dict(size=4), line=dict(color='steelblue', width=1.5),
            name=joint_names[d] if joint_names else f"q{d}",
            showlegend=(d == 0),
        ), row=row, col=1)
        fig.add_hline(y=float(q_start[d]), line=dict(color='green', dash='dash', width=1),
                      row=row, col=1)
        fig.add_hline(y=float(q_goal[d]), line=dict(color='red', dash='dash', width=1),
                      row=row, col=1)
    fig.update_layout(
        title=dict(text=f"Joint Trajectory — {label}", font=dict(size=14)),
        height=180 * ndim, width=900,
    )
    fig.update_xaxes(title_text="Waypoint index", row=ndim, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Interactive 3D Visualization (plotly)
# ═══════════════════════════════════════════════════════════════════════════

def _plotly_box_mesh(mn, mx, color='red', opacity=0.40, name='obstacle'):
    """返回一个 plotly Mesh3d 半透明 AABB."""
    import plotly.graph_objects as go
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])
    # 8 个顶点 (标准 AABB 顶点排列)
    #  0:(x0,y0,z0)  1:(x1,y0,z0)  2:(x1,y1,z0)  3:(x0,y1,z0)
    #  4:(x0,y0,z1)  5:(x1,y0,z1)  6:(x1,y1,z1)  7:(x0,y1,z1)
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]
    # 12 三角面 (6 面各 2 三角)
    # bottom (z0):  (0,1,2),(0,2,3)
    # top    (z1):  (4,5,6),(4,6,7)
    # front  (y0):  (0,1,5),(0,5,4)
    # back   (y1):  (2,3,7),(2,7,6)
    # left   (x0):  (0,3,7),(0,7,4)
    # right  (x1):  (1,2,6),(1,6,5)
    i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]
    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=i, j=j, k=k,
        color=color, opacity=opacity, name=name,
        hoverinfo='name', flatshading=True,
    )


def _plotly_arm_traces(positions, color='#2196F3', width=5, name='arm',
                       opacity=1.0, showlegend=True):
    """返回 plotly traces (线 + 关节点 + 基座 + 末端)."""
    import plotly.graph_objects as go
    xs = [float(p[0]) for p in positions]
    ys = [float(p[1]) for p in positions]
    zs = [float(p[2]) for p in positions]
    traces = []
    # 连杆链
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines+markers',
        line=dict(color=color, width=width),
        marker=dict(size=3, color=color, opacity=opacity),
        name=name, opacity=opacity, showlegend=showlegend,
        hoverinfo='name+text',
        text=[f"Link {i}" for i in range(len(xs))],
    ))
    # 基座
    traces.append(go.Scatter3d(
        x=[xs[0]], y=[ys[0]], z=[zs[0]], mode='markers',
        marker=dict(size=6, color='black', symbol='square', opacity=opacity),
        name=f'{name} base', showlegend=False,
    ))
    # 末端
    traces.append(go.Scatter3d(
        x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode='markers',
        marker=dict(size=5, color=color, symbol='diamond', opacity=opacity),
        name=f'{name} EE', showlegend=False,
    ))
    return traces


def _plotly_obstacle_traces(scene):
    """返回场景中所有障碍物的 Mesh3d traces."""
    traces = []
    if scene is None:
        return traces
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        if abs(mx[2] - mn[2]) > 100:
            continue
        traces.append(_plotly_box_mesh(mn, mx, color='red', opacity=0.40,
                                       name=obs.name))
    return traces


def _obstacle_corner_pts(scene):
    """收集所有障碍物的 min/max 角点, 用于计算坐标范围."""
    pts = []
    if scene is None:
        return pts
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        if abs(mx[2] - mn[2]) > 100:
            continue
        pts.append(mn)
        pts.append(mx)
    return pts


def _plotly_scene_layout(title="Panda Arm", all_pts=None):
    """返回 3D scene 通用 layout."""
    layout = dict(
        title=dict(text=title, font=dict(size=15)),
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=1000, height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if all_pts is not None and len(all_pts) > 0:
        xs = [float(p[0]) for p in all_pts]
        ys = [float(p[1]) for p in all_pts]
        zs = [float(p[2]) for p in all_pts]
        margin = 0.15
        layout['scene']['xaxis'] = dict(range=[min(xs)-margin, max(xs)+margin], title='X (m)')
        layout['scene']['yaxis'] = dict(range=[min(ys)-margin, max(ys)+margin], title='Y (m)')
        layout['scene']['zaxis'] = dict(range=[min(min(zs)-margin, -0.05), max(zs)+margin], title='Z (m)')
    return layout


def plot_arm_scene_html(robot, scene, q_start, q_goal, waypoints=None,
                        title="Panda Arm — Start / Goal"):
    """交互式 3D 场景: 障碍物 + 始末臂型 + 末端轨迹 → plotly Figure."""
    import plotly.graph_objects as go

    traces = _plotly_obstacle_traces(scene)

    pos_start = robot.get_link_positions(np.asarray(q_start, dtype=np.float64))
    pos_goal = robot.get_link_positions(np.asarray(q_goal, dtype=np.float64))
    traces += _plotly_arm_traces(pos_start, color='#4CAF50', width=6, name='Start')
    traces += _plotly_arm_traces(pos_goal, color='#F44336', width=6, name='Goal')

    all_pts = list(pos_start) + list(pos_goal) + _obstacle_corner_pts(scene)

    if waypoints is not None and len(waypoints) >= 2:
        ee_positions = []
        for q in waypoints:
            pos = robot.get_link_positions(np.asarray(q, dtype=np.float64))
            ee_positions.append(pos[-1])
            all_pts.extend(pos)
        ee_x = [float(p[0]) for p in ee_positions]
        ee_y = [float(p[1]) for p in ee_positions]
        ee_z = [float(p[2]) for p in ee_positions]
        traces.append(go.Scatter3d(
            x=ee_x, y=ee_y, z=ee_z, mode='lines+markers',
            line=dict(color='#FF9800', width=4),
            marker=dict(size=2, color='#FF9800'),
            name='EE path', opacity=0.8,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_plotly_scene_layout(title, all_pts))
    return fig


def plot_arm_poses_html(robot, scene, waypoints, n_ghosts=10,
                        title="Panda GCS — Arm Pose Sequence"):
    """交互式 3D 多臂型残影 → plotly Figure."""
    import plotly.graph_objects as go

    traces = _plotly_obstacle_traces(scene)

    n = len(waypoints)
    if n < 2:
        fig = go.Figure(data=traces)
        fig.update_layout(**_plotly_scene_layout(title))
        return fig

    ghost_idxs = np.linspace(0, n - 1, min(n_ghosts, n), dtype=int)

    # coolwarm 色谱: 蓝→红
    import matplotlib.cm as mcm
    colors_rgba = [mcm.coolwarm(float(i) / max(len(ghost_idxs) - 1, 1))
                   for i in range(len(ghost_idxs))]

    all_pts = _obstacle_corner_pts(scene)
    for k, idx in enumerate(ghost_idxs):
        q = np.asarray(waypoints[idx], dtype=np.float64)
        pos = robot.get_link_positions(q)
        rgba = colors_rgba[k]
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        opacity = 0.3 if 0 < k < len(ghost_idxs) - 1 else 0.95
        width = 3 if 0 < k < len(ghost_idxs) - 1 else 7
        label = f"t={idx}"
        if k == 0:
            label = 'Start'
        elif k == len(ghost_idxs) - 1:
            label = 'Goal'
        traces += _plotly_arm_traces(pos, color=hex_color, width=width,
                                     name=label, opacity=opacity,
                                     showlegend=(k == 0 or k == len(ghost_idxs)-1))
        all_pts.extend(pos)

    # 末端轨迹
    ee_positions = []
    for q in waypoints:
        pos = robot.get_link_positions(np.asarray(q, dtype=np.float64))
        ee_positions.append(pos[-1])
    ee_x = [float(p[0]) for p in ee_positions]
    ee_y = [float(p[1]) for p in ee_positions]
    ee_z = [float(p[2]) for p in ee_positions]
    traces.append(go.Scatter3d(
        x=ee_x, y=ee_y, z=ee_z, mode='lines',
        line=dict(color='#FF9800', width=5),
        name='EE trajectory', opacity=0.7,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_plotly_scene_layout(title, all_pts))
    return fig


def create_animation_html(robot, scene, waypoints, n_frames=60,
                          title="Panda GCS Path Animation"):
    """交互式 3D 动画 (plotly frames) — 自动循环播放, 可拖动视角.

    每帧画当前臂型 + 末端轨迹 trail.
    """
    import plotly.graph_objects as go

    smooth_path = resample_path(waypoints, n_frames=n_frames)

    # 预计算所有帧的 link positions
    all_link_pos = []
    for q in smooth_path:
        pos = robot.get_link_positions(np.asarray(q, dtype=np.float64))
        all_link_pos.append(pos)

    all_pts = _obstacle_corner_pts(scene)
    for pos in all_link_pos:
        all_pts.extend(pos)

    # 障碍物 (所有帧通用)
    obs_traces = _plotly_obstacle_traces(scene)
    n_obs = len(obs_traces)

    # 初始帧: 障碍物 + 当前臂 + EE trail
    pos0 = all_link_pos[0]
    arm_xs = [float(p[0]) for p in pos0]
    arm_ys = [float(p[1]) for p in pos0]
    arm_zs = [float(p[2]) for p in pos0]

    arm_trace = go.Scatter3d(
        x=arm_xs, y=arm_ys, z=arm_zs,
        mode='lines+markers',
        line=dict(color='#2196F3', width=7),
        marker=dict(size=4, color='#2196F3'),
        name='Current Arm',
    )
    ee_trace = go.Scatter3d(
        x=[arm_xs[-1]], y=[arm_ys[-1]], z=[arm_zs[-1]],
        mode='lines+markers',
        line=dict(color='#FF9800', width=3),
        marker=dict(size=2, color='#FF9800'),
        name='EE Trail', opacity=0.8,
    )
    base_trace = go.Scatter3d(
        x=[arm_xs[0]], y=[arm_ys[0]], z=[arm_zs[0]],
        mode='markers',
        marker=dict(size=6, color='black', symbol='square'),
        name='Base', showlegend=False,
    )
    ee_marker = go.Scatter3d(
        x=[arm_xs[-1]], y=[arm_ys[-1]], z=[arm_zs[-1]],
        mode='markers',
        marker=dict(size=5, color='#F44336', symbol='diamond'),
        name='EE', showlegend=False,
    )

    data = obs_traces + [arm_trace, ee_trace, base_trace, ee_marker]

    # 构造帧
    frames = []
    trail_x, trail_y, trail_z = [], [], []
    for i in range(n_frames):
        pos = all_link_pos[i]
        xs = [float(p[0]) for p in pos]
        ys = [float(p[1]) for p in pos]
        zs = [float(p[2]) for p in pos]
        trail_x.append(xs[-1])
        trail_y.append(ys[-1])
        trail_z.append(zs[-1])
        frame_data = [
            go.Scatter3d(x=xs, y=ys, z=zs),            # arm
            go.Scatter3d(x=list(trail_x), y=list(trail_y), z=list(trail_z)),  # trail
            go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]]),   # base
            go.Scatter3d(x=[xs[-1]], y=[ys[-1]], z=[zs[-1]]),  # ee marker
        ]
        frames.append(go.Frame(
            data=frame_data,
            traces=list(range(n_obs, n_obs + 4)),
            name=str(i),
        ))

    layout = _plotly_scene_layout(title, all_pts)
    # 播放/暂停按钮 + 自动循环
    layout['updatemenus'] = [dict(
        type='buttons', showactive=False,
        x=0.05, y=0.05, xanchor='left', yanchor='bottom',
        buttons=[
            dict(label='▶ Play',
                 method='animate',
                 args=[None, dict(
                     frame=dict(duration=50, redraw=True),
                     fromcurrent=True,
                     mode='immediate',
                     transition=dict(duration=0),
                 )]),
            dict(label='⏸ Pause',
                 method='animate',
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=False),
                     mode='immediate',
                     transition=dict(duration=0),
                 )]),
        ],
    )]
    # slider
    layout['sliders'] = [dict(
        active=0,
        steps=[dict(args=[[str(i)], dict(
            frame=dict(duration=50, redraw=True),
            mode='immediate', transition=dict(duration=0),
        )], label=str(i), method='animate')
               for i in range(n_frames)],
        x=0.05, len=0.9, y=0, xanchor='left',
        currentvalue=dict(prefix='Frame: ', visible=True),
        transition=dict(duration=0),
    )]

    fig = go.Figure(data=data, frames=frames, layout=layout)

    # 注入自动循环播放 JS
    autoplay_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            var gd = document.querySelector('.plotly-graph-div');
            if (gd) {
                Plotly.animate(gd, null, {
                    frame: {duration: 50, redraw: true},
                    fromcurrent: true,
                    mode: 'immediate',
                    transition: {duration: 0}
                });
            }
        }, 1000);
        // 循环: 到末帧后重新开始
        var checkLoop = setInterval(function() {
            var gd = document.querySelector('.plotly-graph-div');
            if (gd && gd._fullLayout && gd._fullLayout._currentFrame) {
                var cur = parseInt(gd._fullLayout._currentFrame);
                if (cur >= """ + str(n_frames - 1) + """) {
                    Plotly.animate(gd, null, {
                        frame: {duration: 50, redraw: true},
                        mode: 'immediate',
                        transition: {duration: 0}
                    });
                }
            }
        }, 200);
    });
    </script>
    """
    fig._autoplay_js = autoplay_js  # 保存到 figure 对象, 写 HTML 时注入
    return fig


def _save_plotly_html(fig, filepath):
    """保存 plotly figure 为 HTML, 并注入自动播放 JS (如果有)."""
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    # 注入自动循环 JS (如果动画)
    autoplay_js = getattr(fig, '_autoplay_js', '')
    if autoplay_js:
        html_str = html_str.replace('</body>', autoplay_js + '\n</body>')
    Path(filepath).write_text(html_str, encoding='utf-8')


def generate_report(cfg, scene, results, q_start, q_goal, ndim,
                    out_dir, total_s, prep_info, viz_files=None):
    """生成多方法对比报告."""
    from datetime import datetime
    lines = []
    w = lines.append

    w("=" * 70)
    w("  Panda 7-DOF Path Planner — Multi-Method Comparison")
    w("=" * 70)
    w(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Seed       : {cfg.seed}")
    w(f"  DOF        : {ndim}")
    dist = float(np.linalg.norm(q_goal - q_start))
    w(f"  q_start    : {np.array2string(q_start, precision=4)}")
    w(f"  q_goal     : {np.array2string(q_goal, precision=4)}")
    w(f"  config dist: {dist:.4f} rad")
    w("")

    w("--- Scene ---")
    w(f"  obstacles  : {scene.n_obstacles}")
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        sz = mx - mn
        w(f"    {obs.name}: center=({(mn[0]+mx[0])/2:.3f}, {(mn[1]+mx[1])/2:.3f}, "
          f"{(mn[2]+mx[2])/2:.3f})  size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")
    w("")

    w("--- Config ---")
    w(f"  max_consecutive_miss : {cfg.max_consecutive_miss}")
    w(f"  max_boxes (7D cap)   : {cfg.max_boxes}")
    w(f"  min_box_size         : {cfg.min_box_size}")
    w(f"  goal_bias            : {cfg.goal_bias}")
    w(f"  guided_sample_ratio  : {cfg.guided_sample_ratio}")
    w(f"  corridor_hops        : {cfg.corridor_hops}")
    w(f"  coarsen_max_rounds   : {cfg.coarsen_max_rounds}")
    w("")

    w("--- Shared Forest ---")
    w(f"  grown boxes      : {prep_info['n_grown']}")
    cs = prep_info.get('coarsen_stats')
    if cs:
        w(f"  after coarsen    : {cs.n_after} "
          f"({cs.n_merges} merges, {cs.n_rounds} rounds)")
    w(f"  AABB cache nodes : {prep_info['n_cache_nodes']}")
    w(f"  grow time        : {prep_info['grow_ms']:8.1f} ms")
    w(f"  cache time       : {prep_info['cache_ms']:8.1f} ms  (parallel)")
    w(f"  coarsen time     : {prep_info['coarsen_ms']:8.1f} ms")
    shared_ms = prep_info['grow_ms'] + prep_info['coarsen_ms']  # cache is parallel
    w(f"  shared total     : {shared_ms:8.1f} ms  (cache parallel, not added)")
    w("")

    # ─── grow_forest 详细分解 ───
    gd = prep_info.get('grow_detail')
    if gd:
        w("--- Grow Forest Breakdown ---")
        w(f"  warmup_fk        : {gd.get('warmup_ms', 0):8.1f} ms")
        w(f"  sample_batch     : {gd['sample_ms']:8.1f} ms  "
          f"({gd['n_sample_calls']} batches)")
        w(f"  is_occupied      : {gd['is_occupied_ms']:8.1f} ms  "
          f"({gd['n_is_occ_calls']} calls)")
        w(f"  can_expand       : {gd.get('probe_ms', 0):8.1f} ms  "
          f"({gd.get('n_probe_calls', 0)} calls, {gd.get('n_probe_reject', 0)} rejected)")
        w(f"  find_free_box    : {gd['find_free_box_ms']:8.1f} ms  "
          f"({gd['n_ffb_calls']} calls, {gd['n_ffb_none']} none)")
        w(f"  volume_check     : {gd['volume_check_ms']:8.1f} ms")
        w(f"  add_box          : {gd['add_box_ms']:8.1f} ms  "
          f"({gd['n_absorbed']} absorbed)")
        w(f"  overhead/other   : {gd['overhead_ms']:8.1f} ms")
        w("")

    # ─── 对比表格 ───
    w("=" * 70)
    w("  METHOD COMPARISON")
    w("=" * 70)
    header = f"  {'Method':<15s} {'Cost':>8s} {'WP':>4s} {'Bridge ms':>10s} {'Plan ms':>10s} {'Total ms':>10s}"
    w(header)
    w("  " + "-" * 65)
    for res in results:
        if res is None:
            continue
        name = res['method']
        ok = res.get('success', False)
        if ok:
            cost_s = f"{res['cost']:.4f}"
            wp_s = str(len(res['waypoints']))
        else:
            cost_s = "FAIL"
            wp_s = "-"
        bridge_ms = res.get('bridge_ms', 0.0)
        plan_ms = res.get('plan_ms', 0.0)
        total_method = bridge_ms + res.get('adj_ms', 0.0) + plan_ms
        w(f"  {name:<15s} {cost_s:>8s} {wp_s:>4s} {bridge_ms:>10.0f} {plan_ms:>10.0f} {total_method:>10.0f}")
    w("")

    # ─── 每个方法的详细信息 ───
    for res in results:
        if res is None:
            continue
        name = res['method']
        ok = res.get('success', False)
        w(f"--- {name} {'SUCCESS' if ok else 'FAILED'} ---")
        if ok:
            w(f"  cost       : {res['cost']:.4f}")
            w(f"  waypoints  : {len(res['waypoints'])}")
        w(f"  bridge ms  : {res.get('bridge_ms', 0):.1f}")
        w(f"  adj ms     : {res.get('adj_ms', 0):.1f}")
        w(f"  plan ms    : {res.get('plan_ms', 0):.1f}")
        if res.get('bridge_edges'):
            w(f"  bridge edges : {len(res['bridge_edges'])}")
        if res.get('n_before_islands'):
            w(f"  islands    : {res['n_before_islands']} -> {res.get('n_after_islands', '?')}")
        w("")

    w("--- Visualization ---")
    if viz_files:
        for name, ms in viz_files:
            w(f"  {name:30s}: {ms:7.0f} ms")
    w(f"")
    w(f"  Total elapsed      : {total_s:.1f} s")
    w("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    report_path = out_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Report saved -> {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():

    t_total_start = time.perf_counter()

    cfg = PandaGCSConfig()
    # seed=0 → 用当前时间戳, 每次运行不同场景
    if cfg.seed == 0:
        cfg.seed = int(time.time()) % (2**31)
    robot = load_robot("panda")
    ndim = robot.n_joints
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    rng = np.random.default_rng(cfg.seed)

    dist = float(np.linalg.norm(q_goal - q_start))
    print(f"Panda {ndim}-DOF Planner — Dijkstra")
    print(f"  seed        = {cfg.seed}")
    print(f"  q_start     = {np.array2string(q_start, precision=3)}")
    print(f"  q_goal      = {np.array2string(q_goal, precision=3)}")
    print(f"  config dist = {dist:.3f} rad")
    print(f"  obstacles   = {cfg.n_obstacles}")
    print(f"  max_boxes   = {cfg.max_boxes}")

    # ── 1) Scene ──
    t0 = time.perf_counter()
    print("\nBuilding scene ...")
    scene = build_panda_scene(rng, cfg, robot, q_start, q_goal)
    scene_ms = (time.perf_counter() - t0) * 1000
    print(f"  {scene.n_obstacles} obstacles  ({scene_ms:.1f} ms)")
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        sz = mx - mn
        print(f"    {obs.name}: center=({(mn[0]+mx[0])/2:.3f}, {(mn[1]+mx[1])/2:.3f}, "
              f"{(mn[2]+mx[2])/2:.3f})  size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")

    # ── 2) Shared: grow + cache + coarsen ──
    print("\n" + "=" * 60)
    print("  Pipeline: grow + cache + coarsen + bridge + Dijkstra")
    print("=" * 60)
    prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)

    # ── 3) Dijkstra + SOCP refine ──
    print("\n" + "=" * 60)
    print("  Dijkstra + SOCP refine")
    print("=" * 60)
    t0_b = time.perf_counter()
    result_dij = run_method_with_bridge(
        _solve_method_dijkstra, "Dijkstra", prep, cfg, q_start, q_goal, ndim)
    ms_b = (time.perf_counter() - t0_b) * 1000
    if result_dij and result_dij['success']:
        print(f"  Dijkstra: cost={result_dij['cost']:.4f}, "
              f"{len(result_dij['waypoints'])} wp, total {ms_b:.0f}ms")
    else:
        print(f"  Dijkstra: FAILED ({ms_b:.0f}ms)")

    # ── 4) Collect results ──
    all_results = [result_dij]
    best = result_dij if (result_dij and result_dij.get('success')) else None

    # ── 4b) Wait for cache save thread ──
    cache_thread = prep.get('_cache_thread')
    if cache_thread is not None:
        cache_thread.join()
        cr = prep.get('_cache_result', {})
        cache_ms = cr.get('ms', 0.0)
        cache_path = cr.get('path', '?')
        prep['cache_ms'] = cache_ms
        print(f"    [cache] done: {Path(cache_path).name} ({cache_ms:.0f} ms, parallel)")

    # ── 5) Visualization ──
    out_dir = make_output_dir("visualizations", "gcs_panda")
    print(f"\n{'=' * 60}")
    print(f"  Visualization")
    print(f"{'=' * 60}")
    print(f"  Output: {out_dir}")

    viz_files = []
    if best and best['success']:
        wps = best['waypoints']
        method_name = best['method']
        joint_names = [f"J{i+1}" for i in range(ndim)]

        # (a) Joint trajectory
        t0 = time.perf_counter()
        fig_jt = plot_joint_trajectory(
            wps, q_start, q_goal,
            label=f"Panda ({method_name}) cost={best['cost']:.2f}",
            joint_names=joint_names)
        if fig_jt:
            p = out_dir / "joint_trajectory.html"
            _save_plotly_html(fig_jt, p)
            ms = (time.perf_counter() - t0) * 1000
            print(f"    joint_trajectory  {ms:7.0f} ms  -> {p.name}")
            viz_files.append(("joint_trajectory.html", ms))

        # (b) 3D scene
        t0 = time.perf_counter()
        fig_scene = plot_arm_scene_html(
            robot, scene, q_start, q_goal, waypoints=wps,
            title=f"Panda {method_name} — cost={best['cost']:.2f}")
        p = out_dir / "arm_scene.html"
        _save_plotly_html(fig_scene, p)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    arm_scene         {ms:7.0f} ms  -> {p.name}")
        viz_files.append(("arm_scene.html", ms))

        # (c) Ghost poses
        t0 = time.perf_counter()
        fig_ghost = plot_arm_poses_html(
            robot, scene, wps, n_ghosts=10,
            title=f"Panda {method_name} — Arm Pose Sequence")
        p = out_dir / "arm_poses.html"
        _save_plotly_html(fig_ghost, p)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    arm_poses         {ms:7.0f} ms  -> {p.name}")
        viz_files.append(("arm_poses.html", ms))

        # (d) Animation (F: reduced frames for speed)
        t0 = time.perf_counter()
        fig_anim = create_animation_html(
            robot, scene, wps, n_frames=60,
            title=f"Panda {method_name} Path Animation")
        p = out_dir / "animation.html"
        _save_plotly_html(fig_anim, p)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    animation (HTML)  {ms:7.0f} ms  -> {p.name}")
        viz_files.append(("animation.html", ms))

    # ── 6) Report ──
    total_s = time.perf_counter() - t_total_start
    print("\n")
    generate_report(cfg, scene, all_results, q_start, q_goal, ndim,
                    out_dir, total_s, prep, viz_files)


if __name__ == "__main__":
    main()
