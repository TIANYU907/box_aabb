"""
planner/pipeline.py - Panda 7-DOF 多方法规划管线

从 v2/examples/panda_planner.py 提取的核心逻辑：
- PandaGCSConfig: 管线配置
- grow_forest(): forest 生长（串行/并行）
- 邻接+连通分量构建
- Solver A: GCS SOCP (凸松弛 + rounding + refine)
- Solver B: Dijkstra on box graph + SOCP refine
- Solver C: Visibility Graph (seed configs + collision check)
- grow_and_prepare(): 共享前半段管线
- run_method_with_bridge(): 带桥接的求解调度
"""

import heapq
import math
import os
import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None  # GCS solver 不可用

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.models import BoxNode
from forest.connectivity import find_islands, bridge_islands, UnionFind
from forest.hier_aabb_tree import HierAABBTree, build_kd_partitions
from forest.safe_box_forest import SafeBoxForest
from planner.sbf_planner import SBFPlanner
from planner.models import SBFConfig, gmean_edge_length
from forest.coarsen import coarsen_forest

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PandaGCSConfig:
    seed: int = 0  # 0 → 运行时随机

    # Panda start/goal — 远距离构型
    q_start: List[float] = field(
        default_factory=lambda: [0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal: List[float] = field(
        default_factory=lambda: [-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])

    # scene
    n_obstacles: int = 6
    workspace_radius: float = 0.85
    workspace_z_range: Tuple[float, float] = (0.0, 1.0)
    obs_size_range: Tuple[float, float] = (0.08, 0.25)

    # forest growth
    max_consecutive_miss: int = 20
    min_box_size: float = 0.1
    max_boxes: int = 500
    goal_bias: float = 0.10
    guided_sample_ratio: float = 0.6
    boundary_expand: bool = True
    boundary_expand_max_failures: int = 5
    boundary_expand_epsilon: float = 0.01
    parallel_grow: bool = False
    n_partitions_depth: int = 3
    parallel_workers: int = 4

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

def build_panda_scene(
    rng: np.random.Generator,
    cfg: PandaGCSConfig,
    robot=None,
    q_start=None,
    q_goal=None,
    max_trials: int = 100,
) -> Scene:
    """生成 3D 随机障碍物场景, 确保起终点无碰撞."""
    for _ in range(max_trials):
        scene = Scene()
        for i in range(cfg.n_obstacles):
            r = rng.uniform(0.25 * cfg.workspace_radius,
                            0.85 * cfg.workspace_radius)
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
            if (checker.check_config_collision(q_start)
                    or checker.check_config_collision(q_goal)):
                continue
        return scene
    raise RuntimeError("Failed to build collision-free scene for Panda")


def make_planner_config(cfg: PandaGCSConfig) -> SBFConfig:
    """从 PandaGCSConfig 构建 SBFConfig."""
    return SBFConfig(
        max_iterations=999999,
        max_box_nodes=999999,
        min_box_size=cfg.min_box_size,
        goal_bias=cfg.goal_bias,
        guided_sample_ratio=cfg.guided_sample_ratio,
        segment_collision_resolution=0.05,
        connection_radius=1.5,
        verbose=False,
        forest_path=None,
        boundary_expand_enabled=cfg.boundary_expand,
        boundary_expand_max_failures=cfg.boundary_expand_max_failures,
        boundary_expand_epsilon=cfg.boundary_expand_epsilon,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parallel grow worker (top-level for pickle)
# ═══════════════════════════════════════════════════════════════════════════

def _grow_partition_worker(payload: Dict) -> Dict:
    """ProcessPool worker: 在单个 KD 分区内独立生长 boxes."""
    robot = payload["robot"]
    obstacles = payload["obstacles"]
    q_start = np.asarray(payload["q_start"], dtype=np.float64)
    q_goal = np.asarray(payload["q_goal"], dtype=np.float64)
    partition_intervals = payload["intervals"]
    max_boxes = int(payload["max_boxes"])
    max_miss = int(payload["max_miss"])
    min_box_size = float(payload["min_box_size"])
    goal_bias = float(payload.get("goal_bias", 0.1))
    seed_val = payload.get("seed")

    ndim = len(partition_intervals)
    rng = np.random.default_rng(seed_val)

    local_tree = HierAABBTree(robot, joint_limits=partition_intervals)
    obs_packed = local_tree._prepack_obstacles_c(obstacles)
    local_tree.warmup_fk_cache(max_depth=6)

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

    for qs in [q_start, q_goal]:
        in_part = all(
            lo <= qs[d] <= hi
            for d, (lo, hi) in enumerate(partition_intervals))
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
# Forest grow
# ═══════════════════════════════════════════════════════════════════════════

def grow_forest(
    planner: SBFPlanner,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    seed: int,
    max_miss: int = 20,
    ndim: int = 7,
    max_boxes: int = 1500,
    parallel_grow: bool = False,
    n_partitions_depth: int = 3,
    parallel_workers: int = 4,
):
    """生长 forest.

    终止条件:
      - 主条件: 连续 max_miss 个 seed 未命中
      - 安全上限: max_boxes 防止 7D 空间无限增长

    Returns:
        (boxes, forest, timing_detail)
    """
    rng = np.random.default_rng(seed)
    forest = planner._load_or_create_forest()
    forest.hier_tree = planner.hier_tree

    # FK 缓存预热
    t_warmup_start = time.perf_counter()
    n_warmed = planner.hier_tree.warmup_fk_cache(max_depth=6)
    t_warmup = (time.perf_counter() - t_warmup_start) * 1000
    print(f"    [warmup] FK cache: {n_warmed} nodes ({t_warmup:.0f} ms)")

    obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)

    # ── 并行分区生长 ──
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
        workers = min(workers, n_parts)
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
            for p in payloads:
                local_results.append(_grow_partition_worker(p))

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

    # ── 串行生长 ──
    t_sample = 0.0
    t_is_occ = 0.0
    t_probe = 0.0
    t_ffb = 0.0
    t_add = 0.0
    t_vol = 0.0
    n_sample_calls = 0
    n_is_occ_calls = 0
    n_probe_calls = 0
    n_probe_reject = 0
    n_ffb_calls = 0
    n_ffb_none = 0
    n_absorbed = 0

    intervals = planner.joint_limits
    lows = np.array([lo for lo, _ in intervals], dtype=np.float64)
    highs = np.array([hi for _, hi in intervals], dtype=np.float64)
    goal_bias = planner.config.goal_bias
    guided_ratio = getattr(planner.config, 'guided_sample_ratio', 0.6)
    has_hier_tree = hasattr(planner, 'hier_tree') and planner.hier_tree is not None
    batch_size = 32

    def _sample_batch():
        rolls = rng.uniform(size=batch_size)
        configs = np.empty((batch_size, ndim), dtype=np.float64)
        goal_mask = rolls < goal_bias
        if has_hier_tree:
            guided_mask = (~goal_mask) & (rolls < goal_bias + guided_ratio)
        else:
            guided_mask = np.zeros(batch_size, dtype=bool)
        uniform_mask = ~(goal_mask | guided_mask)

        n_goal = int(goal_mask.sum())
        if n_goal > 0:
            noise = rng.normal(0.0, 0.3, size=(n_goal, ndim))
            configs[goal_mask] = np.clip(q_goal + noise, lows, highs)

        n_uni = int(uniform_mask.sum())
        if n_uni > 0:
            configs[uniform_mask] = rng.uniform(lows, highs, size=(n_uni, ndim))

        guided_idxs = np.flatnonzero(guided_mask)
        for i in guided_idxs:
            try:
                q = planner.hier_tree.sample_unoccupied_seed(rng)
            except ValueError:
                q = None
            if q is None:
                q = rng.uniform(lows, highs)
            configs[i] = q

        collisions = planner.collision_checker.check_config_collision_batch(
            configs)
        return [configs[i] for i in range(batch_size) if not collisions[i]]

    seed_buffer: List[np.ndarray] = []

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

    # boundary expand state
    boundary_on = planner.config.boundary_expand_enabled
    boundary_max_fail = planner.config.boundary_expand_max_failures
    expand_target = None
    expand_fails = 0
    n_boundary_attempts = 0
    n_boundary_ok = 0
    t_boundary = 0.0

    consec = 0
    t0 = time.perf_counter()
    terminated_by = "miss"
    while consec < max_miss:
        if forest.n_boxes >= max_boxes:
            terminated_by = "max_boxes"
            break

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
            _ts = time.perf_counter()
            if not seed_buffer:
                seed_buffer = _sample_batch()
                n_sample_calls += 1
            if not seed_buffer:
                t_sample += time.perf_counter() - _ts
                consec += 1
                continue
            q = seed_buffer.pop()
            t_sample += time.perf_counter() - _ts

            _ts = time.perf_counter()
            occ = planner.hier_tree.is_occupied(q)
            t_is_occ += time.perf_counter() - _ts
            n_is_occ_calls += 1

            if occ:
                consec += 1
                continue

            _ts = time.perf_counter()
            can = planner.hier_tree.can_expand(q, obs_packed=obs_packed)
            t_probe += time.perf_counter() - _ts
            n_probe_calls += 1

            if not can:
                n_probe_reject += 1
                consec += 1
                continue

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

        _ts = time.perf_counter()
        vol = 1.0
        for lo, hi in ffb.intervals:
            vol *= max(hi - lo, 0)
        too_small = gmean_edge_length(vol, ndim) < planner.config.min_box_size
        t_vol += time.perf_counter() - _ts

        if too_small:
            consec += 1
            continue

        _ts = time.perf_counter()
        box = BoxNode(node_id=nid, joint_intervals=ffb.intervals,
                      seed_config=q.copy(), volume=vol)
        if ffb.absorbed_box_ids:
            n_absorbed += len(ffb.absorbed_box_ids)
            forest.remove_boxes(ffb.absorbed_box_ids)
        forest.add_box_direct(box)
        t_add += time.perf_counter() - _ts

        if boundary_on:
            expand_target = box
            expand_fails = 0
            n_boundary_ok += 1

        consec = 0
        if forest.n_boxes % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    [grow] {forest.n_boxes} boxes, {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"    [grow] terminated by {terminated_by}: "
          f"{forest.n_boxes} boxes, {elapsed:.1f}s")

    # 后处理: 双向 stepping stone 铺路
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
            return sb, tb

        def _grow_one_box(q_cfg):
            if planner.collision_checker.check_config_collision(q_cfg):
                return None
            nid2 = forest.allocate_id()
            ffb2 = planner.hier_tree.find_free_box(
                q_cfg, planner.obstacles, mark_occupied=True,
                forest_box_id=nid2, obs_packed=obs_packed)
            if ffb2 is None:
                return None
            vol2 = 1.0
            for lo2, hi2 in ffb2.intervals:
                vol2 *= max(hi2 - lo2, 0)
            if gmean_edge_length(vol2, ndim) < planner.config.min_box_size:
                return None
            box2 = BoxNode(node_id=nid2, joint_intervals=ffb2.intervals,
                           seed_config=q_cfg.copy(), volume=vol2)
            if ffb2.absorbed_box_ids:
                forest.remove_boxes(ffb2.absorbed_box_ids)
            forest.add_box_direct(box2)
            return box2

        def _pave_toward(start_c, end_c, max_steps=300):
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
                for attempt in range(15):
                    if attempt == 0:
                        q_try = pos.copy()
                    else:
                        q_try = pos + rng.normal(0, 0.05, size=ndim)
                    for d in range(ndim):
                        q_try[d] = np.clip(q_try[d],
                                           planner.joint_limits[d][0],
                                           planner.joint_limits[d][1])
                    box_new = _grow_one_box(q_try)
                    if box_new is not None:
                        n_placed += 1
                        min_edge = min(hi - lo
                                       for lo, hi in box_new.joint_intervals)
                        step_size = max(0.01, min(min_edge * 0.4, 0.1))
                        pos = pos + direction * step_size
                        placed = True
                        break
                if not placed:
                    pos = pos + direction * 0.03
                step_idx += 1
            return n_placed

        _gr = 0  # track rounds for reporting
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

                n1 = _pave_toward(_st_c, _tgt_c, max_steps=200)
                n2 = _pave_toward(_tgt_c, _st_c, max_steps=200)
                n_gap_fill += n1 + n2

        gap_ms = (time.perf_counter() - _ts_connect) * 1000
        t_boundary += (time.perf_counter() - _ts_connect)
        if n_gap_fill > 0:
            print(f"    [gap fill] +{n_gap_fill} boxes in {_gr + 1} rounds, "
                  f"total={forest.n_boxes} ({gap_ms:.0f}ms)")
        elif _gr > 0:
            print(f"    [gap fill] s/t disconnected, 0 gap boxes "
                  f"({gap_ms:.0f}ms)")

    # 详细计时报告
    timing = dict(
        warmup_ms=t_warmup,
        sample_ms=t_sample * 1000,
        boundary_ms=t_boundary * 1000,
        is_occupied_ms=t_is_occ * 1000,
        probe_ms=t_probe * 1000,
        find_free_box_ms=t_ffb * 1000,
        volume_check_ms=t_vol * 1000,
        add_box_ms=t_add * 1000,
        overhead_ms=(elapsed - t_sample - t_boundary - t_is_occ - t_probe
                     - t_ffb - t_vol - t_add) * 1000,
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
    """O(N²) 向量化 overlap → adjacency dict + UnionFind + islands."""
    ids = list(boxes.keys())
    n = len(ids)
    adj: Dict[int, Set[int]] = {bid: set() for bid in ids}
    uf = UnionFind(ids)

    if n < 2:
        return adj, uf, uf.components()

    ndim = next(iter(boxes.values())).n_dims
    lo = np.empty((n, ndim), dtype=np.float64)
    hi = np.empty((n, ndim), dtype=np.float64)
    for k, bid in enumerate(ids):
        ivs = boxes[bid].joint_intervals
        for d in range(ndim):
            lo[k, d] = ivs[d][0]
            hi[k, d] = ivs[d][1]

    eps = 1e-9
    overlap_ij = ((hi[:, None, :] >= lo[None, :, :] - eps)
                  & (hi[None, :, :] >= lo[:, None, :] - eps))
    overlap_all = np.all(overlap_ij, axis=2)

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
# GCS helpers (from gcs_planner_2dof.py)
# ═══════════════════════════════════════════════════════════════════════════

def find_box_containing(
    q: np.ndarray,
    boxes: Dict[int, BoxNode],
) -> Optional[int]:
    """返回包含 q 的 box id, 若无则返回距离最近的。"""
    best_id, best_d = None, float("inf")
    for bid, b in boxes.items():
        if b.contains(q):
            return bid
        d = b.distance_to_config(q)
        if d < best_d:
            best_d = d
            best_id = bid
    return best_id


def extract_connected_subgraph(
    adj: Dict[int, Set[int]],
    source_id: int,
    target_id: int,
) -> Optional[Set[int]]:
    """BFS 找 source 所在连通分量, 若 target 不在其中返回 None."""
    visited = set()
    queue = [source_id]
    visited.add(source_id)
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, set()):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    if target_id not in visited:
        return None
    return visited


def corridor_prune(
    adj: Dict[int, Set[int]],
    source_id: int,
    target_id: int,
    hops: int = 2,
) -> Optional[Set[int]]:
    """保留最短路径 ± hops 跳范围内的顶点 (corridor 子图)."""
    from collections import deque

    parent: Dict[int, Optional[int]] = {source_id: None}
    queue = deque([source_id])
    while queue:
        u = queue.popleft()
        if u == target_id:
            break
        for v in adj.get(u, set()):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if target_id not in parent:
        return None

    path_verts: Set[int] = set()
    cur = target_id
    while cur is not None:
        path_verts.add(cur)
        cur = parent[cur]

    corridor: Set[int] = set(path_verts)
    for pv in path_verts:
        frontier = {pv}
        for _ in range(hops):
            next_frontier: Set[int] = set()
            for u in frontier:
                for v in adj.get(u, set()):
                    if v not in corridor:
                        next_frontier.add(v)
            corridor |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
    return corridor


def _deperspective_point(xv_val, phi_val, ei, boxes, v, ndim):
    """从 perspective 变量恢复实际坐标并 clip 到 box."""
    ph = max(phi_val[ei], 1e-10)
    pt = xv_val[ei] / ph
    box_v = boxes[v]
    for d in range(ndim):
        lo, hi = box_v.joint_intervals[d]
        pt[d] = np.clip(pt[d], lo, hi)
    return pt


def _round_greedy(
    phi_val, xu_val, xv_val,
    edges, out_edges, id2idx,
    boxes, source_id, target_id, q_start, q_goal, ndim, nv,
) -> Optional[Tuple[bool, float, List[np.ndarray], List[int]]]:
    """Greedy flow tracing rounding."""
    waypoints = [q_start.copy()]
    box_seq = [source_id]
    current = source_id
    visited_edges = set()
    max_steps = nv + 10

    for _ in range(max_steps):
        if current == target_id:
            break
        ci = id2idx[current]
        best_ei, best_phi = -1, -1e-9
        for ei in out_edges[ci]:
            if ei not in visited_edges and phi_val[ei] > best_phi:
                best_phi = phi_val[ei]
                best_ei = ei
        if best_ei < 0 or best_phi < 1e-8:
            break
        visited_edges.add(best_ei)
        u, v = edges[best_ei]
        pt_v = _deperspective_point(xv_val, phi_val, best_ei, boxes, v, ndim)
        waypoints.append(pt_v.copy())
        box_seq.append(v)
        current = v

    if current != target_id:
        return None

    waypoints[-1] = q_goal.copy()
    total_cost = sum(
        float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
        for i in range(len(waypoints) - 1))
    return True, total_cost, waypoints, box_seq


def _round_dfs(
    phi_val, xu_val, xv_val,
    edges, out_edges, id2idx,
    boxes, source_id, target_id, q_start, q_goal, ndim, nv,
    phi_threshold: float = 1e-4,
) -> Optional[Tuple[bool, float, List[np.ndarray], List[int]]]:
    """DFS with backtracking rounding."""
    sorted_out: List[List[int]] = [[] for _ in range(nv)]
    for vi in range(nv):
        eis = [ei for ei in out_edges[vi] if phi_val[ei] > phi_threshold]
        eis.sort(key=lambda ei: -phi_val[ei])
        sorted_out[vi] = eis

    src_idx = id2idx[source_id]
    stack = [(source_id, 0, {source_id})]
    path_edges: List[int] = []

    while stack:
        cur, ei_pos, visited = stack[-1]
        ci = id2idx[cur]
        if cur == target_id:
            break
        found_next = False
        while ei_pos < len(sorted_out[ci]):
            ei = sorted_out[ci][ei_pos]
            ei_pos += 1
            stack[-1] = (cur, ei_pos, visited)
            u, v = edges[ei]
            if v not in visited:
                path_edges.append(ei)
                new_visited = visited | {v}
                stack.append((v, 0, new_visited))
                found_next = True
                break
        if not found_next:
            stack.pop()
            if path_edges:
                path_edges.pop()

    if not stack or stack[-1][0] != target_id:
        return None

    waypoints = [q_start.copy()]
    box_seq = [source_id]
    for ei in path_edges:
        u, v = edges[ei]
        pt_v = _deperspective_point(xv_val, phi_val, ei, boxes, v, ndim)
        waypoints.append(pt_v.copy())
        box_seq.append(v)
    waypoints[-1] = q_goal.copy()
    total_cost = sum(
        float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
        for i in range(len(waypoints) - 1))
    return True, total_cost, waypoints, box_seq


def _round_random_once(
    phi_val, xu_val, xv_val,
    edges, out_edges, id2idx,
    boxes, source_id, target_id, q_start, q_goal, ndim, nv,
    rng: np.random.Generator,
    phi_threshold: float = 1e-4,
    noise_scale: float = 0.3,
) -> Optional[Tuple[bool, float, List[np.ndarray], List[int]]]:
    """Randomized DFS rounding."""
    sorted_out: List[List[int]] = [[] for _ in range(nv)]
    for vi in range(nv):
        eis = [ei for ei in out_edges[vi] if phi_val[ei] > phi_threshold]
        if not eis:
            sorted_out[vi] = []
            continue
        phi_arr = np.array([phi_val[ei] for ei in eis])
        noise = rng.uniform(0, noise_scale * phi_arr.max(), size=len(eis))
        order = np.argsort(-(phi_arr + noise))
        sorted_out[vi] = [eis[i] for i in order]

    stack = [(source_id, 0, {source_id})]
    path_edges: List[int] = []

    while stack:
        cur, ei_pos, visited = stack[-1]
        ci = id2idx[cur]
        if cur == target_id:
            break
        found_next = False
        while ei_pos < len(sorted_out[ci]):
            ei = sorted_out[ci][ei_pos]
            ei_pos += 1
            stack[-1] = (cur, ei_pos, visited)
            u, v = edges[ei]
            if v not in visited:
                path_edges.append(ei)
                stack.append((v, 0, visited | {v}))
                found_next = True
                break
        if not found_next:
            stack.pop()
            if path_edges:
                path_edges.pop()

    if not stack or stack[-1][0] != target_id:
        return None

    waypoints = [q_start.copy()]
    box_seq = [source_id]
    for ei in path_edges:
        u, v = edges[ei]
        pt_v = _deperspective_point(xv_val, phi_val, ei, boxes, v, ndim)
        waypoints.append(pt_v.copy())
        box_seq.append(v)
    waypoints[-1] = q_goal.copy()
    total_cost = sum(
        float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
        for i in range(len(waypoints) - 1))
    return True, total_cost, waypoints, box_seq


# ═══════════════════════════════════════════════════════════════════════════
# Solver A: GCS SOCP (凸松弛 + rounding + refine)
# ═══════════════════════════════════════════════════════════════════════════

def solve_gcs(
    boxes: Dict[int, BoxNode],
    adj: Dict[int, Set[int]],
    source_id: int,
    target_id: int,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    ndim: int = 2,
    corridor_hops: int = 2,
) -> Tuple[bool, float, List[np.ndarray], List[int]]:
    """GCS 凸松弛求最短路径 (Marcucci et al. 2023)."""
    if cp is None:
        raise ImportError("cvxpy is required for GCS solver")

    reachable = corridor_prune(adj, source_id, target_id, hops=corridor_hops)
    if reachable is None:
        print(f"    [GCS] source {source_id} and target {target_id} "
              f"are disconnected")
        return False, float("inf"), [], []

    sub_adj: Dict[int, Set[int]] = {
        u: adj[u] & reachable for u in reachable}

    edges = []
    edge_set = set()
    for u, nbrs in sub_adj.items():
        for v in nbrs:
            if (u, v) not in edge_set:
                edges.append((u, v))
                edge_set.add((u, v))
            if (v, u) not in edge_set:
                edges.append((v, u))
                edge_set.add((v, u))

    ne = len(edges)
    if ne == 0:
        return False, float("inf"), [], []

    box_ids = sorted(reachable)
    nv = len(box_ids)
    print(f"    [GCS] subgraph: {nv} vertices, {ne} directed edges")
    id2idx = {bid: i for i, bid in enumerate(box_ids)}
    src_idx = id2idx[source_id]
    tgt_idx = id2idx[target_id]

    phi = cp.Variable(ne, nonneg=True)
    xu = cp.Variable((ne, ndim))
    xv = cp.Variable((ne, ndim))

    constraints = [phi <= 1.0]

    out_edges_list = [[] for _ in range(nv)]
    in_edges_list = [[] for _ in range(nv)]
    for ei, (u, v) in enumerate(edges):
        out_edges_list[id2idx[u]].append(ei)
        in_edges_list[id2idx[v]].append(ei)

    for vi in range(nv):
        flow_out = (cp.sum(phi[out_edges_list[vi]])
                    if out_edges_list[vi] else 0.0)
        flow_in = (cp.sum(phi[in_edges_list[vi]])
                   if in_edges_list[vi] else 0.0)
        if vi == src_idx:
            constraints.append(flow_out - flow_in == 1.0)
        elif vi == tgt_idx:
            constraints.append(flow_out - flow_in == -1.0)
        else:
            constraints.append(flow_out - flow_in == 0.0)

    for ei, (u, v) in enumerate(edges):
        box_u = boxes[u]
        box_v = boxes[v]
        for d in range(ndim):
            lo_u, hi_u = box_u.joint_intervals[d]
            lo_v, hi_v = box_v.joint_intervals[d]
            constraints.append(xu[ei, d] >= lo_u * phi[ei])
            constraints.append(xu[ei, d] <= hi_u * phi[ei])
            constraints.append(xv[ei, d] >= lo_v * phi[ei])
            constraints.append(xv[ei, d] <= hi_v * phi[ei])

    if out_edges_list[src_idx]:
        constraints.append(
            cp.sum(xu[out_edges_list[src_idx], :], axis=0) == q_start)

    if in_edges_list[tgt_idx]:
        constraints.append(
            cp.sum(xv[in_edges_list[tgt_idx], :], axis=0) == q_goal)

    for vi in range(nv):
        if vi == src_idx or vi == tgt_idx:
            continue
        if in_edges_list[vi] and out_edges_list[vi]:
            constraints.append(
                cp.sum(xv[in_edges_list[vi], :], axis=0)
                == cp.sum(xu[out_edges_list[vi], :], axis=0))

    cost_terms = [cp.norm(xu[ei, :] - xv[ei, :], 2) for ei in range(ne)]
    objective = cp.Minimize(cp.sum(cost_terms))

    prob = cp.Problem(objective, constraints)

    for solver_name, solver_kwargs in [
        ("CLARABEL", dict(solver=cp.CLARABEL, verbose=False)),
        ("SCS", dict(solver=cp.SCS, verbose=False, max_iters=20000,
                     eps=1e-6)),
    ]:
        prob.solve(**solver_kwargs)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            print(f"    [GCS] {solver_name}: {prob.status}")
            continue
        print(f"    [GCS] {solver_name}: {prob.status}, "
              f"obj={prob.value:.4f}")

        rounding_args = (
            phi.value, xu.value, xv.value,
            edges, out_edges_list, id2idx,
            boxes, source_id, target_id, q_start, q_goal, ndim, nv,
        )
        candidates = []

        for round_fn, round_name in [
            (_round_greedy, "greedy"),
            (_round_dfs, "DFS"),
        ]:
            result = round_fn(*rounding_args)
            if result is not None:
                candidates.append((round_name, result))

        for trial in range(20):
            rng_trial = np.random.default_rng(trial * 7 + 13)
            result = _round_random_once(*rounding_args, rng=rng_trial)
            if result is not None:
                candidates.append((f"rand{trial}", result))

        if not candidates:
            print(f"    [GCS] {solver_name}: all rounding failed")
            continue

        candidates.sort(key=lambda x: x[1][1])
        best_result = None
        best_cost = float("inf")
        best_name = ""

        for name, (ok, raw_cost, wps, bseq) in candidates[:5]:
            wps_r, refined_cost = _refine_path_in_boxes(
                wps, bseq, boxes, q_start, q_goal, ndim)
            if refined_cost < best_cost:
                best_cost = refined_cost
                best_result = (ok, refined_cost, wps_r, bseq)
                best_name = name

        if best_result is not None:
            print(f"    [GCS] best rounding: {best_name}, "
                  f"refined={best_cost:.4f} "
                  f"(from {len(candidates)} candidates)")
            return best_result
        print(f"    [GCS] {solver_name}: refinement failed")

    return False, float("inf"), [], []


def _refine_path_in_boxes(
    waypoints: List[np.ndarray],
    box_seq: List[int],
    boxes: Dict[int, BoxNode],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    ndim: int,
) -> Tuple[List[np.ndarray], float]:
    """SOCP 精炼: 在已有 box 序列内重新优化 waypoint 位置."""
    if cp is None:
        cost = sum(float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
                   for i in range(len(waypoints) - 1))
        return waypoints, cost

    m = len(waypoints)
    if m <= 2:
        cost = float(np.linalg.norm(q_goal - q_start))
        return waypoints, cost

    n_free = m - 2
    w = cp.Variable((n_free, ndim))

    constraints = []
    for i in range(n_free):
        box = boxes[box_seq[i + 1]]
        for d in range(ndim):
            lo, hi = box.joint_intervals[d]
            constraints.append(w[i, d] >= lo)
            constraints.append(w[i, d] <= hi)

    segs = [cp.norm(w[0, :] - q_start, 2)]
    for i in range(n_free - 1):
        segs.append(cp.norm(w[i + 1, :] - w[i, :], 2))
    segs.append(cp.norm(q_goal - w[n_free - 1, :], 2))

    prob = cp.Problem(cp.Minimize(cp.sum(segs)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        cost = sum(float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
                   for i in range(m - 1))
        return waypoints, cost

    if (prob.status in ("optimal", "optimal_inaccurate")
            and w.value is not None):
        refined = [q_start.copy()]
        for i in range(n_free):
            pt = w.value[i].copy()
            box = boxes[box_seq[i + 1]]
            for d in range(ndim):
                lo, hi = box.joint_intervals[d]
                pt[d] = np.clip(pt[d], lo, hi)
            refined.append(pt)
        refined.append(q_goal.copy())
        cost = sum(float(np.linalg.norm(refined[i + 1] - refined[i]))
                   for i in range(len(refined) - 1))
        return refined, cost

    cost = sum(float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
               for i in range(m - 1))
    return waypoints, cost


# ═══════════════════════════════════════════════════════════════════════════
# Solver B: Dijkstra on box graph + SOCP refine
# ═══════════════════════════════════════════════════════════════════════════

def _dijkstra_box_graph(boxes, adj, src, tgt):
    """Dijkstra 最短路 (edge weight = center-to-center distance)."""
    centers = {}
    for bid, box in boxes.items():
        centers[bid] = np.array(
            [(lo + hi) / 2 for lo, hi in box.joint_intervals])

    dist_map: Dict[int, float] = {bid: float('inf') for bid in boxes}
    prev_map: Dict[int, Optional[int]] = {bid: None for bid in boxes}
    dist_map[src] = 0.0
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

    seq = []
    cur = tgt
    while cur is not None:
        seq.append(cur)
        cur = prev_map[cur]
    seq.reverse()
    return seq, dist_map[tgt]


def _solve_method_gcs(boxes, adj, src, tgt, q_start, q_goal, ndim,
                      corridor_hops=2, label="GCS"):
    """GCS SOCP 求解."""
    t0 = time.perf_counter()
    success, cost, waypoints, box_seq = solve_gcs(
        boxes, adj, src, tgt, q_start, q_goal, ndim,
        corridor_hops=corridor_hops)
    ms = (time.perf_counter() - t0) * 1000
    return dict(method=label, success=success, cost=cost,
                waypoints=waypoints, box_seq=box_seq, plan_ms=ms)


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

    waypoints = [q_start.copy()]
    for bid in box_seq[1:-1]:
        box = boxes[bid]
        c = np.array([(lo + hi) / 2 for lo, hi in box.joint_intervals])
        waypoints.append(c)
    waypoints.append(q_goal.copy())

    refined_wps, refined_cost = _refine_path_in_boxes(
        waypoints, box_seq, boxes, q_start, q_goal, ndim)

    ms = (time.perf_counter() - t0) * 1000
    print(f"    [{label}] {len(box_seq)} boxes, raw_dist={raw_dist:.4f}, "
          f"refined={refined_cost:.4f}, {len(refined_wps)} wp ({ms:.0f}ms)")
    return dict(method=label, success=True, cost=refined_cost,
                waypoints=refined_wps, box_seq=box_seq, plan_ms=ms)


# ═══════════════════════════════════════════════════════════════════════════
# Solver C: Visibility Graph
# ═══════════════════════════════════════════════════════════════════════════

def _solve_method_visgraph(boxes, q_start, q_goal, collision_checker,
                           segment_resolution=0.05, label="VisGraph"):
    """可视图法: box seed configs 为节点, collision check 为边."""
    t0 = time.perf_counter()

    node_ids = ['start', 'goal']
    node_configs = [q_start.copy(), q_goal.copy()]
    for bid, box in boxes.items():
        if box.seed_config is not None:
            node_ids.append(bid)
            node_configs.append(np.asarray(box.seed_config, dtype=np.float64))

    n_nodes = len(node_ids)
    print(f"    [{label}] {n_nodes} nodes, checking edges ...")

    configs_arr = np.array(node_configs)
    k_nearest = min(50, n_nodes - 1)

    vis_adj: Dict[int, List[Tuple[int, float]]] = {
        i: [] for i in range(n_nodes)}
    n_checks = 0
    n_edges = 0

    for i in range(n_nodes):
        qi = configs_arr[i]
        diffs = configs_arr - qi
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = float('inf')

        nearest_idxs = np.argpartition(
            dists, min(k_nearest, n_nodes - 2))[:k_nearest]
        nearest_idxs = nearest_idxs[np.argsort(dists[nearest_idxs])]

        for j_idx in nearest_idxs:
            j = int(j_idx)
            d = float(dists[j])
            if d == float('inf'):
                continue
            if any(nbr == i for nbr, _ in vis_adj[j]):
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

    dist_map = [float('inf')] * n_nodes
    prev_map = [-1] * n_nodes
    dist_map[0] = 0.0
    pq = [(0.0, 0)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist_map[u]:
            continue
        if u == 1:
            break
        for v, w in vis_adj[u]:
            nd = d + w
            if nd < dist_map[v]:
                dist_map[v] = nd
                prev_map[v] = u
                heapq.heappush(pq, (nd, v))

    if dist_map[1] == float('inf'):
        ms = (time.perf_counter() - t0) * 1000
        print(f"    [{label}] no path found ({ms:.0f}ms)")
        return dict(method=label, success=False, cost=float('inf'),
                    waypoints=[], box_seq=[], plan_ms=ms)

    path_idxs = []
    cur = 1
    while cur != -1:
        path_idxs.append(cur)
        cur = prev_map[cur]
    path_idxs.reverse()

    raw_waypoints = [node_configs[i].copy() for i in path_idxs]
    raw_cost = dist_map[1]

    shortcut_wps = _greedy_shortcut(
        raw_waypoints, collision_checker, segment_resolution)

    final_cost = sum(
        float(np.linalg.norm(shortcut_wps[i + 1] - shortcut_wps[i]))
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

def grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim,
                     no_cache=False):
    """共享前半段: grow → cache (异步) → coarsen.

    返回 dict 包含 planner, boxes, forest_obj, timing 等.
    """
    planner_cfg = make_planner_config(cfg)
    planner = SBFPlanner(robot=robot, scene=scene, config=planner_cfg,
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

    # AABB cache: 后台线程保存
    cache_result = {}
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

    # coarsen (与 cache save 并行)
    n_before_coarsen = len(forest_obj.boxes)
    coarsen_stats = coarsen_forest(
        tree=planner.hier_tree, forest=forest_obj,
        obstacles=planner.obstacles, safety_margin=0.0,
        max_rounds=cfg.coarsen_max_rounds,
    )
    n_after_coarsen = len(forest_obj.boxes)
    coarsen_ms = coarsen_stats.time_ms
    print(f"    [coarsen] {n_before_coarsen} -> {n_after_coarsen} boxes "
          f"({coarsen_stats.n_merges} merges in "
          f"{coarsen_stats.n_rounds} rounds, {coarsen_ms:.0f}ms)")

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


def run_method_with_bridge(method_fn, method_name, prep, cfg, q_start,
                           q_goal, ndim, **method_kwargs):
    """运行需要 adjacency 的方法, 含 lazy bridge."""
    boxes = prep['boxes']
    planner = prep['planner']
    forest_obj = prep['forest_obj']

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
        print(f"    [{method_name}] s-t disconnected "
              f"({n_before_islands} islands), bridging ...")
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
        bridge_edges_res, final_islands, _, bridge_boxes_res, discarded = (
            bridge_result)
        bridge_ms = (time.perf_counter() - t0) * 1000
        bridge_edges = bridge_edges_res
        bridge_boxes_list = bridge_boxes_res
        boxes = forest_obj.boxes

        for bb in bridge_boxes_list:
            if bb.node_id not in adj:
                neighbor_ids = forest_obj._adjacent_existing_ids_from_cache(
                    bb, tol=forest_obj.config.adjacency_tolerance)
                adj[bb.node_id] = set(neighbor_ids)
                for nb in neighbor_ids:
                    adj.setdefault(nb, set()).add(bb.node_id)
        _add_bridge_to_adj(adj, bridge_edges, uf)
        n_after_islands = len(uf.components())
        print(f"    [bridge] islands: {n_before_islands} -> "
              f"{n_after_islands}  ({len(bridge_edges)} edges, "
              f"{len(bridge_boxes_list)} bridge-boxes, {bridge_ms:.0f} ms)")

        if src not in boxes:
            src = find_box_containing(q_start, boxes)
        if tgt not in boxes:
            tgt = find_box_containing(q_goal, boxes)
    else:
        print(f"    [{method_name}] s-t already connected! Skipping bridge.")
        n_after_islands = n_before_islands

    if src is None or tgt is None:
        print(f"    [{method_name}] ERROR: start or goal not in any box "
              f"after bridge")
        return None

    plan_result = method_fn(
        boxes=boxes, adj=adj, src=src, tgt=tgt,
        q_start=q_start, q_goal=q_goal, ndim=ndim,
        label=method_name, **method_kwargs)

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
    """运行 Visibility Graph 方法."""
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
