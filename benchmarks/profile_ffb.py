#!/usr/bin/env python
"""
benchmarks/profile_ffb.py - find_free_box 内部耗时剖析

精确测量 find_free_box 中每个环节的耗时：
  - _compute_aabb（interval FK）单次调用耗时
  - _link_aabbs_collide（碰撞检测）单次调用耗时
  - _split（切分 + 2次FK）单次调用耗时
  - _propagate_up 耗时
  - 上行合并 + promotion 耗时
  - 下行 vs 上行占比

用法：
    python -m benchmarks.profile_ffb
    python -m benchmarks.profile_ffb --n-obs 10 --max-boxes 50
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np

from box_aabb.robot import Robot, load_robot
from planner.models import BoxNode, PlannerConfig, gmean_edge_length
from planner.obstacles import Scene
from planner.collision import CollisionChecker
from planner.box_forest import BoxForest
from planner.hier_aabb_tree import HierAABBTree, HierAABBNode

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("profile_ffb")
logger.setLevel(logging.INFO)


def _normalized_size(vol: float, ndim: int) -> float:
    if vol <= 0 or ndim <= 0:
        return 0.0
    return vol ** (1.0 / ndim)


def random_scene_3d(robot: Robot, n_obs: int, rng: np.random.Generator) -> Scene:
    """随机 3D 障碍物场景"""
    scene = Scene()
    for i in range(n_obs):
        cx = rng.uniform(-0.8, 0.8)
        cy = rng.uniform(-0.8, 0.8)
        cz = rng.uniform(0.0, 1.0)
        sx = rng.uniform(0.05, 0.25)
        sy = rng.uniform(0.05, 0.25)
        sz = rng.uniform(0.05, 0.25)
        scene.add_obstacle(
            min_point=[cx - sx/2, cy - sy/2, cz - sz/2],
            max_point=[cx + sx/2, cy + sy/2, cz + sz/2],
            name=f"obs_{i}",
        )
    return scene


class InstrumentedHierTree:
    """为 HierAABBTree 的 find_free_box 添加详细计时"""

    def __init__(self, hier_tree: HierAABBTree):
        self.ht = hier_tree
        # 累积计时
        self.t_compute_aabb = 0.0
        self.t_collide_check = 0.0
        self.t_split = 0.0
        self.t_propagate_up = 0.0
        self.t_upward_merge = 0.0
        self.t_mark_occupied = 0.0
        # 计数
        self.n_compute_aabb = 0
        self.n_collide_check = 0
        self.n_split = 0
        self.n_propagate_up = 0
        self.n_ffb_calls = 0
        # 深度统计
        self.depths: List[int] = []
        self.path_lengths: List[int] = []

    def find_free_box_instrumented(
        self,
        seed: np.ndarray,
        obstacles: list,
        max_depth: int = 40,
        min_edge_length: float = 0.05,
        mark_occupied: bool = False,
        forest_box_id: Optional[int] = None,
    ):
        """带详细计时的 find_free_box"""
        self.n_ffb_calls += 1
        ht = self.ht

        node = ht.root
        # ensure root aabb
        t0 = time.perf_counter()
        ht._ensure_aabb(node)
        dt_ensure = time.perf_counter() - t0
        self.t_compute_aabb += dt_ensure
        if dt_ensure > 1e-6:
            self.n_compute_aabb += 1

        path: List[HierAABBNode] = []

        # prepack obstacles
        obs_packed = ht._prepack_obstacles(obstacles)

        # ── 下行 ──
        while True:
            if node.occupied:
                return None

            path.append(node)

            aabb = node.refined_aabb if node.refined_aabb is not None else node.raw_aabb

            # collide check
            t0 = time.perf_counter()
            collides = ht._link_aabbs_collide(aabb, obs_packed)
            self.t_collide_check += time.perf_counter() - t0
            self.n_collide_check += 1

            if not collides and node.subtree_occupied == 0:
                break

            if node.depth >= max_depth:
                return None

            split_dim = node.depth % ht.n_dims
            edge = node.intervals[split_dim][1] - node.intervals[split_dim][0]
            if min_edge_length > 0 and edge < min_edge_length * 2:
                return None

            # split
            t0 = time.perf_counter()
            ht._split(node)
            self.t_split += time.perf_counter() - t0
            self.n_split += 1

            if seed[node.split_dim] < node.split_val:
                node = node.left
            else:
                node = node.right

        self.depths.append(node.depth)
        self.path_lengths.append(len(path))

        # ── propagate_up ──
        t0 = time.perf_counter()
        if node.parent is not None:
            ht._propagate_up(node.parent)
        self.t_propagate_up += time.perf_counter() - t0
        self.n_propagate_up += 1

        # ── 上行合并 ──
        t0 = time.perf_counter()
        result_node = node
        absorbed_ids: Set[int] = set()
        for i in range(len(path) - 2, -1, -1):
            parent = path[i]
            aabb = parent.refined_aabb if parent.refined_aabb is not None else parent.raw_aabb

            tc0 = time.perf_counter()
            parent_collides = ht._link_aabbs_collide(aabb, obs_packed)
            dt_c = time.perf_counter() - tc0
            self.t_collide_check += dt_c
            self.n_collide_check += 1

            if parent.subtree_occupied > 0:
                if parent_collides:
                    break
                absorbed_ids |= ht._collect_forest_ids(parent)
                ht._clear_subtree_occupation(parent)
                result_node = parent
            else:
                if not parent_collides:
                    result_node = parent
                else:
                    break
        self.t_upward_merge += time.perf_counter() - t0

        result_intervals = list(result_node.intervals)

        # mark occupied
        t0 = time.perf_counter()
        if mark_occupied:
            ht._mark_occupied(result_node, forest_box_id)
        self.t_mark_occupied += time.perf_counter() - t0

        from planner.hier_aabb_tree import FindFreeBoxResult
        return FindFreeBoxResult(
            intervals=result_intervals,
            absorbed_box_ids=absorbed_ids,
        )

    def report(self) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("  find_free_box 内部耗时剖析")
        lines.append("=" * 80)

        t_total = (self.t_compute_aabb + self.t_collide_check + self.t_split
                   + self.t_propagate_up + self.t_upward_merge + self.t_mark_occupied)

        def pct(t):
            return t / t_total * 100 if t_total > 0 else 0

        lines.append(f"  总 ffb 调用次数: {self.n_ffb_calls}")
        lines.append(f"  总 ffb 聚合耗时: {t_total:.4f}s")
        lines.append("")
        lines.append("  ── 各环节耗时 ──")
        lines.append(f"  {'环节':<25s}  {'耗时':>8s}  {'占比':>6s}  {'次数':>6s}  {'均耗时':>10s}")
        lines.append("  " + "-" * 65)

        items = [
            ("_split (含2×FK)", self.t_split, self.n_split),
            ("_compute_aabb (FK)", self.t_compute_aabb, self.n_compute_aabb),
            ("_link_aabbs_collide", self.t_collide_check, self.n_collide_check),
            ("_propagate_up", self.t_propagate_up, self.n_propagate_up),
            ("upward_merge", self.t_upward_merge, self.n_ffb_calls),
            ("_mark_occupied", self.t_mark_occupied, self.n_ffb_calls),
        ]
        for name, t, n in items:
            avg = t / n * 1000 if n > 0 else 0  # ms
            lines.append(f"  {name:<25s}  {t:>7.4f}s  {pct(t):>5.1f}%  {n:>6d}  {avg:>8.3f}ms")

        lines.append("")
        lines.append("  ── FK 统计 ──")
        # _split 每次产生 2 个 FK 调用
        n_fk_from_split = self.n_split * 2
        n_fk_from_ensure = self.n_compute_aabb
        n_fk_total = n_fk_from_split + n_fk_from_ensure
        t_fk_total = self.t_split + self.t_compute_aabb
        lines.append(f"  FK 总调用次数: {n_fk_total}  (split={n_fk_from_split}, ensure={n_fk_from_ensure})")
        if n_fk_total > 0:
            avg_fk = t_fk_total / n_fk_total * 1000
            lines.append(f"  FK 均耗时: {avg_fk:.3f}ms")
            lines.append(f"  FK 总耗时: {t_fk_total:.4f}s  ({pct(t_fk_total):.1f}%)")

        lines.append("")
        lines.append("  ── 路径深度统计 ──")
        if self.depths:
            lines.append(f"  找到 box 的节点深度: mean={np.mean(self.depths):.1f}  "
                         f"median={np.median(self.depths):.0f}  "
                         f"min={min(self.depths)}  max={max(self.depths)}")
            lines.append(f"  下行路径长度: mean={np.mean(self.path_lengths):.1f}  "
                         f"max={max(self.path_lengths)}")
        lines.append("")

        # 每个 box 平均耗时
        if self.n_ffb_calls > 0:
            avg_per_box = t_total / self.n_ffb_calls * 1000
            lines.append(f"  每次 ffb 平均耗时: {avg_per_box:.2f}ms")

        lines.append("=" * 80)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="find_free_box 内部耗时剖析")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--max-boxes", type=int, default=200)
    parser.add_argument("--max-seeds", type=int, default=3000)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--boundary-batch", type=int, default=6)
    parser.add_argument("--farthest-k", type=int, default=12)
    parser.add_argument("--use-cache", action="store_true", default=False,
                        help="使用缓存 (默认: 冷启动)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    robot = load_robot("panda")
    joint_limits = list(robot.joint_limits)
    ndim = len(joint_limits)

    scene = random_scene_3d(robot, args.n_obs, rng)
    checker = CollisionChecker(robot, scene)
    obstacles = scene.get_obstacles()

    logger.info("加载 HierAABBTree...")
    if args.use_cache:
        hier_tree = HierAABBTree.auto_load(robot, joint_limits)
    else:
        hier_tree = HierAABBTree(robot, joint_limits)

    config = PlannerConfig(hard_overlap_reject=True, verbose=False)
    forest = BoxForest(robot.fingerprint(), joint_limits, config)
    forest.hier_tree = hier_tree

    instrumented = InstrumentedHierTree(hier_tree)

    logger.info("n_obs=%d  max_boxes=%d  use_cache=%s", args.n_obs, args.max_boxes, args.use_cache)

    # ── 采样循环 ──
    n_find_none = 0
    n_find_tiny = 0
    n_collision = 0
    n_inside = 0
    t_sampling = 0.0
    t_collision_check = 0.0

    def _sample_boundary(box_intervals, n, rng_):
        seeds = []
        for _ in range(n):
            dim = rng_.integers(0, ndim)
            side = rng_.integers(0, 2)
            q = np.array([rng_.uniform(lo, hi) for lo, hi in box_intervals])
            q[dim] = box_intervals[dim][side]
            offset = 0.01 if side == 1 else -0.01
            q[dim] = np.clip(q[dim] + offset, joint_limits[dim][0], joint_limits[dim][1])
            seeds.append(q)
        return seeds

    last_box_ivs = None
    global_stalls = 0

    t_main_start = time.time()

    for seed_iter in range(args.max_seeds):
        if forest.n_boxes >= args.max_boxes:
            break
        if global_stalls > 60:
            break

        added = False

        # DFS boundary
        if last_box_ivs is not None:
            ts0 = time.perf_counter()
            bseeds = _sample_boundary(last_box_ivs, args.boundary_batch, rng)
            t_sampling += time.perf_counter() - ts0

            for bq in bseeds:
                if forest.n_boxes >= args.max_boxes:
                    break
                tc0 = time.perf_counter()
                coll = checker.check_config_collision(bq)
                t_collision_check += time.perf_counter() - tc0
                if coll:
                    n_collision += 1
                    continue
                if hier_tree.is_occupied(bq):
                    n_inside += 1
                    continue

                nid = forest.allocate_id()
                ffb = instrumented.find_free_box_instrumented(
                    bq, obstacles, max_depth=args.max_depth,
                    min_edge_length=args.min_edge,
                    mark_occupied=True, forest_box_id=nid)
                if ffb is None:
                    n_find_none += 1
                    continue
                ivs = ffb.intervals
                vol = 1.0
                for lo, hi in ivs:
                    vol *= max(hi - lo, 0.0)
                nsize = _normalized_size(vol, ndim)
                if nsize < 1e-4:
                    n_find_tiny += 1
                    continue
                if ffb.absorbed_box_ids:
                    forest.remove_boxes(ffb.absorbed_box_ids)
                box = BoxNode(node_id=nid, joint_intervals=ivs,
                              seed_config=bq.copy(), volume=vol)
                forest.add_box_direct(box)
                last_box_ivs = ivs
                added = True
                global_stalls = 0

        # farthest point
        if not added and forest.n_boxes > 0:
            boxes_list = list(forest.boxes.values())
            centers = np.array([b.center for b in boxes_list])
            candidates = []
            for _ in range(args.farthest_k):
                q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
                candidates.append(q)
            if candidates:
                candidates = np.array(candidates)
                best_q = None
                best_dist = -1
                for q in candidates:
                    d = np.min(np.linalg.norm(centers - q, axis=1))
                    if d > best_dist:
                        best_dist = d
                        best_q = q

                tc0 = time.perf_counter()
                coll = checker.check_config_collision(best_q)
                t_collision_check += time.perf_counter() - tc0
                if not coll and not hier_tree.is_occupied(best_q):
                    nid = forest.allocate_id()
                    ffb = instrumented.find_free_box_instrumented(
                        best_q, obstacles, max_depth=args.max_depth,
                        min_edge_length=args.min_edge,
                        mark_occupied=True, forest_box_id=nid)
                    if ffb is not None:
                        ivs = ffb.intervals
                        vol = 1.0
                        for lo, hi in ivs:
                            vol *= max(hi - lo, 0.0)
                        nsize = _normalized_size(vol, ndim)
                        if nsize >= 1e-4:
                            if ffb.absorbed_box_ids:
                                forest.remove_boxes(ffb.absorbed_box_ids)
                            box = BoxNode(node_id=nid, joint_intervals=ivs,
                                          seed_config=best_q.copy(), volume=vol)
                            forest.add_box_direct(box)
                            last_box_ivs = ivs
                            added = True
                            global_stalls = 0

        # random fallback
        if not added:
            ts0 = time.perf_counter()
            q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
            t_sampling += time.perf_counter() - ts0

            tc0 = time.perf_counter()
            coll = checker.check_config_collision(q)
            t_collision_check += time.perf_counter() - tc0
            if not coll and not hier_tree.is_occupied(q):
                nid = forest.allocate_id()
                ffb = instrumented.find_free_box_instrumented(
                    q, obstacles, max_depth=args.max_depth,
                    min_edge_length=args.min_edge,
                    mark_occupied=True, forest_box_id=nid)
                if ffb is not None:
                    ivs = ffb.intervals
                    vol = 1.0
                    for lo, hi in ivs:
                        vol *= max(hi - lo, 0.0)
                    nsize = _normalized_size(vol, ndim)
                    if nsize >= 1e-4:
                        if ffb.absorbed_box_ids:
                            forest.remove_boxes(ffb.absorbed_box_ids)
                        box = BoxNode(node_id=nid, joint_intervals=ivs,
                                      seed_config=q.copy(), volume=vol)
                        forest.add_box_direct(box)
                        last_box_ivs = ivs
                        added = True
                        global_stalls = 0

        if not added:
            global_stalls += 1

    dt_main = time.time() - t_main_start

    # ── 报告 ──
    print()
    print(instrumented.report())
    print()
    print(f"  总循环数: {seed_iter + 1}  boxes: {forest.n_boxes}  "
          f"总耗时: {dt_main:.2f}s")
    print(f"  采样: {t_sampling:.4f}s  碰撞检测: {t_collision_check:.4f}s")
    print(f"  find_none: {n_find_none}  find_tiny: {n_find_tiny}  "
          f"collision: {n_collision}  inside: {n_inside}")

    # ── interval FK 单次调用微基准 ──
    print()
    print("=" * 80)
    print("  Interval FK 微基准 (100 次随机区间)")
    print("=" * 80)
    from box_aabb.aabb_calculator import compute_interval_aabb
    zero_len = hier_tree._zero_length_links

    # 不同宽度的区间
    for width_label, width_factor in [("全范围", 1.0), ("1/4范围", 0.25), ("1/16范围", 0.0625)]:
        times = []
        for _ in range(100):
            center = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
            ivs = []
            for d, (lo, hi) in enumerate(joint_limits):
                half = (hi - lo) * width_factor / 2
                c = center[d]
                ivs.append((max(lo, c - half), min(hi, c + half)))
            t0 = time.perf_counter()
            compute_interval_aabb(
                robot=robot, intervals=ivs,
                zero_length_links=zero_len,
                skip_zero_length=True, n_sub=1)
            times.append(time.perf_counter() - t0)
        times = np.array(times) * 1000  # ms
        print(f"  {width_label:>10s}: mean={np.mean(times):.3f}ms  "
              f"median={np.median(times):.3f}ms  "
              f"p95={np.percentile(times, 95):.3f}ms  "
              f"max={np.max(times):.3f}ms")

    print()


if __name__ == "__main__":
    main()
