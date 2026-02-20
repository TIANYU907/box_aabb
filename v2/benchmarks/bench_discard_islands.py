"""
bench_discard_islands.py — 评估舍弃小岛对 bridge_islands 耗时的影响

实验设计:
  - 多个随机 seed 生成不同场景
  - 每个场景分别以 min_island_size=0 (不舍弃) 和 min_island_size=0.5 (舍弃) 运行 bridge_islands
  - 对比桥接耗时、bridge 数量、被舍弃的岛/box 数量
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List

import matplotlib
matplotlib.use("Agg")

import numpy as np

# Ensure v2 is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "v2", "src"))
sys.path.insert(0, os.path.join(_root, "v2"))
from v2._bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from planner.box_rrt import BoxRRT
from planner.models import PlannerConfig, gmean_edge_length
from forest.models import BoxNode
from forest.connectivity import find_islands, bridge_islands


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [20260219, 42, 123, 7777, 88888, 314159, 271828, 100, 2026, 999]
MIN_ISLAND_SIZES = [0.0, 0.5]   # 对比组
N_OBSTACLES = 8
MAX_CONSECUTIVE_MISS = 20
ROBOT_NAME = "2dof_planar"
Q_START_FACTOR = [0.8, 0.2]     # × π
Q_GOAL_FACTOR = [-0.7, -0.4]    # × π (for start/goal per scene)


@dataclass
class RunResult:
    seed: int
    min_island_size: float
    n_boxes: int
    n_islands_before: int
    n_islands_after: int
    n_box_bridges: int
    n_seg_bridges: int
    n_discarded_islands: int
    n_discarded_boxes: int
    bridge_time_ms: float
    total_time_ms: float       # forest grow + bridge


def build_scene(robot, q_start, q_goal, rng, n_obs=8, max_trials=200):
    for _ in range(max_trials):
        scene = Scene()
        for i in range(n_obs):
            cx = float(rng.uniform(-1.6, 1.6))
            cy = float(rng.uniform(-1.6, 1.6))
            w = float(rng.uniform(0.25, 0.65))
            h = float(rng.uniform(0.25, 0.65))
            scene.add_obstacle([cx - w * 0.5, cy - h * 0.5],
                               [cx + w * 0.5, cy + h * 0.5], name=f"obs_{i}")
        checker = CollisionChecker(robot=robot, scene=scene)
        if (not checker.check_config_collision(q_start)
                and not checker.check_config_collision(q_goal)
                and checker.check_segment_collision(q_start, q_goal, 0.03)):
            return scene
    return None


def grow_forest(planner, q_start, q_goal, seed, max_miss=20):
    """生长 forest，返回 (boxes_dict_copy, forest_obj, grow_time_ms)."""
    rng = np.random.default_rng(seed)
    forest = planner._load_or_create_forest()
    forest.hier_tree = planner.hier_tree

    t0 = time.perf_counter()

    # seed boxes
    for qs in [q_start, q_goal]:
        if not planner.hier_tree.is_occupied(qs):
            nid = forest.allocate_id()
            ffb = planner.hier_tree.find_free_box(
                qs, planner.obstacles, mark_occupied=True, forest_box_id=nid)
            if ffb:
                vol = 1.0
                for lo, hi in ffb.intervals:
                    vol *= max(hi - lo, 0)
                forest.add_box_direct(BoxNode(
                    node_id=nid, joint_intervals=ffb.intervals,
                    seed_config=qs.copy(), volume=vol))

    consec = 0
    while consec < max_miss:
        q = planner._sample_seed(q_start, q_goal, rng)
        if q is None:
            consec += 1; continue
        if planner.hier_tree.is_occupied(q):
            consec += 1; continue
        nid = forest.allocate_id()
        ffb = planner.hier_tree.find_free_box(
            q, planner.obstacles, mark_occupied=True, forest_box_id=nid)
        if ffb is None:
            consec += 1; continue
        vol = 1.0
        for lo, hi in ffb.intervals:
            vol *= max(hi - lo, 0)
        if gmean_edge_length(vol, 2) < 0.001:
            consec += 1; continue
        box = BoxNode(node_id=nid, joint_intervals=ffb.intervals,
                      seed_config=q.copy(), volume=vol)
        if ffb.absorbed_box_ids:
            forest.remove_boxes(ffb.absorbed_box_ids)
        forest.add_box_direct(box)
        consec = 0

    grow_time = (time.perf_counter() - t0) * 1000.0

    # deep-copy boxes for benchmark reuse
    boxes_copy = {}
    for bid, b in forest.boxes.items():
        boxes_copy[bid] = BoxNode(
            node_id=b.node_id,
            joint_intervals=[tuple(iv) for iv in b.joint_intervals],
            seed_config=b.seed_config.copy(),
            volume=b.volume,
        )
    return boxes_copy, forest, grow_time


def run_bridge(boxes_template, planner, forest, period, min_island_size):
    """对 boxes 深拷贝后运行 bridge_islands，返回 (RunResult-like dict, time)."""
    # 深拷贝以免污染
    boxes = {}
    for bid, b in boxes_template.items():
        boxes[bid] = BoxNode(
            node_id=b.node_id,
            joint_intervals=[tuple(iv) for iv in b.joint_intervals],
            seed_config=b.seed_config.copy(),
            volume=b.volume,
        )

    t0 = time.perf_counter()
    bridge_edges, final_islands, n_before, bridge_boxes, discarded = bridge_islands(
        boxes=boxes,
        collision_checker=planner.collision_checker,
        segment_resolution=0.03,
        max_pairs_per_island_pair=10,
        max_rounds=5,
        period=period,
        hier_tree=planner.hier_tree,
        obstacles=planner.obstacles,
        forest=forest,
        min_box_size=0.001,
        n_bridge_seeds=7,
        min_island_size=min_island_size,
    )
    bridge_time = (time.perf_counter() - t0) * 1000.0

    return {
        "n_islands_before": n_before,
        "n_islands_after": len(final_islands),
        "n_box_bridges": len(bridge_boxes),
        "n_seg_bridges": len(bridge_edges),
        "n_discarded_islands": len(discarded),
        "n_discarded_boxes": sum(len(s) for s in discarded),
        "bridge_time_ms": bridge_time,
    }


def main():
    robot = load_robot(ROBOT_NAME)
    pi = np.pi
    jl = robot.joint_limits[0]
    period = float(jl[1] - jl[0])

    results: List[RunResult] = []

    header = (
        f"{'seed':>10} | {'min_isl':>7} | {'boxes':>5} | "
        f"{'isl_b':>5} {'isl_a':>5} | {'box_br':>6} {'seg_br':>6} | "
        f"{'disc_i':>6} {'disc_b':>6} | {'bridge_ms':>10} {'total_ms':>10}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        q_start = np.array([Q_START_FACTOR[0] * pi, Q_START_FACTOR[1]])
        q_goal = np.array([Q_GOAL_FACTOR[0] * pi, Q_GOAL_FACTOR[1]])

        scene = build_scene(robot, q_start, q_goal, rng, n_obs=N_OBSTACLES)
        if scene is None:
            print(f"  seed={seed}: failed to build scene, skip")
            continue

        planner_cfg = PlannerConfig(
            min_box_size=0.001,
            goal_bias=0.15,
            guided_sample_ratio=0.6,
            segment_collision_resolution=0.03,
        )
        planner = BoxRRT(robot=robot, scene=scene, config=planner_cfg)

        # 只生长一次 forest
        boxes_template, forest_obj, grow_ms = grow_forest(
            planner, q_start, q_goal, seed, max_miss=MAX_CONSECUTIVE_MISS,
        )
        n_boxes = len(boxes_template)

        for mis in MIN_ISLAND_SIZES:
            info = run_bridge(boxes_template, planner, forest_obj, period, mis)
            total_ms = grow_ms + info["bridge_time_ms"]

            r = RunResult(
                seed=seed,
                min_island_size=mis,
                n_boxes=n_boxes,
                n_islands_before=info["n_islands_before"],
                n_islands_after=info["n_islands_after"],
                n_box_bridges=info["n_box_bridges"],
                n_seg_bridges=info["n_seg_bridges"],
                n_discarded_islands=info["n_discarded_islands"],
                n_discarded_boxes=info["n_discarded_boxes"],
                bridge_time_ms=info["bridge_time_ms"],
                total_time_ms=total_ms,
            )
            results.append(r)

            row = (
                f"{r.seed:>10} | {r.min_island_size:>7.2f} | {r.n_boxes:>5} | "
                f"{r.n_islands_before:>5} {r.n_islands_after:>5} | "
                f"{r.n_box_bridges:>6} {r.n_seg_bridges:>6} | "
                f"{r.n_discarded_islands:>6} {r.n_discarded_boxes:>6} | "
                f"{r.bridge_time_ms:>10.2f} {r.total_time_ms:>10.2f}"
            )
            print(row)

    # ---------- 汇总统计 ----------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for mis in MIN_ISLAND_SIZES:
        group = [r for r in results if r.min_island_size == mis]
        if not group:
            continue
        avg_bridge = np.mean([r.bridge_time_ms for r in group])
        avg_total = np.mean([r.total_time_ms for r in group])
        avg_isl_before = np.mean([r.n_islands_before for r in group])
        avg_isl_after = np.mean([r.n_islands_after for r in group])
        avg_disc_i = np.mean([r.n_discarded_islands for r in group])
        avg_disc_b = np.mean([r.n_discarded_boxes for r in group])
        avg_seg = np.mean([r.n_seg_bridges for r in group])
        print(
            f"  min_island_size={mis:.2f}: "
            f"avg_bridge={avg_bridge:.2f}ms, avg_total={avg_total:.2f}ms, "
            f"avg_islands={avg_isl_before:.1f}->{avg_isl_after:.1f}, "
            f"avg_discarded={avg_disc_i:.1f} islands ({avg_disc_b:.1f} boxes), "
            f"avg_seg_bridges={avg_seg:.1f}"
        )

    # 对比 speedup
    g0 = [r for r in results if r.min_island_size == 0.0]
    g1 = [r for r in results if r.min_island_size == 0.5]
    if g0 and g1:
        seeds_both = set(r.seed for r in g0) & set(r.seed for r in g1)
        speedups = []
        for s in sorted(seeds_both):
            t0 = next(r.bridge_time_ms for r in g0 if r.seed == s)
            t1 = next(r.bridge_time_ms for r in g1 if r.seed == s)
            if t1 > 0:
                speedups.append(t0 / t1)
        if speedups:
            print(f"\n  Bridge speedup (0.0 vs 0.5): "
                  f"mean={np.mean(speedups):.2f}x, "
                  f"min={np.min(speedups):.2f}x, "
                  f"max={np.max(speedups):.2f}x")


if __name__ == "__main__":
    main()
