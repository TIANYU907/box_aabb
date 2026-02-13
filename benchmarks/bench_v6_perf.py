#!/usr/bin/env python
"""
benchmarks/bench_v6_perf.py - v6 优化微基准

测量 Cython SAT 碰撞 + 增量保存的性能指标。

用法：
    python -m benchmarks.bench_v6_perf
    python -m benchmarks.bench_v6_perf --n-obs 10 --n-calls 500
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from box_aabb.robot import Robot, load_robot
from planner.hier_aabb_tree import HierAABBTree
from planner.obstacles import Scene

# ─────────────────────────────────────────────────────────
#  场景生成
# ─────────────────────────────────────────────────────────

def random_scene_3d(n_obs: int, rng: np.random.Generator) -> Scene:
    scene = Scene()
    for i in range(n_obs):
        r = rng.uniform(0.20, 0.72)
        theta = rng.uniform(-math.pi, math.pi)
        cx, cy = r * math.cos(theta), r * math.sin(theta)
        cz = rng.uniform(0.15, 0.85)
        hx = rng.uniform(0.06, 0.18)
        hy = rng.uniform(0.06, 0.18)
        hz = rng.uniform(0.06, 0.18)
        scene.add_obstacle(
            min_point=[cx - hx, cy - hy, cz - hz],
            max_point=[cx + hx, cy + hy, cz + hz],
            name=f"obs_{i}",
        )
    return scene


# ─────────────────────────────────────────────────────────
#  Benchmark 1: find_free_box 吞吐量
# ─────────────────────────────────────────────────────────

def bench_find_free_box(robot, joint_limits, n_obs, n_calls, seed=42):
    rng = np.random.default_rng(seed)
    scene = random_scene_3d(n_obs, rng)
    obstacles = scene.get_obstacles()

    tree = HierAABBTree(robot, joint_limits)
    ndim = robot.n_joints

    # 预生成随机种子点
    seeds = []
    lo = np.array([jl[0] for jl in joint_limits])
    hi = np.array([jl[1] for jl in joint_limits])
    for _ in range(n_calls * 3):
        q = rng.uniform(lo, hi)
        seeds.append(q)

    t0 = time.perf_counter()
    found = 0
    call_i = 0
    nid = 0
    for q in seeds:
        ffb = tree.find_free_box(
            q, obstacles, max_depth=25,
            min_edge_length=0.05,
            mark_occupied=True, forest_box_id=nid,
        )
        call_i += 1
        if ffb is not None:
            found += 1
            nid += 1
        if found >= n_calls:
            break
    dt = time.perf_counter() - t0

    return {
        "total_calls": call_i,
        "found": found,
        "elapsed_s": dt,
        "per_call_ms": dt / max(call_i, 1) * 1000,
        "found_per_s": found / max(dt, 1e-9),
        "tree_nodes": tree.n_nodes,
        "fk_calls": tree.n_fk_calls,
    }


# ─────────────────────────────────────────────────────────
#  Benchmark 2: save_binary vs save_incremental
# ─────────────────────────────────────────────────────────

def bench_save(robot, joint_limits, n_obs, n_boxes_phase1, n_boxes_phase2, seed=42):
    rng = np.random.default_rng(seed)
    scene = random_scene_3d(n_obs, rng)
    obstacles = scene.get_obstacles()

    tree = HierAABBTree(robot, joint_limits)
    lo = np.array([jl[0] for jl in joint_limits])
    hi = np.array([jl[1] for jl in joint_limits])

    # Phase 1: 建树
    nid = 0
    for _ in range(n_boxes_phase1 * 3):
        q = rng.uniform(lo, hi)
        ffb = tree.find_free_box(q, obstacles, max_depth=25,
                                 min_edge_length=0.05,
                                 mark_occupied=True, forest_box_id=nid)
        if ffb is not None:
            nid += 1
        if nid >= n_boxes_phase1:
            break

    nodes_phase1 = tree.n_nodes

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "bench.hcache")

        # 全量保存 phase1
        t0 = time.perf_counter()
        tree.save_binary(path)
        t_full_save_1 = time.perf_counter() - t0
        file_size_1 = os.path.getsize(path)

        # 加载回来
        tree2 = HierAABBTree.load_binary(path, robot)

        # Phase 2: 少量增长
        for _ in range(n_boxes_phase2 * 3):
            q = rng.uniform(lo, hi)
            ffb = tree2.find_free_box(q, obstacles, max_depth=25,
                                      min_edge_length=0.05,
                                      mark_occupied=True, forest_box_id=nid)
            if ffb is not None:
                nid += 1
            if nid >= n_boxes_phase1 + n_boxes_phase2:
                break

        nodes_phase2 = tree2.n_nodes

        # 全量保存 phase2
        path_full = str(Path(tmpdir) / "bench_full.hcache")
        t0 = time.perf_counter()
        tree2.save_binary(path_full)
        t_full_save_2 = time.perf_counter() - t0
        file_size_2 = os.path.getsize(path_full)

        # 重新加载 phase1 → 增量保存
        tree3 = HierAABBTree.load_binary(path, robot)
        for _ in range(n_boxes_phase2 * 3):
            q = rng.uniform(lo, hi)
            ffb = tree3.find_free_box(q, obstacles, max_depth=25,
                                      min_edge_length=0.05,
                                      mark_occupied=True, forest_box_id=nid)
            if ffb is not None:
                nid += 1
            if tree3.n_nodes >= nodes_phase2:
                break

        t0 = time.perf_counter()
        tree3.save_incremental(path)
        t_incr_save = time.perf_counter() - t0
        file_size_incr = os.path.getsize(path)

        # 验证增量保存结果一致
        loaded = HierAABBTree.load_binary(path, robot)
        ok = loaded.n_nodes == tree3.n_nodes

    return {
        "nodes_phase1": nodes_phase1,
        "nodes_phase2": nodes_phase2,
        "full_save_1_ms": t_full_save_1 * 1000,
        "full_save_2_ms": t_full_save_2 * 1000,
        "incr_save_ms": t_incr_save * 1000,
        "speedup": t_full_save_2 / max(t_incr_save, 1e-9),
        "file_size_1_kb": file_size_1 / 1024,
        "file_size_2_kb": file_size_2 / 1024,
        "file_size_incr_kb": file_size_incr / 1024,
        "verify_ok": ok,
    }


# ─────────────────────────────────────────────────────────
#  Benchmark 3: prepack_obstacles_c 开销
# ─────────────────────────────────────────────────────────

def bench_prepack(robot, joint_limits, n_obs, n_iters=1000, seed=42):
    rng = np.random.default_rng(seed)
    scene = random_scene_3d(n_obs, rng)
    obstacles = scene.get_obstacles()
    tree = HierAABBTree(robot, joint_limits)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        packed = tree._prepack_obstacles_c(obstacles, 0.0)
    dt = time.perf_counter() - t0

    n_tuples = len(packed) if packed else 0
    return {
        "n_obs": n_obs,
        "n_active_links": sum(1 for z in tree._zl_list if not z),
        "n_tuples": n_tuples,
        "iters": n_iters,
        "total_ms": dt * 1000,
        "per_call_us": dt / n_iters * 1e6,
    }


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v6 微基准")
    parser.add_argument("--n-obs", type=int, default=5)
    parser.add_argument("--n-calls", type=int, default=200,
                        help="find_free_box 成功次数目标")
    parser.add_argument("--n-phase1", type=int, default=100,
                        help="save 基准 Phase1 box 数")
    parser.add_argument("--n-phase2", type=int, default=50,
                        help="save 基准 Phase2 增量 box 数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    robot = load_robot("panda")
    joint_limits = [
        (p.get("q_min", -2.8973), p.get("q_max", 2.8973))
        for p in robot.dh_params
    ]

    print("=" * 72)
    print("  v6 性能微基准 — Panda 7-DOF")
    print("=" * 72)
    print(f"  n_obs={args.n_obs}  seed={args.seed}")
    print(f"  stride={robot.n_joints + (1 if robot.tool_frame else 0)} links")
    print()

    # ── Benchmark 1 ──
    print("── [1] find_free_box 碰撞 (Cython SAT) ──")
    r1 = bench_find_free_box(robot, joint_limits, args.n_obs,
                             args.n_calls, seed=args.seed)
    print(f"  总调用 = {r1['total_calls']},  成功 = {r1['found']}")
    print(f"  总耗时 = {r1['elapsed_s']:.3f} s")
    print(f"  每次调用 = {r1['per_call_ms']:.3f} ms")
    print(f"  吞吐量 = {r1['found_per_s']:.1f} boxes/s")
    print(f"  tree_nodes = {r1['tree_nodes']},  fk_calls = {r1['fk_calls']}")
    print()

    # ── Benchmark 2 ──
    print("── [2] save_binary vs save_incremental ──")
    r2 = bench_save(robot, joint_limits, args.n_obs,
                    args.n_phase1, args.n_phase2, seed=args.seed)
    print(f"  Phase1 nodes = {r2['nodes_phase1']},  "
          f"Phase2 nodes = {r2['nodes_phase2']}")
    print(f"  全量保存 Phase1 = {r2['full_save_1_ms']:.2f} ms  "
          f"({r2['file_size_1_kb']:.0f} KB)")
    print(f"  全量保存 Phase2 = {r2['full_save_2_ms']:.2f} ms  "
          f"({r2['file_size_2_kb']:.0f} KB)")
    print(f"  增量保存 Phase2 = {r2['incr_save_ms']:.2f} ms  "
          f"({r2['file_size_incr_kb']:.0f} KB)")
    print(f"  加速比 = {r2['speedup']:.2f}x")
    print(f"  验证 = {'OK' if r2['verify_ok'] else 'FAIL'}")
    print()

    # ── Benchmark 3 ──
    print("── [3] _prepack_obstacles_c 开销 ──")
    r3 = bench_prepack(robot, joint_limits, args.n_obs, seed=args.seed)
    print(f"  n_obs={r3['n_obs']}, active_links={r3['n_active_links']}, "
          f"交叉积={r3['n_tuples']} tuples")
    print(f"  {r3['iters']} 次调用 = {r3['total_ms']:.2f} ms  "
          f"({r3['per_call_us']:.1f} µs/call)")
    print()

    print("=" * 72)
    print("  完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
