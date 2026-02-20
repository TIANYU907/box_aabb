"""
rrt_family_panda.py — RRT 系列算法对比 (Panda 7-DOF, 同场景)

复用 panda_planner.py 完全相同的:
  - Panda 7-DOF 机器人
  - 起/终点配置
  - 随机障碍物场景 (通过 seed 复现)
  - 碰撞检测 (box_aabb CollisionChecker)

实现的 RRT 系列算法 (纯 Python, 共用碰撞检测后端):
  1. RRT          — 基础单向 RRT
  2. RRT-Connect  — 双向 RRT (Kuffner & LaValle, 2000)
  3. RRT*         — 渐近最优 RRT (Karaman & Frazzoli, 2011)
  4. Informed-RRT* — 椭球采样 RRT* (Gammell et al., 2014)
  5. BiRRT*       — 双向 RRT*

每种方法使用相同碰撞检测后端, 确保公平对比。

用法:
    python -m v2.examples.rrt_family_panda
    python -m v2.examples.rrt_family_panda --seed 12345 --timeout 30
    python -m v2.examples.rrt_family_panda --trials 5 --algorithms RRT RRTConnect
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys, os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
from _bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from common.output import make_output_dir

# 复用 panda_planner 中的场景构建
from v2.examples.panda_planner import (
    PandaGCSConfig,
    build_panda_scene,
)


# ═══════════════════════════════════════════════════════════════
# KD-Tree for nearest-neighbor (simple, exact)
# ═══════════════════════════════════════════════════════════════

class _NodePool:
    """用 numpy 数组存储 RRT 节点, 避免 Python 对象开销."""
    __slots__ = ('configs', 'parents', 'costs', 'n', 'cap', 'ndim')

    def __init__(self, ndim: int, cap: int = 8192):
        self.ndim = ndim
        self.cap = cap
        self.configs = np.empty((cap, ndim), dtype=np.float64)
        self.parents = np.full(cap, -1, dtype=np.int32)
        self.costs = np.zeros(cap, dtype=np.float64)
        self.n = 0

    def add(self, config: np.ndarray, parent: int, cost: float) -> int:
        if self.n >= self.cap:
            self.cap *= 2
            new_configs = np.empty((self.cap, self.ndim), dtype=np.float64)
            new_configs[:self.n] = self.configs[:self.n]
            self.configs = new_configs
            new_parents = np.full(self.cap, -1, dtype=np.int32)
            new_parents[:self.n] = self.parents[:self.n]
            self.parents = new_parents
            new_costs = np.zeros(self.cap, dtype=np.float64)
            new_costs[:self.n] = self.costs[:self.n]
            self.costs = new_costs
        idx = self.n
        self.configs[idx] = config
        self.parents[idx] = parent
        self.costs[idx] = cost
        self.n += 1
        return idx

    def nearest(self, config: np.ndarray) -> int:
        diffs = self.configs[:self.n] - config
        dists = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dists))

    def near(self, config: np.ndarray, radius: float) -> List[int]:
        diffs = self.configs[:self.n] - config
        dists = np.sum(diffs * diffs, axis=1)
        r_sq = radius * radius
        return list(np.where(dists <= r_sq)[0])

    def extract_path(self, goal_idx: int) -> List[np.ndarray]:
        path = []
        idx = goal_idx
        while idx >= 0:
            path.append(self.configs[idx].copy())
            idx = int(self.parents[idx])
        path.reverse()
        return path


# ═══════════════════════════════════════════════════════════════
# Common helpers
# ═══════════════════════════════════════════════════════════════

def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    """向目标方向延伸 step_size 距离, 不超过终点."""
    diff = q_to - q_from
    dist = np.linalg.norm(diff)
    if dist <= step_size:
        return q_to.copy()
    return q_from + (step_size / dist) * diff


def _collision_free_segment(
    checker: CollisionChecker,
    q_from: np.ndarray,
    q_to: np.ndarray,
    resolution: float,
) -> bool:
    """True = 线段无碰撞."""
    return not checker.check_segment_collision(q_from, q_to, resolution)


def _path_length(waypoints: Sequence[np.ndarray]) -> float:
    if len(waypoints) < 2:
        return 0.0
    return float(sum(
        np.linalg.norm(waypoints[i] - waypoints[i - 1])
        for i in range(1, len(waypoints))
    ))


def _shortcut(path: List[np.ndarray], checker: CollisionChecker,
              resolution: float, max_iters: int = 200,
              rng: np.random.Generator = None) -> List[np.ndarray]:
    """随机 shortcut 路径平滑."""
    if rng is None:
        rng = np.random.default_rng()
    if len(path) <= 2:
        return path
    result = list(path)
    for _ in range(max_iters):
        if len(result) <= 2:
            break
        i = rng.integers(0, len(result) - 2)
        j = rng.integers(i + 2, len(result))
        if _collision_free_segment(checker, result[i], result[j], resolution):
            result = result[:i + 1] + result[j:]
    return result


def _rrt_rewiring_radius(ndim: int, n_nodes: int, gamma: float = None) -> float:
    """RRT* 的搜索半径 r_n = gamma * (log(n)/n)^(1/d)."""
    if gamma is None:
        # 常见取值: 基于单位球体积
        unit_ball = math.pi ** (ndim / 2.0) / math.gamma(ndim / 2.0 + 1.0)
        gamma = 2.0 * (1.0 + 1.0 / ndim) ** (1.0 / ndim) * \
                (1.0 / unit_ball) ** (1.0 / ndim) * 2.0
    if n_nodes < 2:
        return float('inf')
    r = gamma * (math.log(n_nodes) / n_nodes) ** (1.0 / ndim)
    return r


# ═══════════════════════════════════════════════════════════════
# 1. RRT (basic, single-tree)
# ═══════════════════════════════════════════════════════════════

def plan_rrt(
    q_start: np.ndarray, q_goal: np.ndarray,
    joint_limits: List[Tuple[float, float]],
    checker: CollisionChecker,
    timeout: float = 30.0,
    step_size: float = 0.5,
    goal_bias: float = 0.05,
    goal_tol: float = 0.3,
    resolution: float = 0.05,
    seed: int = 42,
) -> Dict:
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)
    pool = _NodePool(ndim)
    pool.add(q_start, -1, 0.0)

    n_collision_checks = 0
    t0 = time.perf_counter()
    n_iters = 0

    while time.perf_counter() - t0 < timeout:
        n_iters += 1

        # sample
        if rng.uniform() < goal_bias:
            q_rand = q_goal.copy()
        else:
            q_rand = rng.uniform(lows, highs)

        # nearest
        idx_near = pool.nearest(q_rand)
        q_near = pool.configs[idx_near]

        # steer
        q_new = _steer(q_near, q_rand, step_size)

        # collision check
        n_collision_checks += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            continue

        cost_new = pool.costs[idx_near] + float(np.linalg.norm(q_new - q_near))
        idx_new = pool.add(q_new, idx_near, cost_new)

        # goal check
        if np.linalg.norm(q_new - q_goal) < goal_tol:
            # connect to exact goal
            n_collision_checks += 1
            if _collision_free_segment(checker, q_new, q_goal, resolution):
                cost_goal = cost_new + float(np.linalg.norm(q_goal - q_new))
                idx_goal = pool.add(q_goal, idx_new, cost_goal)
                dt = time.perf_counter() - t0
                path = pool.extract_path(idx_goal)
                return {
                    "success": True, "plan_time_s": dt,
                    "path_length": _path_length(path),
                    "n_nodes": pool.n, "n_iters": n_iters,
                    "n_collision_checks": n_collision_checks,
                    "waypoints": path,
                }

    dt = time.perf_counter() - t0
    return {
        "success": False, "plan_time_s": dt,
        "path_length": float("nan"),
        "n_nodes": pool.n, "n_iters": n_iters,
        "n_collision_checks": n_collision_checks,
        "waypoints": [],
    }


# ═══════════════════════════════════════════════════════════════
# 2. RRT-Connect (bidirectional)
# ═══════════════════════════════════════════════════════════════

def plan_rrt_connect(
    q_start: np.ndarray, q_goal: np.ndarray,
    joint_limits: List[Tuple[float, float]],
    checker: CollisionChecker,
    timeout: float = 30.0,
    step_size: float = 0.5,
    resolution: float = 0.05,
    seed: int = 42,
) -> Dict:
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)

    # two trees
    tree_a = _NodePool(ndim)
    tree_b = _NodePool(ndim)
    tree_a.add(q_start, -1, 0.0)
    tree_b.add(q_goal, -1, 0.0)

    n_collision_checks = 0
    t0 = time.perf_counter()
    n_iters = 0
    swapped = False  # tracks whether tree_a is actually the goal tree

    def _extend(tree, q_target):
        nonlocal n_collision_checks
        idx_near = tree.nearest(q_target)
        q_near = tree.configs[idx_near]
        q_new = _steer(q_near, q_target, step_size)
        n_collision_checks += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            return None, None
        cost = tree.costs[idx_near] + float(np.linalg.norm(q_new - q_near))
        idx_new = tree.add(q_new, idx_near, cost)
        return idx_new, q_new

    def _connect(tree, q_target):
        """反复 extend 直到到达目标或被阻挡."""
        while True:
            idx, q_new = _extend(tree, q_target)
            if idx is None:
                return None, None
            if np.linalg.norm(q_new - q_target) < 1e-6:
                return idx, q_new

    while time.perf_counter() - t0 < timeout:
        n_iters += 1

        # random sample
        q_rand = rng.uniform(lows, highs)

        # extend tree_a toward random sample
        idx_a, q_new_a = _extend(tree_a, q_rand)
        if idx_a is None:
            continue

        # connect tree_b toward the new node
        idx_b, q_new_b = _connect(tree_b, q_new_a)
        if idx_b is not None:
            # connected! build path
            dt = time.perf_counter() - t0
            path_a = tree_a.extract_path(idx_a)
            path_b = tree_b.extract_path(idx_b)
            path_b.reverse()
            if swapped:
                path_a, path_b = path_b, path_a
                path_a.reverse()
                path_b.reverse()
            full_path = path_a + path_b[1:]
            return {
                "success": True, "plan_time_s": dt,
                "path_length": _path_length(full_path),
                "n_nodes": tree_a.n + tree_b.n, "n_iters": n_iters,
                "n_collision_checks": n_collision_checks,
                "waypoints": full_path,
            }

        # swap trees
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    dt = time.perf_counter() - t0
    return {
        "success": False, "plan_time_s": dt,
        "path_length": float("nan"),
        "n_nodes": tree_a.n + tree_b.n, "n_iters": n_iters,
        "n_collision_checks": n_collision_checks,
        "waypoints": [],
    }


# ═══════════════════════════════════════════════════════════════
# 3. RRT* (asymptotically optimal)
# ═══════════════════════════════════════════════════════════════

def plan_rrt_star(
    q_start: np.ndarray, q_goal: np.ndarray,
    joint_limits: List[Tuple[float, float]],
    checker: CollisionChecker,
    timeout: float = 30.0,
    step_size: float = 0.5,
    goal_bias: float = 0.05,
    goal_tol: float = 0.3,
    resolution: float = 0.05,
    seed: int = 42,
) -> Dict:
    """渐近最优 RRT* (anytime): 持续改进路径质量."""
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)
    pool = _NodePool(ndim)
    pool.add(q_start, -1, 0.0)

    best_goal_idx = -1
    best_cost = float('inf')
    n_collision_checks = 0
    t0 = time.perf_counter()
    n_iters = 0
    cost_history: List[Tuple[float, float]] = []   # (elapsed_s, best_cost)
    first_solution_time = float('nan')

    while time.perf_counter() - t0 < timeout:
        n_iters += 1

        # sample
        if rng.uniform() < goal_bias:
            q_rand = q_goal.copy()
        else:
            q_rand = rng.uniform(lows, highs)

        # nearest + steer
        idx_near = pool.nearest(q_rand)
        q_near = pool.configs[idx_near]
        q_new = _steer(q_near, q_rand, step_size)

        n_collision_checks += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            continue

        # find best parent in neighborhood
        r = min(step_size * 2.0, _rrt_rewiring_radius(ndim, pool.n))
        near_idxs = pool.near(q_new, r)

        best_parent = idx_near
        best_cost_new = pool.costs[idx_near] + float(np.linalg.norm(q_new - q_near))

        for ni in near_idxs:
            d = float(np.linalg.norm(q_new - pool.configs[ni]))
            c = pool.costs[ni] + d
            if c < best_cost_new:
                n_collision_checks += 1
                if _collision_free_segment(checker, pool.configs[ni], q_new, resolution):
                    best_parent = ni
                    best_cost_new = c

        idx_new = pool.add(q_new, best_parent, best_cost_new)

        # rewire neighbors
        for ni in near_idxs:
            if ni == best_parent:
                continue
            d = float(np.linalg.norm(pool.configs[ni] - q_new))
            c_through_new = best_cost_new + d
            if c_through_new < pool.costs[ni]:
                n_collision_checks += 1
                if _collision_free_segment(checker, q_new, pool.configs[ni], resolution):
                    pool.parents[ni] = idx_new
                    pool.costs[ni] = c_through_new

        # goal check
        d_goal = float(np.linalg.norm(q_new - q_goal))
        if d_goal < goal_tol:
            n_collision_checks += 1
            if _collision_free_segment(checker, q_new, q_goal, resolution):
                cost_goal = best_cost_new + d_goal
                if cost_goal < best_cost:
                    elapsed = time.perf_counter() - t0
                    if best_goal_idx < 0:
                        first_solution_time = elapsed
                    idx_goal = pool.add(q_goal, idx_new, cost_goal)
                    best_goal_idx = idx_goal
                    best_cost = cost_goal
                    cost_history.append((elapsed, best_cost))

    dt = time.perf_counter() - t0
    if best_goal_idx >= 0:
        path = pool.extract_path(best_goal_idx)
        return {
            "success": True, "plan_time_s": dt,
            "path_length": _path_length(path),
            "n_nodes": pool.n, "n_iters": n_iters,
            "n_collision_checks": n_collision_checks,
            "waypoints": path,
            "cost_history": cost_history,
            "first_solution_time": first_solution_time,
        }
    return {
        "success": False, "plan_time_s": dt,
        "path_length": float("nan"),
        "n_nodes": pool.n, "n_iters": n_iters,
        "n_collision_checks": n_collision_checks,
        "waypoints": [],
        "cost_history": cost_history,
        "first_solution_time": first_solution_time,
    }


# ═══════════════════════════════════════════════════════════════
# 4. Informed-RRT* (ellipsoidal sampling)
# ═══════════════════════════════════════════════════════════════

def _informed_sample(
    q_start: np.ndarray, q_goal: np.ndarray,
    c_best: float, lows: np.ndarray, highs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """从 start-goal 椭球内均匀采样 (Gammell et al., 2014)."""
    ndim = len(q_start)
    c_min = float(np.linalg.norm(q_goal - q_start))

    if c_best >= 1e10 or c_best <= c_min + 1e-10:
        # no solution yet or optimal → uniform
        return rng.uniform(lows, highs)

    # rotation matrix: align x-axis with start→goal
    a1 = (q_goal - q_start) / c_min
    # Gram-Schmidt for remaining axes
    M = np.eye(ndim)
    M[:, 0] = a1
    # QR decomposition gives orthonormal basis
    Q, _ = np.linalg.qr(M)
    # ensure first column is a1
    if np.dot(Q[:, 0], a1) < 0:
        Q[:, 0] = -Q[:, 0]

    # semi-axes
    r1 = c_best / 2.0
    ri = math.sqrt(max(0, c_best * c_best - c_min * c_min)) / 2.0
    L = np.diag([r1] + [ri] * (ndim - 1))

    center = (q_start + q_goal) / 2.0

    # sample from unit ball
    u = rng.standard_normal(ndim)
    u = u / np.linalg.norm(u)
    r = rng.uniform() ** (1.0 / ndim)
    x_ball = r * u

    q = center + Q @ L @ x_ball
    return np.clip(q, lows, highs)


def plan_informed_rrt_star(
    q_start: np.ndarray, q_goal: np.ndarray,
    joint_limits: List[Tuple[float, float]],
    checker: CollisionChecker,
    timeout: float = 30.0,
    step_size: float = 0.5,
    goal_bias: float = 0.05,
    goal_tol: float = 0.3,
    resolution: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Informed-RRT* (anytime): 椭球采样渐近最优."""
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)
    pool = _NodePool(ndim)
    pool.add(q_start, -1, 0.0)

    best_goal_idx = -1
    best_cost = float('inf')
    n_collision_checks = 0
    t0 = time.perf_counter()
    n_iters = 0
    cost_history: List[Tuple[float, float]] = []
    first_solution_time = float('nan')

    while time.perf_counter() - t0 < timeout:
        n_iters += 1

        # informed sampling
        if best_cost < float('inf') and rng.uniform() > goal_bias:
            q_rand = _informed_sample(q_start, q_goal, best_cost, lows, highs, rng)
        elif rng.uniform() < goal_bias:
            q_rand = q_goal.copy()
        else:
            q_rand = rng.uniform(lows, highs)

        idx_near = pool.nearest(q_rand)
        q_near = pool.configs[idx_near]
        q_new = _steer(q_near, q_rand, step_size)

        n_collision_checks += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            continue

        r = min(step_size * 2.0, _rrt_rewiring_radius(ndim, pool.n))
        near_idxs = pool.near(q_new, r)

        best_parent = idx_near
        best_cost_new = pool.costs[idx_near] + float(np.linalg.norm(q_new - q_near))

        for ni in near_idxs:
            d = float(np.linalg.norm(q_new - pool.configs[ni]))
            c = pool.costs[ni] + d
            if c < best_cost_new:
                n_collision_checks += 1
                if _collision_free_segment(checker, pool.configs[ni], q_new, resolution):
                    best_parent = ni
                    best_cost_new = c

        idx_new = pool.add(q_new, best_parent, best_cost_new)

        # rewire
        for ni in near_idxs:
            if ni == best_parent:
                continue
            d = float(np.linalg.norm(pool.configs[ni] - q_new))
            c_through_new = best_cost_new + d
            if c_through_new < pool.costs[ni]:
                n_collision_checks += 1
                if _collision_free_segment(checker, q_new, pool.configs[ni], resolution):
                    pool.parents[ni] = idx_new
                    pool.costs[ni] = c_through_new

        # goal check
        d_goal = float(np.linalg.norm(q_new - q_goal))
        if d_goal < goal_tol:
            n_collision_checks += 1
            if _collision_free_segment(checker, q_new, q_goal, resolution):
                cost_goal = best_cost_new + d_goal
                if cost_goal < best_cost:
                    elapsed = time.perf_counter() - t0
                    if best_goal_idx < 0:
                        first_solution_time = elapsed
                    idx_goal = pool.add(q_goal, idx_new, cost_goal)
                    best_goal_idx = idx_goal
                    best_cost = cost_goal
                    cost_history.append((elapsed, best_cost))

    dt = time.perf_counter() - t0
    if best_goal_idx >= 0:
        path = pool.extract_path(best_goal_idx)
        return {
            "success": True, "plan_time_s": dt,
            "path_length": _path_length(path),
            "n_nodes": pool.n, "n_iters": n_iters,
            "n_collision_checks": n_collision_checks,
            "waypoints": path,
            "cost_history": cost_history,
            "first_solution_time": first_solution_time,
        }
    return {
        "success": False, "plan_time_s": dt,
        "path_length": float("nan"),
        "n_nodes": pool.n, "n_iters": n_iters,
        "n_collision_checks": n_collision_checks,
        "waypoints": [],
        "cost_history": cost_history,
        "first_solution_time": first_solution_time,
    }


# ═══════════════════════════════════════════════════════════════
# 5. Bi-RRT* (bidirectional RRT*)
# ═══════════════════════════════════════════════════════════════

def plan_birrt_star(
    q_start: np.ndarray, q_goal: np.ndarray,
    joint_limits: List[Tuple[float, float]],
    checker: CollisionChecker,
    timeout: float = 30.0,
    step_size: float = 0.5,
    resolution: float = 0.05,
    seed: int = 42,
) -> Dict:
    """双向 RRT* (anytime): 两棵树交替生长 + 邻域重接."""
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)

    tree_a = _NodePool(ndim)
    tree_b = _NodePool(ndim)
    tree_a.add(q_start, -1, 0.0)
    tree_b.add(q_goal, -1, 0.0)

    best_cost = float('inf')
    best_path = None
    n_collision_checks = 0
    t0 = time.perf_counter()
    n_iters = 0
    swapped = False
    cost_history: List[Tuple[float, float]] = []
    first_solution_time = float('nan')

    while time.perf_counter() - t0 < timeout:
        n_iters += 1
        q_rand = rng.uniform(lows, highs)

        # extend tree_a
        idx_near = tree_a.nearest(q_rand)
        q_near = tree_a.configs[idx_near]
        q_new = _steer(q_near, q_rand, step_size)

        n_collision_checks += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            tree_a, tree_b = tree_b, tree_a
            swapped = not swapped
            continue

        # choose best parent (RRT* style)
        r = min(step_size * 2.0, _rrt_rewiring_radius(ndim, tree_a.n))
        near_idxs = tree_a.near(q_new, r)
        best_parent = idx_near
        best_cost_new = tree_a.costs[idx_near] + float(np.linalg.norm(q_new - q_near))

        for ni in near_idxs:
            d = float(np.linalg.norm(q_new - tree_a.configs[ni]))
            c = tree_a.costs[ni] + d
            if c < best_cost_new:
                n_collision_checks += 1
                if _collision_free_segment(checker, tree_a.configs[ni], q_new, resolution):
                    best_parent = ni
                    best_cost_new = c

        idx_new = tree_a.add(q_new, best_parent, best_cost_new)

        # try connect to tree_b
        idx_b_near = tree_b.nearest(q_new)
        q_b_near = tree_b.configs[idx_b_near]
        d_connect = float(np.linalg.norm(q_new - q_b_near))

        if d_connect < step_size * 2:
            n_collision_checks += 1
            if _collision_free_segment(checker, q_new, q_b_near, resolution):
                total = best_cost_new + d_connect + tree_b.costs[idx_b_near]
                if total < best_cost:
                    elapsed = time.perf_counter() - t0
                    if best_path is None:
                        first_solution_time = elapsed
                    best_cost = total
                    path_a = tree_a.extract_path(idx_new)
                    path_b = tree_b.extract_path(idx_b_near)
                    path_b.reverse()
                    if swapped:
                        path_a, path_b = path_b, path_a
                        path_a.reverse()
                        path_b.reverse()
                    best_path = path_a + path_b[1:]
                    cost_history.append((elapsed, best_cost))

        # swap
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    dt = time.perf_counter() - t0
    if best_path is not None:
        return {
            "success": True, "plan_time_s": dt,
            "path_length": _path_length(best_path),
            "n_nodes": tree_a.n + tree_b.n, "n_iters": n_iters,
            "n_collision_checks": n_collision_checks,
            "waypoints": best_path,
            "cost_history": cost_history,
            "first_solution_time": first_solution_time,
        }
    return {
        "success": False, "plan_time_s": dt,
        "path_length": float("nan"),
        "n_nodes": tree_a.n + tree_b.n, "n_iters": n_iters,
        "n_collision_checks": n_collision_checks,
        "waypoints": [],
        "cost_history": cost_history,
        "first_solution_time": first_solution_time,
    }


# ═══════════════════════════════════════════════════════════════
# Algorithm registry
# ═══════════════════════════════════════════════════════════════

ALGORITHMS = {
    "RRT": plan_rrt,
    "RRTConnect": plan_rrt_connect,
    "RRT*": plan_rrt_star,
    "Informed-RRT*": plan_informed_rrt_star,
    "BiRRT*": plan_birrt_star,
}


def _run_algorithm(
    algo_name: str,
    robot,
    checker: CollisionChecker,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    timeout: float,
    resolution: float = 0.05,
    step_size: float = 0.5,
    seed: int = 42,
) -> Dict:
    fn = ALGORITHMS[algo_name]
    kwargs = dict(
        q_start=q_start, q_goal=q_goal,
        joint_limits=robot.joint_limits,
        checker=checker,
        timeout=timeout,
        step_size=step_size,
        resolution=resolution,
        seed=seed,
    )
    # some planners accept extra kwargs
    if algo_name in ("RRT", "RRT*", "Informed-RRT*"):
        kwargs["goal_bias"] = 0.05
        kwargs["goal_tol"] = 0.3

    result = fn(**kwargs)
    result["algorithm"] = algo_name

    # path smoothing (shortcut)
    if result["success"] and len(result["waypoints"]) > 2:
        raw_length = result["path_length"]
        rng_sc = np.random.default_rng(seed + 9999)
        smooth_path = _shortcut(result["waypoints"], checker, resolution,
                                max_iters=300, rng=rng_sc)
        smooth_length = _path_length(smooth_path)
        result["raw_path_length"] = raw_length
        result["raw_n_waypoints"] = len(result["waypoints"])
        result["waypoints"] = smooth_path
        result["path_length"] = smooth_length
        result["n_waypoints"] = len(smooth_path)
    else:
        result["n_waypoints"] = len(result.get("waypoints", []))
        result["raw_path_length"] = result.get("path_length", float("nan"))
        result["raw_n_waypoints"] = result["n_waypoints"]

    # ensure cost_history / first_solution_time present
    if "cost_history" not in result:
        result["cost_history"] = []
    if "first_solution_time" not in result:
        result["first_solution_time"] = result["plan_time_s"] if result["success"] else float("nan")

    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark driver
# ═══════════════════════════════════════════════════════════════

def run_benchmark(
    seed: int,
    algorithms: List[str],
    timeout: float = 30.0,
    n_trials: int = 3,
    cfg: Optional[PandaGCSConfig] = None,
) -> Dict:
    if cfg is None:
        cfg = PandaGCSConfig()
    cfg.seed = seed

    robot = load_robot("panda")
    ndim = robot.n_joints
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    rng = np.random.default_rng(seed)

    scene = build_panda_scene(rng, cfg, robot, q_start, q_goal)
    checker = CollisionChecker(robot=robot, scene=scene)

    dist = float(np.linalg.norm(q_goal - q_start))
    n_obs = scene.n_obstacles

    print(f"\n{'=' * 70}")
    print(f"  RRT Family Benchmark — Panda {ndim}-DOF")
    print(f"{'=' * 70}")
    print(f"  seed          = {seed}")
    print(f"  q_start       = {np.array2string(q_start, precision=3)}")
    print(f"  q_goal        = {np.array2string(q_goal, precision=3)}")
    print(f"  config dist   = {dist:.3f} rad")
    print(f"  obstacles     = {n_obs}")
    print(f"  timeout       = {timeout:.1f}s per solver")
    print(f"  trials        = {n_trials}")
    print(f"  algorithms    = {algorithms}")
    print()

    obs_info = []
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        obs_info.append({"name": obs.name, "min": mn.tolist(), "max": mx.tolist()})

    all_results = {}
    for algo in algorithms:
        print(f"  [{algo}]", end="", flush=True)
        trial_results = []

        for trial in range(n_trials):
            trial_seed = seed + trial * 1000 + hash(algo) % 10000
            r = _run_algorithm(
                algo_name=algo, robot=robot, checker=checker,
                q_start=q_start, q_goal=q_goal,
                timeout=timeout, seed=trial_seed,
            )
            trial_results.append(r)
            sym = "." if r["success"] else "x"
            print(f" {sym}", end="", flush=True)

        successes = [t for t in trial_results if t["success"]]
        n_succ = len(successes)

        if n_succ > 0:
            avg_plan = sum(t["plan_time_s"] for t in successes) / n_succ
            avg_length = sum(t["path_length"] for t in successes) / n_succ
            avg_raw_length = sum(t["raw_path_length"] for t in successes) / n_succ
            avg_checks = sum(t["n_collision_checks"] for t in successes) / n_succ
            avg_nodes = sum(t["n_nodes"] for t in successes) / n_succ
            min_length = min(t["path_length"] for t in successes)
            best_time = min(t["plan_time_s"] for t in successes)
            avg_first_sol = sum(t["first_solution_time"] for t in successes) / n_succ
        else:
            avg_plan = avg_length = avg_raw_length = float("nan")
            avg_checks = avg_nodes = min_length = best_time = float("nan")
            avg_first_sol = float("nan")

        summary = {
            "n_trials": n_trials,
            "n_success": n_succ,
            "success_rate": n_succ / n_trials,
            "avg_plan_time_s": avg_plan,
            "best_plan_time_s": best_time,
            "avg_first_solution_time": avg_first_sol,
            "avg_path_length": avg_length,
            "avg_raw_path_length": avg_raw_length,
            "min_path_length": min_length,
            "avg_n_nodes": avg_nodes,
            "avg_collision_checks": avg_checks,
        }
        all_results[algo] = {
            "summary": summary,
            "trials": [
                {k: v for k, v in t.items() if k != "waypoints"}
                for t in trial_results
            ],
        }

        avg_first_sol = summary.get("avg_first_solution_time", float("nan"))

        if n_succ > 0:
            first_s = f"1st={avg_first_sol:.3f}s  " if not math.isnan(avg_first_sol) else ""
            print(f"  {n_succ}/{n_trials} ok  "
                  f"{first_s}"
                  f"plan={avg_plan:.3f}s  "
                  f"len={avg_length:.3f} (raw {avg_raw_length:.3f})  "
                  f"nodes={avg_nodes:.0f}  checks={avg_checks:.0f}")
        else:
            print(f"  {n_succ}/{n_trials} ok  (all failed)")

        # print cost evolution for anytime algorithms
        for i, t in enumerate(trial_results):
            ch = t.get("cost_history", [])
            if ch:
                hist_str = "  ".join(f"{et:.3f}s:{c:.3f}" for et, c in ch)
                print(f"    trial {i} cost evolution: {hist_str}")

    return {
        "seed": seed, "ndim": ndim, "n_obstacles": n_obs,
        "timeout_s": timeout, "n_trials": n_trials,
        "config_dist": dist,
        "q_start": q_start.tolist(), "q_goal": q_goal.tolist(),
        "obstacles": obs_info,
        "results": all_results,
    }


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

def print_summary_table(bench: Dict):
    results = bench["results"]
    algos = list(results.keys())

    print(f"\n{'=' * 90}")
    print(f"  Summary — seed={bench['seed']}, "
          f"{bench['n_obstacles']} obs, timeout={bench['timeout_s']}s, "
          f"{bench['n_trials']} trials")
    print(f"{'=' * 90}")

    header = (f"{'Algorithm':<18} {'Success':>8} {'1st Sol':>10} {'Plan(s)':>10} "
              f"{'Length':>10} {'Raw Len':>10} {'Nodes':>8} {'Checks':>10}")
    print(header)
    print("-" * len(header))

    for algo in algos:
        s = results[algo]["summary"]
        rate = f"{s['n_success']}/{s['n_trials']}"
        if s["n_success"] > 0:
            fst = s.get('avg_first_solution_time', float('nan'))
            fst_s = f"{fst:>10.4f}" if not math.isnan(fst) else f"{'—':>10}"
            print(f"{algo:<18} {rate:>8} "
                  f"{fst_s} "
                  f"{s['avg_plan_time_s']:>10.4f} "
                  f"{s['avg_path_length']:>10.3f} "
                  f"{s['avg_raw_path_length']:>10.3f} "
                  f"{s['avg_n_nodes']:>8.0f} "
                  f"{s['avg_collision_checks']:>10.0f}")
        else:
            print(f"{algo:<18} {rate:>8} {'—':>10} {'—':>10} {'—':>10} "
                  f"{'—':>10} {'—':>8} {'—':>10}")
    print()


def save_results(bench: Dict, out_dir: Path):
    json_path = out_dir / "rrt_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    md_path = out_dir / "rrt_benchmark.md"
    results = bench["results"]
    algos = list(results.keys())

    lines = [
        f"# RRT Family Benchmark — Panda 7-DOF",
        f"",
        f"- **Seed**: {bench['seed']}",
        f"- **Obstacles**: {bench['n_obstacles']}",
        f"- **Timeout**: {bench['timeout_s']}s",
        f"- **Trials**: {bench['n_trials']}",
        f"- **Config distance**: {bench['config_dist']:.3f} rad",
        f"",
        f"## Results",
        f"",
        f"| Algorithm | Success | Avg Plan (s) | Avg Length | Best Length "
        f"| Raw Length | Avg Nodes | Avg Checks |",
        f"|-----------|---------|-------------|------------|-------------|"
        f"-----------|-----------|------------|",
    ]
    for algo in algos:
        s = results[algo]["summary"]
        rate = f"{s['n_success']}/{s['n_trials']}"
        if s["n_success"] > 0:
            lines.append(
                f"| {algo} | {rate} | "
                f"{s['avg_plan_time_s']:.4f} | "
                f"{s['avg_path_length']:.3f} | "
                f"{s['min_path_length']:.3f} | "
                f"{s['avg_raw_path_length']:.3f} | "
                f"{s['avg_n_nodes']:.0f} | "
                f"{s['avg_collision_checks']:.0f} |"
            )
        else:
            lines.append(
                f"| {algo} | {rate} | — | — | — | — | — | — |")

    lines.extend(["", "## Per-trial Details", ""])
    for algo in algos:
        trials = results[algo]["trials"]
        lines.append(f"### {algo}")
        lines.append("")
        lines.append(f"| Trial | OK | Plan(s) | 1st Sol(s) | Length | Raw Len | Nodes | Checks |")
        lines.append(f"|-------|----|---------|------------|--------|---------|-------|--------|")
        for i, t in enumerate(trials):
            ok = "Y" if t["success"] else "N"
            fst = t.get("first_solution_time", float("nan"))
            fst_s = f"{fst:.4f}" if not math.isnan(fst) else "—"
            if t["success"]:
                lines.append(
                    f"| {i} | {ok} | {t['plan_time_s']:.4f} | {fst_s} | "
                    f"{t['path_length']:.3f} | {t['raw_path_length']:.3f} | "
                    f"{t['n_nodes']} | {t['n_collision_checks']} |")
            else:
                lines.append(
                    f"| {i} | {ok} | {t['plan_time_s']:.4f} | {fst_s} | — | — | "
                    f"{t['n_nodes']} | {t['n_collision_checks']} |")

        # cost evolution sub-table
        has_history = any(t.get("cost_history") for t in trials)
        if has_history:
            lines.append("")
            lines.append(f"**Cost evolution (time → path cost):**")
            lines.append("")
            for i, t in enumerate(trials):
                ch = t.get("cost_history", [])
                if ch:
                    steps = " → ".join(f"{et:.3f}s: {c:.3f}" for et, c in ch)
                    lines.append(f"- Trial {i}: {steps}")
                else:
                    lines.append(f"- Trial {i}: (no solution found)")

        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {md_path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RRT family benchmark on Panda 7-DOF (same scene as gcs_planner_panda)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (0 = timestamp)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Solver timeout per algorithm (seconds)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per algorithm")
    parser.add_argument("--algorithms", nargs="+",
                        default=list(ALGORITHMS.keys()),
                        help=f"Algorithms to test: {list(ALGORITHMS.keys())}")
    parser.add_argument("--obstacles", type=int, default=6,
                        help="Number of obstacles")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="RRT step size (rad)")
    args = parser.parse_args()

    seed = args.seed if args.seed != 0 else int(time.time()) % (2**31)

    cfg = PandaGCSConfig()
    cfg.n_obstacles = args.obstacles

    t0 = time.perf_counter()
    bench = run_benchmark(
        seed=seed,
        algorithms=args.algorithms,
        timeout=args.timeout,
        n_trials=args.trials,
        cfg=cfg,
    )
    total_s = time.perf_counter() - t0

    print_summary_table(bench)

    out_dir = make_output_dir("benchmarks", "rrt_family_panda")
    save_results(bench, out_dir)

    print(f"\nTotal benchmark time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
