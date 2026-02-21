"""
baselines/rrt_family.py — RRT 系列算法 (纯 Python)

从 v2/examples/rrt_family_panda.py 重构:
- _NodePool: numpy-backed RRT 节点存储
- plan_rrt, plan_rrt_connect, plan_rrt_star,
  plan_informed_rrt_star, plan_birrt_star

统一封装为 BasePlanner 接口:
- RRTPlanner(algorithm="RRT")
- RRTPlanner(algorithm="RRTConnect")
- RRTPlanner(algorithm="RRT*")
- RRTPlanner(algorithm="Informed-RRT*")
- RRTPlanner(algorithm="BiRRT*")
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import BasePlanner, PlanningResult
from forest.collision import CollisionChecker


# ═══════════════════════════════════════════════════════════════════════════
# _NodePool — RRT 节点存储
# ═══════════════════════════════════════════════════════════════════════════

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
            new_c = np.empty((self.cap, self.ndim), dtype=np.float64)
            new_c[:self.n] = self.configs[:self.n]
            self.configs = new_c
            new_p = np.full(self.cap, -1, dtype=np.int32)
            new_p[:self.n] = self.parents[:self.n]
            self.parents = new_p
            new_k = np.zeros(self.cap, dtype=np.float64)
            new_k[:self.n] = self.costs[:self.n]
            self.costs = new_k
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
        return list(np.where(dists <= radius * radius)[0])

    def extract_path(self, goal_idx: int) -> List[np.ndarray]:
        path = []
        idx = goal_idx
        while idx >= 0:
            path.append(self.configs[idx].copy())
            idx = int(self.parents[idx])
        path.reverse()
        return path


# ═══════════════════════════════════════════════════════════════════════════
# Common helpers
# ═══════════════════════════════════════════════════════════════════════════

def _steer(q_from: np.ndarray, q_to: np.ndarray,
           step_size: float) -> np.ndarray:
    diff = q_to - q_from
    dist = np.linalg.norm(diff)
    if dist <= step_size:
        return q_to.copy()
    return q_from + (step_size / dist) * diff


def _collision_free_segment(checker: CollisionChecker,
                            q_from: np.ndarray, q_to: np.ndarray,
                            resolution: float) -> bool:
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


def _rrt_rewiring_radius(ndim: int, n_nodes: int,
                         gamma: float = None) -> float:
    if gamma is None:
        unit_ball = math.pi ** (ndim / 2.0) / math.gamma(ndim / 2.0 + 1.0)
        gamma = (2.0 * (1.0 + 1.0 / ndim) ** (1.0 / ndim)
                 * (1.0 / unit_ball) ** (1.0 / ndim) * 2.0)
    if n_nodes < 2:
        return float('inf')
    return gamma * (math.log(n_nodes) / n_nodes) ** (1.0 / ndim)


# ═══════════════════════════════════════════════════════════════════════════
# 1. RRT (basic, single-tree)
# ═══════════════════════════════════════════════════════════════════════════

def plan_rrt(q_start, q_goal, joint_limits, checker, *,
             timeout=30.0, step_size=0.5, goal_bias=0.05,
             goal_tol=0.3, resolution=0.05, seed=42):
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)
    pool = _NodePool(ndim)
    pool.add(q_start, -1, 0.0)

    n_cc = 0
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < timeout:
        q_rand = q_goal.copy() if rng.uniform() < goal_bias else rng.uniform(lows, highs)
        idx_near = pool.nearest(q_rand)
        q_near = pool.configs[idx_near]
        q_new = _steer(q_near, q_rand, step_size)

        n_cc += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            continue

        cost_new = pool.costs[idx_near] + float(np.linalg.norm(q_new - q_near))
        idx_new = pool.add(q_new, idx_near, cost_new)

        if np.linalg.norm(q_new - q_goal) < goal_tol:
            n_cc += 1
            if _collision_free_segment(checker, q_new, q_goal, resolution):
                cost_goal = cost_new + float(np.linalg.norm(q_goal - q_new))
                idx_goal = pool.add(q_goal, idx_new, cost_goal)
                dt = time.perf_counter() - t0
                path = pool.extract_path(idx_goal)
                return dict(success=True, plan_time_s=dt,
                            path_length=_path_length(path),
                            n_nodes=pool.n, n_collision_checks=n_cc,
                            waypoints=path, first_solution_time=dt)

    dt = time.perf_counter() - t0
    return dict(success=False, plan_time_s=dt, path_length=float("nan"),
                n_nodes=pool.n, n_collision_checks=n_cc, waypoints=[],
                first_solution_time=float("nan"))


# ═══════════════════════════════════════════════════════════════════════════
# 2. RRT-Connect (bidirectional)
# ═══════════════════════════════════════════════════════════════════════════

def plan_rrt_connect(q_start, q_goal, joint_limits, checker, *,
                     timeout=30.0, step_size=0.5, resolution=0.05,
                     seed=42):
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)

    tree_a, tree_b = _NodePool(ndim), _NodePool(ndim)
    tree_a.add(q_start, -1, 0.0)
    tree_b.add(q_goal, -1, 0.0)

    n_cc = 0
    t0 = time.perf_counter()
    swapped = False

    def _extend(tree, q_target):
        nonlocal n_cc
        idx_near = tree.nearest(q_target)
        q_near = tree.configs[idx_near]
        q_new = _steer(q_near, q_target, step_size)
        n_cc += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            return None, None
        cost = tree.costs[idx_near] + float(np.linalg.norm(q_new - q_near))
        idx_new = tree.add(q_new, idx_near, cost)
        return idx_new, q_new

    def _connect(tree, q_target):
        while True:
            idx, q_new = _extend(tree, q_target)
            if idx is None:
                return None, None
            if np.linalg.norm(q_new - q_target) < 1e-6:
                return idx, q_new

    while time.perf_counter() - t0 < timeout:
        q_rand = rng.uniform(lows, highs)
        idx_a, q_new_a = _extend(tree_a, q_rand)
        if idx_a is None:
            continue
        idx_b, _ = _connect(tree_b, q_new_a)
        if idx_b is not None:
            dt = time.perf_counter() - t0
            path_a = tree_a.extract_path(idx_a)
            path_b = tree_b.extract_path(idx_b)
            path_b.reverse()
            if swapped:
                path_a, path_b = path_b, path_a
                path_a.reverse(); path_b.reverse()
            full = path_a + path_b[1:]
            return dict(success=True, plan_time_s=dt,
                        path_length=_path_length(full),
                        n_nodes=tree_a.n + tree_b.n,
                        n_collision_checks=n_cc,
                        waypoints=full, first_solution_time=dt)
        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    dt = time.perf_counter() - t0
    return dict(success=False, plan_time_s=dt, path_length=float("nan"),
                n_nodes=tree_a.n + tree_b.n, n_collision_checks=n_cc,
                waypoints=[], first_solution_time=float("nan"))


# ═══════════════════════════════════════════════════════════════════════════
# 3. RRT* (asymptotically optimal)
# ═══════════════════════════════════════════════════════════════════════════

def plan_rrt_star(q_start, q_goal, joint_limits, checker, *,
                  timeout=30.0, step_size=0.5, goal_bias=0.05,
                  goal_tol=0.3, resolution=0.05, seed=42):
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)
    pool = _NodePool(ndim)
    pool.add(q_start, -1, 0.0)

    best_goal_idx = -1
    best_cost = float('inf')
    n_cc = 0
    t0 = time.perf_counter()
    first_sol_t = float('nan')

    while time.perf_counter() - t0 < timeout:
        q_rand = q_goal.copy() if rng.uniform() < goal_bias else rng.uniform(lows, highs)
        idx_near = pool.nearest(q_rand)
        q_near = pool.configs[idx_near]
        q_new = _steer(q_near, q_rand, step_size)

        n_cc += 1
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
                n_cc += 1
                if _collision_free_segment(checker, pool.configs[ni], q_new, resolution):
                    best_parent, best_cost_new = ni, c

        idx_new = pool.add(q_new, best_parent, best_cost_new)

        for ni in near_idxs:
            if ni == best_parent:
                continue
            d = float(np.linalg.norm(pool.configs[ni] - q_new))
            c_thru = best_cost_new + d
            if c_thru < pool.costs[ni]:
                n_cc += 1
                if _collision_free_segment(checker, q_new, pool.configs[ni], resolution):
                    pool.parents[ni] = idx_new
                    pool.costs[ni] = c_thru

        d_goal = float(np.linalg.norm(q_new - q_goal))
        if d_goal < goal_tol:
            n_cc += 1
            if _collision_free_segment(checker, q_new, q_goal, resolution):
                cost_goal = best_cost_new + d_goal
                if cost_goal < best_cost:
                    if best_goal_idx < 0:
                        first_sol_t = time.perf_counter() - t0
                    idx_goal = pool.add(q_goal, idx_new, cost_goal)
                    best_goal_idx = idx_goal
                    best_cost = cost_goal

    dt = time.perf_counter() - t0
    if best_goal_idx >= 0:
        path = pool.extract_path(best_goal_idx)
        return dict(success=True, plan_time_s=dt,
                    path_length=_path_length(path),
                    n_nodes=pool.n, n_collision_checks=n_cc,
                    waypoints=path, first_solution_time=first_sol_t)
    return dict(success=False, plan_time_s=dt, path_length=float("nan"),
                n_nodes=pool.n, n_collision_checks=n_cc, waypoints=[],
                first_solution_time=first_sol_t)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Informed-RRT*
# ═══════════════════════════════════════════════════════════════════════════

def _informed_sample(q_start, q_goal, c_best, lows, highs, rng):
    ndim = len(q_start)
    c_min = float(np.linalg.norm(q_goal - q_start))
    if c_best >= 1e10 or c_best <= c_min + 1e-10:
        return rng.uniform(lows, highs)
    a1 = (q_goal - q_start) / c_min
    M = np.eye(ndim); M[:, 0] = a1
    Q, _ = np.linalg.qr(M)
    if np.dot(Q[:, 0], a1) < 0:
        Q[:, 0] = -Q[:, 0]
    r1 = c_best / 2.0
    ri = math.sqrt(max(0, c_best**2 - c_min**2)) / 2.0
    L = np.diag([r1] + [ri] * (ndim - 1))
    center = (q_start + q_goal) / 2.0
    u = rng.standard_normal(ndim)
    u /= np.linalg.norm(u)
    r = rng.uniform() ** (1.0 / ndim)
    return np.clip(center + Q @ L @ (r * u), lows, highs)


def plan_informed_rrt_star(q_start, q_goal, joint_limits, checker, *,
                           timeout=30.0, step_size=0.5, goal_bias=0.05,
                           goal_tol=0.3, resolution=0.05, seed=42):
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)
    pool = _NodePool(ndim)
    pool.add(q_start, -1, 0.0)

    best_goal_idx = -1
    best_cost = float('inf')
    n_cc = 0
    t0 = time.perf_counter()
    first_sol_t = float('nan')

    while time.perf_counter() - t0 < timeout:
        if best_cost < float('inf') and rng.uniform() > goal_bias:
            q_rand = _informed_sample(q_start, q_goal, best_cost,
                                      lows, highs, rng)
        elif rng.uniform() < goal_bias:
            q_rand = q_goal.copy()
        else:
            q_rand = rng.uniform(lows, highs)

        idx_near = pool.nearest(q_rand)
        q_near = pool.configs[idx_near]
        q_new = _steer(q_near, q_rand, step_size)

        n_cc += 1
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
                n_cc += 1
                if _collision_free_segment(checker, pool.configs[ni], q_new, resolution):
                    best_parent, best_cost_new = ni, c

        idx_new = pool.add(q_new, best_parent, best_cost_new)

        for ni in near_idxs:
            if ni == best_parent:
                continue
            d = float(np.linalg.norm(pool.configs[ni] - q_new))
            c_thru = best_cost_new + d
            if c_thru < pool.costs[ni]:
                n_cc += 1
                if _collision_free_segment(checker, q_new, pool.configs[ni], resolution):
                    pool.parents[ni] = idx_new
                    pool.costs[ni] = c_thru

        d_goal = float(np.linalg.norm(q_new - q_goal))
        if d_goal < goal_tol:
            n_cc += 1
            if _collision_free_segment(checker, q_new, q_goal, resolution):
                cost_goal = best_cost_new + d_goal
                if cost_goal < best_cost:
                    if best_goal_idx < 0:
                        first_sol_t = time.perf_counter() - t0
                    idx_goal = pool.add(q_goal, idx_new, cost_goal)
                    best_goal_idx = idx_goal
                    best_cost = cost_goal

    dt = time.perf_counter() - t0
    if best_goal_idx >= 0:
        path = pool.extract_path(best_goal_idx)
        return dict(success=True, plan_time_s=dt,
                    path_length=_path_length(path),
                    n_nodes=pool.n, n_collision_checks=n_cc,
                    waypoints=path, first_solution_time=first_sol_t)
    return dict(success=False, plan_time_s=dt, path_length=float("nan"),
                n_nodes=pool.n, n_collision_checks=n_cc, waypoints=[],
                first_solution_time=first_sol_t)


# ═══════════════════════════════════════════════════════════════════════════
# 5. BiRRT* (bidirectional RRT*)
# ═══════════════════════════════════════════════════════════════════════════

def plan_birrt_star(q_start, q_goal, joint_limits, checker, *,
                    timeout=30.0, step_size=0.5, resolution=0.05,
                    seed=42):
    ndim = len(q_start)
    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float64)
    rng = np.random.default_rng(seed)

    tree_a, tree_b = _NodePool(ndim), _NodePool(ndim)
    tree_a.add(q_start, -1, 0.0)
    tree_b.add(q_goal, -1, 0.0)

    best_cost = float('inf')
    best_path = None
    n_cc = 0
    t0 = time.perf_counter()
    swapped = False
    first_sol_t = float('nan')

    while time.perf_counter() - t0 < timeout:
        q_rand = rng.uniform(lows, highs)
        idx_near = tree_a.nearest(q_rand)
        q_near = tree_a.configs[idx_near]
        q_new = _steer(q_near, q_rand, step_size)

        n_cc += 1
        if not _collision_free_segment(checker, q_near, q_new, resolution):
            tree_a, tree_b = tree_b, tree_a
            swapped = not swapped
            continue

        r = min(step_size * 2.0, _rrt_rewiring_radius(ndim, tree_a.n))
        near_idxs = tree_a.near(q_new, r)
        best_parent = idx_near
        best_cost_new = tree_a.costs[idx_near] + float(np.linalg.norm(q_new - q_near))
        for ni in near_idxs:
            d = float(np.linalg.norm(q_new - tree_a.configs[ni]))
            c = tree_a.costs[ni] + d
            if c < best_cost_new:
                n_cc += 1
                if _collision_free_segment(checker, tree_a.configs[ni], q_new, resolution):
                    best_parent, best_cost_new = ni, c

        idx_new = tree_a.add(q_new, best_parent, best_cost_new)

        idx_b_near = tree_b.nearest(q_new)
        q_b_near = tree_b.configs[idx_b_near]
        d_connect = float(np.linalg.norm(q_new - q_b_near))

        if d_connect < step_size * 2:
            n_cc += 1
            if _collision_free_segment(checker, q_new, q_b_near, resolution):
                total = best_cost_new + d_connect + tree_b.costs[idx_b_near]
                if total < best_cost:
                    if best_path is None:
                        first_sol_t = time.perf_counter() - t0
                    best_cost = total
                    path_a = tree_a.extract_path(idx_new)
                    path_b = tree_b.extract_path(idx_b_near)
                    path_b.reverse()
                    if swapped:
                        path_a, path_b = path_b, path_a
                        path_a.reverse(); path_b.reverse()
                    best_path = path_a + path_b[1:]

        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    dt = time.perf_counter() - t0
    if best_path is not None:
        return dict(success=True, plan_time_s=dt,
                    path_length=_path_length(best_path),
                    n_nodes=tree_a.n + tree_b.n, n_collision_checks=n_cc,
                    waypoints=best_path, first_solution_time=first_sol_t)
    return dict(success=False, plan_time_s=dt, path_length=float("nan"),
                n_nodes=tree_a.n + tree_b.n, n_collision_checks=n_cc,
                waypoints=[], first_solution_time=first_sol_t)


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm registry
# ═══════════════════════════════════════════════════════════════════════════

ALGORITHMS = {
    "RRT": plan_rrt,
    "RRTConnect": plan_rrt_connect,
    "RRT*": plan_rrt_star,
    "Informed-RRT*": plan_informed_rrt_star,
    "BiRRT*": plan_birrt_star,
}


# ═══════════════════════════════════════════════════════════════════════════
# BasePlanner 适配器
# ═══════════════════════════════════════════════════════════════════════════

class RRTPlanner(BasePlanner):
    """RRT 系列算法的 BasePlanner 封装.

    Args:
        algorithm: 'RRT' | 'RRTConnect' | 'RRT*' | 'Informed-RRT*' | 'BiRRT*'
    """

    def __init__(self, algorithm: str = "RRTConnect"):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                             f"Choose from {list(ALGORITHMS.keys())}")
        self._algorithm = algorithm
        self._fn = ALGORITHMS[algorithm]
        self._robot = None
        self._scene = None
        self._checker: Optional[CollisionChecker] = None
        self._config: dict = {}

    @property
    def name(self) -> str:
        return self._algorithm

    @property
    def supports_reuse(self) -> bool:
        return False

    def setup(self, robot, scene, config: dict) -> None:
        self._robot = robot
        self._scene = scene
        self._checker = CollisionChecker(robot=robot, scene=scene)
        self._config = config

    def plan(self, start: np.ndarray, goal: np.ndarray,
             timeout: float = 30.0) -> PlanningResult:
        q_start = np.asarray(start, dtype=np.float64)
        q_goal = np.asarray(goal, dtype=np.float64)

        # build kwargs from config + defaults
        kwargs = dict(
            q_start=q_start, q_goal=q_goal,
            joint_limits=self._robot.joint_limits,
            checker=self._checker,
            timeout=timeout,
            step_size=self._config.get('step_size', 0.5),
            resolution=self._config.get('resolution', 0.05),
            seed=self._config.get('seed', 42),
        )
        if self._algorithm in ("RRT", "RRT*", "Informed-RRT*"):
            kwargs["goal_bias"] = self._config.get('goal_bias', 0.05)
            kwargs["goal_tol"] = self._config.get('goal_tol', 0.3)

        raw = self._fn(**kwargs)

        # path shortcut
        if raw["success"] and len(raw["waypoints"]) > 2:
            rng_sc = np.random.default_rng(
                self._config.get('seed', 42) + 9999)
            raw_length = raw["path_length"]
            smooth = _shortcut(raw["waypoints"], self._checker,
                               kwargs["resolution"], max_iters=300,
                               rng=rng_sc)
            raw["waypoints"] = smooth
            raw["path_length"] = _path_length(smooth)
            raw["raw_path_length"] = raw_length

        # convert to PlanningResult
        wps = raw.get("waypoints", [])
        path = np.array(wps, dtype=np.float64) if wps else None

        first_sol = raw.get("first_solution_time", float("nan"))

        return PlanningResult(
            success=raw["success"],
            path=path,
            cost=raw.get("path_length", float("inf")),
            planning_time=raw["plan_time_s"],
            first_solution_time=first_sol,
            collision_checks=raw.get("n_collision_checks", 0),
            nodes_explored=raw.get("n_nodes", 0),
            metadata={
                "algorithm": self._algorithm,
                "raw_path_length": raw.get("raw_path_length",
                                           raw.get("path_length")),
            },
        )

    def reset(self) -> None:
        pass  # RRT is stateless between plan() calls
