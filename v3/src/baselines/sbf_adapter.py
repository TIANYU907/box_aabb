"""
baselines/sbf_adapter.py — SafeBoxForest 管线适配器

将 pipeline.py 的 grow_and_prepare / run_method_with_bridge 封装为
BasePlanner 接口, 支持跨查询复用 (supports_reuse=True).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BasePlanner, PlanningResult

from forest.collision import CollisionChecker
from planner.pipeline import (
    PandaGCSConfig,
    make_planner_config,
    grow_and_prepare,
    run_method_with_bridge,
    run_method_visgraph,
    _solve_method_dijkstra,
    _solve_method_gcs,
)


class SBFAdapter(BasePlanner):
    """SafeBoxForest pipeline adapter.

    默认使用 Dijkstra + SOCP refine 方法.
    ``supports_reuse=True``: setup 构建 forest 后, 多次 plan()
    复用同一 forest / adjacency.
    """

    def __init__(self, method: str = "dijkstra"):
        """
        Args:
            method: 'dijkstra' | 'gcs' | 'visgraph'
        """
        self._method = method
        self._robot = None
        self._scene = None
        self._cfg: Optional[PandaGCSConfig] = None
        self._prep: Optional[dict] = None
        self._checker: Optional[CollisionChecker] = None
        self._ndim: int = 0

    # ── BasePlanner 接口 ───────────────────────────────────────

    @property
    def name(self) -> str:
        return f"SBF-{self._method.capitalize()}"

    @property
    def supports_reuse(self) -> bool:
        return True

    def setup(self, robot, scene, config: dict) -> None:
        self._robot = robot
        self._scene = scene
        self._ndim = robot.n_joints

        # build PandaGCSConfig from dict
        cfg = PandaGCSConfig()
        for k, v in config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        self._cfg = cfg

        self._checker = CollisionChecker(robot=robot, scene=scene)
        self._prep = None  # lazy: first plan() will build

    def plan(self, start: np.ndarray, goal: np.ndarray,
             timeout: float = 30.0) -> PlanningResult:
        t_total = time.perf_counter()
        q_start = np.asarray(start, dtype=np.float64)
        q_goal = np.asarray(goal, dtype=np.float64)

        # ── 1) grow forest (first call only, or after reset) ──
        phase_times = {}
        if self._prep is None:
            t0 = time.perf_counter()
            self._prep = grow_and_prepare(
                self._robot, self._scene, self._cfg,
                q_start, q_goal, self._ndim,
                no_cache=self._cfg.no_cache if hasattr(self._cfg, 'no_cache') else False,
            )
            # wait for cache thread
            ct = self._prep.get('_cache_thread')
            if ct is not None:
                ct.join()
                cr = self._prep.get('_cache_result', {})
                self._prep['cache_ms'] = cr.get('ms', 0.0)
            phase_times['grow'] = time.perf_counter() - t0

        # ── 2) solve ──
        t0 = time.perf_counter()
        if self._method == "visgraph":
            raw = run_method_visgraph(
                self._prep, self._cfg,
                q_start, q_goal, self._checker, self._ndim)
        else:
            fn = (_solve_method_gcs if self._method == "gcs"
                  else _solve_method_dijkstra)
            raw = run_method_with_bridge(
                fn, self._method.capitalize(),
                self._prep, self._cfg,
                q_start, q_goal, self._ndim)
        phase_times['solve'] = time.perf_counter() - t0

        total_s = time.perf_counter() - t_total

        # ── 3) convert to PlanningResult ──
        if raw is None or not raw.get('success', False):
            return PlanningResult.failure(
                planning_time=total_s,
                nodes_explored=len(self._prep.get('boxes', {})),
                method=self._method,
            )

        wps = raw.get('waypoints', [])
        path = np.array(wps, dtype=np.float64) if wps else None
        cost = raw.get('cost', float('inf'))

        return PlanningResult(
            success=True,
            path=path,
            cost=cost,
            planning_time=total_s,
            first_solution_time=total_s,  # SBF 不是 anytime
            collision_checks=0,           # not tracked at pipeline level
            nodes_explored=len(self._prep.get('boxes', {})),
            phase_times=phase_times,
            metadata={
                'method': self._method,
                'n_grown': self._prep.get('n_grown', 0),
                'grow_ms': self._prep.get('grow_ms', 0),
                'coarsen_ms': self._prep.get('coarsen_ms', 0),
                'bridge_ms': raw.get('bridge_ms', 0),
                'plan_ms': raw.get('plan_ms', 0),
            },
        )

    def reset(self) -> None:
        """清除 forest — 下次 plan() 将重新 grow."""
        self._prep = None
    def update_scene_incremental(
        self,
        new_scene,
        added_obstacles: list = None,
        removed_obstacle_names: list = None,
        regrow_budget: int = 60,
        rng=None,
    ) -> dict:
        """增量更新障碍物, 复用已有 forest.

        Args:
            new_scene: 更新后的 Scene (仅用于引用, 实际修改由 pipeline 完成)
            added_obstacles: [{'min_point': [...], 'max_point': [...], 'name': ...}]
            removed_obstacle_names: ['obs_name', ...]
            regrow_budget: 补种预算
            rng: 随机数生成器

        Returns:
            增量更新统计信息 dict
        """
        from planner.pipeline import incremental_obstacle_update

        if self._prep is None:
            raise RuntimeError("必须先调用 plan() 构建 forest 才能增量更新")

        return incremental_obstacle_update(
            prep=self._prep,
            scene=self._scene,
            added_obstacles=added_obstacles or [],
            removed_obstacle_names=removed_obstacle_names or [],
            regrow_budget=regrow_budget,
            rng=rng,
        )