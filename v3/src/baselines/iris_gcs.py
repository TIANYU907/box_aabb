"""
baselines/iris_gcs.py — Drake IRIS + GCS (Marcucci) baseline adapter

使用 SBF forest 作为 GCS 输入 (复用 planner/gcs_optimizer.py 中的 Drake
GraphOfConvexSets 求解器).

需要 pydrake 安装. 若不可用, import 时不会报错,
但 plan() 会返回 failure.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .base import BasePlanner, PlanningResult

logger = logging.getLogger(__name__)

# 延迟检测 Drake 可用性
_HAS_DRAKE = None


def _check_drake() -> bool:
    global _HAS_DRAKE
    if _HAS_DRAKE is None:
        try:
            from pydrake.geometry.optimization import GraphOfConvexSets  # noqa: F401
            _HAS_DRAKE = True
        except ImportError:
            _HAS_DRAKE = False
    return _HAS_DRAKE


class IRISGCSPlanner(BasePlanner):
    """Drake IRIS + GCS planner.

    使用 SBFPlanner 构建 box forest, 然后通过 Drake
    GraphOfConvexSets 求解最优路径.

    ``supports_reuse``: partial — IRIS regions (boxes) 可复用,
    但 GCS 求解每次重新运行.
    """

    def __init__(self):
        self._robot = None
        self._scene = None
        self._config: dict = {}
        self._planner = None  # SBFPlanner instance
        self._forest = None
        self._boxes = None

    @property
    def name(self) -> str:
        return "IRIS-GCS"

    @property
    def supports_reuse(self) -> bool:
        return True  # forest / boxes 可跨查询复用

    def setup(self, robot, scene, config: dict) -> None:
        self._robot = robot
        self._scene = scene
        self._config = config
        self._planner = None
        self._forest = None
        self._boxes = None

    def plan(self, start: np.ndarray, goal: np.ndarray,
             timeout: float = 30.0) -> PlanningResult:
        if not _check_drake():
            logger.error("pydrake not available — IRIS-GCS cannot run")
            return PlanningResult.failure(
                planning_time=0.0,
                reason="pydrake not installed")

        q_start = np.asarray(start, dtype=np.float64)
        q_goal = np.asarray(goal, dtype=np.float64)
        t_total = time.perf_counter()
        phase_times = {}

        # ── 1) Build forest (first call only) ──
        if self._planner is None:
            from planner.sbf_planner import SBFPlanner
            from planner.models import SBFConfig

            cfg_dict = dict(self._config)
            sbf_cfg = SBFConfig.from_dict(cfg_dict) if cfg_dict else SBFConfig()
            self._planner = SBFPlanner(
                robot=self._robot, scene=self._scene,
                config=sbf_cfg, no_cache=True)

            t0 = time.perf_counter()
            result = self._planner.plan(q_start, q_goal,
                                        seed=self._config.get('seed', 42))
            phase_times['sbf_build'] = time.perf_counter() - t0

            if result.forest is None:
                dt = time.perf_counter() - t_total
                return PlanningResult.failure(
                    planning_time=dt,
                    reason="SBF forest build failed")

            self._forest = result.forest
            self._boxes = result.forest.boxes

        # ── 2) GCS optimization via Drake ──
        try:
            from planner.gcs_optimizer import GCSOptimizer
        except ImportError:
            dt = time.perf_counter() - t_total
            return PlanningResult.failure(
                planning_time=dt,
                reason="GCSOptimizer not available")

        t0 = time.perf_counter()
        try:
            # Validate boxes and build adjacency
            colliding = self._forest.validate_boxes(
                self._planner.collision_checker)
            valid_boxes = {bid: b for bid, b in self._boxes.items()
                          if bid not in colliding}
            valid_adj = {
                bid: (nbrs - colliding)
                for bid, nbrs in self._forest.adjacency.items()
                if bid not in colliding
            }

            adj_edges = self._planner.connector.build_adjacency_edges(
                valid_boxes, valid_adj)
            ep_edges, src_id, tgt_id = (
                self._planner.connector.connect_endpoints_to_forest(
                    q_start, q_goal, valid_boxes))

            if src_id is None or tgt_id is None:
                dt = time.perf_counter() - t_total
                return PlanningResult.failure(
                    planning_time=dt,
                    nodes_explored=len(valid_boxes),
                    reason="start/goal not in any box")

            graph = self._planner.connector.build_forest_graph(
                adj_edges, ep_edges,
                q_start, q_goal, src_id, tgt_id, valid_boxes)

            gcs = GCSOptimizer(fallback=False, bezier_degree=3)
            path = gcs.optimize(graph, valid_boxes, q_start, q_goal)
            phase_times['gcs_solve'] = time.perf_counter() - t0

        except Exception as e:
            dt = time.perf_counter() - t_total
            logger.warning("IRIS-GCS solve error: %s", e)
            return PlanningResult.failure(
                planning_time=dt,
                reason=str(e))

        dt = time.perf_counter() - t_total

        if path is None:
            return PlanningResult.failure(
                planning_time=dt,
                nodes_explored=len(self._boxes),
                reason="Drake GCS solver returned None")

        path_arr = np.array(path, dtype=np.float64)
        cost = float(sum(
            np.linalg.norm(path_arr[i] - path_arr[i - 1])
            for i in range(1, len(path_arr))
        ))

        return PlanningResult(
            success=True,
            path=path_arr,
            cost=cost,
            planning_time=dt,
            first_solution_time=dt,
            collision_checks=0,
            nodes_explored=len(self._boxes),
            phase_times=phase_times,
            metadata={"algorithm": "IRIS-GCS"},
        )

    def reset(self) -> None:
        self._planner = None
        self._forest = None
        self._boxes = None
