"""planner/sbf_query.py - fast planning query on existing SafeBoxForest."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .models import SBFConfig, SBFResult
from forest.models import BoxNode
from forest.safe_box_forest import SafeBoxForest
from forest.scene import Scene
from forest.collision import CollisionChecker
from .path_smoother import PathSmoother, compute_path_length
from .gcs_optimizer import GCSOptimizer


class SBFQuery:
    """Query planner using an existing SafeBoxForest without global rebuild."""

    def __init__(
        self,
        forest: SafeBoxForest,
        robot,
        scene: Scene,
        config: Optional[SBFConfig] = None,
    ) -> None:
        self.forest = forest
        self.robot = robot
        self.scene = scene
        self.config = config or SBFConfig()
        self.collision_checker = CollisionChecker(robot=robot, scene=scene)
        self.path_smoother = PathSmoother(
            collision_checker=self.collision_checker,
            segment_resolution=self.config.segment_collision_resolution,
        )
        self.gcs_optimizer = GCSOptimizer(fallback=True, bezier_degree=self.config.gcs_bezier_degree)

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, seed: Optional[int] = None) -> SBFResult:
        t0 = time.time()
        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)
        result = SBFResult()

        if self.collision_checker.check_config_collision(q_start):
            result.message = "起始配置存在碰撞"
            result.computation_time = time.time() - t0
            return result
        if self.collision_checker.check_config_collision(q_goal):
            result.message = "目标配置存在碰撞"
            result.computation_time = time.time() - t0
            return result

        if not self.collision_checker.check_segment_collision(
            q_start, q_goal, self.config.segment_collision_resolution
        ):
            result.success = True
            result.path = [q_start.copy(), q_goal.copy()]
            result.path_length = float(np.linalg.norm(q_goal - q_start))
            result.message = "直连成功"
            result.computation_time = time.time() - t0
            return result

        start_box = self.forest.find_containing(q_start) or self.forest.find_nearest(q_start)
        goal_box = self.forest.find_containing(q_goal) or self.forest.find_nearest(q_goal)
        if start_box is None or goal_box is None:
            result.message = "无法在 forest 中定位始末点"
            result.computation_time = time.time() - t0
            return result

        # fallback path: nearest-point waypoint sequence via Dijkstra on forest adjacency
        graph = {
            "start": "start",
            "goal": "goal",
            "edges": {
                "start": [(start_box.node_id, 0.0, None)],
                "goal": [],
            },
        }
        for bid, neighbors in self.forest.adjacency.items():
            graph["edges"].setdefault(bid, [])
            for nb in neighbors:
                if nb in self.forest.boxes:
                    a = self.forest.boxes[bid].center
                    b = self.forest.boxes[nb].center
                    graph["edges"][bid].append((nb, float(np.linalg.norm(a - b)), None))
        graph["edges"].setdefault(goal_box.node_id, []).append(("goal", 0.0, None))

        path = self.gcs_optimizer._optimize_fallback(
            graph,
            self.forest.boxes,
            q_start,
            q_goal,
        )
        if not path:
            result.message = "forest 查询未找到路径"
            result.computation_time = time.time() - t0
            return result

        path = self.path_smoother.shortcut(path, max_iters=self.config.path_shortcut_iters)
        path = self.path_smoother.smooth_moving_average(path)

        result.success = True
        result.path = path
        result.path_length = compute_path_length(path)
        result.n_boxes_created = self.forest.n_boxes
        result.n_collision_checks = self.collision_checker.n_collision_checks
        result.message = "forest 查询成功"
        result.computation_time = time.time() - t0
        return result
