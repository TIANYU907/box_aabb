"""
planner/box_query.py - 基于已有森林的快速查询规划

给定预构建的 BoxForest，通过连接始末点、图搜索、路径平滑
实现快速路径规划（无需重新拓展 box）。

典型用法：
    forest = BoxForest.load("forest.pkl", robot, scene)
    query = BoxForestQuery(forest)
    result = query.plan(q_start, q_goal, seed=42)
"""

import time
import logging
from typing import List, Tuple, Optional

import numpy as np

from box_aabb.robot import Robot
from .models import PlannerConfig, PlannerResult, BoxNode, gmean_edge_length
from .obstacles import Scene
from .collision import CollisionChecker
from .hier_aabb_tree import HierAABBTree
from .box_tree import BoxTreeManager
from .connector import TreeConnector
from .path_smoother import PathSmoother, compute_path_length
from .gcs_optimizer import GCSOptimizer
from .box_forest import BoxForest

logger = logging.getLogger(__name__)


class BoxForestQuery:
    """基于已有 BoxForest 的快速查询规划器

    在查询阶段：
    1. 复用已构建的 box 森林
    2. 在始末点附近做少量补充拓展（可选）
    3. 连接始末点到森林
    4. 图搜索 + 路径平滑

    Args:
        forest: 预构建的 BoxForest
        config: 规划参数（可覆盖森林中的默认配置）
    """

    def __init__(
        self,
        forest: BoxForest,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.forest = forest
        self.config = config or forest.config

        self.collision_checker = CollisionChecker(
            robot=forest.robot,
            scene=forest.scene,
        )
        self.hier_tree = HierAABBTree.auto_load(
            forest.robot, forest.joint_limits)
        self.obstacles = forest.scene.get_obstacles()
        self.connector = TreeConnector(
            tree_manager=forest.tree_manager,
            collision_checker=self.collision_checker,
            max_attempts=self.config.connection_max_attempts,
            connection_radius=self.config.connection_radius,
            segment_resolution=self.config.segment_collision_resolution,
        )
        self.path_smoother = PathSmoother(
            collision_checker=self.collision_checker,
            segment_resolution=self.config.segment_collision_resolution,
        )
        self.gcs_optimizer = GCSOptimizer(
            fallback=True,
            bezier_degree=self.config.gcs_bezier_degree,
        )

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        seed: Optional[int] = None,
        expand_budget: Optional[int] = None,
    ) -> PlannerResult:
        """使用已有森林进行快速路径规划

        Args:
            q_start: 起始配置
            q_goal: 目标配置
            seed: 随机数种子
            expand_budget: 始末点附近补充拓展的最大 box 数（默认 config.query_expand_budget）

        Returns:
            PlannerResult
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)

        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)
        result = PlannerResult()
        budget = expand_budget or self.config.query_expand_budget

        # ---- Step 0: 验证始末点 ----
        if self.collision_checker.check_config_collision(q_start):
            result.message = "起始配置存在碰撞"
            result.computation_time = time.time() - t0
            return result

        if self.collision_checker.check_config_collision(q_goal):
            result.message = "目标配置存在碰撞"
            result.computation_time = time.time() - t0
            return result

        # ---- Step 0.5: 尝试直连 ----
        if not self.collision_checker.check_segment_collision(
            q_start, q_goal, self.config.segment_collision_resolution
        ):
            result.success = True
            result.path = [q_start.copy(), q_goal.copy()]
            result.path_length = float(np.linalg.norm(q_goal - q_start))
            result.message = "直连成功"
            result.computation_time = time.time() - t0
            return result

        # ---- Step 1: 补充拓展（始末点附近）----
        self._local_expand(q_start, budget // 2, rng)
        self._local_expand(q_goal, budget - budget // 2, rng)

        # ---- Step 2: 连接 ----
        tree_mgr = self.forest.tree_manager
        intra_edges = self.connector.connect_within_trees()
        inter_edges = self.connector.connect_between_trees()
        endpoint_edges, start_box_id, goal_box_id = \
            self.connector.connect_endpoints(q_start, q_goal)

        all_edges = intra_edges + inter_edges + endpoint_edges
        result.edges = all_edges

        if start_box_id is None or goal_box_id is None:
            result.message = "无法将始末点连接到 box forest"
            result.computation_time = time.time() - t0
            result.box_trees = tree_mgr.get_all_trees()
            result.n_boxes_created = tree_mgr.total_nodes
            result.n_collision_checks = self.collision_checker.n_collision_checks
            return result

        # ---- Step 3: 图搜索 ----
        boxes_dict = {b.node_id: b for b in tree_mgr.get_all_boxes()}
        graph = self.connector.build_adjacency_graph(
            all_edges, q_start, q_goal, start_box_id, goal_box_id)

        path = self._graph_search(graph, boxes_dict, q_start, q_goal)

        if path is None or len(path) < 2:
            result.message = "图搜索未找到路径"
            result.computation_time = time.time() - t0
            result.box_trees = tree_mgr.get_all_trees()
            result.n_boxes_created = tree_mgr.total_nodes
            result.n_collision_checks = self.collision_checker.n_collision_checks
            return result

        # ---- Step 4: 路径后处理 ----
        path = self.path_smoother.shortcut(
            path, max_iters=self.config.path_shortcut_iters, rng=rng)
        path = self.path_smoother.smooth_moving_average(path)

        # ---- 组装结果 ----
        result.success = True
        result.path = path
        result.path_length = compute_path_length(path)
        result.box_trees = tree_mgr.get_all_trees()
        result.n_boxes_created = tree_mgr.total_nodes
        result.n_collision_checks = self.collision_checker.n_collision_checks
        result.computation_time = time.time() - t0
        result.message = (
            f"Forest 查询成功: {len(path)} 个路径点, "
            f"路径长度 {result.path_length:.4f}, "
            f"查询耗时 {result.computation_time:.3f}s "
            f"(森林构建 {self.forest.build_time:.2f}s)")
        logger.info(result.message)
        return result

    def _local_expand(
        self,
        q_center: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> None:
        """在指定配置附近做局部 box 拓展"""
        tree_mgr = self.forest.tree_manager
        n_added = 0

        for _ in range(budget * 3):  # 最多尝试 3x 次
            if n_added >= budget:
                break

            # 在 q_center 附近采样
            noise = rng.normal(0, 0.5, size=len(q_center))
            q_seed = q_center + noise
            for d in range(len(q_seed)):
                lo, hi = self.forest.joint_limits[d]
                q_seed[d] = np.clip(q_seed[d], lo, hi)

            if self.collision_checker.check_config_collision(q_seed):
                continue
            if tree_mgr.find_containing_box(q_seed) is not None:
                continue

            ivs = self.hier_tree.find_free_box(
                q_seed, self.obstacles, mark_occupied=True)
            if ivs is None:
                continue
            vol = 1.0
            for lo, hi in ivs.intervals:
                vol *= max(hi - lo, 0.0)
            ndim = len(ivs.intervals)
            if gmean_edge_length(vol, ndim) < self.config.min_box_size:
                continue

            node_id = tree_mgr.allocate_node_id()
            box = BoxNode(
                node_id=node_id,
                joint_intervals=ivs.intervals,
                seed_config=q_seed.copy(),
                volume=vol,
            )

            # 加入森林
            containing = tree_mgr.find_containing_box(q_seed)
            if containing is not None:
                tree_mgr.add_box(containing.tree_id, box,
                                 parent_id=containing.node_id)
            else:
                nearest = tree_mgr.find_nearest_box(q_seed)
                if nearest is not None and \
                   nearest.distance_to_config(q_seed) < self.config.connection_radius:
                    tree_mgr.add_box(nearest.tree_id, box,
                                     parent_id=nearest.node_id)
                else:
                    tree_mgr.create_tree(box)
            n_added += 1

    def _graph_search(self, graph, boxes_dict, q_start, q_goal):
        """图搜索"""
        if self.config.use_gcs:
            return self.gcs_optimizer.optimize(
                graph, boxes_dict, q_start, q_goal)
        return self.gcs_optimizer._optimize_fallback(
            graph, boxes_dict, q_start, q_goal)
