"""
planner/box_rrt.py - Box-RRT 主规划器

基于关节区间(box)拓展的 RRT 路径规划算法。

算法流程：
1. 初始化碰撞检测器、box 拓展器、树管理器
2. 验证始末点无碰撞
3. 主采样循环：
   a. 随机/目标偏向采样 seed 点
   b. 拓展 box
   c. 创建/加入 box tree
   d. 在叶子 box 边界上再采样并拓展
   e. 检查停止条件
4. 连接各树 + 始末点
5. 图搜索 / GCS 优化
6. 路径平滑后处理
"""

import time
import logging
from typing import List, Tuple, Optional

import numpy as np

from ..robot import Robot
from .models import PlannerConfig, PlannerResult, BoxNode
from .obstacles import Scene
from .collision import CollisionChecker
from .box_expansion import BoxExpander
from .box_tree import BoxTreeManager
from .connector import TreeConnector
from .path_smoother import PathSmoother, compute_path_length
from .gcs_optimizer import GCSOptimizer

logger = logging.getLogger(__name__)


class BoxRRT:
    """Box-RRT 路径规划器

    在关节空间中通过拓展无碰撞 box 来快速覆盖 C-free，
    然后在 box graph 上搜索/优化路径。

    Args:
        robot: 机器人模型
        scene: 障碍物场景
        config: 规划参数配置
        joint_limits: 关节限制（默认从 robot.joint_limits 获取）

    Example:
        >>> robot = load_robot('2dof_planar')
        >>> scene = Scene()
        >>> scene.add_obstacle([1.0, 0.5], [1.5, 1.0])
        >>> planner = BoxRRT(robot, scene)
        >>> result = planner.plan(q_start, q_goal)
        >>> if result.success:
        ...     print(f"路径长度: {result.path_length:.4f}")
    """

    def __init__(
        self,
        robot: Robot,
        scene: Scene,
        config: Optional[PlannerConfig] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.config = config or PlannerConfig()

        # 关节限制
        if joint_limits is not None:
            self.joint_limits = list(joint_limits)
        elif robot.joint_limits is not None:
            self.joint_limits = list(robot.joint_limits)
        else:
            # 默认 [-pi, pi]
            self.joint_limits = [(-np.pi, np.pi)] * robot.n_joints

        self._n_dims = len(self.joint_limits)

        # 初始化子模块
        self.collision_checker = CollisionChecker(
            robot=robot,
            scene=scene,
        )
        # 高自由度机器人自动启用采样辅助 box 拓展
        _use_sampling = robot.n_joints > 4
        self.box_expander = BoxExpander(
            robot=robot,
            collision_checker=self.collision_checker,
            joint_limits=self.joint_limits,
            expansion_resolution=self.config.expansion_resolution,
            max_rounds=self.config.max_expansion_rounds,
            jacobian_delta=self.config.jacobian_delta,
            use_sampling=_use_sampling,
        )
        self.tree_manager = BoxTreeManager()
        self.connector = TreeConnector(
            tree_manager=self.tree_manager,
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
    ) -> PlannerResult:
        """执行路径规划

        Args:
            q_start: 起始配置 (n_joints,)
            q_goal: 目标配置 (n_joints,)
            seed: 随机数种子（可选，用于可重复性）

        Returns:
            PlannerResult 规划结果
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)

        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)

        result = PlannerResult()

        # ---- Step 0: 验证始末点 ----
        if self.collision_checker.check_config_collision(q_start):
            result.message = "起始配置存在碰撞"
            result.computation_time = time.time() - t0
            logger.error(result.message)
            return result

        if self.collision_checker.check_config_collision(q_goal):
            result.message = "目标配置存在碰撞"
            result.computation_time = time.time() - t0
            logger.error(result.message)
            return result

        # ---- Step 0.5: 尝试直连 ----
        if not self.collision_checker.check_segment_collision(
            q_start, q_goal, self.config.segment_collision_resolution
        ):
            result.success = True
            result.path = [q_start.copy(), q_goal.copy()]
            result.path_length = float(np.linalg.norm(q_goal - q_start))
            result.message = "直连成功（无需 box 拓展）"
            result.computation_time = time.time() - t0
            logger.info(result.message)
            return result

        # ---- Step 1: 从始末点创建初始 box tree ----
        self._create_initial_trees(q_start, q_goal, rng)

        # ---- Step 2: 主采样循环 ----
        n_boxes = self.tree_manager.total_nodes
        for iteration in range(self.config.max_iterations):
            if n_boxes >= self.config.max_box_nodes:
                break

            # 采样 seed 点
            q_seed = self._sample_seed(q_start, q_goal, rng)
            if q_seed is None:
                continue

            # 拓展 box
            node_id = self.tree_manager.allocate_node_id()
            box = self.box_expander.expand(q_seed, node_id=node_id, rng=rng)
            if box is None or box.volume < self.config.min_box_volume:
                continue

            # 加入最近的树或创建新树
            self._add_box_to_tree(box, rng)
            n_boxes = self.tree_manager.total_nodes

            # 边界再采样和拓展
            if box.tree_id >= 0:
                self._boundary_expand(box.tree_id, rng)
                n_boxes = self.tree_manager.total_nodes

            # 每隔一段时间尝试桥接采样（针对窄通道）
            if (iteration + 1) % 10 == 0:
                self._bridge_sampling(rng)
                n_boxes = self.tree_manager.total_nodes

            # 定期检查连通性
            if (iteration + 1) % 20 == 0 and self.config.verbose:
                logger.info(
                    "迭代 %d: %d 棵树, %d 个 box, 总体积 %.4f",
                    iteration + 1, self.tree_manager.n_trees,
                    n_boxes, self.tree_manager.get_total_volume(),
                )

        # ---- Step 3: 连接各树 ----
        intra_edges = self.connector.connect_within_trees()
        inter_edges = self.connector.connect_between_trees()
        endpoint_edges, start_box_id, goal_box_id = \
            self.connector.connect_endpoints(q_start, q_goal)

        all_edges = intra_edges + inter_edges + endpoint_edges
        result.edges = all_edges

        if start_box_id is None or goal_box_id is None:
            result.message = "无法将始末点连接到 box tree"
            result.computation_time = time.time() - t0
            result.box_trees = self.tree_manager.get_all_trees()
            result.n_boxes_created = self.tree_manager.total_nodes
            result.n_collision_checks = self.collision_checker.n_collision_checks
            logger.warning(result.message)
            return result

        # ---- Step 4: 图搜索/GCS 优化 ----
        boxes_dict = {b.node_id: b for b in self.tree_manager.get_all_boxes()}
        graph = self.connector.build_adjacency_graph(
            all_edges, q_start, q_goal, start_box_id, goal_box_id,
        )

        path = self._graph_search(graph, boxes_dict, q_start, q_goal)

        # ---- Step 4b: 连通性修复 ----
        if path is None or len(path) < 2:
            logger.info("初次搜索失败，尝试连通性修复...")
            bridge_edges = self._bridge_disconnected(
                graph, boxes_dict, q_start, q_goal,
                start_box_id, goal_box_id,
            )
            if bridge_edges:
                all_edges = all_edges + bridge_edges
                result.edges = all_edges
                graph = self.connector.build_adjacency_graph(
                    all_edges, q_start, q_goal, start_box_id, goal_box_id,
                )
                path = self._graph_search(graph, boxes_dict, q_start, q_goal)

        if path is None or len(path) < 2:
            result.message = "图搜索/GCS 优化未找到路径"
            result.computation_time = time.time() - t0
            result.box_trees = self.tree_manager.get_all_trees()
            result.n_boxes_created = self.tree_manager.total_nodes
            result.n_collision_checks = self.collision_checker.n_collision_checks
            logger.warning(result.message)
            return result

        # ---- Step 5: 路径后处理 ----
        path = self.path_smoother.shortcut(
            path, max_iters=self.config.path_shortcut_iters, rng=rng,
        )
        path = self.path_smoother.smooth_moving_average(path)

        # ---- 组装结果 ----
        result.success = True
        result.path = path
        result.path_length = compute_path_length(path)
        result.box_trees = self.tree_manager.get_all_trees()
        result.n_boxes_created = self.tree_manager.total_nodes
        result.n_collision_checks = self.collision_checker.n_collision_checks
        result.computation_time = time.time() - t0
        result.message = (
            f"规划成功: {len(path)} 个路径点, "
            f"路径长度 {result.path_length:.4f}, "
            f"{self.tree_manager.n_trees} 棵树, "
            f"{result.n_boxes_created} 个 box"
        )
        logger.info(result.message)
        return result

    # ==================== 辅助方法 ====================

    def _graph_search(
        self,
        graph: dict,
        boxes_dict: dict,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """在邻接图上搜索路径"""
        if self.config.use_gcs:
            return self.gcs_optimizer.optimize(
                graph, boxes_dict, q_start, q_goal,
            )
        return self.gcs_optimizer._optimize_fallback(
            graph, boxes_dict, q_start, q_goal,
        )

    def _bridge_disconnected(
        self,
        graph: dict,
        boxes_dict: dict,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        start_box_id: int,
        goal_box_id: int,
    ) -> List:
        """修复断开的图连通性

        反复通过 BFS 找到从 start 可达的节点集合，
        如果 goal 不可达，在可达集合与不可达集合的最近 box 对
        之间尝试线段碰撞检测连接，直到 goal 可达或耗尽尝试。

        Returns:
            新建的 Edge 列表
        """
        from .models import Edge

        all_new_edges: List = []
        max_rounds = 5  # 最多 5 轮修复

        # 维护一份邻接表，逐轮更新
        adj = {}
        for k, v in graph['edges'].items():
            adj[k] = list(v)

        for round_i in range(max_rounds):
            # BFS from start
            reachable = set()
            queue = ['start']
            reachable.add('start')
            while queue:
                u = queue.pop(0)
                for v, _, _ in adj.get(u, []):
                    if v not in reachable:
                        reachable.add(v)
                        queue.append(v)

            if 'goal' in reachable:
                break  # connected

            logger.info(
                "连通性修复 (轮 %d): start 可达 %d 节点, goal 不可达",
                round_i + 1, len(reachable),
            )

            # Separate into reachable/unreachable box sets
            reachable_boxes = [
                boxes_dict[n] for n in reachable
                if isinstance(n, int) and n in boxes_dict
            ]
            unreachable_boxes = [
                b for b in boxes_dict.values()
                if b.node_id not in reachable
            ]

            if not reachable_boxes or not unreachable_boxes:
                break

            # Find closest pairs between reachable and unreachable
            candidates = []
            for rb in reachable_boxes:
                for ub in unreachable_boxes:
                    dist = float(np.linalg.norm(rb.center - ub.center))
                    candidates.append((rb, ub, dist))
            candidates.sort(key=lambda x: x[2])

            new_edges_this_round = 0
            max_candidates = min(len(candidates), 200)

            for rb, ub, dist in candidates[:max_candidates]:
                # Try connecting nearest points on box surfaces
                q_r = rb.nearest_point_to(ub.center)
                q_u = ub.nearest_point_to(rb.center)

                if not self.collision_checker.check_segment_collision(
                    q_r, q_u, self.config.segment_collision_resolution,
                ):
                    edge = Edge(
                        edge_id=self.connector._allocate_edge_id(),
                        source_box_id=rb.node_id,
                        target_box_id=ub.node_id,
                        source_config=q_r,
                        target_config=q_u,
                        source_tree_id=rb.tree_id,
                        target_tree_id=ub.tree_id,
                        is_collision_free=True,
                    )
                    all_new_edges.append(edge)
                    # Update adjacency in place for next round
                    adj.setdefault(rb.node_id, []).append(
                        (ub.node_id, edge.cost, edge))
                    adj.setdefault(ub.node_id, []).append(
                        (rb.node_id, edge.cost, edge))
                    new_edges_this_round += 1
                    if new_edges_this_round >= 20:
                        break

            logger.info(
                "连通性修复 (轮 %d): 新增 %d 条桥接边",
                round_i + 1, new_edges_this_round,
            )

            if new_edges_this_round == 0:
                break  # no progress

        logger.info("连通性修复: 共新增 %d 条桥接边", len(all_new_edges))
        return all_new_edges

    # ==================== 内部方法 ====================

    def _create_initial_trees(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """从始末点创建初始 box tree"""
        # 起始点 box
        start_id = self.tree_manager.allocate_node_id()
        start_box = self.box_expander.expand(q_start, node_id=start_id, rng=rng)
        if start_box is not None and start_box.volume >= self.config.min_box_volume:
            self.tree_manager.create_tree(start_box)
            logger.info("起始 box: 体积 %.6f", start_box.volume)

        # 目标点 box
        goal_id = self.tree_manager.allocate_node_id()
        goal_box = self.box_expander.expand(q_goal, node_id=goal_id, rng=rng)
        if goal_box is not None and goal_box.volume >= self.config.min_box_volume:
            self.tree_manager.create_tree(goal_box)
            logger.info("目标 box: 体积 %.6f", goal_box.volume)

    def _sample_seed(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        """采样一个无碰撞 seed 点

        策略：
        - 以 goal_bias 概率朝目标采样
        - 否则在关节限制内均匀随机采样
        - 重试若干次以找到无碰撞点
        """
        max_attempts = 20

        for _ in range(max_attempts):
            if rng.uniform() < self.config.goal_bias:
                # 目标偏向：在 goal 附近采样
                noise = rng.normal(0, 0.3, size=self._n_dims)
                q = q_goal + noise
                # 裁剪到关节限制
                for i in range(self._n_dims):
                    q[i] = np.clip(q[i], self.joint_limits[i][0],
                                   self.joint_limits[i][1])
            else:
                # 均匀随机采样
                q = np.array([
                    rng.uniform(lo, hi) for lo, hi in self.joint_limits
                ])

            if not self.collision_checker.check_config_collision(q):
                return q

        return None

    def _add_box_to_tree(
        self,
        box: BoxNode,
        rng: np.random.Generator,
    ) -> None:
        """将新 box 加入最近的树，或创建新树

        如果新 box 与某棵树的某个 box 有交集，则加入该树；
        否则创建新树。
        """
        seed = box.seed_config

        # 找是否有包含 seed 的现有 box
        containing_box = self.tree_manager.find_containing_box(seed)
        if containing_box is not None:
            # 加入该 box 所在的树
            self.tree_manager.add_box(
                containing_box.tree_id, box,
                parent_id=containing_box.node_id,
            )
            return

        # 找最近的 box
        nearest = self.tree_manager.find_nearest_box(seed)
        if nearest is not None:
            dist = nearest.distance_to_config(seed)
            if dist < self.config.connection_radius:
                # 足够近，加入该树
                self.tree_manager.add_box(
                    nearest.tree_id, box,
                    parent_id=nearest.node_id,
                )
                return

        # 创建新树
        self.tree_manager.create_tree(box)

    def _boundary_expand(
        self,
        tree_id: int,
        rng: np.random.Generator,
    ) -> None:
        """在树的叶子 box 边界上再采样并拓展

        这是 Box-RRT 的关键步骤：从已有 box 的边界出发，
        采样新 seed 并拓展，使树逐渐生长覆盖更多空间。
        """
        samples = self.tree_manager.get_boundary_samples(
            tree_id, n_samples=self.config.seed_batch_size, rng=rng,
        )

        for q_seed in samples:
            if self.tree_manager.total_nodes >= self.config.max_box_nodes:
                break

            if self.collision_checker.check_config_collision(q_seed):
                continue

            node_id = self.tree_manager.allocate_node_id()
            new_box = self.box_expander.expand(q_seed, node_id=node_id, rng=rng)
            if new_box is None or new_box.volume < self.config.min_box_volume:
                continue

            # 找到距 seed 最近的同树 box 作为父节点
            nearest = self.tree_manager.find_nearest_box_in_tree(tree_id, q_seed)
            if nearest is not None:
                self.tree_manager.add_box(
                    tree_id, new_box, parent_id=nearest.node_id,
                )

    def _bridge_sampling(self, rng: np.random.Generator) -> None:
        """桥接采样：在不同树之间的间隙区域定向采样

        找到两棵树最近的边界区域，在该区域附近集中采样 seed，
        尝试在窄通道中建立连接。
        """
        trees = self.tree_manager.get_all_trees()
        if len(trees) < 2:
            return
        if self.tree_manager.total_nodes >= self.config.max_box_nodes:
            return

        # 每对树之间尝试桥接
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                if self.tree_manager.total_nodes >= self.config.max_box_nodes:
                    return

                # 找两棵树最近的 box 对
                tree_a, tree_b = trees[i], trees[j]
                best_dist = float('inf')
                best_pair = None
                for ba in tree_a.nodes.values():
                    for bb in tree_b.nodes.values():
                        d = ba.distance_to_config(bb.center)
                        if d < best_dist:
                            best_dist = d
                            best_pair = (ba, bb)

                if best_pair is None or best_dist < 1e-10:
                    continue  # 已重叠或无 box

                ba, bb = best_pair
                # 在两个 box 的最近点之间的中间区域采样
                q_a = ba.nearest_point_to(bb.center)
                q_b = bb.nearest_point_to(ba.center)
                midpoint = (q_a + q_b) / 2.0

                for _ in range(3):
                    if self.tree_manager.total_nodes >= self.config.max_box_nodes:
                        break

                    # 在中间区域加随机噪声
                    noise = rng.normal(0, best_dist * 0.3, size=self._n_dims)
                    q_seed = midpoint + noise
                    # 裁剪到关节限制
                    for d in range(self._n_dims):
                        q_seed[d] = np.clip(q_seed[d],
                                            self.joint_limits[d][0],
                                            self.joint_limits[d][1])

                    if self.collision_checker.check_config_collision(q_seed):
                        continue

                    node_id = self.tree_manager.allocate_node_id()
                    new_box = self.box_expander.expand(q_seed, node_id=node_id, rng=rng)
                    if new_box is None or new_box.volume < self.config.min_box_volume:
                        continue

                    self._add_box_to_tree(new_box, rng)
