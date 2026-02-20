"""
planner/box_planner.py - BoxPlanner 主规划器

基于关节区间(box)拓展与分解的路径规划算法。

v5.0 更新：
- 集成 BoxForest 扁平无重叠 box 集合
- 扩展时边界截断（避免重叠已有 box）
- 邻接图搜索 + 共享面 waypoint 优化
- 路径限制在 box 内（box-aware shortcut + smoothing）
- BoxForest 跨场景持久化

算法流程：
1. 加载 BoxForest（如有缓存）+ AABB 缓存
2. 验证始末点无碰撞
3. 尝试直连
4. 扩展 box（边界截断避免重叠）
5. 邻接图构建
6. 惰性碰撞验证
7. Dijkstra 搜 box 序列
8. 共享面 waypoint 优化
9. Box-aware shortcut + smoothing
10. 保存 BoxForest + AABB 缓存
"""

import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Set, Dict

import numpy as np

from aabb.robot import Robot
from .models import PlannerConfig, PlannerResult, gmean_edge_length
from forest.models import BoxNode
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.hier_aabb_tree import HierAABBTree, build_kd_partitions
from .box_tree import BoxTreeManager
from forest.box_forest import BoxForest
from .connector import TreeConnector
from .path_smoother import PathSmoother, compute_path_length
from .gcs_optimizer import GCSOptimizer
from forest.deoverlap import compute_adjacency

logger = logging.getLogger(__name__)


def _partition_expand_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """多进程 worker：在单分区内扩展局部 boxes。"""
    robot = payload["robot"]
    obstacles = payload["obstacles"]
    q_start = np.asarray(payload["q_start"], dtype=np.float64)
    q_goal = np.asarray(payload["q_goal"], dtype=np.float64)
    partition_meta = payload["partition_meta"]
    max_iterations = int(payload["max_iterations"])
    min_box_size = float(payload["min_box_size"])
    active_split_dims = payload.get("active_split_dims")
    seed = payload.get("seed")

    intervals = partition_meta["intervals"]
    if not isinstance(intervals, list):
        return {"partition_id": int(partition_meta.get("partition_id", -1)), "boxes": []}

    local_tree = HierAABBTree(
        robot,
        joint_limits=intervals,
        active_split_dims=active_split_dims,
    )
    rng = np.random.default_rng(seed)
    local_boxes: Dict[int, Dict[str, object]] = {}
    next_local_id = 0
    n_dims = len(intervals)

    lows = np.array([lo for lo, _ in intervals], dtype=np.float64)
    highs = np.array([hi for _, hi in intervals], dtype=np.float64)

    def _sample_seed_local() -> np.ndarray:
        use_goal = bool(rng.uniform() < 0.1)
        if use_goal:
            noise = rng.normal(0.0, 0.3, size=(n_dims,))
            return np.clip(q_goal[:n_dims] + noise, lows, highs)
        return rng.uniform(lows, highs)

    def _try_expand(seed_q: np.ndarray) -> None:
        nonlocal next_local_id
        local_id = next_local_id
        ffb = local_tree.find_free_box(
            seed_q,
            obstacles,
            mark_occupied=True,
            forest_box_id=local_id,
            constrained_intervals=intervals,
        )
        if ffb is None:
            return
        vol = 1.0
        for lo, hi in ffb.intervals:
            vol *= max(hi - lo, 0.0)
        if gmean_edge_length(vol, n_dims) < min_box_size:
            return
        if ffb.absorbed_box_ids:
            for absorbed_id in ffb.absorbed_box_ids:
                local_boxes.pop(int(absorbed_id), None)
        local_boxes[local_id] = {
            "joint_intervals": ffb.intervals,
            "seed_config": seed_q.copy(),
            "volume": vol,
        }
        next_local_id += 1

    if local_tree._is_config_in_intervals(q_start, intervals):
        _try_expand(q_start[:n_dims])
    if local_tree._is_config_in_intervals(q_goal, intervals):
        _try_expand(q_goal[:n_dims])

    for _ in range(max_iterations):
        _try_expand(_sample_seed_local())

    return {
        "partition_id": int(partition_meta.get("partition_id", -1)),
        "boxes": [local_boxes[k] for k in sorted(local_boxes.keys())],
    }


class BoxPlanner:
    """BoxPlanner — 基于 box 分解的路径规划器

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
        >>> planner = BoxPlanner(robot, scene)
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
        # 使用 HierAABBTree 做 box 拓展（自顶向下切分 + 缓存精化）
        self.hier_tree = HierAABBTree.auto_load(
            robot,
            self.joint_limits,
            active_split_dims=self.config.parallel_partition_dims,
        )
        self.obstacles = scene.get_obstacles()
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

    def _prepare_partitions(self) -> List[Dict[str, object]]:
        """构建 KD 子空间分区元数据（供后续并行扩展使用）。"""
        depth = max(0, int(self.config.parallel_partition_depth))
        dims = self.config.parallel_partition_dims
        partitions = build_kd_partitions(self.joint_limits, depth, dims=dims)

        result: List[Dict[str, object]] = []
        for pid, ivs in enumerate(partitions):
            result.append({
                "partition_id": pid,
                "intervals": ivs,
            })

        # 预计算相邻分区对（共享切分面的候选）
        adjacent_pairs: List[Tuple[int, int]] = []
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                if self._partitions_adjacent(partitions[i], partitions[j]):
                    adjacent_pairs.append((i, j))

        for entry in result:
            entry["adjacent_pairs"] = adjacent_pairs
        return result

    def _expand_partition_worker(
        self,
        partition_meta: Dict[str, object],
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_iterations: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """在单个子空间中扩展 box（worker 逻辑，当前进程内执行）。"""
        intervals = partition_meta["intervals"]
        if not isinstance(intervals, list):
            return []

        local_tree = HierAABBTree(
            self.robot,
            joint_limits=intervals,
            active_split_dims=self.config.parallel_partition_dims,
        )
        rng = np.random.default_rng(seed)
        local_boxes: Dict[int, Dict[str, object]] = {}
        next_local_id = 0

        def _try_expand(seed_q: np.ndarray) -> None:
            nonlocal next_local_id
            local_id = next_local_id
            ffb = local_tree.find_free_box(
                seed_q,
                self.obstacles,
                mark_occupied=True,
                forest_box_id=local_id,
                constrained_intervals=intervals,
            )
            if ffb is None:
                return
            vol = 1.0
            for lo, hi in ffb.intervals:
                vol *= max(hi - lo, 0.0)
            if gmean_edge_length(vol, self._n_dims) < self.config.min_box_size:
                return
            if ffb.absorbed_box_ids:
                for absorbed_id in ffb.absorbed_box_ids:
                    local_boxes.pop(int(absorbed_id), None)
            local_boxes[local_id] = {
                "joint_intervals": ffb.intervals,
                "seed_config": seed_q.copy(),
                "volume": vol,
            }
            next_local_id += 1

        if self.hier_tree._is_config_in_intervals(q_start, intervals):
            _try_expand(q_start)
        if self.hier_tree._is_config_in_intervals(q_goal, intervals):
            _try_expand(q_goal)

        for _ in range(max_iterations):
            q_seed = self._sample_seed_in_partition(q_start, q_goal, rng, intervals)
            if q_seed is None:
                continue
            _try_expand(q_seed)

        return [local_boxes[k] for k in sorted(local_boxes.keys())]

    def _sample_seed_in_partition(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        rng: np.random.Generator,
        partition_intervals: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        return self._sample_seed(
            q_start,
            q_goal,
            rng,
            sampling_intervals=partition_intervals,
        )

    def _merge_connect_partitions(
        self,
        forest: BoxForest,
        local_results: List[Dict[str, object]],
        partitions: List[Dict[str, object]],
    ) -> List:
        partition_box_ids = forest.merge_partition_forests(local_results)

        if self.config.parallel_cross_partition_connect:
            pair_set: Set[Tuple[int, int]] = set()
            for p in partitions:
                for a, b in p.get("adjacent_pairs", []):
                    pair_set.add((int(a), int(b)))
            cross_edges = self.connector.connect_across_partitions(
                sorted(pair_set),
                forest.boxes,
                partition_box_ids,
            )
        else:
            cross_edges = []

        forest.validate_invariants(strict=True)
        return cross_edges

    @staticmethod
    def _partitions_adjacent(
        a: List[Tuple[float, float]],
        b: List[Tuple[float, float]],
        tol: float = 1e-10,
    ) -> bool:
        """判断两个子空间是否共享切分边界（用于跨区补边候选）。"""
        if len(a) != len(b):
            return False

        touching_dims = 0
        for (a_lo, a_hi), (b_lo, b_hi) in zip(a, b):
            overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
            if overlap < -tol:
                return False
            if abs(overlap) <= tol:
                touching_dims += 1
        return touching_dims >= 1

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

        try:
            return self._plan_impl(q_start, q_goal, rng, t0, result)
        finally:
            pass

    def _plan_impl(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        rng: np.random.Generator,
        t0: float,
        result: PlannerResult,
    ) -> PlannerResult:
        """plan() 的实际实现（由 plan 调用，try/finally 确保缓存保存）"""
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

        # ---- Step 1: 加载/初始化 BoxForest ----
        forest = self._load_or_create_forest()
        forest.hier_tree = self.hier_tree

        # ---- Step 2: 验证已有 box 在当前场景的碰撞状态 ----
        colliding_ids = forest.validate_boxes(self.collision_checker)
        # 构建有效 box 集合（排除碰撞的）
        valid_boxes = {
            bid: box for bid, box in forest.boxes.items()
            if bid not in colliding_ids
        }

        # ---- Step 3: 扩展新 box ----
        existing_list = list(valid_boxes.values())
        raw_new_boxes: List[BoxNode] = []
        partition_cross_edges: List = []
        parallel_mode = self.config.parallel_expand and self.config.parallel_workers != 1

        # 从始末点扩展（单区模式）
        # 并行分区模式下由各分区 worker 在 constrained_intervals 内处理始末点，
        # 避免先全空间扩展再合并分区结果导致重叠。
        if not parallel_mode:
            for q_seed in [q_start, q_goal]:
                if self.hier_tree.is_occupied(q_seed):
                    continue
                nid = forest.allocate_id()
                ffb_result = self.hier_tree.find_free_box(
                    q_seed, self.obstacles, mark_occupied=True,
                    forest_box_id=nid)
                if ffb_result is None:
                    continue
                ivs = ffb_result.intervals
                vol = 1.0
                for lo, hi in ivs:
                    vol *= max(hi - lo, 0.0)
                if gmean_edge_length(vol, self._n_dims) < self.config.min_box_size:
                    continue
                if ffb_result.absorbed_box_ids:
                    forest.remove_boxes(ffb_result.absorbed_box_ids)
                box = BoxNode(
                    node_id=nid,
                    joint_intervals=ivs,
                    seed_config=q_seed.copy(),
                    volume=vol,
                )
                forest.add_box_direct(box)
                raw_new_boxes.append(box)

        # 主采样循环（并行分区模式 / 单区模式）
        n_boxes = len(existing_list) + len(raw_new_boxes)
        if parallel_mode:
            partitions = self._prepare_partitions()
            if partitions:
                per_partition_iters = max(1, self.config.max_iterations // len(partitions))
                local_results: List[Dict[str, object]] = []
                workers = self.config.parallel_workers if self.config.parallel_workers > 0 else len(partitions)
                try:
                    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
                        futs = []
                        for pm in partitions:
                            part_seed = int(rng.integers(0, 2**31 - 1))
                            payload = {
                                "robot": self.robot,
                                "obstacles": self.obstacles,
                                "q_start": q_start,
                                "q_goal": q_goal,
                                "partition_meta": pm,
                                "max_iterations": per_partition_iters,
                                "min_box_size": self.config.min_box_size,
                                "active_split_dims": self.config.parallel_partition_dims,
                                "seed": part_seed,
                            }
                            futs.append(ex.submit(_partition_expand_worker, payload))
                        for fut in as_completed(futs):
                            local_results.append(fut.result())
                except Exception as e:
                    logger.warning("ProcessPool 执行失败，回退进程内扩展: %s", e)
                    local_results = []
                    for pm in partitions:
                        part_seed = int(rng.integers(0, 2**31 - 1))
                        local_boxes = self._expand_partition_worker(
                            pm, q_start, q_goal,
                            max_iterations=per_partition_iters,
                            seed=part_seed,
                        )
                        local_results.append({
                            "partition_id": int(pm["partition_id"]),
                            "boxes": local_boxes,
                        })

                if local_results:
                    partition_cross_edges = self._merge_connect_partitions(
                        forest,
                        local_results,
                        partitions,
                    )
                    raw_new_boxes = list(forest.boxes.values())
                    n_boxes = len(forest.boxes)
            else:
                logger.warning("parallel_expand 启用但未生成分区，回退单区扩展")

        if not parallel_mode:
            for iteration in range(self.config.max_iterations):
                if n_boxes >= self.config.max_box_nodes:
                    break

                q_seed = self._sample_seed(q_start, q_goal, rng)
                if q_seed is None:
                    continue

                # 用树的占用状态检查（O(depth)）
                if self.hier_tree.is_occupied(q_seed):
                    continue

                nid = forest.allocate_id()
                ffb_result = self.hier_tree.find_free_box(
                    q_seed, self.obstacles, mark_occupied=True,
                    forest_box_id=nid)
                if ffb_result is None:
                    continue
                ivs = ffb_result.intervals
                vol = 1.0
                for lo, hi in ivs:
                    vol *= max(hi - lo, 0.0)
                if gmean_edge_length(vol, self._n_dims) < self.config.min_box_size:
                    continue
                if ffb_result.absorbed_box_ids:
                    forest.remove_boxes(ffb_result.absorbed_box_ids)

                box = BoxNode(
                    node_id=nid,
                    joint_intervals=ivs,
                    seed_config=q_seed.copy(),
                    volume=vol,
                )
                forest.add_box_direct(box)
                raw_new_boxes.append(box)
                n_boxes = len(existing_list) + len(raw_new_boxes)

                if (iteration + 1) % 20 == 0 and self.config.verbose:
                    logger.info(
                        "迭代 %d: %d 已有 box + %d 新 box",
                        iteration + 1, len(existing_list), len(raw_new_boxes),
                    )

        # ---- Step 4: 验证 ----
        # 简化：对所有 box 做惰性验证（新增的通常很少）
        all_colliding = forest.validate_boxes(self.collision_checker)

        valid_boxes = {
            bid: box for bid, box in forest.boxes.items()
            if bid not in all_colliding
        }
        valid_adjacency = {
            bid: neighbors - all_colliding
            for bid, neighbors in forest.adjacency.items()
            if bid not in all_colliding
        }

        result.n_boxes_created = len(forest.boxes)

        # ---- Step 5: 构建搜索图 + 连接端点 ----
        adj_edges = self.connector.build_adjacency_edges(
            valid_boxes, valid_adjacency)
        if partition_cross_edges:
            adj_edges = adj_edges + partition_cross_edges
        endpoint_edges, start_box_id, goal_box_id = \
            self.connector.connect_endpoints_to_forest(
                q_start, q_goal, valid_boxes)

        all_edges = adj_edges + endpoint_edges
        result.edges = all_edges

        if start_box_id is None or goal_box_id is None:
            result.message = "无法将始末点连接到 box forest"
            result.computation_time = time.time() - t0
            result.n_collision_checks = self.collision_checker.n_collision_checks
            logger.warning(result.message)
            return result

        graph = self.connector.build_forest_graph(
            adj_edges, endpoint_edges,
            q_start, q_goal, start_box_id, goal_box_id,
            valid_boxes,
        )

        # ---- Step 6: Dijkstra 搜索 box 序列 ----
        path_nodes = self.gcs_optimizer._dijkstra(graph)
        if path_nodes is None:
            # 尝试桥接修复
            logger.info("Dijkstra 失败，尝试桥接修复...")
            bridge_edges = self._bridge_disconnected(
                graph, valid_boxes, q_start, q_goal,
                start_box_id, goal_box_id,
            )
            if bridge_edges:
                all_edges = all_edges + bridge_edges
                result.edges = all_edges
                graph = self.connector.build_forest_graph(
                    adj_edges + bridge_edges, endpoint_edges,
                    q_start, q_goal, start_box_id, goal_box_id,
                    valid_boxes,
                )
                path_nodes = self.gcs_optimizer._dijkstra(graph)

        if path_nodes is None:
            result.message = "图搜索未找到路径"
            result.computation_time = time.time() - t0
            result.n_collision_checks = self.collision_checker.n_collision_checks
            logger.warning(result.message)
            return result

        # 提取 box 序列
        box_sequence = []
        for node_id in path_nodes:
            if node_id in ('start', 'goal'):
                continue
            if node_id in valid_boxes:
                box_sequence.append(valid_boxes[node_id])

        # ---- Step 7: 共享面 waypoint 优化 ----
        path = self.gcs_optimizer.optimize_box_sequence(
            box_sequence, q_start, q_goal,
            allow_scipy=self.config.use_gcs,
        )

        if path is None or len(path) < 2:
            result.message = "路径优化失败"
            result.computation_time = time.time() - t0
            result.n_collision_checks = self.collision_checker.n_collision_checks
            return result

        # ---- Step 8: Box-aware 路径后处理 ----
        # 构建路径点与 box 的对应关系
        path_boxes = self._assign_boxes_to_path(path, box_sequence)

        path, path_boxes = self.path_smoother.shortcut_in_boxes(
            path, path_boxes,
            max_iters=self.config.path_shortcut_iters, rng=rng,
        )
        path = self.path_smoother.smooth_in_boxes(
            path, path_boxes,
        )

        # ---- Step 9: 保存 BoxForest ----
        self._save_forest(forest)

        # ---- 组装结果 ----
        result.success = True
        result.path = path
        result.path_length = compute_path_length(path)
        result.forest = forest
        result.box_trees = self.tree_manager.get_all_trees()  # legacy 兼容
        result.n_boxes_created = len(forest.boxes)
        result.n_collision_checks = self.collision_checker.n_collision_checks
        result.computation_time = time.time() - t0
        result.message = (
            f"规划成功: {len(path)} 个路径点, "
            f"路径长度 {result.path_length:.4f}, "
            f"{len(forest.boxes)} 个 box (forest), "
            f"{sum(len(v) for v in forest.adjacency.values()) // 2} 条邻接边"
        )
        logger.info(result.message)
        return result

    def _load_or_create_forest(self) -> BoxForest:
        """加载或创建 BoxForest"""
        forest_path = self.config.forest_path
        if forest_path:
            try:
                forest = BoxForest.load(forest_path, self.robot)
                logger.info("从 %s 加载 BoxForest", forest_path)
                return forest
            except Exception as e:
                logger.warning("加载 BoxForest 失败: %s, 创建新实例", e)

        # period 从实际 joint_limits 计算，而非 2π（避免浮点截断误差）
        period = float(self.joint_limits[0][1] - self.joint_limits[0][0])
        return BoxForest(
            robot_fingerprint=self.robot.fingerprint(),
            joint_limits=self.joint_limits,
            config=self.config,
            period=period,
        )

    def _save_forest(self, forest: BoxForest) -> None:
        """保存 BoxForest（如果配置了路径）"""
        forest_path = self.config.forest_path
        if forest_path:
            try:
                forest.save(forest_path)
            except Exception as e:
                logger.warning("保存 BoxForest 失败: %s", e)

    def _assign_boxes_to_path(
        self,
        path: List[np.ndarray],
        box_sequence: List[BoxNode],
    ) -> List[BoxNode]:
        """为路径的每个点分配对应的 box

        分配策略：
        - 首先尝试在 box_sequence 中找包含该点的 box
        - 找不到就用最近的 box
        """
        result = []
        for pt in path:
            best = None
            for box in box_sequence:
                if box.contains(pt):
                    best = box
                    break
            if best is None and box_sequence:
                # 找最近的
                best = min(box_sequence,
                           key=lambda b: b.distance_to_config(pt))
            result.append(best if best is not None else box_sequence[0])
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
        """从始末点创建初始 box tree（使用 HierAABBTree）"""
        for q_seed, label in [(q_start, "起始"), (q_goal, "目标")]:
            nid = self.tree_manager.allocate_node_id()
            ffb_result = self.hier_tree.find_free_box(
                q_seed, self.obstacles, mark_occupied=True,
                forest_box_id=nid)
            if ffb_result is None:
                continue
            ivs = ffb_result.intervals
            vol = 1.0
            for lo, hi in ivs:
                vol *= max(hi - lo, 0.0)
            if gmean_edge_length(vol, self._n_dims) < self.config.min_box_size:
                continue
            box = BoxNode(
                node_id=nid,
                joint_intervals=ivs,
                seed_config=q_seed.copy(),
                volume=vol,
            )
            self.tree_manager.create_tree(box)
            logger.info("%s box: 几何平均边长 %.6f", label, gmean_edge_length(vol, self._n_dims))

    def _sample_seed(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        rng: np.random.Generator,
        sampling_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[np.ndarray]:
        """采样一个无碰撞 seed 点

        策略（三层优先级）：
        1. 以 goal_bias 概率朝目标采样
        2. 以 guided_sample_ratio 概率使用 KD 树引导采样（偏向未占用区域）
        3. 其余情况均匀随机采样
        逐个检测碰撞，找到首个无碰撞配置即返回（早停）。
        """
        max_attempts = 20

        intervals = sampling_intervals if sampling_intervals is not None else self.joint_limits
        lows = np.array([lo for lo, _ in intervals], dtype=np.float64)
        highs = np.array([hi for _, hi in intervals], dtype=np.float64)

        goal_bias = self.config.goal_bias
        guided_ratio = getattr(self.config, 'guided_sample_ratio', 0.6)
        has_hier_tree = hasattr(self, 'hier_tree') and self.hier_tree is not None

        for _ in range(max_attempts):
            roll = rng.uniform()
            if roll < goal_bias:
                # goal 偏向
                noise = rng.normal(0.0, 0.3, size=self._n_dims)
                q = np.clip(q_goal + noise, lows, highs)
            elif has_hier_tree and roll < goal_bias + guided_ratio:
                # KD 树引导采样
                q = self.hier_tree.sample_unoccupied_seed(rng)
                if q is None:
                    q = rng.uniform(lows, highs)
            else:
                # 均匀随机
                q = rng.uniform(lows, highs)

            if not self.collision_checker.check_config_collision(q):
                return q.copy()

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
        """在树的叶子 box 边界上再采样并拓展（使用 HierAABBTree）"""
        samples = self.tree_manager.get_boundary_samples(
            tree_id, n_samples=self.config.seed_batch_size, rng=rng,
        )

        for q_seed in samples:
            if self.tree_manager.total_nodes >= self.config.max_box_nodes:
                break

            if self.collision_checker.check_config_collision(q_seed):
                continue

            node_id = self.tree_manager.allocate_node_id()
            ffb_result = self.hier_tree.find_free_box(
                q_seed, self.obstacles, mark_occupied=True,
                forest_box_id=node_id)
            if ffb_result is None:
                continue
            ivs = ffb_result.intervals
            vol = 1.0
            for lo, hi in ivs:
                vol *= max(hi - lo, 0.0)
            if gmean_edge_length(vol, self._n_dims) < self.config.min_box_size:
                continue

            new_box = BoxNode(
                node_id=node_id,
                joint_intervals=ivs,
                seed_config=q_seed.copy(),
                volume=vol,
            )

            # 找到距 seed 最近的同树 box 作为父节点
            nearest = self.tree_manager.find_nearest_box_in_tree(tree_id, q_seed)
            if nearest is not None:
                self.tree_manager.add_box(
                    tree_id, new_box, parent_id=nearest.node_id,
                )

    def _bridge_sampling(self, rng: np.random.Generator) -> None:
        """桥接采样：在不同树之间的间隙区域定向采样（使用 HierAABBTree）"""
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
                    ffb_result = self.hier_tree.find_free_box(
                        q_seed, self.obstacles, mark_occupied=True,
                        forest_box_id=node_id)
                    if ffb_result is None:
                        continue
                    ivs = ffb_result.intervals
                    vol = 1.0
                    for lo, hi in ivs:
                        vol *= max(hi - lo, 0.0)
                    if gmean_edge_length(vol, self._n_dims) < self.config.min_box_size:
                        continue

                    new_box = BoxNode(
                        node_id=node_id,
                        joint_intervals=ivs,
                        seed_config=q_seed.copy(),
                        volume=vol,
                    )
                    self._add_box_to_tree(new_box, rng)


# 向后兼容别名
BoxRRT = BoxPlanner
