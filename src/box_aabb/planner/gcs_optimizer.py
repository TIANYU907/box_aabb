"""
planner/gcs_optimizer.py - GCS 路径优化

轻量集成 Drake 的 GraphOfConvexSets 求解器，在 box graph 上
执行凸优化寻找最短路径。

当 Drake 不可用时，回退到 Dijkstra + scipy 简化方案。

参考:
    Marcucci et al., "Motion planning around obstacles with convex optimization",
    Science Robotics, 2023. DOI: 10.1126/scirobotics.adf7843
"""

import logging
import heapq
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

from .models import BoxNode, Edge

logger = logging.getLogger(__name__)

# 尝试导入 Drake
try:
    from pydrake.geometry.optimization import (
        GraphOfConvexSets,
        HPolyhedron,
        Point as DrakePoint,
    )
    from pydrake.solvers import MosekSolver, GurobiSolver, CommonSolverOption
    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False
    logger.debug("Drake 不可用，GCS 优化器将使用 fallback 模式")

# 尝试导入 scipy（用于 fallback）
try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class GCSOptimizer:
    """GCS 路径优化器

    在 box graph 上使用凸优化寻找最优路径。

    - **Drake 模式**：使用 ``pydrake.geometry.optimization.GraphOfConvexSets``
      构建凸优化问题，求解松弛 + rounding。
    - **Fallback 模式**：使用 Dijkstra 最短路径 + scipy 局部优化。

    Args:
        fallback: Drake 不可用时是否启用 fallback
        bezier_degree: GCS Bézier 曲线阶数

    Example:
        >>> optimizer = GCSOptimizer()
        >>> path = optimizer.optimize(graph, q_start, q_goal)
    """

    def __init__(
        self,
        fallback: bool = True,
        bezier_degree: int = 3,
    ) -> None:
        self.fallback = fallback
        self.bezier_degree = bezier_degree
        self.use_drake = HAS_DRAKE

    def optimize(
        self,
        adjacency_graph: Dict,
        boxes: Dict[int, BoxNode],
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """在 box graph 上优化路径

        Args:
            adjacency_graph: 由 TreeConnector.build_adjacency_graph 构建的图
            boxes: {node_id: BoxNode} 映射
            q_start: 起始配置
            q_goal: 目标配置

        Returns:
            优化后的路径点列表，或 None（失败）
        """
        if self.use_drake:
            try:
                return self._optimize_drake(adjacency_graph, boxes, q_start, q_goal)
            except Exception as e:
                logger.warning("Drake GCS 优化失败: %s, 回退到 fallback", e)
                if self.fallback:
                    return self._optimize_fallback(adjacency_graph, boxes, q_start, q_goal)
                return None

        if self.fallback:
            return self._optimize_fallback(adjacency_graph, boxes, q_start, q_goal)

        logger.error("Drake 不可用且 fallback 禁用，无法优化")
        return None

    # ==================== Drake GCS ====================

    def _optimize_drake(
        self,
        adjacency_graph: Dict,
        boxes: Dict[int, BoxNode],
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """使用 Drake GraphOfConvexSets 优化"""
        if not HAS_DRAKE:
            raise RuntimeError("Drake 不可用")

        gcs = GraphOfConvexSets()
        n_dims = len(q_start)

        # 为每个 box 创建 GCS vertex
        box_to_vertex = {}
        for node_id, box in boxes.items():
            # 构建 HPolyhedron: lb <= x <= ub
            lb = np.array([lo for lo, hi in box.joint_intervals])
            ub = np.array([hi for lo, hi in box.joint_intervals])

            # A·x <= b 形式: [I; -I] x <= [ub; -lb]
            A = np.vstack([np.eye(n_dims), -np.eye(n_dims)])
            b = np.concatenate([ub, -lb])
            hpoly = HPolyhedron(A, b)

            v = gcs.AddVertex(hpoly, f"box_{node_id}")
            box_to_vertex[node_id] = v

        # 添加 start/goal 点
        source = gcs.AddVertex(DrakePoint(q_start), "start")
        target = gcs.AddVertex(DrakePoint(q_goal), "goal")

        # 添加边
        adj_dict = adjacency_graph['edges']
        added_edges = set()

        for src_id, neighbors in adj_dict.items():
            for tgt_id, cost, edge_obj in neighbors:
                # 跳过 start/goal 特殊节点
                if src_id == 'start':
                    src_vertex = source
                elif src_id == 'goal':
                    src_vertex = target
                elif src_id in box_to_vertex:
                    src_vertex = box_to_vertex[src_id]
                else:
                    continue

                if tgt_id == 'start':
                    tgt_vertex = source
                elif tgt_id == 'goal':
                    tgt_vertex = target
                elif tgt_id in box_to_vertex:
                    tgt_vertex = box_to_vertex[tgt_id]
                else:
                    continue

                # 避免重复边
                edge_key = (id(src_vertex), id(tgt_vertex))
                if edge_key in added_edges:
                    continue
                added_edges.add(edge_key)

                gcs_edge = gcs.AddEdge(src_vertex, tgt_vertex)

                # 添加路径长度代价
                xu = gcs_edge.xu()
                xv = gcs_edge.xv()
                # L2 cost: ||xu - xv||^2
                gcs_edge.AddCost((xu - xv).dot(xu - xv))

                # 连续性约束: xu == xv (对应 GCS 论文中的 ri,m = rj,0)
                for d in range(n_dims):
                    gcs_edge.AddConstraint(xu[d] == xv[d])

        # 求解
        result = gcs.SolveShortestPath(source, target, True)
        if not result.is_success():
            logger.warning("Drake GCS 求解失败")
            return None

        # 提取路径
        path = [q_start.copy()]
        # TODO: 从 result 提取中间路径点
        # 目前简化版直接提取激活边上的点
        for edge in gcs.Edges():
            if result.GetSolution(edge.phi()) > 0.5:
                xu_val = result.GetSolution(edge.xu())
                path.append(xu_val)

        path.append(q_goal.copy())
        logger.info("Drake GCS 优化完成: %d 个路径点", len(path))
        return path

    # ==================== Fallback: Dijkstra + scipy ====================

    def _optimize_fallback(
        self,
        adjacency_graph: Dict,
        boxes: Dict[int, BoxNode],
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """Fallback: Dijkstra 最短路径 + scipy 局部优化"""
        logger.info("使用 fallback 模式: Dijkstra + 局部优化")

        # Dijkstra 找最短路径（以边代价为权重）
        path_nodes = self._dijkstra(adjacency_graph)
        if path_nodes is None:
            logger.warning("Dijkstra 找不到路径")
            return None

        # 将 box 序列转为路径点（每个 box 取中心或最近点）
        path = [q_start.copy()]
        prev_point = q_start

        for node_id in path_nodes:
            if node_id in ('start', 'goal'):
                continue
            if node_id in boxes:
                box = boxes[node_id]
                # 取 box 内离前一个点最近的点
                nearest = box.nearest_point_to(prev_point)
                path.append(nearest)
                prev_point = nearest

        path.append(q_goal.copy())

        # scipy 局部优化（可选）
        if HAS_SCIPY and len(path) > 2:
            path = self._scipy_smooth(path, boxes, path_nodes)

        logger.info("Fallback 优化完成: %d 个路径点", len(path))
        return path

    def _dijkstra(self, graph: Dict) -> Optional[List]:
        """Dijkstra 最短路径

        Returns:
            节点 ID 序列 ['start', box_id_1, ..., box_id_n, 'goal']
        """
        start = graph['start']
        goal = graph['goal']
        adj = graph['edges']

        if start not in adj or goal not in adj:
            logger.debug("Dijkstra: start 或 goal 不在图中")
            return None

        dist: Dict = {start: 0.0}
        prev: Dict = {}
        counter = 0  # 防止 heap 比较时类型不匹配
        heap = [(0.0, counter, start)]
        visited = set()

        while heap:
            d, _, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            if u == goal:
                # 重建路径
                path = []
                node = goal
                while node in prev:
                    path.append(node)
                    node = prev[node]
                path.append(start)
                return list(reversed(path))

            for v, cost, _ in adj.get(u, []):
                if v in visited:
                    continue
                new_dist = d + max(cost, 1e-6)
                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    prev[v] = u
                    counter += 1
                    heapq.heappush(heap, (new_dist, counter, v))

        return None

    def _scipy_smooth(
        self,
        path: List[np.ndarray],
        boxes: Dict[int, BoxNode],
        path_nodes: List,
    ) -> List[np.ndarray]:
        """使用 scipy 对路径做局部优化

        最小化路径长度，约束每个中间点在对应 box 内。
        """
        if not HAS_SCIPY:
            return path

        n_pts = len(path)
        if n_pts <= 2:
            return path

        n_dims = len(path[0])

        # 只优化中间点（固定 start/goal）
        x0 = np.concatenate(path[1:-1])

        # 收集中间点对应的 box
        mid_boxes = []
        for node_id in path_nodes:
            if node_id in ('start', 'goal'):
                continue
            if node_id in boxes:
                mid_boxes.append(boxes[node_id])
        # 确保长度匹配
        while len(mid_boxes) < n_pts - 2:
            mid_boxes.append(None)

        def objective(x):
            pts = [path[0]] + [x[i * n_dims:(i + 1) * n_dims]
                                for i in range(n_pts - 2)] + [path[-1]]
            length = 0.0
            for i in range(1, len(pts)):
                length += np.linalg.norm(pts[i] - pts[i - 1])
            return length

        # box 约束
        bounds = []
        for i in range(n_pts - 2):
            if i < len(mid_boxes) and mid_boxes[i] is not None:
                box = mid_boxes[i]
                for d in range(n_dims):
                    lo, hi = box.joint_intervals[d]
                    bounds.append((lo, hi))
            else:
                for d in range(n_dims):
                    bounds.append((None, None))

        try:
            result = scipy_minimize(
                objective, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6},
            )
            if result.success:
                optimized = [path[0]]
                for i in range(n_pts - 2):
                    optimized.append(result.x[i * n_dims:(i + 1) * n_dims])
                optimized.append(path[-1])
                return optimized
        except Exception as e:
            logger.warning("scipy 优化失败: %s", e)

        return path
