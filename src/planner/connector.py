"""
planner/connector.py - 树间连接与始末点连接

负责：
1. 连接不同 box tree（构建 graph of convex sets 的边）
2. 基于无重叠邻接图构建边（v5.0）
3. 连接规划起始点/目标点到最近的 box
4. 构建整体图用于路径搜索

连接策略（v5.0 更新）：
- 邻接模式：从 deoverlap 后的邻接表直接构建边，transition 在共享面中心
- Legacy 模式：兼容旧的 overlap-based 和 segment-based 连接
"""

import logging
from typing import List, Tuple, Dict, Optional, Set

import numpy as np

from .models import BoxNode, BoxTree, Edge
from .box_tree import BoxTreeManager
from .collision import CollisionChecker

logger = logging.getLogger(__name__)


class TreeConnector:
    """树间连接器

    构建 box tree 之间以及 box 之间的连接关系，形成 Graph of Convex Sets。

    Args:
        tree_manager: box 树管理器
        collision_checker: 碰撞检测器
        max_attempts: 每对树最大连接尝试次数
        connection_radius: 最大连接距离（关节空间 L2）
        segment_resolution: 线段碰撞检测分辨率

    Example:
        >>> connector = TreeConnector(manager, checker)
        >>> edges = connector.connect_all_trees()
        >>> graph = connector.build_adjacency_graph(q_start, q_goal, edges)
    """

    def __init__(
        self,
        tree_manager: BoxTreeManager,
        collision_checker: CollisionChecker,
        max_attempts: int = 50,
        connection_radius: float = 2.0,
        segment_resolution: float = 0.05,
    ) -> None:
        self.tree_manager = tree_manager
        self.collision_checker = collision_checker
        self.max_attempts = max_attempts
        self.connection_radius = connection_radius
        self.segment_resolution = segment_resolution
        self._next_edge_id = 0

    def _allocate_edge_id(self) -> int:
        eid = self._next_edge_id
        self._next_edge_id += 1
        return eid

    # ==================== 树内连接 ====================

    def connect_within_trees(self) -> List[Edge]:
        """连接每棵树内有交集的 box 对

        父子关系以及兄弟关系中有交集的 box 自动建立连接。

        Returns:
            树内连接边列表
        """
        edges: List[Edge] = []
        for tree in self.tree_manager.get_all_trees():
            nodes = list(tree.nodes.values())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if nodes[i].overlap_with(nodes[j]):
                        # 交集点取两个 box 交集区域的中心
                        mid = self._intersection_center(nodes[i], nodes[j])
                        edge = Edge(
                            edge_id=self._allocate_edge_id(),
                            source_box_id=nodes[i].node_id,
                            target_box_id=nodes[j].node_id,
                            source_config=mid,
                            target_config=mid,
                            source_tree_id=tree.tree_id,
                            target_tree_id=tree.tree_id,
                            is_collision_free=True,  # 在 box 内，自然无碰撞
                        )
                        edges.append(edge)
        logger.info("树内连接: %d 条边", len(edges))
        return edges

    # ==================== 树间连接 ====================

    def connect_between_trees(self) -> List[Edge]:
        """连接不同树之间的 box

        策略：
        1. 先找跨树有交集的 box 对（trivially collision-free）
        2. 再对没有交集的找最近 box 对、用线段碰撞检测验证

        Returns:
            树间连接边列表
        """
        edges: List[Edge] = []
        trees = self.tree_manager.get_all_trees()

        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                # 第一步：找跨树的重叠 box 对
                overlap_edges = self._connect_overlapping_between(trees[i], trees[j])
                edges.extend(overlap_edges)
                # 第二步：如果没有重叠，尝试线段连接
                if not overlap_edges:
                    new_edges = self._try_connect_two_trees(trees[i], trees[j])
                    edges.extend(new_edges)

        logger.info("树间连接: %d 条边", len(edges))
        return edges

    def _connect_overlapping_between(
        self, tree_a: BoxTree, tree_b: BoxTree,
    ) -> List[Edge]:
        """找两棵树中有交集的 box 对，直接建立连接"""
        edges: List[Edge] = []
        max_edges = 20  # 限制数量避免爆炸

        for box_a in tree_a.nodes.values():
            for box_b in tree_b.nodes.values():
                if box_a.overlap_with(box_b):
                    mid = self._intersection_center(box_a, box_b)
                    edge = Edge(
                        edge_id=self._allocate_edge_id(),
                        source_box_id=box_a.node_id,
                        target_box_id=box_b.node_id,
                        source_config=mid,
                        target_config=mid,
                        source_tree_id=tree_a.tree_id,
                        target_tree_id=tree_b.tree_id,
                        is_collision_free=True,
                    )
                    edges.append(edge)
                    if len(edges) >= max_edges:
                        return edges
        return edges

    def _try_connect_two_trees(
        self, tree_a: BoxTree, tree_b: BoxTree,
    ) -> List[Edge]:
        """尝试连接两棵树

        策略：
        1. 找到两棵树中距离最近的 box 对
        2. 在 box 表面采样连接点
        3. 线段碰撞检测验证
        """
        edges: List[Edge] = []

        # 找距离最近的 k 对 box
        candidates = self._find_closest_box_pairs(tree_a, tree_b, k=self.max_attempts)

        for box_a, box_b, dist in candidates:
            if dist > self.connection_radius:
                continue

            # 在 box_a 表面取离 box_b 最近的点
            q_a = box_a.nearest_point_to(box_b.center)
            q_b = box_b.nearest_point_to(box_a.center)

            # 验证线段无碰撞
            if not self.collision_checker.check_segment_collision(
                q_a, q_b, self.segment_resolution
            ):
                edge = Edge(
                    edge_id=self._allocate_edge_id(),
                    source_box_id=box_a.node_id,
                    target_box_id=box_b.node_id,
                    source_config=q_a,
                    target_config=q_b,
                    source_tree_id=tree_a.tree_id,
                    target_tree_id=tree_b.tree_id,
                    is_collision_free=True,
                )
                edges.append(edge)
                logger.debug("连接树 %d 节点 %d → 树 %d 节点 %d，距离 %.4f",
                             tree_a.tree_id, box_a.node_id,
                             tree_b.tree_id, box_b.node_id, dist)
                # 允许多条连接以增加鲁棒性
                if len(edges) >= 3:
                    break

        return edges

    def _find_closest_box_pairs(
        self, tree_a: BoxTree, tree_b: BoxTree, k: int,
    ) -> List[Tuple[BoxNode, BoxNode, float]]:
        """找两棵树中距离最近的 k 对 box"""
        pairs: List[Tuple[BoxNode, BoxNode, float]] = []

        for box_a in tree_a.nodes.values():
            for box_b in tree_b.nodes.values():
                dist = float(np.linalg.norm(box_a.center - box_b.center))
                pairs.append((box_a, box_b, dist))

        pairs.sort(key=lambda x: x[2])
        return pairs[:k]

    @staticmethod
    def _intersection_center(a: BoxNode, b: BoxNode) -> np.ndarray:
        """计算两个 box 交集区域的中心"""
        center = np.empty(a.n_dims)
        for i in range(a.n_dims):
            lo = max(a.joint_intervals[i][0], b.joint_intervals[i][0])
            hi = min(a.joint_intervals[i][1], b.joint_intervals[i][1])
            center[i] = (lo + hi) / 2.0
        return center

    # ==================== 始末点连接 ====================

    def connect_endpoints(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Tuple[List[Edge], Optional[int], Optional[int]]:
        """将规划起始点和目标点连接到 box tree

        对每个端点：
        1. 检查是否在某个 box 内 → 直接连接
        2. 否则找最近的 box，用线段连接

        Args:
            q_start: 起始配置
            q_goal: 目标配置

        Returns:
            (edges, start_box_id, goal_box_id)
            start_box_id: 始点连接到的 box ID（None 表示连接失败）
            goal_box_id: 终点连接到的 box ID
        """
        edges: List[Edge] = []

        start_box_id = self._connect_point(q_start, "start", edges)
        goal_box_id = self._connect_point(q_goal, "goal", edges)

        return edges, start_box_id, goal_box_id

    def _connect_point(
        self,
        config: np.ndarray,
        label: str,
        edges: List[Edge],
    ) -> Optional[int]:
        """将一个点连接到最近的 box"""
        # 1. 检查是否在某个 box 内
        box = self.tree_manager.find_containing_box(config)
        if box is not None:
            logger.info("%s 点在 box %d (树 %d) 内", label, box.node_id, box.tree_id)
            return box.node_id

        # 2. 找最近的 box
        box = self.tree_manager.find_nearest_box(config)
        if box is None:
            logger.warning("%s 点无法连接：无可用 box", label)
            return None

        # 在 box 表面取最近点
        q_box = box.nearest_point_to(config)

        # 验证线段
        if self.collision_checker.check_segment_collision(
            config, q_box, self.segment_resolution
        ):
            logger.warning("%s 点到最近 box %d 的连接线段存在碰撞", label, box.node_id)
            # 尝试其他 box
            for candidate in self._find_candidates_for_point(config, exclude=box.node_id):
                q_cand = candidate.nearest_point_to(config)
                if not self.collision_checker.check_segment_collision(
                    config, q_cand, self.segment_resolution
                ):
                    edge = Edge(
                        edge_id=self._allocate_edge_id(),
                        source_box_id=-1,  # 端点用 -1 标记
                        target_box_id=candidate.node_id,
                        source_config=config.copy(),
                        target_config=q_cand,
                        source_tree_id=-1,
                        target_tree_id=candidate.tree_id,
                        is_collision_free=True,
                    )
                    edges.append(edge)
                    return candidate.node_id
            return None

        edge = Edge(
            edge_id=self._allocate_edge_id(),
            source_box_id=-1,
            target_box_id=box.node_id,
            source_config=config.copy(),
            target_config=q_box,
            source_tree_id=-1,
            target_tree_id=box.tree_id,
            is_collision_free=True,
        )
        edges.append(edge)
        return box.node_id

    def _find_candidates_for_point(
        self, config: np.ndarray, exclude: int, max_candidates: int = 10,
    ) -> List[BoxNode]:
        """找出离 config 最近的若干 box（排除指定 box）"""
        all_boxes = self.tree_manager.get_all_boxes()
        candidates = [(b, b.distance_to_config(config))
                      for b in all_boxes if b.node_id != exclude]
        candidates.sort(key=lambda x: x[1])
        return [b for b, _ in candidates[:max_candidates]]

    # ==================== 邻接图模式 (v5.0) ====================

    def build_adjacency_edges(
        self,
        boxes: Dict[int, BoxNode],
        adjacency: Dict[int, Set[int]],
    ) -> List[Edge]:
        """从无重叠邻接图构建边

        每对邻接 box 的共享面中心作为 transition 点。

        Args:
            boxes: {node_id: BoxNode}
            adjacency: {node_id: set of neighbor_ids}

        Returns:
            邻接边列表
        """
        from .deoverlap import shared_face_center

        edges: List[Edge] = []
        seen: Set[Tuple[int, int]] = set()

        for box_id, neighbors in adjacency.items():
            if box_id not in boxes:
                continue
            for nb_id in neighbors:
                if nb_id not in boxes:
                    continue
                key = (min(box_id, nb_id), max(box_id, nb_id))
                if key in seen:
                    continue
                seen.add(key)

                wp = shared_face_center(boxes[box_id], boxes[nb_id])
                if wp is None:
                    # fallback: 交集中心（微小重叠场景）
                    wp = self._intersection_center(boxes[box_id], boxes[nb_id])

                edge = Edge(
                    edge_id=self._allocate_edge_id(),
                    source_box_id=box_id,
                    target_box_id=nb_id,
                    source_config=wp,
                    target_config=wp,
                    source_tree_id=boxes[box_id].tree_id,
                    target_tree_id=boxes[nb_id].tree_id,
                    is_collision_free=True,  # box 内，自然无碰撞
                )
                edges.append(edge)

        logger.info("邻接图边: %d 条", len(edges))
        return edges

    def connect_endpoints_to_forest(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        boxes: Dict[int, BoxNode],
    ) -> Tuple[List[Edge], Optional[int], Optional[int]]:
        """将始末点连接到 BoxForest 的 box

        优先检查包含，然后尝试自由空间线段连接。

        Args:
            q_start, q_goal: 起始和目标配置
            boxes: 所有有效（非碰撞）的 box

        Returns:
            (edges, start_box_id, goal_box_id)
        """
        edges: List[Edge] = []

        start_box_id = self._connect_point_to_boxes(
            q_start, "start", edges, boxes)
        goal_box_id = self._connect_point_to_boxes(
            q_goal, "goal", edges, boxes)

        return edges, start_box_id, goal_box_id

    def _connect_point_to_boxes(
        self,
        config: np.ndarray,
        label: str,
        edges: List[Edge],
        boxes: Dict[int, BoxNode],
    ) -> Optional[int]:
        """将一个点连接到 box 集合中最近的 box"""
        # 1. 检查是否在某个 box 内
        for box in boxes.values():
            if box.contains(config):
                logger.info("%s 点在 box %d 内", label, box.node_id)
                return box.node_id

        # 2. 找最近的 box + 线段碰撞检测
        candidates = sorted(
            boxes.values(),
            key=lambda b: b.distance_to_config(config),
        )

        for box in candidates[:20]:
            q_box = box.nearest_point_to(config)
            if not self.collision_checker.check_segment_collision(
                config, q_box, self.segment_resolution
            ):
                edge = Edge(
                    edge_id=self._allocate_edge_id(),
                    source_box_id=-1,
                    target_box_id=box.node_id,
                    source_config=config.copy(),
                    target_config=q_box,
                    source_tree_id=-1,
                    target_tree_id=box.tree_id,
                    is_collision_free=True,
                )
                edges.append(edge)
                logger.info(
                    "%s 点通过线段连接到 box %d (距离 %.4f)",
                    label, box.node_id, box.distance_to_config(config),
                )
                return box.node_id

        logger.warning("%s 点无法连接到任何 box", label)
        return None

    # ==================== 图构建 ====================

    def build_forest_graph(
        self,
        adjacency_edges: List[Edge],
        endpoint_edges: List[Edge],
        q_start: np.ndarray,
        q_goal: np.ndarray,
        start_box_id: Optional[int],
        goal_box_id: Optional[int],
        boxes: Dict[int, BoxNode],
    ) -> Dict:
        """构建基于 BoxForest 邻接图的搜索图

        与 build_adjacency_graph 格式相同，但节点来自 boxes 参数
        而非 tree_manager。

        Args:
            adjacency_edges: 邻接边
            endpoint_edges: 端点连接边
            q_start, q_goal: 起终点
            start_box_id, goal_box_id: 端点连接的 box ID
            boxes: 有效 box 字典

        Returns:
            搜索图字典
        """
        all_edges = adjacency_edges + endpoint_edges

        nodes: Set = set()
        adj: Dict = {}

        # 收集所有 box 节点
        for box_id in boxes:
            nodes.add(box_id)
            adj.setdefault(box_id, [])

        # 添加边
        for edge in all_edges:
            src, tgt = edge.source_box_id, edge.target_box_id

            if src == -1:
                src = 'start' if np.allclose(edge.source_config, q_start) else 'goal'
                nodes.add(src)
                adj.setdefault(src, [])

            if tgt == -1:
                tgt = 'start' if np.allclose(edge.target_config, q_start) else 'goal'
                nodes.add(tgt)
                adj.setdefault(tgt, [])

            adj.setdefault(src, []).append((tgt, edge.cost, edge))
            adj.setdefault(tgt, []).append((src, edge.cost, edge))

        # 确保 start/goal 在图中
        if start_box_id is not None:
            nodes.add('start')
            adj.setdefault('start', []).append((start_box_id, 0.0, None))
            adj.setdefault(start_box_id, []).append(('start', 0.0, None))

        if goal_box_id is not None:
            nodes.add('goal')
            adj.setdefault('goal', []).append((goal_box_id, 0.0, None))
            adj.setdefault(goal_box_id, []).append(('goal', 0.0, None))

        return {
            'nodes': nodes,
            'edges': adj,
            'start': 'start',
            'goal': 'goal',
            'start_box': start_box_id,
            'goal_box': goal_box_id,
        }

    # ==================== 图构建 (legacy) ====================

    def build_adjacency_graph(
        self,
        all_edges: List[Edge],
        q_start: np.ndarray,
        q_goal: np.ndarray,
        start_box_id: Optional[int],
        goal_box_id: Optional[int],
    ) -> Dict:
        """构建邻接图用于路径搜索

        图的节点为 box node_id 加上特殊的 'start' 和 'goal' 节点。
        图的边为连接关系及代价。

        Returns:
            {
                'nodes': set of node_ids (int, plus 'start', 'goal'),
                'edges': {node_id: [(neighbor_id, cost, edge), ...]},
                'start': 'start',
                'goal': 'goal',
                'start_box': start_box_id,
                'goal_box': goal_box_id,
            }
        """
        nodes: Set = set()
        adj: Dict = {}

        # 收集所有 box 节点
        for box in self.tree_manager.get_all_boxes():
            nodes.add(box.node_id)
            if box.node_id not in adj:
                adj[box.node_id] = []

        # 添加边
        for edge in all_edges:
            src, tgt = edge.source_box_id, edge.target_box_id

            if src == -1:
                src = 'start' if np.allclose(edge.source_config, q_start) else 'goal'
                nodes.add(src)
                if src not in adj:
                    adj[src] = []

            if tgt == -1:
                tgt = 'start' if np.allclose(edge.target_config, q_start) else 'goal'
                nodes.add(tgt)
                if tgt not in adj:
                    adj[tgt] = []

            adj.setdefault(src, []).append((tgt, edge.cost, edge))
            adj.setdefault(tgt, []).append((src, edge.cost, edge))

        # 确保 start/goal 在图中
        if start_box_id is not None:
            nodes.add('start')
            adj.setdefault('start', []).append((start_box_id, 0.0, None))
            adj.setdefault(start_box_id, []).append(('start', 0.0, None))

        if goal_box_id is not None:
            nodes.add('goal')
            adj.setdefault('goal', []).append((goal_box_id, 0.0, None))
            adj.setdefault(goal_box_id, []).append(('goal', 0.0, None))

        return {
            'nodes': nodes,
            'edges': adj,
            'start': 'start',
            'goal': 'goal',
            'start_box': start_box_id,
            'goal_box': goal_box_id,
        }
