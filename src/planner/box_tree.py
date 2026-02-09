"""
planner/box_tree.py - Box 树管理

管理多棵 box tree 的创建、生长和查询。
每棵树以一个根 box 为起点，通过边界采样和拓展生长。
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

from .models import BoxNode, BoxTree

logger = logging.getLogger(__name__)


class BoxTreeManager:
    """Box 树管理器

    管理多棵 box tree 的生命周期：创建、添加节点、查询、边界采样。

    Example:
        >>> manager = BoxTreeManager()
        >>> tree_id = manager.create_tree(root_box)
        >>> manager.add_box(tree_id, child_box, parent_id=root_box.node_id)
        >>> samples = manager.get_boundary_samples(tree_id, n_samples=5)
    """

    def __init__(self) -> None:
        self._trees: Dict[int, BoxTree] = {}
        self._next_tree_id: int = 0
        self._next_node_id: int = 0  # 全局节点 ID 分配

    @property
    def n_trees(self) -> int:
        return len(self._trees)

    @property
    def total_nodes(self) -> int:
        return sum(t.n_nodes for t in self._trees.values())

    def allocate_node_id(self) -> int:
        """分配全局唯一节点 ID"""
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def create_tree(self, root_box: BoxNode) -> int:
        """创建新树

        Args:
            root_box: 根 box 节点

        Returns:
            新树的 tree_id
        """
        tree_id = self._next_tree_id
        self._next_tree_id += 1

        root_box.tree_id = tree_id
        root_box.parent_id = None

        tree = BoxTree(
            tree_id=tree_id,
            nodes={root_box.node_id: root_box},
            root_id=root_box.node_id,
        )
        self._trees[tree_id] = tree
        logger.debug("创建树 %d，根节点 %d，体积 %.6f",
                      tree_id, root_box.node_id, root_box.volume)
        return tree_id

    def add_box(self, tree_id: int, box: BoxNode, parent_id: int) -> int:
        """向树中添加子节点

        Args:
            tree_id: 目标树 ID
            box: 新 box 节点
            parent_id: 父节点 ID

        Returns:
            新节点的 node_id
        """
        tree = self._trees[tree_id]
        if parent_id not in tree.nodes:
            raise ValueError(f"父节点 {parent_id} 不在树 {tree_id} 中")

        box.tree_id = tree_id
        box.parent_id = parent_id
        tree.nodes[box.node_id] = box
        tree.nodes[parent_id].children_ids.append(box.node_id)

        logger.debug("树 %d: 添加节点 %d (父=%d)，体积 %.6f",
                      tree_id, box.node_id, parent_id, box.volume)
        return box.node_id

    def get_tree(self, tree_id: int) -> BoxTree:
        """获取指定树"""
        return self._trees[tree_id]

    def get_all_trees(self) -> List[BoxTree]:
        """获取所有树"""
        return list(self._trees.values())

    def get_all_boxes(self) -> List[BoxNode]:
        """获取所有树的所有 box 节点"""
        boxes: List[BoxNode] = []
        for tree in self._trees.values():
            boxes.extend(tree.nodes.values())
        return boxes

    def get_total_volume(self) -> float:
        """所有树的总体积"""
        return sum(t.total_volume for t in self._trees.values())

    def find_containing_box(self, config: np.ndarray) -> Optional[BoxNode]:
        """找到包含给定配置的 box

        Args:
            config: 关节配置

        Returns:
            包含该配置的 BoxNode，或 None
        """
        for tree in self._trees.values():
            for box in tree.nodes.values():
                if box.contains(config):
                    return box
        return None

    def find_nearest_box(self, config: np.ndarray) -> Optional[BoxNode]:
        """找到离给定配置最近的 box

        Args:
            config: 关节配置

        Returns:
            最近的 BoxNode，或 None（无树时）
        """
        best_box = None
        best_dist = float('inf')

        for tree in self._trees.values():
            for box in tree.nodes.values():
                dist = box.distance_to_config(config)
                if dist < best_dist:
                    best_dist = dist
                    best_box = box

        return best_box

    def find_nearest_box_in_tree(
        self, tree_id: int, config: np.ndarray,
    ) -> Optional[BoxNode]:
        """在指定树中找到最近的 box"""
        tree = self._trees[tree_id]
        best_box = None
        best_dist = float('inf')

        for box in tree.nodes.values():
            dist = box.distance_to_config(config)
            if dist < best_dist:
                best_dist = dist
                best_box = box

        return best_box

    def get_boundary_samples(
        self,
        tree_id: int,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[np.ndarray]:
        """在树的叶子 box 边界上采样新 seed 点

        采样策略：
        - 概率正比于 box 体积（大 box 获得更多采样）
        - 在 box 边界面上均匀采样（随机选一个面，面上均匀）

        Args:
            tree_id: 树 ID
            n_samples: 采样数量
            rng: 随机数生成器

        Returns:
            新 seed 配置列表
        """
        if rng is None:
            rng = np.random.default_rng()

        tree = self._trees[tree_id]
        leaves = tree.get_leaf_nodes()
        if not leaves:
            leaves = list(tree.nodes.values())

        # 按体积加权选择 box
        volumes = np.array([max(n.volume, 1e-12) for n in leaves])
        probs = volumes / volumes.sum()

        samples: List[np.ndarray] = []
        for _ in range(n_samples):
            # 选一个 box
            box_idx = rng.choice(len(leaves), p=probs)
            box = leaves[box_idx]

            sample = self._sample_on_boundary(box, rng)
            samples.append(sample)

        return samples

    def _sample_on_boundary(
        self, box: BoxNode, rng: np.random.Generator,
    ) -> np.ndarray:
        """在 box 的边界面上均匀采样一个点

        1. 随机选一个维度和一个方向（lo 或 hi）
        2. 固定该维度到边界值
        3. 其余维度在 box 内均匀采样
        """
        n_dims = box.n_dims
        sample = np.empty(n_dims)

        # 选有效维度（宽度 > 0 的维度）
        valid_dims = [i for i in range(n_dims)
                      if box.joint_intervals[i][1] - box.joint_intervals[i][0] > 1e-10]

        if not valid_dims:
            # 退化为点
            return box.center.copy()

        # 随机选一个维度
        dim = rng.choice(valid_dims)
        # 随机选 lo 或 hi
        side = rng.choice(2)  # 0=lo, 1=hi

        for i in range(n_dims):
            lo, hi = box.joint_intervals[i]
            if i == dim:
                sample[i] = lo if side == 0 else hi
            else:
                if hi - lo < 1e-10:
                    sample[i] = (lo + hi) / 2.0
                else:
                    sample[i] = rng.uniform(lo, hi)

        return sample

    def get_tree_for_box(self, node_id: int) -> Optional[int]:
        """查找包含指定节点的树 ID"""
        for tree_id, tree in self._trees.items():
            if node_id in tree.nodes:
                return tree_id
        return None
