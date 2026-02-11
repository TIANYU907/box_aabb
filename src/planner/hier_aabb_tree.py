"""
planner/hier_aabb_tree.py - 层级自适应 AABB 缓存树

基于 KD-tree 式二叉空间切分的 AABB 包络缓存。
C-space 被递归二分（按维度轮转、取中点），每个节点惰性计算
interval FK AABB。随着查询次数增加，树自动加深、
父节点的 refined_aabb（子节点 union）单调变紧。

核心特性：
- **惰性求值**：仅在查询路径上创建节点和计算 AABB
- **渐进精化**：refined_aabb = union(children) ≤ raw_aabb（单调变紧）
- **跨场景复用**：仅绑定机器人运动学，障碍物场景在查询时传入
- **持久化**：pickle 保存/加载，跨会话累积缓存

使用方式：
    tree = HierAABBTree(robot)
    box = tree.find_free_box(seed, obstacles, max_depth=40)
    tree.save("hier_cache.pkl")

    # 后续加载
    tree = HierAABBTree.load("hier_cache.pkl", robot)
"""

from __future__ import annotations

import pickle
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np

from box_aabb.robot import Robot
from box_aabb.models import LinkAABBInfo
from box_aabb.interval_fk import compute_interval_aabb

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
#  节点
# ─────────────────────────────────────────────────────

@dataclass
class HierAABBNode:
    """KD-tree 节点

    Attributes:
        intervals: 此节点覆盖的 C-space 超矩形 [(lo, hi), ...]
        depth: 深度（0 = root）
        raw_aabb: 直接 interval FK 得到的保守 AABB（松）
        refined_aabb: 子节点 union 精化后的 AABB（更紧或相同）
        split_dim: 切分维度 (depth % n_dims)
        split_val: 切分值（中点）
        left: 左子节点 (dim < split_val)
        right: 右子节点 (dim >= split_val)
        parent: 父节点引用
    """
    intervals: List[Tuple[float, float]]
    depth: int = 0
    raw_aabb: Optional[List[LinkAABBInfo]] = field(default=None, repr=False)
    refined_aabb: Optional[List[LinkAABBInfo]] = field(default=None, repr=False)
    split_dim: Optional[int] = None
    split_val: Optional[float] = None
    left: Optional['HierAABBNode'] = field(default=None, repr=False)
    right: Optional['HierAABBNode'] = field(default=None, repr=False)
    parent: Optional['HierAABBNode'] = field(default=None, repr=False)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def volume(self) -> float:
        v = 1.0
        for lo, hi in self.intervals:
            v *= max(hi - lo, 0.0)
        return v

    @property
    def widths(self) -> List[float]:
        return [hi - lo for lo, hi in self.intervals]

    @property
    def center(self) -> np.ndarray:
        return np.array([(lo + hi) / 2 for lo, hi in self.intervals])


# ─────────────────────────────────────────────────────
#  树
# ─────────────────────────────────────────────────────

class HierAABBTree:
    """层级自适应 AABB 缓存树

    Attributes:
        robot: 机器人模型
        joint_limits: 关节限制
        n_dims: 关节维数
        root: 根节点
        n_nodes: 当前节点总数
        n_fk_calls: 累计 interval FK 调用次数
    """

    def __init__(
        self,
        robot: Robot,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.robot = robot
        self.robot_fingerprint = robot.fingerprint()
        self._zero_length_links: Set[int] = robot.zero_length_links.copy()

        if joint_limits is not None:
            self.joint_limits = list(joint_limits)
        elif robot.joint_limits is not None:
            self.joint_limits = list(robot.joint_limits)
        else:
            self.joint_limits = [(-np.pi, np.pi)] * robot.n_joints

        self.n_dims = len(self.joint_limits)
        self.root = HierAABBNode(intervals=list(self.joint_limits), depth=0)
        self.n_nodes = 1
        self.n_fk_calls = 0

    # ──────────────────────────────────────────────
    #  内部：AABB 计算
    # ──────────────────────────────────────────────

    def _compute_aabb(
        self, intervals: List[Tuple[float, float]]
    ) -> List[LinkAABBInfo]:
        """调用 interval FK 计算保守 AABB"""
        self.n_fk_calls += 1
        link_aabbs, _ = compute_interval_aabb(
            robot=self.robot,
            intervals=intervals,
            zero_length_links=self._zero_length_links,
            skip_zero_length=True,
            n_sub=1,
        )
        return link_aabbs

    def _ensure_aabb(self, node: HierAABBNode) -> None:
        """确保节点的 raw_aabb 和 refined_aabb 已计算"""
        if node.raw_aabb is None:
            node.raw_aabb = self._compute_aabb(node.intervals)
            if node.is_leaf():
                node.refined_aabb = node.raw_aabb
            # 如果已有子节点，refined 由子节点决定（不覆盖）

    @staticmethod
    def _union_aabb(
        a: List[LinkAABBInfo], b: List[LinkAABBInfo]
    ) -> List[LinkAABBInfo]:
        """合并两组 link AABB（逐 link 取 min/max）"""
        result: List[LinkAABBInfo] = []
        for la, lb in zip(a, b):
            min_pt = [min(la.min_point[k], lb.min_point[k]) for k in range(3)]
            max_pt = [max(la.max_point[k], lb.max_point[k]) for k in range(3)]
            result.append(LinkAABBInfo(
                link_index=la.link_index,
                link_name=la.link_name,
                min_point=min_pt,
                max_point=max_pt,
                is_zero_length=la.is_zero_length and lb.is_zero_length,
            ))
        return result

    # ──────────────────────────────────────────────
    #  内部：切分
    # ──────────────────────────────────────────────

    def _split(self, node: HierAABBNode) -> None:
        """将叶节点二分裂

        切分维度 = depth % n_dims，切分点 = 中点。
        两个子节点立即计算 AABB，然后向上传播精化。
        """
        if not node.is_leaf():
            return

        dim = node.depth % self.n_dims
        lo, hi = node.intervals[dim]
        mid = (lo + hi) / 2.0

        node.split_dim = dim
        node.split_val = mid

        left_ivs = list(node.intervals)
        left_ivs[dim] = (lo, mid)
        right_ivs = list(node.intervals)
        right_ivs[dim] = (mid, hi)

        node.left = HierAABBNode(
            intervals=left_ivs, depth=node.depth + 1, parent=node)
        node.right = HierAABBNode(
            intervals=right_ivs, depth=node.depth + 1, parent=node)
        self.n_nodes += 2

        # 两个子节点都计算 AABB（多花 1 次 FK，但填充缓存+启用精化）
        self._ensure_aabb(node.left)
        self._ensure_aabb(node.right)

        # 立即精化本节点
        node.refined_aabb = self._union_aabb(
            node.left.refined_aabb, node.right.refined_aabb)

        # 沿路径向上传播
        self._propagate_up(node.parent)

    def _propagate_up(self, node: Optional[HierAABBNode]) -> None:
        """从 node 向根方向更新 refined_aabb"""
        while node is not None:
            if node.left is None or node.right is None:
                break
            if (node.left.refined_aabb is None
                    or node.right.refined_aabb is None):
                break
            new_refined = self._union_aabb(
                node.left.refined_aabb, node.right.refined_aabb)
            # 如果没有变化则可以提前停止
            node.refined_aabb = new_refined
            node = node.parent

    # ──────────────────────────────────────────────
    #  碰撞检测辅助
    # ──────────────────────────────────────────────

    @staticmethod
    def _link_aabbs_collide(
        link_aabbs: List[LinkAABBInfo],
        obstacles: list,
        safety_margin: float = 0.0,
    ) -> bool:
        """检测 link AABB 集合是否与任何障碍物重叠"""
        for la in link_aabbs:
            if la.is_zero_length:
                continue
            la_min = np.array(la.min_point)
            la_max = np.array(la.max_point)
            for obs in obstacles:
                obs_min = obs.min_point - safety_margin
                obs_max = obs.max_point + safety_margin
                # 分离轴测试
                separated = False
                for k in range(min(len(la_min), len(obs_min))):
                    if la_max[k] < obs_min[k] - 1e-10 or obs_max[k] < la_min[k] - 1e-10:
                        separated = True
                        break
                if not separated:
                    return True
        return False

    # ──────────────────────────────────────────────
    #  核心 API：找无碰撞 box
    # ──────────────────────────────────────────────

    def find_free_box(
        self,
        seed: np.ndarray,
        obstacles: list,
        max_depth: int = 40,
        safety_margin: float = 0.0,
        min_edge_length: float = 0.0,
        post_expand_fn=None,
    ) -> Optional[List[Tuple[float, float]]]:
        """从顶向下切分，找到包含 seed 的最大无碰撞 box

        算法：
        1. 下行：从 root 出发，如果当前节点 AABB 碰撞则切分，
           走向包含 seed 的子节点，直到找到无碰撞节点或达到 max_depth。
        2. 上行：回溯路径，尝试用父节点的 refined_aabb 判断
           能否合并为更大的无碰撞 box。

        Args:
            seed: 种子配置（必须在 joint_limits 内且无碰撞）
            obstacles: 场景障碍物列表（Scene.get_obstacles()）
            max_depth: 最大切分深度
            safety_margin: 碰撞检测安全裕度
            min_edge_length: 最小分割边长，当待分割维度宽度 < 此值时停止
            post_expand_fn: 可选的后处理扩张函数（预留接口 B）
                签名: (intervals, seed, obstacles) -> intervals
                若提供，会对切分结果做进一步扩张

        Returns:
            无碰撞 box 的 intervals，或 None
        """
        node = self.root
        self._ensure_aabb(node)
        path: List[HierAABBNode] = []

        # ── 下行：沿 seed 方向切分直到无碰撞 ──
        while True:
            path.append(node)

            aabb = node.refined_aabb or node.raw_aabb
            if not self._link_aabbs_collide(aabb, obstacles, safety_margin):
                break  # 整个节点无碰撞

            if node.depth >= max_depth:
                return None  # 达到最大深度仍碰撞

            # 检查最小边长：待分割维度宽度过小则停止
            split_dim = node.depth % self.n_dims
            edge = node.intervals[split_dim][1] - node.intervals[split_dim][0]
            if min_edge_length > 0 and edge < min_edge_length * 2:
                return None  # 再分就低于最小边长

            # 惰性切分
            self._split(node)

            # 走向包含 seed 的子节点
            if seed[node.split_dim] < node.split_val:
                node = node.left
            else:
                node = node.right

        # ── 上行：尝试合并为更大 box ──
        result_node = node
        for i in range(len(path) - 2, -1, -1):
            parent = path[i]
            aabb = parent.refined_aabb or parent.raw_aabb
            if not self._link_aabbs_collide(aabb, obstacles, safety_margin):
                result_node = parent
            else:
                break

        result_intervals = list(result_node.intervals)

        # ── 可选：后处理扩张（接口 B 预留）──
        if post_expand_fn is not None:
            result_intervals = post_expand_fn(
                result_intervals, seed, obstacles)

        return result_intervals

    # ──────────────────────────────────────────────
    #  通用 AABB 查询
    # ──────────────────────────────────────────────

    def query_aabb(
        self, query_intervals: List[Tuple[float, float]]
    ) -> Optional[List[LinkAABBInfo]]:
        """查询任意 box 的保守 AABB（利用缓存树）

        沿树下行，收集与 query_intervals 重叠的叶节点 AABB，
        取 union。如果叶节点尚未计算则惰性计算。

        比直接调用 interval FK 更紧（当树足够深时）。
        """
        return self._query_recursive(self.root, query_intervals)

    def _query_recursive(
        self,
        node: HierAABBNode,
        query: List[Tuple[float, float]],
    ) -> Optional[List[LinkAABBInfo]]:
        """递归查询"""
        # 检查 node 与 query 是否有交集
        for (nlo, nhi), (qlo, qhi) in zip(node.intervals, query):
            if nhi <= qlo or qhi <= nlo:
                return None  # 不相交

        # 如果是叶节点，确保有 AABB 并返回
        if node.is_leaf():
            self._ensure_aabb(node)
            return node.refined_aabb

        # 内部节点：递归子树
        left_a = self._query_recursive(node.left, query)
        right_a = self._query_recursive(node.right, query)

        if left_a is None:
            return right_a
        if right_a is None:
            return left_a
        return self._union_aabb(left_a, right_a)

    # ──────────────────────────────────────────────
    #  统计
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """返回树的统计信息"""
        n_leaves = 0
        max_depth = 0
        depths: List[int] = []

        def _walk(node: HierAABBNode):
            nonlocal n_leaves, max_depth
            if node.is_leaf():
                n_leaves += 1
                depths.append(node.depth)
                if node.depth > max_depth:
                    max_depth = node.depth
            else:
                if node.left:
                    _walk(node.left)
                if node.right:
                    _walk(node.right)

        _walk(self.root)
        return {
            'n_nodes': self.n_nodes,
            'n_leaves': n_leaves,
            'max_depth': max_depth,
            'avg_depth': float(np.mean(depths)) if depths else 0,
            'n_fk_calls': self.n_fk_calls,
        }

    # ──────────────────────────────────────────────
    #  持久化
    # ──────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """保存树到文件（pickle）

        去掉 parent 引用和 robot 对象以避免循环引用。
        """
        def _strip_parent(node: HierAABBNode):
            node.parent = None  # pickle 不序列化 parent
            if node.left:
                _strip_parent(node.left)
            if node.right:
                _strip_parent(node.right)

        _strip_parent(self.root)

        data = {
            'robot_fingerprint': self.robot_fingerprint,
            'joint_limits': self.joint_limits,
            'n_dims': self.n_dims,
            'root': self.root,
            'n_nodes': self.n_nodes,
            'n_fk_calls': self.n_fk_calls,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 恢复 parent 引用
        self._rebuild_parents(self.root, None)

        logger.info(
            "HierAABBTree 已保存到 %s (%d nodes, %d FK calls)",
            filepath, self.n_nodes, self.n_fk_calls,
        )

    @classmethod
    def load(cls, filepath: str, robot: Robot) -> 'HierAABBTree':
        """从文件加载

        Args:
            filepath: 缓存文件路径
            robot: 机器人模型（需与构建时一致）

        Raises:
            ValueError: 机器人指纹不匹配
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if data['robot_fingerprint'] != robot.fingerprint():
            raise ValueError(
                f"机器人指纹不匹配: "
                f"文件中为 {data['robot_fingerprint'][:16]}..., "
                f"当前为 {robot.fingerprint()[:16]}...",
            )

        tree = cls.__new__(cls)
        tree.robot = robot
        tree.robot_fingerprint = data['robot_fingerprint']
        tree._zero_length_links = robot.zero_length_links.copy()
        tree.joint_limits = data['joint_limits']
        tree.n_dims = data['n_dims']
        tree.root = data['root']
        tree.n_nodes = data['n_nodes']
        tree.n_fk_calls = data['n_fk_calls']

        # 重建 parent 引用
        tree._rebuild_parents(tree.root, None)

        stats = tree.get_stats()
        logger.info(
            "HierAABBTree 从 %s 加载: %d nodes, depth=%d, %d FK calls",
            filepath, stats['n_nodes'], stats['max_depth'],
            tree.n_fk_calls,
        )
        return tree

    @staticmethod
    def _rebuild_parents(node: HierAABBNode, parent: Optional[HierAABBNode]):
        """重建 parent 引用（load 后调用）"""
        node.parent = parent
        if node.left:
            HierAABBTree._rebuild_parents(node.left, node)
        if node.right:
            HierAABBTree._rebuild_parents(node.right, node)

    # ──────────────────────────────────────────────
    #  全局缓存
    # ──────────────────────────────────────────────

    _CACHE_DIR_NAME = ".cache"
    _CACHE_SUBDIR = "hier_aabb"

    @classmethod
    def _global_cache_dir(cls) -> Path:
        """返回全局缓存目录（项目根 / .cache / hier_aabb）"""
        # 沿 src/planner/hier_aabb_tree.py → 项目根
        project_root = Path(__file__).resolve().parent.parent.parent
        d = project_root / cls._CACHE_DIR_NAME / cls._CACHE_SUBDIR
        return d

    @classmethod
    def _cache_filename(cls, robot: Robot) -> str:
        fp = robot.fingerprint()[:16]
        return f"{robot.name}_{fp}.pkl"

    @classmethod
    def auto_load(
        cls,
        robot: Robot,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ) -> 'HierAABBTree':
        """自动从全局缓存加载，若不存在则新建空树

        缓存按 robot fingerprint 索引，跨场景/跨会话复用。
        """
        cache_dir = cls._global_cache_dir()
        cache_file = cache_dir / cls._cache_filename(robot)

        if cache_file.exists():
            try:
                tree = cls.load(str(cache_file), robot)
                # 如果 joint_limits 不同，丢弃缓存
                if joint_limits is not None:
                    jl = list(joint_limits)
                    if len(jl) == len(tree.joint_limits):
                        match = all(
                            abs(a[0] - b[0]) < 1e-10 and abs(a[1] - b[1]) < 1e-10
                            for a, b in zip(jl, tree.joint_limits)
                        )
                        if not match:
                            logger.info(
                                "joint_limits 不匹配，忽略缓存，新建空树")
                            return cls(robot, joint_limits)
                return tree
            except Exception as e:
                logger.warning("全局缓存加载失败 (%s): %s，新建空树",
                               cache_file, e)

        logger.info("未找到全局缓存，新建 HierAABBTree (%s)", robot.name)
        return cls(robot, joint_limits)

    def auto_save(self) -> str:
        """保存到全局缓存目录，返回保存路径"""
        cache_dir = self._global_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / self._cache_filename(self.robot)
        self.save(str(cache_file))
        return str(cache_file)
