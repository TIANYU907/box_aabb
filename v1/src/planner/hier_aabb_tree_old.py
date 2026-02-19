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
- **持久化**：numpy (.npz) 保存/加载，跨会话累积缓存

使用方式：
    tree = HierAABBTree(robot)
    box = tree.find_free_box(seed, obstacles, max_depth=40)
    tree.save("hier_cache.npz")

    # 后续加载
    tree = HierAABBTree.load("hier_cache.npz", robot)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np

from box_aabb.robot import Robot
from box_aabb.models import LinkAABBInfo
from box_aabb.interval_fk import compute_interval_aabb
from box_aabb.interval_fk_fast import (
    compute_fk_full,
    compute_fk_incremental,
)

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
        raw_aabb: 直接 interval FK 得到的保守 AABB（松）— (n_links, 6) float32
        refined_aabb: 子节点 union 精化后的 AABB（更紧）— (n_links, 6) float32
        split_dim: 切分维度 (depth % n_dims)
        split_val: 切分值（中点）
        left: 左子节点 (dim < split_val)
        right: 右子节点 (dim >= split_val)
        parent: 父节点引用
    """
    intervals: List[Tuple[float, float]]
    depth: int = 0
    raw_aabb: Optional[np.ndarray] = field(default=None, repr=False)
    refined_aabb: Optional[np.ndarray] = field(default=None, repr=False)
    split_dim: Optional[int] = None
    split_val: Optional[float] = None
    left: Optional['HierAABBNode'] = field(default=None, repr=False)
    right: Optional['HierAABBNode'] = field(default=None, repr=False)
    parent: Optional['HierAABBNode'] = field(default=None, repr=False)
    occupied: bool = False
    subtree_occupied: int = 0
    forest_box_id: Optional[int] = None  # 对应 BoxForest 中的 BoxNode.node_id
    _fk_cache: Optional[tuple] = field(default=None, repr=False)
    # (prefix_lo, prefix_hi, joints_lo, joints_hi) — 各 ndarray
    _arr_idx: Optional[int] = field(default=None, repr=False)
    # lazy load 时的数组索引

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


@dataclass
class FindFreeBoxResult:
    """find_free_box 的返回结果

    Attributes:
        intervals: 无碰撞 box 的关节区间
        absorbed_box_ids: 被提升（promotion）吸收的旧 BoxNode ID 集合。
            若非空，调用方应先 forest.remove_boxes(absorbed_box_ids)，
            再 forest.add_box_direct(new_box)。
    """
    intervals: List[Tuple[float, float]]
    absorbed_box_ids: Set[int] = field(default_factory=set)


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
        self._last_ffb_none_reason: Optional[str] = None  # 上次 ffb 返回 None 的原因
        self._init_lazy_state()
        self._init_link_metadata()

    # ──────────────────────────────────────────────
    #  Lazy load 状态
    # ──────────────────────────────────────────────

    def _init_lazy_state(self) -> None:
        """初始化 lazy load 相关属性（__init__ 和 load 调用）"""
        self._lazy_intervals: Optional[np.ndarray] = None   # (n, n_dims, 2)
        self._lazy_raw_aabbs: Optional[np.ndarray] = None   # (n_raw, n_links, 6)
        self._lazy_raw_pos: Optional[dict] = None           # {arr_idx: pos}
        self._lazy_refined_aabbs: Optional[np.ndarray] = None
        self._lazy_refined_pos: Optional[dict] = None

    def _ensure_node_data(self, node: HierAABBNode) -> None:
        """按需物化 lazy 节点的 intervals / raw_aabb / refined_aabb

        对已物化节点（intervals is not None）为 O(1) 空操作。
        """
        if node.intervals is not None:
            return
        idx = node._arr_idx
        if idx is None or self._lazy_intervals is None:
            return
        arr = self._lazy_intervals
        nd = self.n_dims
        node.intervals = [
            (float(arr[idx, d, 0]), float(arr[idx, d, 1])) for d in range(nd)
        ]
        if self._lazy_raw_pos is not None:
            k = self._lazy_raw_pos.get(idx)
            if k is not None:
                node.raw_aabb = self._lazy_raw_aabbs[k]
        if self._lazy_refined_pos is not None:
            k = self._lazy_refined_pos.get(idx)
            if k is not None:
                node.refined_aabb = self._lazy_refined_aabbs[k]
        # 叶节点修复
        if node.is_leaf() and node.raw_aabb is None and node.refined_aabb is not None:
            node.raw_aabb = node.refined_aabb

    # ──────────────────────────────────────────────
    #  内部：link 元数据
    # ──────────────────────────────────────────────

    def _init_link_metadata(self) -> None:
        """初始化 link 元数据（n_links 和 zero-length 掩码）

        由 __init__ 和 load 调用，确保在任何 AABB 操作之前就绪。
        """
        # n_links = n_joints + (1 if tool_frame else 0)
        n_joints = len(self.robot.dh_params)
        has_tool = self.robot.tool_frame is not None
        self._n_links = n_joints + (1 if has_tool else 0)
        self._zl_mask = np.array(
            [i + 1 in self._zero_length_links for i in range(self._n_links)],
            dtype=bool,
        )

    # ──────────────────────────────────────────────
    #  内部：AABB 计算（紧凑 numpy 格式）
    # ──────────────────────────────────────────────

    def _extract_compact(
        self, prefix_lo: np.ndarray, prefix_hi: np.ndarray,
    ) -> np.ndarray:
        """从 prefix transforms 提取 (n_links, 6) float32 紧凑 AABB

        link i (1-based) 的 AABB = union(translation[i-1], translation[i])。
        列布局: [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        n = self._n_links
        s_lo = prefix_lo[:n, :3, 3]       # (n, 3)
        s_hi = prefix_hi[:n, :3, 3]
        e_lo = prefix_lo[1:n + 1, :3, 3]  # (n, 3)
        e_hi = prefix_hi[1:n + 1, :3, 3]
        mins = np.minimum(s_lo, e_lo)
        maxs = np.maximum(s_hi, e_hi)
        return np.hstack([mins, maxs]).astype(np.float32)

    def _compute_aabb(
        self, node: HierAABBNode,
    ) -> np.ndarray:
        """使用高速 interval FK 计算保守 AABB，返回 (n_links, 6) float32"""
        self.n_fk_calls += 1
        prefix_lo, prefix_hi, joints_lo, joints_hi = compute_fk_full(
            self.robot, node.intervals)
        node._fk_cache = (prefix_lo, prefix_hi, joints_lo, joints_hi)
        return self._extract_compact(prefix_lo, prefix_hi)

    def _compute_aabb_incremental(
        self, node: HierAABBNode,
        parent: HierAABBNode,
        changed_joint: int,
    ) -> np.ndarray:
        """增量 FK：复用 parent 的前缀变换，返回 (n_links, 6) float32"""
        pcache = parent._fk_cache
        if pcache is None:
            return self._compute_aabb(node)
        self.n_fk_calls += 1
        p_plo, p_phi, p_jlo, p_jhi = pcache
        prefix_lo, prefix_hi, joints_lo, joints_hi = compute_fk_incremental(
            self.robot, node.intervals,
            p_plo, p_phi, p_jlo, p_jhi, changed_joint)
        node._fk_cache = (prefix_lo, prefix_hi, joints_lo, joints_hi)
        return self._extract_compact(prefix_lo, prefix_hi)

    def _ensure_aabb(self, node: HierAABBNode) -> None:
        """确保节点的 raw_aabb 和 refined_aabb 已计算"""
        self._ensure_node_data(node)
        if node.raw_aabb is None:
            node.raw_aabb = self._compute_aabb(node)
            if node.is_leaf():
                node.refined_aabb = node.raw_aabb
            # 如果已有子节点，refined 由子节点决定（不覆盖）

    @staticmethod
    def _union_aabb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """合并两组 AABB：逐 link 取 min/max，全向量化

        a, b: (n_links, 6) float32 — 列 0:3 = min, 列 3:6 = max
        """
        result = np.empty_like(a)
        result[:, :3] = np.minimum(a[:, :3], b[:, :3])
        result[:, 3:] = np.maximum(a[:, 3:], b[:, 3:])
        return result

    # ──────────────────────────────────────────────
    #  内部：切分
    # ──────────────────────────────────────────────

    def _split(self, node: HierAABBNode) -> None:
        """将叶节点二分裂，使用增量 FK 复用父节点的前缀变换。"""
        if not node.is_leaf():
            return
        self._ensure_node_data(node)

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

        # 增量 FK：复用 node 的前缀变换，只重算分裂维度
        if node._fk_cache is not None:
            p_plo, p_phi, p_jlo, p_jhi = node._fk_cache
            # left child
            self.n_fk_calls += 1
            l_plo, l_phi, l_jlo, l_jhi = compute_fk_incremental(
                self.robot, left_ivs,
                p_plo, p_phi, p_jlo, p_jhi, dim)
            node.left._fk_cache = (l_plo, l_phi, l_jlo, l_jhi)
            node.left.raw_aabb = self._extract_compact(l_plo, l_phi)
            node.left.refined_aabb = node.left.raw_aabb
            # right child
            self.n_fk_calls += 1
            r_plo, r_phi, r_jlo, r_jhi = compute_fk_incremental(
                self.robot, right_ivs,
                p_plo, p_phi, p_jlo, p_jhi, dim)
            node.right._fk_cache = (r_plo, r_phi, r_jlo, r_jhi)
            node.right.raw_aabb = self._extract_compact(r_plo, r_phi)
            node.right.refined_aabb = node.right.raw_aabb
        else:
            # 无父缓存，回退到全量
            self._ensure_aabb(node.left)
            self._ensure_aabb(node.right)

        # 精化本节点
        node.refined_aabb = self._union_aabb(
            node.left.refined_aabb, node.right.refined_aabb)

        # 注意: 不再在此调用 _propagate_up，由 find_free_box 在上行前批量传播

    def _propagate_up(self, node: Optional[HierAABBNode]) -> None:
        """从 node 向根方向更新 refined_aabb（含 early-stop）"""
        while node is not None:
            if node.left is None or node.right is None:
                break
            self._ensure_node_data(node.left)
            self._ensure_node_data(node.right)
            if (node.left.refined_aabb is None
                    or node.right.refined_aabb is None):
                break
            new_refined = self._union_aabb(
                node.left.refined_aabb, node.right.refined_aabb)
            # early-stop: 如果 refined_aabb 没有变化则停止传播
            if node.refined_aabb is not None and self._aabb_equal(
                    node.refined_aabb, new_refined):
                break
            node.refined_aabb = new_refined
            node = node.parent

    @staticmethod
    def _aabb_equal(a: np.ndarray, b: np.ndarray) -> bool:
        """比较两组 AABB 是否完全相同（向量化）"""
        return np.array_equal(a, b)

    # ──────────────────────────────────────────────
    #  占用跟踪（HierAABBTree 保证子节点无重叠）
    # ──────────────────────────────────────────────

    def _mark_occupied(
        self, node: HierAABBNode, forest_box_id: Optional[int] = None,
    ) -> None:
        """标记节点为已占用，向上传播计数

        标记后该节点对应的 C-space 区域被视为已归入 BoxForest，
        后续 find_free_box 不会返回重叠区域。

        Args:
            node: 要标记的节点
            forest_box_id: 对应的 BoxNode.node_id（用于 promotion）
        """
        node.occupied = True
        node.forest_box_id = forest_box_id
        node.subtree_occupied += 1
        p = node.parent
        while p is not None:
            p.subtree_occupied += 1
            p = p.parent

    def _reset_occupation(self, node: HierAABBNode) -> None:
        """重置整棵树的占用状态（加载全局缓存后调用）"""
        node.occupied = False
        node.subtree_occupied = 0
        node.forest_box_id = None
        if node.left:
            self._reset_occupation(node.left)
        if node.right:
            self._reset_occupation(node.right)

    def _collect_forest_ids(self, node: HierAABBNode) -> Set[int]:
        """递归收集子树中所有已占用节点的 forest_box_id"""
        ids: Set[int] = set()
        if node.occupied and node.forest_box_id is not None:
            ids.add(node.forest_box_id)
        if node.left:
            ids |= self._collect_forest_ids(node.left)
        if node.right:
            ids |= self._collect_forest_ids(node.right)
        return ids

    def _clear_subtree_occupation(self, node: HierAABBNode) -> int:
        """清除子树的占用状态，返回被清除的占用数

        清除 node 及其所有后代的 occupied/forest_box_id，
        并将 subtree_occupied 归零。
        """
        cleared = 0
        if node.occupied:
            cleared += 1
            node.occupied = False
            node.forest_box_id = None
        if node.left:
            cleared += self._clear_subtree_occupation(node.left)
        if node.right:
            cleared += self._clear_subtree_occupation(node.right)
        node.subtree_occupied = 0
        return cleared

    def is_occupied(self, config: np.ndarray) -> bool:
        """检查配置是否在已占用区域内

        沿树下行，O(tree_depth) 复杂度。
        比 BoxForest.find_containing 的 O(N) 更快。
        """
        return self.find_containing_box_id(config) is not None

    def find_containing_box_id(self, config: np.ndarray) -> Optional[int]:
        """找到包含 config 的已占用节点对应的 forest_box_id

        沿树下行，O(tree_depth) 复杂度。
        返回 forest_box_id 或 None（无占用节点包含该点）。
        """
        node = self.root
        while True:
            if node.occupied:
                return node.forest_box_id
            if node.is_leaf() or node.subtree_occupied == 0:
                return None
            if config[node.split_dim] < node.split_val:
                node = node.left
            else:
                node = node.right

    # ──────────────────────────────────────────────
    #  碰撞检测辅助
    # ──────────────────────────────────────────────

    @staticmethod
    def _prepack_obstacles(
        obstacles: list, safety_margin: float = 0.0,
    ) -> Optional[tuple]:
        """将障碍物列表预打包为 numpy 数组，供 _link_aabbs_collide 使用

        Returns:
            (obs_mins, obs_maxs) 各为 (M, D) ndarray，或无障碍物时返回 None
        """
        if not obstacles:
            return None
        obs_mins = np.array([obs.min_point for obs in obstacles]) - safety_margin
        obs_maxs = np.array([obs.max_point for obs in obstacles]) + safety_margin
        return (obs_mins, obs_maxs)

    def _link_aabbs_collide(
        self,
        aabb: np.ndarray,
        obs_packed: Optional[tuple],
    ) -> bool:
        """检测 link AABB 集合是否与任何障碍物重叠（全向量化 SAT）

        Args:
            aabb: (n_links, 6) float32 — 列 0:3 = min, 列 3:6 = max
            obs_packed: _prepack_obstacles 的返回值 (obs_mins, obs_maxs)
        """
        if obs_packed is None:
            return False
        obs_mins, obs_maxs = obs_packed  # (M, D)
        n_dims = obs_mins.shape[1]  # 2 or 3

        # 只检测非零长度的 link
        active = ~self._zl_mask  # (n_links,)
        la_mins = aabb[active, :n_dims]       # (K, D)
        la_maxs = aabb[active, 3:3 + n_dims]  # (K, D)

        # 广播: (K, 1, D) vs (1, M, D) → (K, M, D)
        separated = (
            np.any(la_maxs[:, np.newaxis, :] < obs_mins[np.newaxis, :, :] - 1e-10, axis=2)
            | np.any(obs_maxs[np.newaxis, :, :] < la_mins[:, np.newaxis, :] - 1e-10, axis=2)
        )  # (K, M)
        # 碰撞 = 任意 (link, obstacle) 对未分离
        return not np.all(separated)

    # ──────────────────────────────────────────────
    #  核心 API：找无碰撞 box
    # ──────────────────────────────────────────────

    def find_free_box(
        self,
        seed: np.ndarray,
        obstacles: list,
        max_depth: int = 40,
        safety_margin: float = 0.0,
        min_edge_length: float = 0.05,
        post_expand_fn=None,
        mark_occupied: bool = False,
        forest_box_id: Optional[int] = None,
    ) -> Optional[FindFreeBoxResult]:
        """从顶向下切分，找到包含 seed 的最大无碰撞 box

        算法：
        1. 下行：从 root 出发，如果当前节点 AABB 碰撞则切分，
           走向包含 seed 的子节点，直到找到无碰撞节点或达到 max_depth。
        2. 上行：先批量传播精化（_propagate_up），然后回溯路径，
           尝试用父节点的 refined_aabb 合并为更大的无碰撞 box。
           若父节点无碰撞但有已占用后代，执行 promotion（吸收旧 box）。

        Args:
            seed: 种子配置（必须在 joint_limits 内且无碰撞）
            obstacles: 场景障碍物列表（Scene.get_obstacles()）
            max_depth: 最大切分深度
            safety_margin: 碰撞检测安全裕度
            min_edge_length: 最小分割边长，当待分割维度宽度 < 此值时停止
            post_expand_fn: 可选的后处理扩张函数（预留接口 B）
                签名: (intervals, seed, obstacles) -> intervals
                若提供，会对切分结果做进一步扩张
            mark_occupied: 是否将结果标记为已占用。
                若 True，后续调用不会返回与此 box 重叠的区域，
                从而免去 BoxForest 的 deoverlap 步骤。
            forest_box_id: 此次产生的 BoxNode.node_id（当 mark_occupied=True）。
                用于 promotion 时追踪 HierAABBNode 与 BoxNode 的映射。

        Returns:
            FindFreeBoxResult（含 intervals 和 absorbed_box_ids），或 None
        """
        node = self.root
        self._ensure_aabb(node)
        path: List[HierAABBNode] = []

        # 预打包障碍物（一次性 numpy 转换）
        obs_packed = self._prepack_obstacles(obstacles, safety_margin)

        # ── 下行：沿 seed 方向切分直到无碰撞 ──
        while True:
            # 已占用区域 → seed 在已有 box 内
            if node.occupied:
                self._last_ffb_none_reason = "occupied"
                return None

            path.append(node)
            self._ensure_node_data(node)

            aabb = node.refined_aabb if node.refined_aabb is not None else node.raw_aabb
            # 无碰撞 且 无已占用后代 → 整个节点可用
            if (not self._link_aabbs_collide(aabb, obs_packed)
                    and node.subtree_occupied == 0):
                break

            if node.depth >= max_depth:
                self._last_ffb_none_reason = "max_depth"
                return None  # 达到最大深度仍碰撞或有占用

            # 检查最小边长：待分割维度宽度过小则停止
            split_dim = node.depth % self.n_dims
            edge = node.intervals[split_dim][1] - node.intervals[split_dim][0]
            if min_edge_length > 0 and edge < min_edge_length * 2:
                self._last_ffb_none_reason = "min_edge"
                return None  # 再分就低于最小边长

            # 惰性切分
            self._split(node)

            # 走向包含 seed 的子节点
            if seed[node.split_dim] < node.split_val:
                node = node.left
            else:
                node = node.right

        # ── 上行前：批量向上传播精化 ──
        if node.parent is not None:
            self._propagate_up(node.parent)

        # ── 上行：尝试合并为更大 box + promotion ──
        result_node = node
        absorbed_ids: Set[int] = set()
        for i in range(len(path) - 2, -1, -1):
            parent = path[i]
            aabb = parent.refined_aabb if parent.refined_aabb is not None else parent.raw_aabb

            if parent.subtree_occupied > 0:
                # 父节点有已占用后代——检查是否可以 promotion
                if self._link_aabbs_collide(aabb, obs_packed):
                    break  # 父节点仍碰撞，停止
                # 父节点无碰撞 → 吸收子树中所有已占用节点
                absorbed_ids |= self._collect_forest_ids(parent)
                self._clear_subtree_occupation(parent)
                result_node = parent
            else:
                if not self._link_aabbs_collide(aabb, obs_packed):
                    result_node = parent
                else:
                    break

        result_intervals = list(result_node.intervals)

        # 标记为已占用（保证后续调用不返回重叠区域）
        if mark_occupied:
            self._mark_occupied(result_node, forest_box_id)

        # ── 可选：后处理扩张（接口 B 预留）──
        if post_expand_fn is not None:
            result_intervals = post_expand_fn(
                result_intervals, seed, obstacles)

        return FindFreeBoxResult(
            intervals=result_intervals,
            absorbed_box_ids=absorbed_ids,
        )

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

        Returns:
            List[LinkAABBInfo]（外部 API 兼容格式），或 None
        """
        result = self._query_recursive(self.root, query_intervals)
        if result is None:
            return None
        return self._compact_to_link_aabbs(result)

    def _query_recursive(
        self,
        node: HierAABBNode,
        query: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """递归查询，返回 (n_links, 6) ndarray 或 None"""
        self._ensure_node_data(node)
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

    @staticmethod
    def _compact_to_link_aabbs(arr: np.ndarray) -> List[LinkAABBInfo]:
        """(n_links, 6) float32 → List[LinkAABBInfo]（外部 API 用）"""
        result: List[LinkAABBInfo] = []
        for i in range(arr.shape[0]):
            result.append(LinkAABBInfo(
                link_index=i + 1,
                link_name=f"Link {i + 1} (Joint {i})",
                min_point=[float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])],
                max_point=[float(arr[i, 3]), float(arr[i, 4]), float(arr[i, 5])],
            ))
        return result

    def save(self, filepath: str) -> None:
        """保存树到文件（numpy 扁平格式，lazy 感知）

        对 lazy 加载的节点直接从缓存数组批量拷贝 intervals / AABB，
        仅对新建节点做 Python 级序列化。对 180K 节点的树，
        保存时间从 ~0.8s 降到 ~0.3s。
        """
        # 1. DFS 展平所有节点
        nodes: List[HierAABBNode] = []
        stack = [self.root]
        while stack:
            nd = stack.pop()
            nodes.append(nd)
            if nd.right is not None:
                stack.append(nd.right)
            if nd.left is not None:
                stack.append(nd.left)

        n = len(nodes)
        node_to_idx = {id(nd): i for i, nd in enumerate(nodes)}

        # 2. 构建结构数组
        structure = np.empty((n, 4), dtype=np.int32)
        split_vals = np.empty(n, dtype=np.float64)
        intervals = np.empty((n, self.n_dims, 2), dtype=np.float64)

        # 分离 lazy 节点和已物化节点
        lazy_dst: List[int] = []    # intervals 数组中的目标行
        lazy_src: List[int] = []    # _lazy_intervals 中的源行
        has_lazy = self._lazy_intervals is not None

        # AABB 收集
        raw_list: List[np.ndarray] = []
        raw_idx_list: List[int] = []
        refined_list: List[np.ndarray] = []
        refined_idx_list: List[int] = []

        for i, nd in enumerate(nodes):
            structure[i, 0] = nd.depth
            structure[i, 1] = nd.split_dim if nd.split_dim is not None else -1
            structure[i, 2] = node_to_idx[id(nd.left)] if nd.left is not None else -1
            structure[i, 3] = node_to_idx[id(nd.right)] if nd.right is not None else -1
            split_vals[i] = nd.split_val if nd.split_val is not None else 0.0

            # intervals
            if nd.intervals is not None:
                # 已物化（新建或已访问） — 从 Python 对象拷贝
                for d in range(self.n_dims):
                    intervals[i, d, 0] = nd.intervals[d][0]
                    intervals[i, d, 1] = nd.intervals[d][1]
            elif has_lazy and nd._arr_idx is not None:
                # 未物化 lazy 节点 — 稍后批量拷贝
                lazy_dst.append(i)
                lazy_src.append(nd._arr_idx)

            # AABB: 先检查已物化，再查 lazy 缓存
            if nd.raw_aabb is not None:
                raw_list.append(nd.raw_aabb)
                raw_idx_list.append(i)
            elif has_lazy and nd._arr_idx is not None and self._lazy_raw_pos:
                k = self._lazy_raw_pos.get(nd._arr_idx)
                if k is not None:
                    raw_list.append(self._lazy_raw_aabbs[k])
                    raw_idx_list.append(i)

            if nd.refined_aabb is not None:
                refined_list.append(nd.refined_aabb)
                refined_idx_list.append(i)
            elif has_lazy and nd._arr_idx is not None and self._lazy_refined_pos:
                k = self._lazy_refined_pos.get(nd._arr_idx)
                if k is not None:
                    refined_list.append(self._lazy_refined_aabbs[k])
                    refined_idx_list.append(i)

        # 批量拷贝 lazy 节点的 intervals（numpy 花式索引）
        if lazy_dst:
            intervals[lazy_dst] = self._lazy_intervals[lazy_src]

        # 3. 堆叠 AABB
        raw_indices = np.array(raw_idx_list, dtype=np.int32) if raw_idx_list else np.empty(0, dtype=np.int32)
        raw_aabbs = np.stack(raw_list) if raw_list else np.empty((0, 0, 6), dtype=np.float32)
        refined_indices = np.array(refined_idx_list, dtype=np.int32) if refined_idx_list else np.empty(0, dtype=np.int32)
        refined_aabbs = np.stack(refined_list) if refined_list else np.empty((0, 0, 6), dtype=np.float32)

        # 4. 保存
        np.savez(
            filepath,
            _format_version_=np.array([3], dtype=np.int32),
            _robot_fingerprint_=np.array([self.robot_fingerprint]),
            _n_nodes_=np.array([self.n_nodes], dtype=np.int64),
            _n_fk_calls_=np.array([self.n_fk_calls], dtype=np.int64),
            _n_dims_=np.array([self.n_dims], dtype=np.int32),
            _joint_limits_=np.array(self.joint_limits, dtype=np.float64),
            structure=structure,
            split_vals=split_vals,
            intervals=intervals,
            raw_indices=raw_indices,
            raw_aabbs=raw_aabbs,
            refined_indices=refined_indices,
            refined_aabbs=refined_aabbs,
        )

        logger.info(
            "HierAABBTree 已保存到 %s (%d nodes, %d FK calls)",
            filepath, self.n_nodes, self.n_fk_calls,
        )

    @classmethod
    def load(cls, filepath: str, robot: Robot) -> 'HierAABBTree':
        """从 numpy 扁平文件加载（lazy：骨架节点 + 按需物化）

        仅创建轻量骨架节点（指针 + depth/split），不分配 intervals
        和 AABB。实际遍历时由 _ensure_node_data 按需物化。
        对 180K 节点的树，加载时间从 ~1.4s 降到 ~0.3s。

        Args:
            filepath: .npz 缓存文件路径
            robot: 机器人模型

        Raises:
            ValueError: 机器人指纹不匹配
        """
        data = np.load(filepath, allow_pickle=False)

        fp_saved = str(data['_robot_fingerprint_'][0])
        if fp_saved != robot.fingerprint():
            raise ValueError(
                f"机器人指纹不匹配: 文件中为 {fp_saved[:16]}..., "
                f"当前为 {robot.fingerprint()[:16]}...",
            )

        tree = cls.__new__(cls)
        tree.robot = robot
        tree.robot_fingerprint = fp_saved
        tree._zero_length_links = robot.zero_length_links.copy()
        tree.n_dims = int(data['_n_dims_'][0])
        jl_arr = data['_joint_limits_']
        tree.joint_limits = [(float(jl_arr[i, 0]), float(jl_arr[i, 1]))
                             for i in range(jl_arr.shape[0])]
        tree.n_nodes = int(data['_n_nodes_'][0])
        tree.n_fk_calls = int(data['_n_fk_calls_'][0])
        tree._last_ffb_none_reason = None
        tree._init_link_metadata()

        # 存储数组用于 lazy 访问
        tree._lazy_intervals = data['intervals']
        tree._lazy_raw_aabbs = data['raw_aabbs']
        raw_indices = data['raw_indices']
        tree._lazy_raw_pos = {int(raw_indices[k]): k
                              for k in range(len(raw_indices))}
        tree._lazy_refined_aabbs = data['refined_aabbs']
        refined_indices = data['refined_indices']
        tree._lazy_refined_pos = {int(refined_indices[k]): k
                                  for k in range(len(refined_indices))}

        # 读取结构数组
        structure = data['structure']
        split_vals = data['split_vals']
        n = structure.shape[0]

        # 创建骨架节点（跳过 dataclass __init__，不分配 intervals/aabb）
        nodes: list = [None] * n
        for i in range(n):
            nd = HierAABBNode.__new__(HierAABBNode)
            nd.depth = int(structure[i, 0])
            sdim = int(structure[i, 1])
            nd.split_dim = sdim if sdim >= 0 else None
            nd.split_val = float(split_vals[i]) if sdim >= 0 else None
            nd.intervals = None     # lazy
            nd.raw_aabb = None      # lazy
            nd.refined_aabb = None  # lazy
            nd.left = None
            nd.right = None
            nd.parent = None
            nd.occupied = False
            nd.subtree_occupied = 0
            nd.forest_box_id = None
            nd._fk_cache = None
            nd._arr_idx = i
            nodes[i] = nd

        # 链接树结构
        for i in range(n):
            left_idx = int(structure[i, 2])
            right_idx = int(structure[i, 3])
            if left_idx >= 0:
                nodes[i].left = nodes[left_idx]
                nodes[left_idx].parent = nodes[i]
            if right_idx >= 0:
                nodes[i].right = nodes[right_idx]
                nodes[right_idx].parent = nodes[i]

        tree.root = nodes[0]

        logger.info(
            "HierAABBTree 从 %s 加载: %d nodes, %d FK calls",
            filepath, tree.n_nodes, tree.n_fk_calls,
        )
        return tree

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
        return f"{robot.name}_{fp}.npz"

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
                if joint_limits is not None:
                    jl = list(joint_limits)
                    if len(jl) == len(tree.joint_limits):
                        match = all(
                            abs(a[0] - b[0]) < 1e-10 and abs(a[1] - b[1]) < 1e-10
                            for a, b in zip(jl, tree.joint_limits)
                        )
                        if not match:
                            logger.info("joint_limits 不匹配，忽略缓存，新建空树")
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

    # ──────────────────────────────────────────────
    #  缓存合并
    # ──────────────────────────────────────────────

    def merge_from(self, other: 'HierAABBTree') -> int:
        """将 other 树的缓存节点合并到当前树（结构性合并）

        切分策略确定性（dim = depth % n_dims, split = midpoint），
        同位置节点 intervals 完全一致，可安全对齐。
        合并保留更深的子树结构，跳过占用状态。

        Args:
            other: 待合并的另一棵树（同一机器人）

        Returns:
            新增的节点数
        """
        if self.robot_fingerprint != other.robot_fingerprint:
            logger.warning("merge_from: fingerprint 不匹配，跳过合并")
            return 0
        added = self._merge_recursive(self.root, other.root)
        self.n_nodes += added
        # FK calls 取较大值（两棵树有重叠计算）
        self.n_fk_calls = max(self.n_fk_calls, other.n_fk_calls)
        logger.info(
            "merge_from: 新增 %d 节点，合并后共 %d 节点",
            added, self.n_nodes,
        )
        return added

    def _merge_recursive(
        self, dst: HierAABBNode, src: HierAABBNode,
    ) -> int:
        """递归合并 src 子树到 dst，返回新增节点数"""
        added = 0
        self._ensure_node_data(dst)

        # 复制 AABB（若 dst 尚未计算）
        if dst.raw_aabb is None and src.raw_aabb is not None:
            dst.raw_aabb = src.raw_aabb
        if (dst.is_leaf() and dst.refined_aabb is None
                and src.refined_aabb is not None):
            dst.refined_aabb = src.refined_aabb

        # src 是叶节点 → 无更深结构可合并
        if src.is_leaf():
            return 0

        # src 有子节点
        if dst.is_leaf():
            # dst 为叶、src 更深 → 将 src 子树接入 dst
            dst.split_dim = src.split_dim
            dst.split_val = src.split_val
            dst.left = src.left
            dst.right = src.right
            n_left = self._adopt_subtree(dst.left, dst)
            n_right = self._adopt_subtree(dst.right, dst)
            added = n_left + n_right
        else:
            # 两者都有子节点 → 递归合并
            added += self._merge_recursive(dst.left, src.left)
            added += self._merge_recursive(dst.right, src.right)

        # 刷新 refined_aabb
        if (dst.left is not None and dst.right is not None
                and dst.left.refined_aabb is not None
                and dst.right.refined_aabb is not None):
            dst.refined_aabb = self._union_aabb(
                dst.left.refined_aabb, dst.right.refined_aabb)

        return added

    def _adopt_subtree(
        self, node: HierAABBNode, parent: HierAABBNode,
    ) -> int:
        """设置 parent 引用、清除占用状态，返回子树节点数（含 node）"""
        node.parent = parent
        node.occupied = False
        node.subtree_occupied = 0
        node.forest_box_id = None
        count = 1
        if node.left:
            count += self._adopt_subtree(node.left, node)
        if node.right:
            count += self._adopt_subtree(node.right, node)
        return count

    def auto_merge_save(self) -> str:
        """加载已有缓存并合并当前树的新节点，然后保存

        用于冷启动场景：不丢失之前累积的缓存节点。
        若无已有缓存则等同于 auto_save。

        Returns:
            保存路径
        """
        cache_dir = self._global_cache_dir()
        cache_file = cache_dir / self._cache_filename(self.robot)

        if cache_file.exists():
            try:
                cached = self.load(str(cache_file), self.robot)
                n_before = cached.n_nodes
                n_added = cached.merge_from(self)
                logger.info(
                    "auto_merge_save: 缓存 %d → 合并后 %d 节点 (+%d)",
                    n_before, cached.n_nodes, n_added,
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached.save(str(cache_file))
                return str(cache_file)
            except Exception as e:
                logger.warning(
                    "auto_merge_save 加载失败 (%s): %s，回退到覆盖保存",
                    cache_file, e,
                )

        return self.auto_save()
