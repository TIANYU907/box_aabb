"""
planner/hier_aabb_tree.py - 层级自适应 AABB 缓存树 (v6: 单 AABB + mmap r+)

基于 KD-tree 式二叉空间切分的 AABB 包络缓存。
C-space 被递归二分（按维度轮转、取中点），每个节点惰性计算
interval FK AABB。随着查询次数增加，树自动加深、
父节点 AABB（子节点 union）单调变紧。

核心特性：
- **惰性求值**：仅在查询路径上创建节点和计算 AABB
- **渐进精化**：parent.aabb = union(children) ≤ raw FK AABB（单调变紧）
- **跨场景复用**：仅绑定机器人运动学，障碍物场景在查询时传入
- **持久化**：HCACHE02 二进制格式，mmap r+ 增量保存

v6 架构：
- 树拓扑（left/right/parent/depth）存储为 Python list[int]，
  单元素存取 ~20ns（numpy 的 10 倍速）。
- **单 AABB**：每节点仅维护一个 AABB 数组 (n_links, 6) float32，
  叶节点存储 FK 结果，内部节点存储 union(children) 精化值。
  不再区分 raw_aabb / refined_aabb。
- intervals 不存储——由 root + split_val 沿路径推导。
- find_free_box 下行中 running_ivs 原地修改，无 list 拷贝。
- HCACHE02 固定 stride 文件格式，支持 mmap r+ 增量写回。

使用方式：
    tree = HierAABBTree(robot)
    box = tree.find_free_box(seed, obstacles, max_depth=40)
    tree.save_binary("hier_cache.hcache")

    # 后续加载
    tree = HierAABBTree.load_binary("hier_cache.hcache", robot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np

from aabb.robot import Robot
from aabb.models import LinkAABBInfo
from aabb.interval_fk import (
    compute_fk_full,
    compute_fk_incremental,
)
try:
    from ._hier_core import NodeStore
except Exception:  # pragma: no cover - optional Cython extension
    class NodeStore:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "forest._hier_core is unavailable. Build Cython extension "
                "or provide a Python fallback before using HierAABBTree."
            )
from ._hier_layout import (
    HCACHE_MAGIC, HCACHE_VERSION, HEADER_SIZE,
    compute_stride,
    fingerprint_sha256, write_header, read_header, update_header_field,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
#  _NodeView — 索引兼容层
# ─────────────────────────────────────────────────────

class _NodeView:
    """将 SoA 数组中的一个索引包装为类似 HierAABBNode 的对象。

    供测试代码通过 ``tree.root.left.intervals`` 等属性访问。
    所有读/写直接委托给 HierAABBTree 的底层数组。
    """
    __slots__ = ('_tree', '_idx')

    def __init__(self, tree: 'HierAABBTree', idx: int):
        object.__setattr__(self, '_tree', tree)
        object.__setattr__(self, '_idx', idx)

    # ── 结构 ──

    def is_leaf(self) -> bool:
        return self._tree._store.get_left(self._idx) == -1

    @property
    def depth(self) -> int:
        return self._tree._store.get_depth(self._idx)

    @property
    def intervals(self) -> List[Tuple[float, float]]:
        return self._tree._get_intervals(self._idx)

    @property
    def split_dim(self) -> Optional[int]:
        if self._tree._store.get_left(self._idx) < 0:
            return None
        return self._tree._store.get_depth(self._idx) % self._tree.n_dims

    @property
    def split_val(self) -> Optional[float]:
        if self._tree._store.get_left(self._idx) < 0:
            return None
        return self._tree._store.get_split_val(self._idx)

    @property
    def left(self) -> Optional['_NodeView']:
        l = self._tree._store.get_left(self._idx)
        return _NodeView(self._tree, l) if l >= 0 else None

    @property
    def right(self) -> Optional['_NodeView']:
        r = self._tree._store.get_right(self._idx)
        return _NodeView(self._tree, r) if r >= 0 else None

    @property
    def parent(self) -> Optional['_NodeView']:
        p = self._tree._store.get_parent(self._idx)
        return _NodeView(self._tree, p) if p >= 0 else None

    # ── AABB ──

    @property
    def aabb(self) -> Optional[np.ndarray]:
        s = self._tree._store
        return s.get_aabb(self._idx) if s.get_has_aabb(self._idx) else None

    @aabb.setter
    def aabb(self, val):
        s = self._tree._store
        if val is not None:
            s.set_aabb(self._idx, val)
        else:
            s.set_has_aabb(self._idx, 0)

    @property
    def raw_aabb(self) -> Optional[np.ndarray]:
        """向后兼容别名 → aabb"""
        return self.aabb

    @raw_aabb.setter
    def raw_aabb(self, val):
        self.aabb = val

    @property
    def refined_aabb(self) -> Optional[np.ndarray]:
        """向后兼容别名 → aabb"""
        return self.aabb

    @refined_aabb.setter
    def refined_aabb(self, val):
        self.aabb = val

    # ── 占用 ──

    @property
    def occupied(self) -> bool:
        return bool(self._tree._store.is_occupied(self._idx))

    @occupied.setter
    def occupied(self, val: bool):
        s = self._tree._store
        if val:
            s.mark_occupied(self._idx, -1)
        else:
            # NodeStore 没有单独 reset 单节点的方法 — 直接设字节
            s._reset_single_occupation(self._idx) if hasattr(s, '_reset_single_occupation') else None

    @property
    def subtree_occupied(self) -> int:
        return self._tree._store.get_subtree_occ(self._idx)

    @subtree_occupied.setter
    def subtree_occupied(self, val: int):
        pass  # NodeStore 自动管理 subtree_occ

    @property
    def forest_box_id(self) -> Optional[int]:
        v = self._tree._store.get_forest_id(self._idx)
        return v if v >= 0 else None

    @forest_box_id.setter
    def forest_box_id(self, val):
        pass  # NodeStore 管理 forest_id

    # ── FK 缓存 ──

    @property
    def _fk_cache(self) -> Optional[tuple]:
        return self._tree._fk_cache.get(self._idx)

    @_fk_cache.setter
    def _fk_cache(self, val):
        if val is not None:
            self._tree._fk_cache[self._idx] = val
        elif self._idx in self._tree._fk_cache:
            del self._tree._fk_cache[self._idx]

    # ── 几何量 ──

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
#  遗留数据类 / 结果
# ─────────────────────────────────────────────────────

@dataclass
class HierAABBNode:
    """遗留数据类——仅保留供 import 兼容。内部不再使用。"""
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
    forest_box_id: Optional[int] = None
    _fk_cache: Optional[tuple] = field(default=None, repr=False)
    _arr_idx: Optional[int] = field(default=None, repr=False)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class FindFreeBoxResult:
    """find_free_box 的返回结果

    Attributes:
        intervals: 无碰撞 box 的关节区间
        absorbed_box_ids: 被提升（promotion）吸收的旧 BoxNode ID 集合。
    """
    intervals: List[Tuple[float, float]]
    absorbed_box_ids: Set[int] = field(default_factory=set)


# ─────────────────────────────────────────────────────
#  树
# ─────────────────────────────────────────────────────

class HierAABBTree:
    """层级自适应 AABB 缓存树

    v6 架构：Python-list 拓扑 + 单 AABB，无 Python 对象节点。
    """

    _INIT_CAP = 256

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
        self._init_link_metadata()

        cap = self._INIT_CAP
        nl = self._n_links
        stride = compute_stride(nl)

        # ── NodeStore: 拓扑 + AABB + 占用，统一管理 ──
        self._store = NodeStore(nl, self.n_dims, stride, cap,
                                self._zero_length_links)

        # ── FK 缓存：稀疏 dict ──
        self._fk_cache: dict = {}

        self.n_nodes = 1
        self.n_fk_calls = 0
        self._last_ffb_none_reason: Optional[str] = None
        self._source_filepath: Optional[str] = None
        self._source_n_alloc: int = 0  # 加载时的节点数，用于增量保存

    # ──────────────────────────────────────────────
    #  内部：link 元数据
    # ──────────────────────────────────────────────

    def _init_link_metadata(self) -> None:
        n_joints = len(self.robot.dh_params)
        has_tool = self.robot.tool_frame is not None
        self._n_links = n_joints + (1 if has_tool else 0)
        self._zl_mask = np.array(
            [i + 1 in self._zero_length_links for i in range(self._n_links)],
            dtype=bool,
        )
        self._zl_list = self._zl_mask.tolist()

    # ──────────────────────────────────────────────
    #  容量管理
    # ──────────────────────────────────────────────

    def _ensure_capacity(self, needed: int) -> None:
        self._store.ensure_capacity(needed)

    # ──────────────────────────────────────────────
    #  root 属性
    # ──────────────────────────────────────────────

    @property
    def root(self) -> _NodeView:
        return _NodeView(self, 0)

    # ──────────────────────────────────────────────
    #  intervals 推导
    # ──────────────────────────────────────────────

    def _get_intervals(self, idx: int) -> List[Tuple[float, float]]:
        """沿 parent 链从 root 推导节点 intervals（O(depth) ≤ 40）"""
        # 构建 root → idx 路径
        store = self._store
        path: list = []
        i = idx
        while i >= 0:
            path.append(i)
            i = store.get_parent(i)
        path.reverse()

        ivs = list(self.joint_limits)
        nd = self.n_dims
        for k in range(len(path) - 1):
            p = path[k]
            child = path[k + 1]
            dim = store.get_depth(p) % nd
            sv = store.get_split_val(p)
            if child == store.get_left(p):
                ivs[dim] = (ivs[dim][0], sv)
            else:
                ivs[dim] = (sv, ivs[dim][1])
        return ivs

    # ──────────────────────────────────────────────
    #  AABB 计算
    # ──────────────────────────────────────────────

    def _extract_compact(
        self, prefix_lo: np.ndarray, prefix_hi: np.ndarray,
    ) -> np.ndarray:
        """从 prefix transforms 提取 (n_links, 6) float32 紧凑 AABB"""
        n = self._n_links
        s_lo = prefix_lo[:n, :3, 3]
        s_hi = prefix_hi[:n, :3, 3]
        e_lo = prefix_lo[1:n + 1, :3, 3]
        e_hi = prefix_hi[1:n + 1, :3, 3]
        result = np.empty((n, 6), dtype=np.float32)
        result[:, :3] = np.minimum(s_lo, e_lo)
        result[:, 3:] = np.maximum(s_hi, e_hi)
        return result

    def _compute_aabb_for(
        self, idx: int, intervals: list,
    ) -> np.ndarray:
        """全量 interval FK 计算 AABB，返回 (n_links, 6) float32"""
        self.n_fk_calls += 1
        prefix_lo, prefix_hi, joints_lo, joints_hi = compute_fk_full(
            self.robot, intervals)
        self._fk_cache[idx] = (prefix_lo, prefix_hi, joints_lo, joints_hi)
        return self._extract_compact(prefix_lo, prefix_hi)

    def _ensure_aabb(self, node_or_idx, intervals=None) -> None:
        """确保节点有 AABB。接受 _NodeView 或 int。"""
        idx = node_or_idx._idx if isinstance(node_or_idx, _NodeView) else node_or_idx
        self._ensure_aabb_at(idx, intervals)

    def _ensure_aabb_at(self, idx: int, intervals=None) -> None:
        """确保节点有 AABB（内部方法，始终接受 int）"""
        if self._store.get_has_aabb(idx):
            return
        if intervals is None:
            intervals = self._get_intervals(idx)
        aabb = self._compute_aabb_for(idx, intervals)
        self._store.set_aabb(idx, aabb)

    @staticmethod
    def _union_aabb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """合并两组 AABB：逐 link 取 min/max"""
        result = np.empty_like(a)
        result[:, :3] = np.minimum(a[:, :3], b[:, :3])
        result[:, 3:] = np.maximum(a[:, 3:], b[:, 3:])
        return result

    # ──────────────────────────────────────────────
    #  切分
    # ──────────────────────────────────────────────

    def _split(self, node_or_idx, intervals=None) -> None:
        """将叶节点二分裂。接受 _NodeView 或 int。"""
        idx = node_or_idx._idx if isinstance(node_or_idx, _NodeView) else node_or_idx

        store = self._store
        if store.get_left(idx) != -1:
            return  # 已分裂

        depth = store.get_depth(idx)
        dim = depth % self.n_dims

        if intervals is None:
            intervals = self._get_intervals(idx)

        lo, hi = intervals[dim]
        mid = (lo + hi) / 2.0

        # 分配子节点
        store.ensure_capacity(store.next_idx + 2)
        left_idx = store.alloc_node(idx, depth + 1)
        right_idx = store.alloc_node(idx, depth + 1)
        self.n_nodes += 2

        # 设置父节点分裂信息
        store.set_split_val(idx, mid)
        store.set_left(idx, left_idx)
        store.set_right(idx, right_idx)

        # 构建子节点 intervals（仅用于 FK 计算，轻量 list 拷贝）
        left_ivs = list(intervals)
        left_ivs[dim] = (lo, mid)
        right_ivs = list(intervals)
        right_ivs[dim] = (mid, hi)

        # 增量 FK：复用父节点的前缀变换
        fk = self._fk_cache.get(idx)
        if fk is not None:
            p_plo, p_phi, p_jlo, p_jhi = fk

            self.n_fk_calls += 1
            l_plo, l_phi, l_jlo, l_jhi = compute_fk_incremental(
                self.robot, left_ivs, p_plo, p_phi, p_jlo, p_jhi, dim)
            self._fk_cache[left_idx] = (l_plo, l_phi, l_jlo, l_jhi)
            aabb_l = self._extract_compact(l_plo, l_phi)
            store.set_aabb(left_idx, aabb_l)

            self.n_fk_calls += 1
            r_plo, r_phi, r_jlo, r_jhi = compute_fk_incremental(
                self.robot, right_ivs, p_plo, p_phi, p_jlo, p_jhi, dim)
            self._fk_cache[right_idx] = (r_plo, r_phi, r_jlo, r_jhi)
            aabb_r = self._extract_compact(r_plo, r_phi)
            store.set_aabb(right_idx, aabb_r)
        else:
            # 无父缓存，回退到全量 FK
            self._ensure_aabb_at(left_idx, left_ivs)
            self._ensure_aabb_at(right_idx, right_ivs)

        # 精化本节点：aabb = union(left, right)
        ref = self._union_aabb(
            store.get_aabb(left_idx), store.get_aabb(right_idx))
        store.set_aabb(idx, ref)

    def _propagate_up(self, node_or_idx) -> None:
        """从 idx 向根方向更新 AABB = union(children)（含 early-stop）"""
        idx = node_or_idx._idx if isinstance(node_or_idx, _NodeView) else node_or_idx
        self._store.propagate_up(idx)

    # ──────────────────────────────────────────────
    #  占用跟踪
    # ──────────────────────────────────────────────

    def _mark_occupied(
        self, node_or_idx, forest_box_id: Optional[int] = None,
    ) -> None:
        """标记节点为已占用。接受 _NodeView 或 int。"""
        idx = node_or_idx._idx if isinstance(node_or_idx, _NodeView) else node_or_idx
        fid = forest_box_id if forest_box_id is not None else -1
        self._store.mark_occupied(idx, fid)

    def _reset_occupation(self, node_or_idx=None) -> None:
        """重置整棵树的占用状态"""
        self._store.reset_all_occupation()

    def _collect_forest_ids(self, node_or_idx) -> Set[int]:
        """递归收集子树中所有已占用节点的 forest_box_id"""
        idx = node_or_idx._idx if isinstance(node_or_idx, _NodeView) else node_or_idx
        return self._store.collect_forest_ids(idx)

    def _clear_subtree_occupation(self, node_or_idx) -> int:
        """清除子树的占用状态，返回被清除的占用数"""
        idx = node_or_idx._idx if isinstance(node_or_idx, _NodeView) else node_or_idx
        self._store.clear_subtree_occupation(idx)
        return 0  # NodeStore 版本不返回计数

    def is_occupied(self, config: np.ndarray) -> bool:
        return self.find_containing_box_id(config) is not None

    def find_containing_box_id(self, config: np.ndarray) -> Optional[int]:
        """找到包含 config 的已占用节点对应的 forest_box_id（O(depth)）"""
        idx = 0
        nd = self.n_dims
        store = self._store
        while True:
            if store.is_occupied(idx):
                fid = store.get_forest_id(idx)
                return fid if fid >= 0 else None
            if store.get_left(idx) < 0 or store.get_subtree_occ(idx) == 0:
                return None
            dim = store.get_depth(idx) % nd
            if config[dim] < store.get_split_val(idx):
                idx = store.get_left(idx)
            else:
                idx = store.get_right(idx)

    # ──────────────────────────────────────────────
    #  碰撞检测辅助
    # ──────────────────────────────────────────────

    def _prepack_obstacles_c(
        self, obstacles: list, safety_margin: float = 0.0,
    ) -> Optional[list]:
        """预打包障碍物为 Cython NodeStore 碰撞检测格式

        返回 list of (link_idx, lo0, hi0, lo1, hi1, lo2, hi2) 元组。
        跳过零长连杆，展开 links × obstacles 交叉积。
        """
        if not obstacles:
            return None
        obs_mins = np.array([obs.min_point for obs in obstacles]) - safety_margin
        obs_maxs = np.array([obs.max_point for obs in obstacles]) + safety_margin
        mn_l = obs_mins.tolist()
        mx_l = obs_maxs.tolist()
        packed: list = []
        zl = self._zl_list
        n_links = len(zl)
        n_obs = len(obstacles)
        for li in range(n_links):
            if zl[li]:
                continue
            for oi in range(n_obs):
                mn = mn_l[oi]
                mx = mx_l[oi]
                packed.append((li, mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]))
        return packed

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
        1. 下行：沿 seed 方向切分，running_ivs 原地更新（零拷贝）
        2. 上行：批量传播精化，回溯路径尝试 promotion

        Returns:
            FindFreeBoxResult 或 None
        """
        idx = 0
        store = self._store
        self._ensure_aabb_at(idx)
        path: list = []

        obs_packed = self._prepack_obstacles_c(obstacles, safety_margin)

        # running_ivs: 原地更新，不做 list 拷贝
        running_ivs = list(self.joint_limits)
        nd = self.n_dims

        # ── 下行 ──
        while True:
            if store.is_occupied(idx):
                self._last_ffb_none_reason = "occupied"
                return None

            path.append(idx)

            if (store.get_has_aabb(idx)
                    and not store.link_aabbs_collide(idx, obs_packed)
                    and store.get_subtree_occ(idx) == 0):
                break

            depth = store.get_depth(idx)
            if depth >= max_depth:
                self._last_ffb_none_reason = "max_depth"
                return None

            split_dim = depth % nd
            edge = running_ivs[split_dim][1] - running_ivs[split_dim][0]
            if min_edge_length > 0 and edge < min_edge_length * 2:
                self._last_ffb_none_reason = "min_edge"
                return None

            # 惰性切分（传入 running_ivs 避免重推导）
            self._split(idx, running_ivs)

            sv = store.get_split_val(idx)
            if seed[split_dim] < sv:
                running_ivs[split_dim] = (running_ivs[split_dim][0], sv)
                idx = store.get_left(idx)
            else:
                running_ivs[split_dim] = (sv, running_ivs[split_dim][1])
                idx = store.get_right(idx)

        # ── 上行前：批量传播精化 ──
        parent_idx = store.get_parent(idx)
        if parent_idx >= 0:
            self._propagate_up(parent_idx)

        # ── 上行：尝试合并 + promotion ──
        result_idx = idx
        absorbed_ids: Set[int] = set()
        for i in range(len(path) - 2, -1, -1):
            pidx = path[i]
            if not store.get_has_aabb(pidx):
                break

            if store.get_subtree_occ(pidx) > 0:
                if store.link_aabbs_collide(pidx, obs_packed):
                    break
                absorbed_ids |= self._collect_forest_ids(pidx)
                self._clear_subtree_occupation(pidx)
                result_idx = pidx
            else:
                if not store.link_aabbs_collide(pidx, obs_packed):
                    result_idx = pidx
                else:
                    break

        # 结果 intervals：从 root 推导（O(depth)，一次性）
        result_intervals = self._get_intervals(result_idx)

        if mark_occupied:
            self._mark_occupied(result_idx, forest_box_id)

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
        result = self._query_recursive(0, query_intervals)
        if result is None:
            return None
        return self._compact_to_link_aabbs(result)

    def _query_recursive(
        self,
        idx: int,
        query: List[Tuple[float, float]],
        node_ivs: Optional[list] = None,
    ) -> Optional[np.ndarray]:
        if node_ivs is None:
            node_ivs = self._get_intervals(idx)

        # 检查是否相交
        for (nlo, nhi), (qlo, qhi) in zip(node_ivs, query):
            if nhi <= qlo or qhi <= nlo:
                return None

        store = self._store
        if store.get_left(idx) == -1:  # leaf
            self._ensure_aabb_at(idx, node_ivs)
            return store.get_aabb(idx) if store.get_has_aabb(idx) else None

        # 内部节点：构建子节点 intervals 并递归
        dim = store.get_depth(idx) % self.n_dims
        sv = store.get_split_val(idx)

        left_ivs = list(node_ivs)
        left_ivs[dim] = (node_ivs[dim][0], sv)
        right_ivs = list(node_ivs)
        right_ivs[dim] = (sv, node_ivs[dim][1])

        left_a = self._query_recursive(store.get_left(idx), query, left_ivs)
        right_a = self._query_recursive(store.get_right(idx), query, right_ivs)

        if left_a is None:
            return right_a
        if right_a is None:
            return left_a
        return self._union_aabb(left_a, right_a)

    # ──────────────────────────────────────────────
    #  统计
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        n_leaves = 0
        max_depth = 0
        depths: list = []
        store = self._store
        stack = [0]
        while stack:
            i = stack.pop()
            left = store.get_left(i)
            if left == -1:
                n_leaves += 1
                d = store.get_depth(i)
                depths.append(d)
                if d > max_depth:
                    max_depth = d
            else:
                stack.append(left)
                stack.append(store.get_right(i))
        return {
            'n_nodes': self.n_nodes,
            'n_leaves': n_leaves,
            'max_depth': max_depth,
            'avg_depth': float(np.mean(depths)) if depths else 0,
            'n_fk_calls': self.n_fk_calls,
        }

    # ──────────────────────────────────────────────
    #  格式转换
    # ──────────────────────────────────────────────

    @staticmethod
    def _compact_to_link_aabbs(arr: np.ndarray) -> List[LinkAABBInfo]:
        result: List[LinkAABBInfo] = []
        for i in range(arr.shape[0]):
            result.append(LinkAABBInfo(
                link_index=i + 1,
                link_name=f"Link {i + 1} (Joint {i})",
                min_point=[float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])],
                max_point=[float(arr[i, 3]), float(arr[i, 4]), float(arr[i, 5])],
            ))
        return result

    # ──────────────────────────────────────────────
    #  持久化（HCACHE02 二进制格式：固定 stride, mmap r+）
    # ──────────────────────────────────────────────

    def save_binary(self, filepath: str) -> None:
        """保存树到 .hcache (HCACHE02 格式)

        格式: [4096B header][node0][node1]...
        每节点固定 stride 字节，包含拓扑 + 单 AABB。
        """
        store = self._store
        n = store.next_idx
        stride = store.stride

        # 清除所有 dirty 标记
        store.clear_all_dirty()

        buf = store.get_raw_buffer()
        used = n * stride

        with open(filepath, 'wb') as f:
            write_header(
                f,
                n_nodes=self.n_nodes,
                n_alloc=n,
                n_dims=self.n_dims,
                n_links=self._n_links,
                n_fk_calls=self.n_fk_calls,
                stride=stride,
                fp_str=self.robot_fingerprint,
                joint_limits=self.joint_limits,
            )
            f.write(bytes(buf[:used]))

        logger.info(
            "HierAABBTree 已保存到 %s (%d nodes, %d FK calls, HCACHE02)",
            filepath, self.n_nodes, self.n_fk_calls,
        )
        self._source_filepath = str(filepath)
        self._source_n_alloc = n

    def save_incremental(self, filepath: str) -> None:
        """增量保存：仅写回 dirty 节点 + 新分配的节点

        前提：filepath 与 _source_filepath 相同，且文件格式兼容。
        当文件有新节点扩展时，先追加新节点区域，再逐个写回 dirty 旧节点。
        比全量 save_binary 减少 10-100× I/O。
        """
        import os
        store = self._store
        n = store.next_idx
        stride = store.stride
        old_n = self._source_n_alloc
        buf = store.get_raw_buffer()

        with open(filepath, 'r+b') as f:
            # 更新 header
            write_header(
                f,
                n_nodes=self.n_nodes,
                n_alloc=n,
                n_dims=self.n_dims,
                n_links=self._n_links,
                n_fk_calls=self.n_fk_calls,
                stride=stride,
                fp_str=self.robot_fingerprint,
                joint_limits=self.joint_limits,
            )

            # 追加新节点（超出旧文件范围的部分）
            if n > old_n:
                f.seek(HEADER_SIZE + old_n * stride)
                f.write(bytes(buf[old_n * stride: n * stride]))

            # 逐个写回 dirty 旧节点（仅已有区域内修改过的）
            for idx in store.iter_dirty():
                if idx < old_n:
                    off = idx * stride
                    f.seek(HEADER_SIZE + off)
                    f.write(bytes(buf[off: off + stride]))

        store.clear_all_dirty()
        self._source_filepath = str(filepath)
        self._source_n_alloc = n

        logger.info(
            "HierAABBTree 增量保存到 %s (%d nodes, dirty+new wrote, HCACHE02)",
            filepath, self.n_nodes,
        )

    @classmethod
    def load_binary(cls, filepath: str, robot: Robot) -> 'HierAABBTree':
        """从 .hcache (HCACHE02) 加载

        AABB 通过 structured array 读取后拷贝到连续数组。
        _cap = n, 首次 _split 触发 _ensure_capacity。
        """
        with open(filepath, 'rb') as f:
            hdr = read_header(f)

        fp_hash_saved = hdr['fp_hash']
        fp_hash_robot = fingerprint_sha256(robot.fingerprint())
        if fp_hash_saved != fp_hash_robot:
            raise ValueError(
                f"机器人指纹不匹配: 文件 SHA256={fp_hash_saved[:8].hex()}..., "
                f"当前 SHA256={fp_hash_robot[:8].hex()}...",
            )

        n = hdr['n_alloc']
        nl = hdr['n_links']
        nd = hdr['n_dims']
        stride = hdr['stride']

        # 读取原始节点字节
        data = np.fromfile(filepath, dtype=np.uint8,
                           offset=HEADER_SIZE, count=n * stride)

        tree = cls.__new__(cls)
        tree.robot = robot
        tree.robot_fingerprint = robot.fingerprint()
        tree._zero_length_links = robot.zero_length_links.copy()
        tree.n_dims = nd
        tree.joint_limits = hdr['joint_limits']
        tree.n_nodes = hdr['n_nodes']
        tree.n_fk_calls = hdr['n_fk_calls']
        tree._last_ffb_none_reason = None
        tree._init_link_metadata()

        # 创建 NodeStore 并绑定加载的缓冲区
        store = NodeStore(nl, nd, stride, 1, tree._zero_length_links)
        store.attach_buffer(data, n)
        store.next_idx = n
        tree._store = store

        # FK 缓存 (空)
        tree._fk_cache = {}
        tree._source_filepath = str(filepath)
        tree._source_n_alloc = n

        logger.info(
            "HierAABBTree 从 %s 加载: %d nodes, %d FK calls (HCACHE02)",
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
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / cls._CACHE_DIR_NAME / cls._CACHE_SUBDIR

    @classmethod
    def _cache_filename(cls, robot: Robot) -> str:
        fp = robot.fingerprint()[:16]
        return f"{robot.name}_{fp}.hcache"

    @classmethod
    def auto_load(
        cls,
        robot: Robot,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ) -> 'HierAABBTree':
        cache_dir = cls._global_cache_dir()
        hcache_file = cache_dir / cls._cache_filename(robot)

        if hcache_file.exists():
            try:
                tree = cls.load_binary(str(hcache_file), robot)
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
                logger.warning("全局缓存加载失败 (%s): %s",
                               hcache_file, e)

        logger.info("未找到全局缓存，新建 HierAABBTree (%s)", robot.name)
        return cls(robot, joint_limits)

    def auto_save(self) -> str:
        cache_dir = self._global_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / self._cache_filename(self.robot)
        cache_path = str(cache_file)

        # 增量保存：仅当从同一文件加载时
        if (self._source_filepath is not None
                and self._source_filepath == cache_path
                and cache_file.exists()):
            try:
                self.save_incremental(cache_path)
                return cache_path
            except Exception as e:
                logger.warning("增量保存失败，回退全量保存: %s", e)

        self.save_binary(cache_path)
        return cache_path

    # ──────────────────────────────────────────────
    #  缓存合并
    # ──────────────────────────────────────────────

    def merge_from(self, other: 'HierAABBTree') -> int:
        """将 other 树的缓存合并到当前树（结构性合并）"""
        if self.robot_fingerprint != other.robot_fingerprint:
            logger.warning("merge_from: fingerprint 不匹配，跳过合并")
            return 0
        added = self._merge_recursive(0, other, 0)
        self.n_nodes += added
        self.n_fk_calls = max(self.n_fk_calls, other.n_fk_calls)
        logger.info(
            "merge_from: 新增 %d 节点，合并后共 %d 节点",
            added, self.n_nodes,
        )
        return added

    def _merge_recursive(
        self, dst_idx: int, other: 'HierAABBTree', src_idx: int,
    ) -> int:
        added = 0
        ds = self._store
        os = other._store

        # 复制 AABB（若 dst 尚未计算）
        if not ds.get_has_aabb(dst_idx) and os.get_has_aabb(src_idx):
            ds.set_aabb(dst_idx, os.get_aabb(src_idx))

        dst_is_leaf = ds.get_left(dst_idx) == -1
        src_is_leaf = os.get_left(src_idx) == -1

        if src_is_leaf:
            return 0

        if dst_is_leaf:
            # dst 为叶、src 更深 → 嫁接 src 子树
            ds.set_split_val(dst_idx, os.get_split_val(src_idx))
            new_left = self._graft_subtree(other, os.get_left(src_idx), dst_idx)
            new_right = self._graft_subtree(other, os.get_right(src_idx), dst_idx)
            ds.set_left(dst_idx, new_left)
            ds.set_right(dst_idx, new_right)
            added = self._count_subtree(new_left) + self._count_subtree(new_right)
        else:
            # 两者都有子节点 → 递归
            added += self._merge_recursive(
                ds.get_left(dst_idx), other, os.get_left(src_idx))
            added += self._merge_recursive(
                ds.get_right(dst_idx), other, os.get_right(src_idx))

        # 刷新 AABB = union(children)
        left = ds.get_left(dst_idx)
        right = ds.get_right(dst_idx)
        if (left >= 0 and right >= 0
                and ds.get_has_aabb(left) and ds.get_has_aabb(right)):
            ds.set_aabb(dst_idx, self._union_aabb(
                ds.get_aabb(left), ds.get_aabb(right)))

        return added

    def _graft_subtree(
        self, other: 'HierAABBTree', src_idx: int, parent_idx: int,
    ) -> int:
        """从 other 树拷贝子树到 self，返回 self 中的新根索引"""
        ds = self._store
        os = other._store

        ds.ensure_capacity(ds.next_idx + 1)
        depth = os.get_depth(src_idx)
        new_idx = ds.alloc_node(parent_idx, depth)
        ds.set_split_val(new_idx, os.get_split_val(src_idx))

        if os.get_has_aabb(src_idx):
            ds.set_aabb(new_idx, os.get_aabb(src_idx))

        src_left = os.get_left(src_idx)
        src_right = os.get_right(src_idx)
        if src_left >= 0:
            ds.set_left(new_idx, self._graft_subtree(other, src_left, new_idx))
        if src_right >= 0:
            ds.set_right(new_idx, self._graft_subtree(other, src_right, new_idx))

        return new_idx

    def _count_subtree(self, idx: int) -> int:
        count = 1
        store = self._store
        left = store.get_left(idx)
        right = store.get_right(idx)
        if left >= 0:
            count += self._count_subtree(left)
        if right >= 0:
            count += self._count_subtree(right)
        return count

    def auto_merge_save(self) -> str:
        """加载已有缓存并合并当前树的新节点，然后保存"""
        cache_dir = self._global_cache_dir()
        cache_file = cache_dir / self._cache_filename(self.robot)

        loaded_cache = None
        if cache_file.exists():
            try:
                loaded_cache = self.load_binary(str(cache_file), self.robot)
            except Exception as e:
                logger.warning("auto_merge_save 加载 %s 失败: %s",
                               cache_file, e)

        if loaded_cache is not None:
            n_before = loaded_cache.n_nodes
            n_added = loaded_cache.merge_from(self)
            logger.info(
                "auto_merge_save: 缓存 %d → 合并后 %d 节点 (+%d)",
                n_before, loaded_cache.n_nodes, n_added,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            loaded_cache.save_binary(str(cache_file))
            return str(cache_file)

        return self.auto_save()
