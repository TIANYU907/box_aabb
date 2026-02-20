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


def build_kd_partitions(
    root_intervals: List[Tuple[float, float]],
    depth: int,
    dims: Optional[List[int]] = None,
) -> List[List[Tuple[float, float]]]:
    """按给定维度顺序对区间做 KD 二分，返回互不重叠子空间。"""
    intervals = [tuple(iv) for iv in root_intervals]
    if depth <= 0:
        return [intervals]

    n_dims = len(intervals)
    if n_dims == 0:
        return [intervals]

    if dims:
        dims_valid = [int(d) for d in dims if 0 <= int(d) < n_dims]
    else:
        dims_valid = list(range(n_dims))
    if not dims_valid:
        dims_valid = list(range(n_dims))

    parts: List[List[Tuple[float, float]]] = [intervals]
    for lv in range(depth):
        split_dim = dims_valid[lv % len(dims_valid)]
        next_parts: List[List[Tuple[float, float]]] = []
        for part in parts:
            lo, hi = part[split_dim]
            mid = (lo + hi) * 0.5
            left = list(part)
            right = list(part)
            left[split_dim] = (lo, mid)
            right[split_dim] = (mid, hi)
            next_parts.append(left)
            next_parts.append(right)
        parts = next_parts
    return parts


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
        return self._tree._split_dim_for_depth(self._tree._store.get_depth(self._idx))

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
        node_idx: 被标记占用 (或可标记) 的 tree 节点索引 (用于 rollback).
    """
    intervals: List[Tuple[float, float]]
    absorbed_box_ids: Set[int] = field(default_factory=set)
    node_idx: int = -1


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
        active_split_dims: Optional[List[int]] = None,
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
        self.active_split_dims = self._resolve_active_split_dims(active_split_dims)

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

    def _resolve_active_split_dims(
        self,
        dims: Optional[List[int]],
    ) -> List[int]:
        if dims:
            base = [int(d) for d in dims if 0 <= int(d) < self.n_dims]
        else:
            base = list(range(self.n_dims))

        if not base:
            base = list(range(self.n_dims))

        relevant_set = self._aabb_relevant_split_dim_set
        filtered: List[int] = []
        seen: Set[int] = set()
        for d in base:
            if d in relevant_set and d not in seen:
                filtered.append(d)
                seen.add(d)

        if filtered:
            return filtered

        # 兜底：若数值判定失败导致无可用维度，回退到全维切分
        return list(range(self.n_dims))

    def _split_dim_for_depth(self, depth: int) -> int:
        dims = self.active_split_dims
        if not dims:
            return depth % self.n_dims
        return dims[depth % len(dims)]

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
        self._aabb_relevant_split_dims = self._infer_aabb_relevant_split_dims(
            self.robot,
            self.n_dims,
            self._n_links,
        )
        self._aabb_relevant_split_dim_set = set(self._aabb_relevant_split_dims)

    @staticmethod
    def _infer_aabb_relevant_split_dims(
        robot: Robot,
        n_dims: int,
        n_links: int,
    ) -> List[int]:
        """推断会影响 AABB 计算的关节维度。

        通过 ``Robot.compute_relevant_joints`` 统计所有连杆位置受影响的关节。
        若推断失败或为空，则回退为全维度，保证行为稳定。
        """
        if n_dims <= 0:
            return []

        try:
            relevant: Set[int] = set()
            for link_idx in range(1, n_links + 1):
                for joint_idx in robot.compute_relevant_joints(link_idx):
                    if 0 <= int(joint_idx) < n_dims:
                        relevant.add(int(joint_idx))

            if relevant:
                return sorted(relevant)
        except Exception as e:
            logger.warning("推断 AABB 相关关节维度失败，回退全维切分: %s", e)

        return list(range(n_dims))

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
        for k in range(len(path) - 1):
            p = path[k]
            child = path[k + 1]
            dim = self._split_dim_for_depth(store.get_depth(p))
            sv = store.get_split_val(p)
            if child == store.get_left(p):
                ivs[dim] = (ivs[dim][0], sv)
            else:
                ivs[dim] = (sv, ivs[dim][1])
        return ivs

    def _get_intervals_from_base(
        self,
        idx: int,
        base_intervals: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        store = self._store
        path: list = []
        i = idx
        while i >= 0:
            path.append(i)
            i = store.get_parent(i)
        path.reverse()

        ivs = list(base_intervals)
        for k in range(len(path) - 1):
            p = path[k]
            child = path[k + 1]
            dim = self._split_dim_for_depth(store.get_depth(p))
            sv = store.get_split_val(p)
            if child == store.get_left(p):
                ivs[dim] = (ivs[dim][0], sv)
            else:
                ivs[dim] = (sv, ivs[dim][1])
        return ivs

    @staticmethod
    def _intersect_intervals(
        a: List[Tuple[float, float]],
        b: List[Tuple[float, float]],
    ) -> Optional[List[Tuple[float, float]]]:
        if len(a) != len(b):
            return None
        out: List[Tuple[float, float]] = []
        for (a_lo, a_hi), (b_lo, b_hi) in zip(a, b):
            lo = max(a_lo, b_lo)
            hi = min(a_hi, b_hi)
            if hi <= lo:
                return None
            out.append((lo, hi))
        return out

    @staticmethod
    def _is_config_in_intervals(
        q: np.ndarray,
        ivs: List[Tuple[float, float]],
        tol: float = 1e-12,
    ) -> bool:
        if len(q) < len(ivs):
            return False
        for i, (lo, hi) in enumerate(ivs):
            if q[i] < lo - tol or q[i] > hi + tol:
                return False
        return True

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

    @staticmethod
    def _refine_aabb(
        old: np.ndarray, union: np.ndarray,
    ) -> np.ndarray:
        """精化 AABB: intersect(old_direct_FK, union(children))

        old 和 union 都是有效的过逼近。它们的交集仍是有效的过逼近，
        且严格不松于任何一方。
          min 维度: 取 max(old, union) → 下界收紧
          max 维度: 取 min(old, union) → 上界收紧
        """
        result = np.empty_like(old)
        result[:, :3] = np.maximum(old[:, :3], union[:, :3])
        result[:, 3:] = np.minimum(old[:, 3:], union[:, 3:])
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
        dim = self._split_dim_for_depth(depth)

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

        # 增量 FK：复用父节点的前缀变换 (Cython accelerated)
        fk = self._fk_cache.get(idx)
        if fk is not None:
            p_plo, p_phi, p_jlo, p_jhi = fk

            self.n_fk_calls += 2
            l_plo, l_phi, l_jlo, l_jhi = compute_fk_incremental(
                self.robot, left_ivs, p_plo, p_phi, p_jlo, p_jhi, dim)
            self._fk_cache[left_idx] = (l_plo, l_phi, l_jlo, l_jhi)
            aabb_l = self._extract_compact(l_plo, l_phi)
            store.set_aabb(left_idx, aabb_l)

            r_plo, r_phi, r_jlo, r_jhi = compute_fk_incremental(
                self.robot, right_ivs, p_plo, p_phi, p_jlo, p_jhi, dim)
            self._fk_cache[right_idx] = (r_plo, r_phi, r_jlo, r_jhi)
            aabb_r = self._extract_compact(r_plo, r_phi)
            store.set_aabb(right_idx, aabb_r)
        else:
            # 无父缓存，回退到全量 FK
            self._ensure_aabb_at(left_idx, left_ivs)
            self._ensure_aabb_at(right_idx, right_ivs)

        # 精化本节点：aabb = intersect(old, union(left, right))
        union_ab = self._union_aabb(
            store.get_aabb(left_idx), store.get_aabb(right_idx))
        if store.get_has_aabb(idx):
            ref = self._refine_aabb(store.get_aabb(idx), union_ab)
        else:
            ref = union_ab
        store.set_aabb(idx, ref)

    def _propagate_up(self, node_or_idx) -> None:
        """从 idx 向根方向更新 AABB = union(children)（含 early-stop）"""
        idx = node_or_idx._idx if isinstance(node_or_idx, _NodeView) else node_or_idx
        self._store.propagate_up(idx)

    # ──────────────────────────────────────────────
    #  Promotion 递归碰撞检测
    # ──────────────────────────────────────────────

    def _promotion_collide_check(
        self,
        idx: int,
        obs_packed,
        promotion_depth: int,
    ) -> bool:
        """递归子树碰撞检测，用于 FFB 上行 promotion。

        promotion_depth=0: 用当前节点的 union AABB 检测 (等价旧行为)。
        promotion_depth=k: 递归到 2^k 个后代节点，分别用各自 AABB 检测。
            所有后代均无碰撞才返回 False (安全)。

        Returns True 表示碰撞 (不安全)。
        """
        store = self._store
        # 优先使用 Cython 加速版本
        if hasattr(store, 'subtree_collide_check'):
            return store.subtree_collide_check(idx, obs_packed, promotion_depth)
        # Python fallback
        return self._promotion_collide_check_py(idx, obs_packed, promotion_depth)

    def _promotion_collide_check_py(
        self,
        idx: int,
        obs_packed,
        remaining_depth: int,
    ) -> bool:
        """Python fallback: 递归子树碰撞检测。"""
        store = self._store
        if remaining_depth <= 0:
            return store.link_aabbs_collide(idx, obs_packed)

        left = store.get_left(idx)
        if left < 0:
            # 叶节点，无法再下潜
            return store.link_aabbs_collide(idx, obs_packed)

        right = store.get_right(idx)

        # 子节点必须有 AABB，否则回退到当前节点
        if not store.get_has_aabb(left) or not store.get_has_aabb(right):
            return store.link_aabbs_collide(idx, obs_packed)

        # 两个子节点都无碰撞才返回 False
        if self._promotion_collide_check_py(left, obs_packed, remaining_depth - 1):
            return True
        if self._promotion_collide_check_py(right, obs_packed, remaining_depth - 1):
            return True
        return False

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

    def unoccupy_boxes(self, box_ids: Set[int]) -> int:
        """清除指定 forest box 在 tree 中的占用状态.

        通过 forest_ids_array 构建 forest_box_id → [node_idx, ...] 映射,
        然后对每个 node 调用 unmark_occupied (mark_occupied 的逆操作).

        Args:
            box_ids: 需要清除占用的 forest box ID 集合

        Returns:
            实际清除的 tree 节点数
        """
        if not box_ids:
            return 0
        store = self._store
        fids = store.forest_ids_array()          # int32 ndarray
        mask = fids >= 0
        valid_idxs = np.flatnonzero(mask)
        valid_fids = fids[valid_idxs]

        box_id_set = set(box_ids)
        n_cleared = 0
        for i in range(len(valid_idxs)):
            if int(valid_fids[i]) in box_id_set:
                store.unmark_occupied(int(valid_idxs[i]))
                n_cleared += 1
        return n_cleared

    def is_occupied(self, config: np.ndarray) -> bool:
        return self.find_containing_box_id(config) is not None

    def find_containing_box_id(self, config: np.ndarray) -> Optional[int]:
        """找到包含 config 的已占用节点对应的 forest_box_id（O(depth)）"""
        idx = 0
        store = self._store
        while True:
            if store.is_occupied(idx):
                fid = store.get_forest_id(idx)
                return fid if fid >= 0 else None
            if store.get_left(idx) < 0 or store.get_subtree_occ(idx) == 0:
                return None
            dim = self._split_dim_for_depth(store.get_depth(idx))
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
    #  引导采样：优先选择未占用区域
    # ──────────────────────────────────────────────

    def sample_unoccupied_seed(
        self,
        rng: np.random.Generator,
        max_walk_depth: int = 12,
    ) -> Optional[np.ndarray]:
        """沿 KD 树按空闲体积权重下行采样，返回落在空闲区域的 seed 点。

        算法：
        1. 从根节点出发，维护当前区间 running_ivs。
        2. 每到一个内部节点（有子节点），按左/右子树的 **空闲体积** 随机选择下行方向。
           空闲体积 = 子节点总体积 - subtree_occ_vol，使采样概率与真实空闲体积成正比。
        3. 到达叶节点（未展开 / 未占用）时，在其区间内均匀采样。
        4. 若到达已占用叶节点则返回 None（概率极低）。

        复杂度：O(max_walk_depth)，与均匀采样相当。

        Args:
            rng: numpy 随机数生成器
            max_walk_depth: 最大下行深度（超过后在当前区间采样）

        Returns:
            采样点 (ndarray) 或 None（若所有区间均已占用）
        """
        store = self._store
        idx = 0

        # 根节点全部占满
        if store.is_occupied(idx):
            return None

        running_ivs = list(self.joint_limits)

        for _ in range(max_walk_depth):
            left_idx = store.get_left(idx)

            # 叶节点：直接在此采样
            if left_idx < 0:
                break

            right_idx = store.get_right(idx)

            # 子节点总体积（占 root 体积的比例）= 2^(-(depth+1))
            child_depth = store.get_depth(idx) + 1
            child_vol = 2.0 ** (-child_depth)

            # 空闲体积 = 总体积 - 已占用体积
            occ_vol_l = store.get_subtree_occ_vol(left_idx)
            occ_vol_r = store.get_subtree_occ_vol(right_idx)
            w_l = max(0.0, child_vol - occ_vol_l)
            w_r = max(0.0, child_vol - occ_vol_r)

            # 若一侧是已占用叶节点（整个子空间被覆盖），强制置零
            if store.is_occupied(left_idx) and store.get_left(left_idx) < 0:
                w_l = 0.0
            if store.is_occupied(right_idx) and store.get_left(right_idx) < 0:
                w_r = 0.0

            if w_l + w_r <= 0:
                break  # 两侧都满，在当前区间采样

            # 按空闲体积比例选择方向
            dim = self._split_dim_for_depth(store.get_depth(idx))
            sv = store.get_split_val(idx)

            go_left = rng.uniform() < (w_l / (w_l + w_r))

            if go_left:
                running_ivs[dim] = (running_ivs[dim][0], sv)
                idx = left_idx
            else:
                running_ivs[dim] = (sv, running_ivs[dim][1])
                idx = right_idx

            # 当前节点已被占用（整个子空间被一个 box 覆盖），退回上层
            if store.is_occupied(idx):
                return None

        # 在 running_ivs 区间内均匀采样
        seed = np.empty(self.n_dims, dtype=np.float64)
        for d in range(self.n_dims):
            lo, hi = running_ivs[d]
            seed[d] = rng.uniform(lo, hi)

        return seed

    # ──────────────────────────────────────────────
    #  核心 API：找无碰撞 box
    # ──────────────────────────────────────────────

    def can_expand(self, seed: np.ndarray, obs_packed=None,
                    obstacles: list = None, safety_margin: float = 0.0,
                    max_probe_depth: int = 4) -> bool:
        """轻量级浅层探测：检查 seed 方向是否还可能展开新 box。

        沿 seed 方向下行最多 max_probe_depth 层，检查：
        - 是否已占用 (occupied)
        - 是否子树全占用 (subtree_occ 大且无空余)
        - 是否 AABB 与障碍物完全无交集 (早停成功)
        返回 True 表示值得尝试 find_free_box。
        """
        store = self._store
        idx = 0
        if obs_packed is None and obstacles is not None:
            obs_packed = self._prepack_obstacles_c(obstacles, safety_margin)

        for _ in range(max_probe_depth):
            if store.is_occupied(idx):
                return False  # 已占用

            left = store.get_left(idx)
            if left < 0:
                # 叶节点：检查 AABB 是否无碰撞
                if store.get_has_aabb(idx) and obs_packed is not None:
                    if not store.link_aabbs_collide(idx, obs_packed):
                        return True  # 无碰撞叶节点，可展开
                return True  # 未展开叶节点，也值得尝试

            # 内部节点：检查子树占用
            if store.get_subtree_occ(idx) > 0:
                # 子树有占用但还有空余 => 可能可以
                pass

            # 检查当前 AABB vs 障碍物
            if (store.get_has_aabb(idx) and obs_packed is not None
                    and not store.link_aabbs_collide(idx, obs_packed)
                    and store.get_subtree_occ(idx) == 0):
                return True  # 无碰撞且无占用 => 当前节点就是好的

            # 继续下行
            dim = self._split_dim_for_depth(store.get_depth(idx))
            sv = store.get_split_val(idx)
            if seed[dim] < sv:
                idx = left
            else:
                idx = store.get_right(idx)

        return True  # 探测深度用尽，还没排除 => 尝试

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
        constrained_intervals: Optional[List[Tuple[float, float]]] = None,
        obs_packed=None,
        promotion_depth: int = 2,
    ) -> Optional[FindFreeBoxResult]:
        """从顶向下切分，找到包含 seed 的最大无碰撞 box

        算法：
        1. 下行：沿 seed 方向切分，running_ivs 原地更新（零拷贝）
        2. 上行：批量传播精化，回溯路径尝试 promotion

        Args:
            obs_packed: 预打包的障碍物 (可复用，避免重复构建)
            promotion_depth: 上行 promotion 时递归碰撞检测的下潜深度。
                0 = 用当前节点的 union AABB 做碰撞检测 (默认，兼容旧行为)。
                1 = 用 2 个子节点的 AABB 分别检测。
                2 = 用 4 个孙子节点的 AABB 分别检测。
                k = 用 2^k 个后代节点的 AABB 检测。
                更大的值使 promotion 碰撞判定更精确 (减少假阳性),
                但每次 promotion 检查的开销从 O(1) 增长到 O(2^k)。

        Returns:
            FindFreeBoxResult 或 None
        """
        store = self._store
        self._ensure_aabb_at(0)

        if obs_packed is None:
            obs_packed = self._prepack_obstacles_c(obstacles, safety_margin)

        # running_ivs: 原地更新，不做 list 拷贝
        if constrained_intervals is not None:
            clipped = self._intersect_intervals(self.joint_limits, list(constrained_intervals))
            if clipped is None:
                self._last_ffb_none_reason = "invalid_constraint"
                return None
            if not self._is_config_in_intervals(seed, clipped):
                self._last_ffb_none_reason = "seed_outside_constraint"
                return None
            base_ivs = clipped
        else:
            base_ivs = list(self.joint_limits)

        # ── 尝试 Cython 加速下行 ──
        _use_cy = getattr(self, '_use_cy_descent', None)
        if _use_cy is None:
            _use_cy = hasattr(store, 'descent_loop')
            self._use_cy_descent = _use_cy
            if _use_cy:
                _sd = self.active_split_dims or list(range(self.n_dims))
                self._split_dims_arr = np.array(_sd, dtype=np.int32)

        if _use_cy:
            # 确保根节点 FK 可用
            root_fk = self._fk_cache.get(0)
            if root_fk is None:
                self._compute_aabb_for(0, list(self.joint_limits))
                root_fk = self._fk_cache[0]

            _r = self.robot
            _sd = self._split_dims_arr
            idx, path, fail_code, n_new, n_fk = store.descent_loop(
                seed, obs_packed,
                _r._dh_alpha, _r._dh_a, _r._dh_d,
                _r._dh_theta, _r._dh_joint_type,
                _r.tool_frame is not None,
                _r._tool_alpha, _r._tool_a, _r._tool_d,
                _sd, len(_sd),
                base_ivs, max_depth, min_edge_length,
                root_fk[0], root_fk[1], root_fk[2], root_fk[3],
            )
            self.n_fk_calls += n_fk
            self.n_nodes += n_new

            if fail_code != 0:
                _reasons = {1: "occupied", 2: "max_depth", 3: "min_edge"}
                self._last_ffb_none_reason = _reasons.get(fail_code, "unknown")
                return None
        else:
            # ── Python 下行 (fallback) ──
            idx = 0
            path = []
            running_ivs = list(base_ivs)

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

                split_dim = self._split_dim_for_depth(depth)
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
        # promotion_depth > 0 时，碰撞检测递归到子树，减少 union AABB 假阳性
        _collide_fn = self._promotion_collide_check
        result_idx = idx
        absorbed_ids: Set[int] = set()
        for i in range(len(path) - 2, -1, -1):
            pidx = path[i]
            if not store.get_has_aabb(pidx):
                break

            if store.get_subtree_occ(pidx) > 0:
                if _collide_fn(pidx, obs_packed, promotion_depth):
                    break
                absorbed_ids |= self._collect_forest_ids(pidx)
                self._clear_subtree_occupation(pidx)
                result_idx = pidx
            else:
                if not _collide_fn(pidx, obs_packed, promotion_depth):
                    result_idx = pidx
                else:
                    break

        # 结果 intervals：从 root 推导（O(depth)，一次性）
        if constrained_intervals is not None:
            result_intervals = self._get_intervals_from_base(result_idx, base_ivs)
        else:
            result_intervals = self._get_intervals(result_idx)

        if mark_occupied:
            self._mark_occupied(result_idx, forest_box_id)

        if post_expand_fn is not None:
            result_intervals = post_expand_fn(
                result_intervals, seed, obstacles)

        return FindFreeBoxResult(
            intervals=result_intervals,
            absorbed_box_ids=absorbed_ids,
            node_idx=result_idx,
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
        dim = self._split_dim_for_depth(store.get_depth(idx))
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
        tree.active_split_dims = tree._resolve_active_split_dims(None)

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

    def warmup_fk_cache(self, max_depth: int = 6) -> int:
        """预热 FK 缓存: 对树的前 max_depth 层计算 FK 并缓存.

        cache 加载后 _fk_cache 为空, 导致首次 _split 回退全量 FK.
        此方法从根节点 BFS 遍历前几层, 为每个已有子节点的内部节点
        计算并缓存 FK, 使后续 _split 可直接使用增量 FK.

        Returns:
            预热的节点数
        """
        store = self._store
        warmed = 0
        # BFS: (node_idx, intervals)
        queue = [(0, list(self.joint_limits))]
        while queue:
            idx, ivs = queue.pop(0)
            depth = store.get_depth(idx)
            if depth >= max_depth:
                continue

            # 计算 FK 并缓存 (如果尚无缓存)
            if idx not in self._fk_cache:
                if not store.get_has_aabb(idx):
                    # 仅对有 AABB 的节点做 (cache 加载的都有)
                    continue
                self._compute_aabb_for(idx, ivs)
                warmed += 1

            # 如有子节点则继续遍历
            left = store.get_left(idx)
            if left < 0:
                continue
            dim = self._split_dim_for_depth(depth)
            sv = store.get_split_val(idx)
            left_ivs = list(ivs)
            left_ivs[dim] = (ivs[dim][0], sv)
            right_idx = store.get_right(idx)
            right_ivs = list(ivs)
            right_ivs[dim] = (sv, ivs[dim][1])
            queue.append((left, left_ivs))
            queue.append((right_idx, right_ivs))

        logger.info("warmup_fk_cache: %d nodes warmed (max_depth=%d)", warmed, max_depth)
        return warmed

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
        active_split_dims: Optional[List[int]] = None,
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
                            return cls(robot, joint_limits, active_split_dims=active_split_dims)
                tree.active_split_dims = tree._resolve_active_split_dims(active_split_dims)
                return tree
            except Exception as e:
                logger.warning("全局缓存加载失败 (%s): %s",
                               hcache_file, e)

        logger.info("未找到全局缓存，新建 HierAABBTree (%s)", robot.name)
        return cls(robot, joint_limits, active_split_dims=active_split_dims)

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

        # 刷新 AABB = intersect(old, union(children))
        left = ds.get_left(dst_idx)
        right = ds.get_right(dst_idx)
        if (left >= 0 and right >= 0
                and ds.get_has_aabb(left) and ds.get_has_aabb(right)):
            union_ab = self._union_aabb(
                ds.get_aabb(left), ds.get_aabb(right))
            if ds.get_has_aabb(dst_idx):
                ds.set_aabb(dst_idx, self._refine_aabb(
                    ds.get_aabb(dst_idx), union_ab))
            else:
                ds.set_aabb(dst_idx, union_ab)

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
