"""
planner/aabb_cache.py - AABB 包络缓存系统

缓存机器人在特定关节区间上的 AABB 包络计算结果，避免重复计算。

核心设计：
- Interval 库：存储 interval FK 的保守结果（体积越小越精确）
- Numerical 库：存储 numerical 方法的紧致结果（体积越大越精确）
- 自适应索引：逐维度 SortedList + NumPy 向量化查询，避免维度爆炸
- 子集合并：任意大区间的包络可由缓存子集在 3D 空间合并得到

数学基础：
    对关节区间 D = D1 ∪ D2（按某维度拆分），各连杆包络满足：
    AABB_ℓ(D) = BoundingBox(AABB_ℓ(D1) ∪ AABB_ℓ(D2))
    即逐连杆取 min_point 的 min、max_point 的 max（3D 空间取并集包围盒）。
"""

import time
import copy
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

import numpy as np

from box_aabb.models import LinkAABBInfo

logger = logging.getLogger(__name__)

# 默认缓存目录（项目根目录下）
DEFAULT_CACHE_DIR = Path(".cache") / "aabb"


def _round_intervals(intervals: Tuple[Tuple[float, float], ...],
                     decimals: int = 6) -> Tuple[Tuple[float, float], ...]:
    """将区间端点四舍五入到指定小数位，避免浮点精度问题"""
    return tuple(
        (round(lo, decimals), round(hi, decimals))
        for lo, hi in intervals
    )


def _intervals_to_key(intervals) -> Tuple[Tuple[float, float], ...]:
    """将区间列表转为可哈希的 key"""
    return _round_intervals(tuple(
        (float(lo), float(hi)) for lo, hi in intervals
    ))


@dataclass
class CacheEntry:
    """缓存条目

    Attributes:
        entry_id: 条目唯一标识
        intervals: 关节区间（round 到 6 位小数）
        link_aabbs: 各连杆 AABB 计算结果
        n_sub: 连杆等分段数
        total_volume: 所有连杆 AABB 体积之和
        method: 计算方法标识 ('interval' 或 'numerical')
        timestamp: 创建时间
    """
    entry_id: int
    intervals: Tuple[Tuple[float, float], ...]
    link_aabbs: List[LinkAABBInfo]
    n_sub: int = 1
    total_volume: float = 0.0
    method: str = 'interval'
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.total_volume == 0.0:
            self.total_volume = sum(
                la.volume for la in self.link_aabbs if not la.is_zero_length
            )


class IntervalStore:
    """自适应区间索引存储

    使用逐维度排序索引 + NumPy 向量化查询的双层结构，
    避免高维空间的维度爆炸问题。

    存储的条目区间可以是任意大小（自适应分辨率）。
    支持子集查询、包含查询、合并查询。

    Args:
        n_dims: 关节空间维度
        store_type: 'interval' 或 'numerical'，决定精度替换逻辑
        max_entries: 最大条目数（LRU 淘汰）
    """

    def __init__(
        self,
        n_dims: int,
        store_type: str = 'interval',
        max_entries: int = 100000,
    ) -> None:
        self.n_dims = n_dims
        self.store_type = store_type
        self.max_entries = max_entries

        # 条目存储
        self._entries: Dict[int, CacheEntry] = {}
        self._exact_index: Dict[Tuple, int] = {}  # intervals_key → entry_id
        self._next_id: int = 0

        # NumPy 向量化层（延迟构建）
        self._np_intervals: Optional[np.ndarray] = None  # (M, N, 2)
        self._np_volumes: Optional[np.ndarray] = None     # (M,)
        self._np_ids: Optional[np.ndarray] = None          # (M,)
        self._np_dirty: bool = True
        self._ops_since_rebuild: int = 0

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    def store(
        self,
        intervals: List[Tuple[float, float]],
        link_aabbs: List[LinkAABBInfo],
        n_sub: int = 1,
        method: str = 'interval',
    ) -> int:
        """存入缓存条目

        精度替换规则：
        - interval 库：同一区间，体积越小越精确 → 小体积替换大体积
        - numerical 库：同一区间，体积越大越精确 → 大体积替换小体积

        Args:
            intervals: 关节区间
            link_aabbs: 连杆 AABB 列表
            n_sub: 连杆等分段数
            method: 计算方法

        Returns:
            条目 ID
        """
        key = _intervals_to_key(intervals)
        new_volume = sum(la.volume for la in link_aabbs if not la.is_zero_length)

        # 精确匹配检查：是否已有相同区间的条目
        if key in self._exact_index:
            existing_id = self._exact_index[key]
            existing = self._entries[existing_id]
            # 精度替换判断
            should_replace = False
            if self.store_type == 'interval':
                # interval 库：体积越小越精确
                should_replace = new_volume < existing.total_volume
            else:
                # numerical 库：体积越大越精确
                should_replace = new_volume > existing.total_volume

            if should_replace:
                entry = CacheEntry(
                    entry_id=existing_id,
                    intervals=key,
                    link_aabbs=link_aabbs,
                    n_sub=n_sub,
                    total_volume=new_volume,
                    method=method,
                )
                self._entries[existing_id] = entry
                self._np_dirty = True
                return existing_id
            return existing_id

        # 新条目
        entry_id = self._next_id
        self._next_id += 1

        entry = CacheEntry(
            entry_id=entry_id,
            intervals=key,
            link_aabbs=link_aabbs,
            n_sub=n_sub,
            total_volume=new_volume,
            method=method,
        )
        self._entries[entry_id] = entry
        self._exact_index[key] = entry_id
        self._np_dirty = True
        self._ops_since_rebuild += 1

        # LRU 淘汰
        if len(self._entries) > self.max_entries:
            self._evict_oldest()

        return entry_id

    def query_exact(
        self, intervals: List[Tuple[float, float]]
    ) -> Optional[CacheEntry]:
        """精确查询：区间完全匹配"""
        key = _intervals_to_key(intervals)
        eid = self._exact_index.get(key)
        if eid is not None and eid in self._entries:
            return self._entries[eid]
        return None

    def query_subsets(
        self, intervals: List[Tuple[float, float]]
    ) -> List[CacheEntry]:
        """子集查询：找缓存中被目标区间包含的所有条目

        即 ∀i: cache_lo_i ≥ query_lo_i 且 cache_hi_i ≤ query_hi_i

        Args:
            intervals: 目标区间

        Returns:
            所有子集条目列表
        """
        self._ensure_numpy()
        if self._np_intervals is None or len(self._np_ids) == 0:
            return []

        query = np.array([(lo, hi) for lo, hi in intervals])  # (N, 2)

        # 向量化子集判断
        lo_ok = self._np_intervals[:, :, 0] >= query[np.newaxis, :, 0] - 1e-9
        hi_ok = self._np_intervals[:, :, 1] <= query[np.newaxis, :, 1] + 1e-9
        mask = np.all(lo_ok & hi_ok, axis=1)

        result_ids = self._np_ids[mask]
        return [self._entries[int(eid)] for eid in result_ids
                if int(eid) in self._entries]

    def query_containing(
        self, sub_intervals: List[Tuple[float, float]]
    ) -> List[CacheEntry]:
        """包含查询：找缓存中包含给定子区间的条目

        即 ∀i: cache_lo_i ≤ query_lo_i 且 cache_hi_i ≥ query_hi_i

        用于 seed 拓展时查找 seed 所在的已缓存区域。

        Args:
            sub_intervals: 子区间（可以是点区间 [q, q]）

        Returns:
            包含该子区间的所有条目列表（按体积降序）
        """
        self._ensure_numpy()
        if self._np_intervals is None or len(self._np_ids) == 0:
            return []

        query = np.array([(lo, hi) for lo, hi in sub_intervals])  # (N, 2)

        lo_ok = self._np_intervals[:, :, 0] <= query[np.newaxis, :, 0] + 1e-9
        hi_ok = self._np_intervals[:, :, 1] >= query[np.newaxis, :, 1] - 1e-9
        mask = np.all(lo_ok & hi_ok, axis=1)

        result_ids = self._np_ids[mask]
        entries = [self._entries[int(eid)] for eid in result_ids
                   if int(eid) in self._entries]
        # 按体积降序排序
        entries.sort(key=lambda e: e.total_volume, reverse=True)
        return entries

    def query_interval_merge(
        self, intervals: List[Tuple[float, float]]
    ) -> Tuple[Optional[List[LinkAABBInfo]], List[CacheEntry], List[Tuple]]:
        """合并查询：找目标区间内所有子集并在 3D 空间合并包络

        对每个连杆，将所有子集条目的 AABB 取 3D 并集包围盒：
        min_point = min(所有子集 min_point)
        max_point = max(所有子集 max_point)

        Args:
            intervals: 目标关节区间

        Returns:
            (merged_aabbs, covered_entries, uncovered_gaps)
            - merged_aabbs: 合并后的连杆 AABB 列表（若无子集则为 None）
            - covered_entries: 命中的缓存条目列表
            - uncovered_gaps: 未覆盖的间隙信息（简化为维度索引列表）
        """
        subsets = self.query_subsets(intervals)
        if not subsets:
            return None, [], [_intervals_to_key(intervals)]

        # 合并所有子集的 link AABBs
        merged = _merge_link_aabbs([e.link_aabbs for e in subsets])

        # 简化间隙检测：检查子集是否完全覆盖目标区间
        # 通过比较子集的区间并集与目标区间来判断
        uncovered = self._find_uncovered_gaps(intervals, subsets)

        return merged, subsets, uncovered

    def remove(self, entry_id: int) -> None:
        """删除条目"""
        if entry_id in self._entries:
            entry = self._entries.pop(entry_id)
            self._exact_index.pop(entry.intervals, None)
            self._np_dirty = True

    def clear(self) -> None:
        """清空所有条目"""
        self._entries.clear()
        self._exact_index.clear()
        self._np_intervals = None
        self._np_volumes = None
        self._np_ids = None
        self._np_dirty = True

    def _ensure_numpy(self) -> None:
        """确保 NumPy 向量化层是最新的"""
        if not self._np_dirty and self._np_intervals is not None:
            return

        if not self._entries:
            self._np_intervals = np.empty((0, self.n_dims, 2))
            self._np_volumes = np.empty(0)
            self._np_ids = np.empty(0, dtype=np.int64)
            self._np_dirty = False
            return

        entries_list = list(self._entries.values())
        M = len(entries_list)

        intervals_arr = np.empty((M, self.n_dims, 2), dtype=np.float64)
        volumes_arr = np.empty(M, dtype=np.float64)
        ids_arr = np.empty(M, dtype=np.int64)

        for idx, entry in enumerate(entries_list):
            for d in range(self.n_dims):
                intervals_arr[idx, d, 0] = entry.intervals[d][0]
                intervals_arr[idx, d, 1] = entry.intervals[d][1]
            volumes_arr[idx] = entry.total_volume
            ids_arr[idx] = entry.entry_id

        self._np_intervals = intervals_arr
        self._np_volumes = volumes_arr
        self._np_ids = ids_arr
        self._np_dirty = False
        self._ops_since_rebuild = 0

    def _evict_oldest(self) -> None:
        """LRU 淘汰：删除最旧的条目"""
        if not self._entries:
            return
        # 找 timestamp 最小的
        oldest_id = min(self._entries, key=lambda k: self._entries[k].timestamp)
        self.remove(oldest_id)

    def _find_uncovered_gaps(
        self,
        intervals: List[Tuple[float, float]],
        subsets: List[CacheEntry],
    ) -> List[Tuple]:
        """简化间隙检测

        检查子集是否在每个维度上完整覆盖目标区间。
        仅做单维度扫描线分析（避免多维组合爆炸）。

        Returns:
            未覆盖间隙的简化描述列表（空列表表示全覆盖）
        """
        if not subsets:
            return [_intervals_to_key(intervals)]

        # 对每个维度做覆盖分析
        for d in range(self.n_dims):
            q_lo, q_hi = intervals[d]
            if q_hi - q_lo < 1e-10:
                continue

            # 收集该维度上所有子集的区间段
            segs = sorted(
                (e.intervals[d][0], e.intervals[d][1]) for e in subsets
            )
            # 扫描线合并
            covered_hi = q_lo
            for seg_lo, seg_hi in segs:
                if seg_lo > covered_hi + 1e-9:
                    # 存在间隙
                    return [('gap', d, covered_hi, seg_lo)]
                covered_hi = max(covered_hi, seg_hi)
            if covered_hi < q_hi - 1e-9:
                return [('gap', d, covered_hi, q_hi)]

        return []  # 全覆盖


def _merge_link_aabbs(
    aabb_lists: List[List[LinkAABBInfo]],
) -> List[LinkAABBInfo]:
    """合并多组连杆 AABB：逐连杆在 3D 空间取并集包围盒

    对每个连杆 ℓ，合并后的 AABB 为：
    min_point = min(所有输入的 min_point)
    max_point = max(所有输入的 max_point)

    Args:
        aabb_lists: 多组 link AABB 列表

    Returns:
        合并后的 link AABB 列表
    """
    if not aabb_lists:
        return []

    if len(aabb_lists) == 1:
        return copy.deepcopy(aabb_lists[0])

    # 按 link_index 聚合
    link_data: Dict[int, Dict] = {}

    for aabb_list in aabb_lists:
        for la in aabb_list:
            lid = la.link_index
            if lid not in link_data:
                link_data[lid] = {
                    'link_name': la.link_name,
                    'min_point': list(la.min_point),
                    'max_point': list(la.max_point),
                    'is_zero_length': la.is_zero_length,
                }
            else:
                d = link_data[lid]
                for axis in range(3):
                    d['min_point'][axis] = min(
                        d['min_point'][axis], la.min_point[axis])
                    d['max_point'][axis] = max(
                        d['max_point'][axis], la.max_point[axis])
                # 如果任一组非零长度，则合并结果非零长度
                if not la.is_zero_length:
                    d['is_zero_length'] = False

    # 构建结果
    result = []
    for lid in sorted(link_data.keys()):
        d = link_data[lid]
        result.append(LinkAABBInfo(
            link_index=lid,
            link_name=d['link_name'],
            min_point=d['min_point'],
            max_point=d['max_point'],
            is_zero_length=d['is_zero_length'],
        ))
    return result


class AABBCache:
    """AABB 包络缓存管理器

    为不同机器人分别维护 interval 和 numerical 两个独立存储库。
    interval 库存放保守安全的包络结果，numerical 库存放紧致非安全的结果。

    使用方式：
        cache = AABBCache()
        cache.store(robot, intervals, link_aabbs, method='interval')
        result = cache.query_exact(robot, intervals, method='interval')

    Args:
        max_entries_per_store: 每个存储库的最大条目数
    """

    def __init__(self, max_entries_per_store: int = 100000) -> None:
        self.max_entries = max_entries_per_store
        # robot_id → {'interval': IntervalStore, 'numerical': IntervalStore}
        self._stores: Dict[str, Dict[str, IntervalStore]] = {}

    def _get_store(self, robot, method: str = 'interval') -> IntervalStore:
        """获取指定机器人和方法的存储库"""
        rid = robot.fingerprint()
        store_type = 'interval' if method == 'interval' else 'numerical'

        if rid not in self._stores:
            n_dims = robot.n_joints
            self._stores[rid] = {
                'interval': IntervalStore(
                    n_dims, store_type='interval',
                    max_entries=self.max_entries),
                'numerical': IntervalStore(
                    n_dims, store_type='numerical',
                    max_entries=self.max_entries),
            }
        return self._stores[rid][store_type]

    def store(
        self,
        robot,
        intervals: List[Tuple[float, float]],
        link_aabbs: List[LinkAABBInfo],
        n_sub: int = 1,
        method: str = 'interval',
    ) -> int:
        """存入缓存

        Args:
            robot: 机器人模型
            intervals: 关节区间
            link_aabbs: 连杆 AABB 列表
            n_sub: 连杆等分段数
            method: 计算方法 ('interval' 或 'numerical')

        Returns:
            条目 ID
        """
        store = self._get_store(robot, method)
        return store.store(intervals, link_aabbs, n_sub, method)

    def query_exact(
        self,
        robot,
        intervals: List[Tuple[float, float]],
        method: str = 'interval',
    ) -> Optional[CacheEntry]:
        """精确查询"""
        store = self._get_store(robot, method)
        return store.query_exact(intervals)

    def query_subsets(
        self,
        robot,
        intervals: List[Tuple[float, float]],
        method: str = 'interval',
    ) -> List[CacheEntry]:
        """子集查询"""
        store = self._get_store(robot, method)
        return store.query_subsets(intervals)

    def query_containing(
        self,
        robot,
        sub_intervals: List[Tuple[float, float]],
        method: str = 'interval',
    ) -> List[CacheEntry]:
        """包含查询（找包含给定子区间的缓存条目）"""
        store = self._get_store(robot, method)
        return store.query_containing(sub_intervals)

    def query_interval_merge(
        self,
        robot,
        intervals: List[Tuple[float, float]],
        method: str = 'interval',
    ) -> Tuple[Optional[List[LinkAABBInfo]], List[CacheEntry], List]:
        """合并查询"""
        store = self._get_store(robot, method)
        return store.query_interval_merge(intervals)

    def has_coverage(
        self,
        robot,
        intervals: List[Tuple[float, float]],
        method: str = 'interval',
    ) -> bool:
        """快速判断缓存中是否有覆盖该区间的条目"""
        store = self._get_store(robot, method)
        # 精确匹配或子集覆盖
        if store.query_exact(intervals) is not None:
            return True
        merged, _, gaps = store.query_interval_merge(intervals)
        return merged is not None and len(gaps) == 0

    def get_stats(self, robot) -> Dict[str, int]:
        """获取缓存统计信息"""
        rid = robot.fingerprint()
        if rid not in self._stores:
            return {'interval': 0, 'numerical': 0}
        return {
            'interval': self._stores[rid]['interval'].n_entries,
            'numerical': self._stores[rid]['numerical'].n_entries,
        }

    def save(self, filepath: str) -> None:
        """持久化到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'max_entries': self.max_entries,
                'stores': self._stores,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("AABB 缓存已保存到 %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> 'AABBCache':
        """从文件加载"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        cache = cls(max_entries_per_store=data.get('max_entries', 100000))
        cache._stores = data.get('stores', {})
        logger.info("AABB 缓存已从 %s 加载", filepath)
        return cache

    def clear(self, robot=None) -> None:
        """清空缓存

        Args:
            robot: 若指定，仅清空该机器人的缓存；否则全清
        """
        if robot is not None:
            rid = robot.fingerprint()
            if rid in self._stores:
                self._stores[rid]['interval'].clear()
                self._stores[rid]['numerical'].clear()
        else:
            self._stores.clear()

    # ==================== 自动持久化 ====================

    @staticmethod
    def cache_path_for_robot(
        robot,
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """获取指定机器人的缓存文件路径

        文件命名格式: ``<robot_name>_<fingerprint[:12]>.pkl``
        目录默认为 ``.cache/aabb/``。

        Args:
            robot: 机器人模型
            cache_dir: 缓存目录路径（默认 DEFAULT_CACHE_DIR）

        Returns:
            缓存文件的 Path 对象
        """
        d = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        # 安全文件名：小写、替换空格
        safe_name = robot.name.lower().replace(' ', '_').replace('-', '_')
        fp_short = robot.fingerprint()[:12]
        return d / f"{safe_name}_{fp_short}.pkl"

    @classmethod
    def auto_load(
        cls,
        robot,
        cache_dir: Optional[Path] = None,
        max_entries_per_store: int = 100000,
    ) -> 'AABBCache':
        """自动加载机器人的包络缓存

        若缓存文件存在且可成功读取，则返回已填充的 AABBCache 实例；
        否则返回空的 AABBCache。

        Args:
            robot: 机器人模型
            cache_dir: 缓存目录（默认 DEFAULT_CACHE_DIR）
            max_entries_per_store: 每个库的最大条目数

        Returns:
            AABBCache 实例

        Example:
            >>> cache = AABBCache.auto_load(robot)
            >>> stats = cache.get_stats(robot)
        """
        path = cls.cache_path_for_robot(robot, cache_dir)
        if path.exists():
            try:
                cache = cls.load(str(path))
                stats = cache.get_stats(robot)
                total = stats.get('interval', 0) + stats.get('numerical', 0)
                logger.info(
                    "自动加载缓存: %s (%d 条目, interval=%d, numerical=%d)",
                    path, total, stats.get('interval', 0),
                    stats.get('numerical', 0),
                )
                return cache
            except Exception as e:
                logger.warning("缓存文件读取失败 (%s): %s，创建空缓存", path, e)
        else:
            logger.info("无已有缓存文件: %s，创建空缓存", path)
        return cls(max_entries_per_store=max_entries_per_store)

    def auto_save(
        self,
        robot,
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """自动保存当前缓存到磁盘

        自动创建缓存目录（若不存在）。

        Args:
            robot: 机器人模型
            cache_dir: 缓存目录（默认 DEFAULT_CACHE_DIR）

        Returns:
            保存的文件路径

        Example:
            >>> cache.auto_save(robot)
            PosixPath('.cache/aabb/panda_abc123def456.pkl')
        """
        path = self.cache_path_for_robot(robot, cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(path))
        stats = self.get_stats(robot)
        total = stats.get('interval', 0) + stats.get('numerical', 0)
        logger.info(
            "自动保存缓存: %s (%d 条目)",
            path, total,
        )
        return path
