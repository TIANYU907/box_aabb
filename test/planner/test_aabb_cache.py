"""
test/planner/test_aabb_cache.py - AABB 缓存系统测试
"""

import os
import tempfile
import pytest
import numpy as np

from box_aabb.robot import load_robot
from box_aabb.interval_fk import compute_interval_aabb
from box_aabb.models import LinkAABBInfo
from planner.aabb_cache import (
    AABBCache,
    IntervalStore,
    CacheEntry,
    _merge_link_aabbs,
    _intervals_to_key,
)


@pytest.fixture
def robot_2dof():
    return load_robot('2dof_planar')


@pytest.fixture
def robot_3dof():
    return load_robot('3dof_planar')


@pytest.fixture
def sample_link_aabbs():
    """生成用于测试的 LinkAABBInfo 列表"""
    return [
        LinkAABBInfo(
            link_index=1, link_name='link1',
            min_point=[0.0, -0.5, 0.0],
            max_point=[1.0, 0.5, 0.0],
            is_zero_length=False,
        ),
        LinkAABBInfo(
            link_index=2, link_name='link2',
            min_point=[0.5, -1.0, 0.0],
            max_point=[2.0, 1.0, 0.0],
            is_zero_length=False,
        ),
    ]


@pytest.fixture
def sample_link_aabbs_2():
    """另一组 LinkAABBInfo"""
    return [
        LinkAABBInfo(
            link_index=1, link_name='link1',
            min_point=[-0.5, -0.3, 0.0],
            max_point=[0.8, 0.3, 0.0],
            is_zero_length=False,
        ),
        LinkAABBInfo(
            link_index=2, link_name='link2',
            min_point=[0.3, -0.8, 0.0],
            max_point=[1.5, 0.8, 0.0],
            is_zero_length=False,
        ),
    ]


class TestIntervalStore:
    """IntervalStore 测试"""

    def test_store_and_query_exact(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval')
        intervals = [(0.0, 1.0), (0.0, 1.0)]

        eid = store.store(intervals, sample_link_aabbs, n_sub=1)
        assert eid == 0
        assert store.n_entries == 1

        entry = store.query_exact(intervals)
        assert entry is not None
        assert entry.entry_id == eid
        assert len(entry.link_aabbs) == 2

    def test_query_exact_miss(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval')
        store.store([(0.0, 1.0), (0.0, 1.0)], sample_link_aabbs)

        result = store.query_exact([(0.0, 0.5), (0.0, 1.0)])
        assert result is None

    def test_precision_replace_interval(self, sample_link_aabbs, sample_link_aabbs_2):
        """interval 库：小体积替换大体积"""
        store = IntervalStore(n_dims=2, store_type='interval')
        intervals = [(0.0, 1.0), (0.0, 1.0)]

        # 先存一个大体积的
        store.store(intervals, sample_link_aabbs, n_sub=1)
        entry1 = store.query_exact(intervals)
        vol1 = entry1.total_volume

        # 再存一个小体积的（应替换）
        store.store(intervals, sample_link_aabbs_2, n_sub=2)
        entry2 = store.query_exact(intervals)
        vol2 = entry2.total_volume

        assert vol2 <= vol1
        assert store.n_entries == 1  # 只有一个条目

    def test_precision_replace_numerical(self, sample_link_aabbs, sample_link_aabbs_2):
        """numerical 库：大体积替换小体积"""
        store = IntervalStore(n_dims=2, store_type='numerical')
        intervals = [(0.0, 1.0), (0.0, 1.0)]

        # 先存小体积的
        store.store(intervals, sample_link_aabbs_2, n_sub=1)
        entry1 = store.query_exact(intervals)
        vol1 = entry1.total_volume

        # 再存大体积的（应替换）
        store.store(intervals, sample_link_aabbs, n_sub=2)
        entry2 = store.query_exact(intervals)
        vol2 = entry2.total_volume

        assert vol2 >= vol1

    def test_query_subsets(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval')

        # 存多个小区间
        store.store([(0.0, 0.5), (0.0, 0.5)], sample_link_aabbs)
        store.store([(0.5, 1.0), (0.0, 0.5)], sample_link_aabbs)
        store.store([(0.0, 0.5), (0.5, 1.0)], sample_link_aabbs)

        # 查询大区间包含的子集
        subsets = store.query_subsets([(0.0, 1.0), (0.0, 1.0)])
        assert len(subsets) == 3

        # 更小的查询区间
        subsets2 = store.query_subsets([(0.0, 0.6), (0.0, 0.6)])
        assert len(subsets2) == 1  # 只有 [0,0.5]x[0,0.5]

    def test_query_containing(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval')

        # 存一个大区间
        store.store([(0.0, 2.0), (0.0, 2.0)], sample_link_aabbs)

        # 查询小区间是否被包含
        containing = store.query_containing([(0.5, 0.5), (0.5, 0.5)])
        assert len(containing) == 1

        # 超出范围
        containing2 = store.query_containing([(2.5, 2.5), (0.5, 0.5)])
        assert len(containing2) == 0

    def test_query_interval_merge(self, sample_link_aabbs, sample_link_aabbs_2):
        store = IntervalStore(n_dims=2, store_type='interval')

        # 存两个相邻子区间
        store.store([(0.0, 0.5), (0.0, 1.0)], sample_link_aabbs)
        store.store([(0.5, 1.0), (0.0, 1.0)], sample_link_aabbs_2)

        # 合并查询
        merged, covered, gaps = store.query_interval_merge(
            [(0.0, 1.0), (0.0, 1.0)])

        assert merged is not None
        assert len(covered) == 2
        assert len(gaps) == 0  # 完全覆盖

    def test_query_interval_merge_with_gap(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval')

        store.store([(0.0, 0.3), (0.0, 1.0)], sample_link_aabbs)
        store.store([(0.7, 1.0), (0.0, 1.0)], sample_link_aabbs)

        merged, covered, gaps = store.query_interval_merge(
            [(0.0, 1.0), (0.0, 1.0)])

        assert merged is not None
        assert len(covered) == 2
        assert len(gaps) > 0  # 有间隙

    def test_remove(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval')
        eid = store.store([(0.0, 1.0), (0.0, 1.0)], sample_link_aabbs)
        assert store.n_entries == 1

        store.remove(eid)
        assert store.n_entries == 0
        assert store.query_exact([(0.0, 1.0), (0.0, 1.0)]) is None

    def test_clear(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval')
        store.store([(0.0, 1.0), (0.0, 1.0)], sample_link_aabbs)
        store.store([(1.0, 2.0), (0.0, 1.0)], sample_link_aabbs)
        assert store.n_entries == 2

        store.clear()
        assert store.n_entries == 0

    def test_lru_eviction(self, sample_link_aabbs):
        store = IntervalStore(n_dims=2, store_type='interval', max_entries=3)

        store.store([(0.0, 1.0), (0.0, 1.0)], sample_link_aabbs)
        store.store([(1.0, 2.0), (0.0, 1.0)], sample_link_aabbs)
        store.store([(2.0, 3.0), (0.0, 1.0)], sample_link_aabbs)
        assert store.n_entries == 3

        # 第 4 个触发淘汰
        store.store([(3.0, 4.0), (0.0, 1.0)], sample_link_aabbs)
        assert store.n_entries == 3


class TestMergeLinkAABBs:
    """合并函数测试"""

    def test_merge_two_lists(self, sample_link_aabbs, sample_link_aabbs_2):
        merged = _merge_link_aabbs([sample_link_aabbs, sample_link_aabbs_2])

        assert len(merged) == 2
        for la in merged:
            if la.link_index == 1:
                assert la.min_point[0] <= -0.5
                assert la.max_point[0] >= 1.0
            elif la.link_index == 2:
                assert la.min_point[0] <= 0.3
                assert la.max_point[0] >= 2.0

    def test_merge_single_list(self, sample_link_aabbs):
        merged = _merge_link_aabbs([sample_link_aabbs])
        assert len(merged) == 2

    def test_merge_empty(self):
        merged = _merge_link_aabbs([])
        assert len(merged) == 0


class TestAABBCache:
    """AABBCache 高级接口测试"""

    def test_store_and_query(self, robot_2dof, sample_link_aabbs):
        cache = AABBCache()
        intervals = [(0.0, 0.5), (0.0, 0.5)]

        eid = cache.store(robot_2dof, intervals, sample_link_aabbs,
                          method='interval')
        assert eid >= 0

        entry = cache.query_exact(robot_2dof, intervals, method='interval')
        assert entry is not None

        # 不同方法不交叉
        entry_num = cache.query_exact(robot_2dof, intervals, method='numerical')
        assert entry_num is None

    def test_different_robots(self, robot_2dof, robot_3dof, sample_link_aabbs):
        cache = AABBCache()
        intervals_2d = [(0.0, 1.0), (0.0, 1.0)]
        intervals_3d = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        aabb_3d = sample_link_aabbs + [
            LinkAABBInfo(link_index=3, link_name='link3',
                         min_point=[0, 0, 0], max_point=[1, 1, 0],
                         is_zero_length=False)]

        cache.store(robot_2dof, intervals_2d, sample_link_aabbs)
        cache.store(robot_3dof, intervals_3d, aabb_3d)

        assert cache.get_stats(robot_2dof)['interval'] == 1
        assert cache.get_stats(robot_3dof)['interval'] == 1

    def test_has_coverage(self, robot_2dof, sample_link_aabbs):
        cache = AABBCache()
        intervals = [(0.0, 1.0), (0.0, 1.0)]
        cache.store(robot_2dof, intervals, sample_link_aabbs)

        assert cache.has_coverage(robot_2dof, intervals)
        assert not cache.has_coverage(robot_2dof, [(2.0, 3.0), (0.0, 1.0)])

    def test_save_and_load(self, robot_2dof, sample_link_aabbs):
        cache = AABBCache()
        cache.store(robot_2dof, [(0.0, 1.0), (0.0, 1.0)], sample_link_aabbs)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            cache.save(filepath)
            loaded = AABBCache.load(filepath)
            entry = loaded.query_exact(robot_2dof, [(0.0, 1.0), (0.0, 1.0)])
            assert entry is not None
        finally:
            os.unlink(filepath)

    def test_clear_specific_robot(self, robot_2dof, sample_link_aabbs):
        cache = AABBCache()
        cache.store(robot_2dof, [(0.0, 1.0), (0.0, 1.0)], sample_link_aabbs)
        assert cache.get_stats(robot_2dof)['interval'] == 1

        cache.clear(robot_2dof)
        assert cache.get_stats(robot_2dof)['interval'] == 0

    def test_clear_all(self, robot_2dof, sample_link_aabbs):
        cache = AABBCache()
        cache.store(robot_2dof, [(0.0, 1.0), (0.0, 1.0)], sample_link_aabbs)
        cache.clear()
        assert cache.get_stats(robot_2dof) == {'interval': 0, 'numerical': 0}


class TestCacheWithRealFK:
    """使用真实区间 FK 的缓存集成测试"""

    def test_cache_real_aabb(self, robot_2dof):
        intervals = [(0.0, 0.5), (0.0, 0.5)]
        link_aabbs, _ = compute_interval_aabb(
            robot=robot_2dof,
            intervals=intervals,
            zero_length_links=robot_2dof.zero_length_links,
            skip_zero_length=True,
            n_sub=1,
        )

        cache = AABBCache()
        cache.store(robot_2dof, intervals, link_aabbs, n_sub=1)

        entry = cache.query_exact(robot_2dof, intervals)
        assert entry is not None
        assert len(entry.link_aabbs) > 0
        assert entry.total_volume >= 0  # 2D 平面机器人 z 范围为 0，体积可为 0
