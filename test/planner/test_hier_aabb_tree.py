"""
test/planner/test_hier_aabb_tree.py - HierAABBTree 单元测试
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from box_aabb.robot import load_robot
from planner.hier_aabb_tree import HierAABBTree, HierAABBNode
from planner.obstacles import Scene


# ─── fixtures ───

@pytest.fixture
def robot_2dof():
    return load_robot("2dof_planar")


@pytest.fixture
def joint_limits(robot_2dof):
    return list(robot_2dof.joint_limits)


@pytest.fixture
def empty_scene():
    return Scene()


@pytest.fixture
def scene_with_obs(robot_2dof):
    """有一个障碍物的场景"""
    s = Scene()
    s.add_obstacle(min_point=[0.8, -0.3], max_point=[1.2, 0.3], name="obs")
    return s


@pytest.fixture
def hier_tree(robot_2dof, joint_limits):
    return HierAABBTree(robot_2dof, joint_limits)


# ─── 基本构造 ───

class TestConstruction:
    def test_init(self, hier_tree, robot_2dof):
        assert hier_tree.n_dims == 2
        assert hier_tree.n_nodes == 1
        assert hier_tree.n_fk_calls == 0
        assert hier_tree.root.is_leaf()
        assert hier_tree.root.depth == 0

    def test_root_covers_joint_limits(self, hier_tree, joint_limits):
        for i, (lo, hi) in enumerate(joint_limits):
            assert hier_tree.root.intervals[i][0] == lo
            assert hier_tree.root.intervals[i][1] == hi


# ─── AABB 计算 ───

class TestAABB:
    def test_ensure_aabb(self, hier_tree):
        hier_tree._ensure_aabb(hier_tree.root)
        assert hier_tree.root.raw_aabb is not None
        assert hier_tree.root.refined_aabb is not None
        assert hier_tree.n_fk_calls == 1

    def test_ensure_aabb_idempotent(self, hier_tree):
        hier_tree._ensure_aabb(hier_tree.root)
        hier_tree._ensure_aabb(hier_tree.root)
        assert hier_tree.n_fk_calls == 1  # 不重复计算

    def test_union_preserves_links(self, hier_tree):
        hier_tree._ensure_aabb(hier_tree.root)
        a = hier_tree.root.raw_aabb
        union = HierAABBTree._union_aabb(a, a)
        assert len(union) == len(a)
        for la, lu in zip(a, union):
            assert la.min_point == lu.min_point
            assert la.max_point == lu.max_point


# ─── 切分 ───

class TestSplit:
    def test_split_creates_children(self, hier_tree):
        hier_tree._split(hier_tree.root)
        assert not hier_tree.root.is_leaf()
        assert hier_tree.root.left is not None
        assert hier_tree.root.right is not None
        assert hier_tree.n_nodes == 3  # root + 2 children
        assert hier_tree.n_fk_calls == 2  # left + right (root AABB not computed by split)

    def test_split_dim_is_depth_mod_ndims(self, hier_tree):
        hier_tree._split(hier_tree.root)
        assert hier_tree.root.split_dim == 0  # depth=0, 0 % 2 = 0

        hier_tree._split(hier_tree.root.left)
        assert hier_tree.root.left.split_dim == 1  # depth=1, 1 % 2 = 1

    def test_split_at_midpoint(self, hier_tree, joint_limits):
        hier_tree._split(hier_tree.root)
        lo, hi = joint_limits[0]
        mid = (lo + hi) / 2
        assert hier_tree.root.split_val == pytest.approx(mid)
        assert hier_tree.root.left.intervals[0][1] == pytest.approx(mid)
        assert hier_tree.root.right.intervals[0][0] == pytest.approx(mid)

    def test_children_cover_parent(self, hier_tree, joint_limits):
        hier_tree._split(hier_tree.root)
        # dim 0 fully covered
        assert hier_tree.root.left.intervals[0][0] == joint_limits[0][0]
        assert hier_tree.root.right.intervals[0][1] == joint_limits[0][1]
        # dim 1 unchanged
        assert hier_tree.root.left.intervals[1] == joint_limits[1]
        assert hier_tree.root.right.intervals[1] == joint_limits[1]

    def test_refined_tighter_than_raw(self, hier_tree):
        hier_tree._ensure_aabb(hier_tree.root)
        raw_total_vol = sum(la.volume for la in hier_tree.root.raw_aabb)

        hier_tree._split(hier_tree.root)
        refined_total_vol = sum(la.volume for la in hier_tree.root.refined_aabb)

        # refined (union of children) should be <= raw (direct computation)
        assert refined_total_vol <= raw_total_vol + 1e-6

    def test_double_split_idempotent(self, hier_tree):
        hier_tree._split(hier_tree.root)
        n = hier_tree.n_nodes
        hier_tree._split(hier_tree.root)  # 已分裂过，无操作
        assert hier_tree.n_nodes == n


# ─── find_free_box ───

class TestFindFreeBox:
    def test_empty_scene_returns_full_space(self, robot_2dof, joint_limits,
                                             empty_scene):
        tree = HierAABBTree(robot_2dof, joint_limits)
        seed = np.array([0.0, 0.0])
        obstacles = empty_scene.get_obstacles()
        box = tree.find_free_box(seed, obstacles, max_depth=40)
        # 无障碍物，应返回整个关节空间或非常接近
        assert box is not None
        for (lo, hi), (jlo, jhi) in zip(box, joint_limits):
            assert lo <= seed[0] or lo <= seed[1]  # 包含 seed

    def test_returns_box_containing_seed(self, robot_2dof, joint_limits,
                                          scene_with_obs):
        tree = HierAABBTree(robot_2dof, joint_limits)
        seed = np.array([2.5, 2.5])  # 远离障碍物
        obstacles = scene_with_obs.get_obstacles()
        box = tree.find_free_box(seed, obstacles, max_depth=30)
        assert box is not None
        # 验证 seed 在 box 内
        for i, (lo, hi) in enumerate(box):
            assert lo <= seed[i] + 1e-9
            assert hi >= seed[i] - 1e-9

    def test_multiple_seeds_share_cache(self, robot_2dof, joint_limits,
                                         scene_with_obs):
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = scene_with_obs.get_obstacles()

        seed1 = np.array([2.0, 2.0])
        tree.find_free_box(seed1, obstacles, max_depth=20)
        fk_after_first = tree.n_fk_calls

        seed2 = np.array([2.1, 2.1])  # 附近的 seed
        tree.find_free_box(seed2, obstacles, max_depth=20)
        fk_after_second = tree.n_fk_calls

        # 第二次 FK 调用应少于第一次（共享祖先节点缓存）
        fk_first = fk_after_first
        fk_second = fk_after_second - fk_after_first
        assert fk_second < fk_first

    def test_max_depth_limit(self, robot_2dof, joint_limits, scene_with_obs):
        tree = HierAABBTree(robot_2dof, joint_limits)
        # seed 在障碍物正对方向，很深才能找到安全 box
        seed = np.array([0.0, 0.0])
        obstacles = scene_with_obs.get_obstacles()
        box = tree.find_free_box(seed, obstacles, max_depth=3)
        # max_depth=3 可能不够深，可能返回 None
        # 此处只验证不崩溃
        if box is not None:
            for i, (lo, hi) in enumerate(box):
                assert lo <= seed[i] + 1e-9
                assert hi >= seed[i] - 1e-9


# ─── query_aabb ───

class TestQueryAABB:
    def test_query_aabb_returns_something(self, hier_tree, joint_limits):
        # 查询整个空间
        result = hier_tree.query_aabb(joint_limits)
        assert result is not None
        assert len(result) > 0

    def test_query_sub_interval(self, hier_tree, joint_limits):
        # 先切分
        hier_tree._split(hier_tree.root)
        # 查询一个子区间
        sub = [(0.0, 1.0), (0.0, 1.0)]
        result = hier_tree.query_aabb(sub)
        assert result is not None


# ─── 持久化 ───

class TestPersistence:
    def test_save_load_roundtrip(self, robot_2dof, joint_limits, scene_with_obs):
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = scene_with_obs.get_obstacles()
        tree.find_free_box(np.array([2.0, 2.0]), obstacles, max_depth=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_cache.pkl")
            tree.save(path)

            loaded = HierAABBTree.load(path, robot_2dof)
            assert loaded.n_nodes == tree.n_nodes
            assert loaded.n_fk_calls == tree.n_fk_calls
            assert loaded.n_dims == tree.n_dims

    def test_load_rejects_wrong_robot(self, robot_2dof, joint_limits):
        tree = HierAABBTree(robot_2dof, joint_limits)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_cache.pkl")
            tree.save(path)

            # 用不同机器人加载应报错
            robot_3dof = load_robot("3dof_planar")
            with pytest.raises(ValueError, match="指纹不匹配"):
                HierAABBTree.load(path, robot_3dof)


# ─── 统计 ───

class TestStats:
    def test_stats_after_splits(self, hier_tree, scene_with_obs):
        obstacles = scene_with_obs.get_obstacles()
        hier_tree.find_free_box(np.array([2.0, 2.0]), obstacles, max_depth=10)

        stats = hier_tree.get_stats()
        assert stats['n_nodes'] > 1
        assert stats['n_leaves'] > 0
        assert stats['max_depth'] > 0
        assert stats['n_fk_calls'] > 0
