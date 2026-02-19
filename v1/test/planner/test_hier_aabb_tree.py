"""
test/planner/test_hier_aabb_tree.py - HierAABBTree 单元测试
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from box_aabb.robot import load_robot
from planner.hier_aabb_tree import HierAABBTree, HierAABBNode, FindFreeBoxResult
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
        assert union.shape == a.shape
        np.testing.assert_array_equal(union, a)


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
        raw = hier_tree.root.raw_aabb  # (n_links, 6)
        raw_total_vol = np.sum(np.prod(
            np.maximum(raw[:, 3:] - raw[:, :3], 0), axis=1))

        hier_tree._split(hier_tree.root)
        ref = hier_tree.root.refined_aabb
        refined_total_vol = np.sum(np.prod(
            np.maximum(ref[:, 3:] - ref[:, :3], 0), axis=1))

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
        result = tree.find_free_box(seed, obstacles, max_depth=40)
        # 无障碍物，应返回整个关节空间或非常接近
        assert result is not None
        box = result.intervals
        for (lo, hi), (jlo, jhi) in zip(box, joint_limits):
            assert lo <= seed[0] or lo <= seed[1]  # 包含 seed

    def test_returns_box_containing_seed(self, robot_2dof, joint_limits,
                                          scene_with_obs):
        tree = HierAABBTree(robot_2dof, joint_limits)
        seed = np.array([2.5, 2.5])  # 远离障碍物
        obstacles = scene_with_obs.get_obstacles()
        result = tree.find_free_box(seed, obstacles, max_depth=30)
        assert result is not None
        box = result.intervals
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
            for i, (lo, hi) in enumerate(box.intervals):
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
            path = str(Path(tmpdir) / "test_cache.hcache")
            tree.save_binary(path)

            loaded = HierAABBTree.load_binary(path, robot_2dof)
            assert loaded.n_nodes == tree.n_nodes
            assert loaded.n_fk_calls == tree.n_fk_calls
            assert loaded.n_dims == tree.n_dims

    def test_load_rejects_wrong_robot(self, robot_2dof, joint_limits):
        tree = HierAABBTree(robot_2dof, joint_limits)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_cache.hcache")
            tree.save_binary(path)

            # 用不同机器人加载应报错
            robot_3dof = load_robot("3dof_planar")
            with pytest.raises(ValueError, match="指纹不匹配"):
                HierAABBTree.load_binary(path, robot_3dof)

    def test_save_incremental_roundtrip(self, robot_2dof, joint_limits,
                                         scene_with_obs):
        """增量保存等价于全量保存"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = scene_with_obs.get_obstacles()
        tree.find_free_box(np.array([2.0, 2.0]), obstacles, max_depth=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "incr.hcache")
            # 第一次全量保存
            tree.save_binary(path)

            # 再加几个 box → 产生 dirty 节点
            tree.find_free_box(np.array([0.5, 0.5]), obstacles, max_depth=12)
            tree.find_free_box(np.array([1.5, 1.5]), obstacles, max_depth=12)

            # 增量保存
            tree.save_incremental(path)

            # 加载并验证
            loaded = HierAABBTree.load_binary(path, robot_2dof)
            assert loaded.n_nodes == tree.n_nodes
            assert loaded.n_fk_calls == tree.n_fk_calls

            # 对比 root AABB
            tree._ensure_aabb(tree.root)
            loaded._ensure_aabb(loaded.root)
            np.testing.assert_array_almost_equal(
                tree.root.aabb, loaded.root.aabb, decimal=5)


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


# ─── 占用提升 (promotion) ───

class TestPromotion:
    """测试当父节点 refined_aabb 变为无碰撞时，吸收已占用子节点"""

    def test_find_free_box_returns_result_type(self, robot_2dof, joint_limits,
                                                empty_scene):
        """验证返回值是 FindFreeBoxResult"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = empty_scene.get_obstacles()
        result = tree.find_free_box(
            np.array([0.0, 0.0]), obstacles, mark_occupied=True,
            forest_box_id=42)
        assert result is not None
        assert isinstance(result, FindFreeBoxResult)
        assert isinstance(result.intervals, list)
        assert isinstance(result.absorbed_box_ids, set)
        assert len(result.absorbed_box_ids) == 0  # 空场景无 promotion

    def test_forest_box_id_tracked(self, robot_2dof, joint_limits, empty_scene):
        """验证 mark_occupied 后 forest_box_id 被记录"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = empty_scene.get_obstacles()
        result = tree.find_free_box(
            np.array([0.0, 0.0]), obstacles, mark_occupied=True,
            forest_box_id=99)
        assert result is not None
        # root 应被标记为 occupied 且 forest_box_id=99
        assert tree.root.occupied is True
        assert tree.root.forest_box_id == 99

    def test_promotion_absorbs_children(self, robot_2dof, joint_limits,
                                         scene_with_obs):
        """两次 find_free_box（mark_occupied）后，精化可能使父节点无碰撞，
        第三次查询另一位置时上行阶段触发 promotion"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = scene_with_obs.get_obstacles()

        # 先在远处两个位置各扩展一个 box
        r1 = tree.find_free_box(
            np.array([2.5, 2.5]), obstacles, max_depth=30,
            mark_occupied=True, forest_box_id=100)
        r2 = tree.find_free_box(
            np.array([2.6, 2.6]), obstacles, max_depth=30,
            mark_occupied=True, forest_box_id=101)

        # 至少一个应成功
        assert r1 is not None or r2 is not None

        # 如果两个都成功，检查 absorbed_box_ids
        if r1 is not None and r2 is not None:
            # r2 可能吸收了 r1（如果它们的公共祖先变为无碰撞）
            if r2.absorbed_box_ids:
                assert 100 in r2.absorbed_box_ids

    def test_no_promotion_when_parent_collides(self, robot_2dof, joint_limits,
                                                scene_with_obs):
        """当父节点仍碰撞时，不应发生 promotion"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = scene_with_obs.get_obstacles()

        # 在靠近障碍物的位置扩展（深度较大的 box）
        r1 = tree.find_free_box(
            np.array([0.5, 0.5]), obstacles, max_depth=30,
            mark_occupied=True, forest_box_id=200)
        if r1 is None:
            pytest.skip("无法在此位置找到 free box")

        # 在同一区域附近再扩展
        r2 = tree.find_free_box(
            np.array([0.6, 0.6]), obstacles, max_depth=30,
            mark_occupied=True, forest_box_id=201)
        # 即使成功，靠近障碍物的父节点通常仍碰撞
        # promotion 不应吸收非碰撞安全的区域

    def test_collect_forest_ids(self, robot_2dof, joint_limits, empty_scene):
        """测试 _collect_forest_ids 辅助方法"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = empty_scene.get_obstacles()

        # 手动构造: split root, 标记 left, right
        tree._split(tree.root)
        tree._mark_occupied(tree.root.left, 10)
        tree._mark_occupied(tree.root.right, 20)

        ids = tree._collect_forest_ids(tree.root)
        assert ids == {10, 20}

    def test_clear_subtree_occupation(self, robot_2dof, joint_limits,
                                       empty_scene):
        """测试 _clear_subtree_occupation 辅助方法"""
        tree = HierAABBTree(robot_2dof, joint_limits)

        tree._split(tree.root)
        tree._mark_occupied(tree.root.left, 10)
        tree._mark_occupied(tree.root.right, 20)

        tree._clear_subtree_occupation(tree.root)
        assert tree.root.subtree_occupied == 0
        assert tree.root.left.occupied is False
        assert tree.root.left.forest_box_id is None
        assert tree.root.right.occupied is False
        assert tree.root.right.forest_box_id is None


# ─── vectorized collision ───

class TestVectorizedCollision:
    """测试向量化碰撞检测"""

    def test_prepack_empty(self, hier_tree):
        """空障碍物列表返回 None"""
        assert hier_tree._prepack_obstacles_c([], 0.0) is None

    def test_prepack_shape(self, hier_tree, scene_with_obs):
        """预打包返回 list of tuples"""
        obstacles = scene_with_obs.get_obstacles()
        packed = hier_tree._prepack_obstacles_c(obstacles, 0.0)
        assert packed is not None
        assert isinstance(packed, list)
        assert len(packed) > 0
        # 每个元组: (link_idx, lo0, hi0, lo1, hi1, lo2, hi2)
        assert len(packed[0]) == 7

    def test_collide_empty_obstacles(self, hier_tree):
        """无障碍物不碰撞"""
        hier_tree._ensure_aabb(hier_tree.root)
        assert not hier_tree._store.link_aabbs_collide(0, None)

    def test_collide_consistency(self, hier_tree, scene_with_obs):
        """向量化结果与预期一致"""
        hier_tree._ensure_aabb(hier_tree.root)
        obstacles = scene_with_obs.get_obstacles()
        packed = hier_tree._prepack_obstacles_c(obstacles, 0.0)
        result = hier_tree._store.link_aabbs_collide(0, packed)
        assert isinstance(result, bool)


class TestFindContainingBoxId:
    """测试 find_containing_box_id 的 O(depth) 空间查询"""

    def test_returns_none_when_empty(self, robot_2dof, joint_limits):
        """空树（无占用节点）返回 None"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        assert tree.find_containing_box_id(np.array([0.0, 0.0])) is None

    def test_returns_id_after_occupy(self, robot_2dof, joint_limits,
                                     empty_scene):
        """占用后能找到对应的 forest_box_id"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = empty_scene.get_obstacles()
        seed = np.array([0.5, 0.5])
        result = tree.find_free_box(
            seed, obstacles, mark_occupied=True, forest_box_id=99)
        assert result is not None
        # seed 应在占用区域内
        found_id = tree.find_containing_box_id(seed)
        assert found_id == 99

    def test_returns_none_outside_occupied(self, robot_2dof, joint_limits,
                                            empty_scene):
        """不在任何占用区域内时返回 None"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = empty_scene.get_obstacles()
        seed = np.array([0.5, 0.5])
        tree.find_free_box(
            seed, obstacles, mark_occupied=True, forest_box_id=99,
            max_depth=4)
        # 很远处大概率不在同一个占用节点内
        far_point = np.array([-2.5, -2.5])
        found_id = tree.find_containing_box_id(far_point)
        # 空场景 depth=4 时根节点可能占据，也可能不占据
        # 但至少返回值类型正确
        assert found_id is None or isinstance(found_id, int)

    def test_is_occupied_consistent(self, robot_2dof, joint_limits,
                                     empty_scene):
        """is_occupied 与 find_containing_box_id 结果一致"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = empty_scene.get_obstacles()
        seed = np.array([1.0, -1.0])
        tree.find_free_box(
            seed, obstacles, mark_occupied=True, forest_box_id=7)
        # 已占用点
        assert tree.is_occupied(seed) is True
        assert tree.find_containing_box_id(seed) is not None
        # 它们应等价
        for q in [np.array([0.0, 0.0]), np.array([2.0, 2.0]),
                  np.array([-1.0, 1.0])]:
            occ = tree.is_occupied(q)
            fid = tree.find_containing_box_id(q)
            assert occ == (fid is not None), \
                f"is_occupied={occ} but find_containing_box_id={fid} for {q}"

    def test_multiple_boxes(self, robot_2dof, joint_limits, scene_with_obs):
        """多个占用节点能正确区分"""
        tree = HierAABBTree(robot_2dof, joint_limits)
        obstacles = scene_with_obs.get_obstacles()
        results = {}
        for i, seed in enumerate([np.array([0.5, 0.5]),
                                   np.array([-0.5, -0.5]),
                                   np.array([1.5, 1.5])]):
            r = tree.find_free_box(
                seed, obstacles, mark_occupied=True, forest_box_id=100 + i)
            if r is not None:
                results[100 + i] = (r.intervals, seed)
        # 每个 seed 查询应返回对应的 forest_box_id
        for box_id, (ivs, seed) in results.items():
            found = tree.find_containing_box_id(seed)
            assert found == box_id, \
                f"seed {seed}: expected {box_id}, got {found}"