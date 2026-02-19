"""test/planner/test_box_tree.py - BoxTreeManager 与 BoxNode/BoxTree 测试"""
import math
import numpy as np
import pytest

from planner.models import BoxNode, BoxTree
from planner.box_tree import BoxTreeManager


class TestBoxNode:
    """BoxNode 数据模型测试"""

    def test_basic_construction(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(-1.0, 1.0), (-0.5, 0.5)],
            seed_config=np.array([0.0, 0.0]),
        )
        assert box.node_id == 0
        assert box.n_dims == 2
        assert box.volume == pytest.approx(2.0)  # 2 * 1 = 2

    def test_contains(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(-1.0, 1.0), (-0.5, 0.5)],
            seed_config=np.array([0.0, 0.0]),
        )
        assert box.contains(np.array([0.0, 0.0]))
        assert box.contains(np.array([1.0, 0.5]))  # on boundary
        assert not box.contains(np.array([1.1, 0.0]))

    def test_center(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(0.0, 2.0), (1.0, 3.0)],
            seed_config=np.array([1.0, 2.0]),
        )
        np.testing.assert_array_almost_equal(box.center, [1.0, 2.0])

    def test_widths(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(0.0, 2.0), (1.0, 4.0)],
            seed_config=np.array([1.0, 2.5]),
        )
        np.testing.assert_array_almost_equal(box.widths, [2.0, 3.0])

    def test_distance_to_config_inside(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(-1.0, 1.0), (-1.0, 1.0)],
            seed_config=np.array([0.0, 0.0]),
        )
        assert box.distance_to_config(np.array([0.0, 0.0])) == pytest.approx(0.0)

    def test_distance_to_config_outside(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(0.0, 1.0), (0.0, 1.0)],
            seed_config=np.array([0.5, 0.5]),
        )
        # 点 (2, 0.5) → 距离 = 1.0 (x 方向)
        assert box.distance_to_config(np.array([2.0, 0.5])) == pytest.approx(1.0)

    def test_overlap_with(self):
        box1 = BoxNode(
            node_id=0,
            joint_intervals=[(0.0, 2.0), (0.0, 2.0)],
            seed_config=np.array([1.0, 1.0]),
        )
        box2 = BoxNode(
            node_id=1,
            joint_intervals=[(1.0, 3.0), (1.0, 3.0)],
            seed_config=np.array([2.0, 2.0]),
        )
        box3 = BoxNode(
            node_id=2,
            joint_intervals=[(5.0, 6.0), (5.0, 6.0)],
            seed_config=np.array([5.5, 5.5]),
        )
        assert box1.overlap_with(box2) is True
        assert box1.overlap_with(box3) is False

    def test_nearest_point_to(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(0.0, 1.0), (0.0, 1.0)],
            seed_config=np.array([0.5, 0.5]),
        )
        # 点在外部 → 最近点是 box 表面上的点
        nearest = box.nearest_point_to(np.array([2.0, 0.5]))
        np.testing.assert_array_almost_equal(nearest, [1.0, 0.5])

        # 点在内部 → 返回自身
        nearest_in = box.nearest_point_to(np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(nearest_in, [0.5, 0.5])

    def test_zero_width_volume(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(1.0, 1.0), (0.0, 1.0)],
            seed_config=np.array([1.0, 0.5]),
        )
        # 零宽度维度（固定关节）不参与体积计算
        # 有效维度 [0.0, 1.0] 宽度 1.0 → 体积 = 1.0
        assert box.volume == 1.0

    def test_all_zero_width_volume(self):
        box = BoxNode(
            node_id=0,
            joint_intervals=[(1.0, 1.0), (2.0, 2.0)],
            seed_config=np.array([1.0, 2.0]),
        )
        # 全部维度零宽度 → 体积 = 0.0
        assert box.volume == 0.0


class TestBoxTree:
    """BoxTree 数据模型测试"""

    def test_basic_tree(self):
        root = BoxNode(
            node_id=0,
            joint_intervals=[(0.0, 1.0), (0.0, 1.0)],
            seed_config=np.array([0.5, 0.5]),
        )
        tree = BoxTree(tree_id=0, nodes={0: root}, root_id=0)
        assert tree.n_nodes == 1
        assert tree.total_volume == pytest.approx(1.0)

    def test_leaf_nodes(self):
        root = BoxNode(node_id=0, joint_intervals=[(0, 1), (0, 1)],
                        seed_config=np.array([0.5, 0.5]), children_ids=[1])
        child = BoxNode(node_id=1, joint_intervals=[(1, 2), (1, 2)],
                         seed_config=np.array([1.5, 1.5]), parent_id=0)
        tree = BoxTree(tree_id=0, nodes={0: root, 1: child}, root_id=0)
        leaves = tree.get_leaf_nodes()
        assert len(leaves) == 1
        assert leaves[0].node_id == 1


class TestBoxTreeManager:
    """BoxTreeManager 测试"""

    def test_create_tree(self):
        manager = BoxTreeManager()
        root = BoxNode(
            node_id=manager.allocate_node_id(),
            joint_intervals=[(0, 1), (0, 1)],
            seed_config=np.array([0.5, 0.5]),
        )
        tree_id = manager.create_tree(root)
        assert tree_id == 0
        assert manager.n_trees == 1
        assert manager.total_nodes == 1

    def test_add_box(self):
        manager = BoxTreeManager()
        nid = manager.allocate_node_id()
        root = BoxNode(node_id=nid, joint_intervals=[(0, 1), (0, 1)],
                        seed_config=np.array([0.5, 0.5]))
        tid = manager.create_tree(root)

        child_id = manager.allocate_node_id()
        child = BoxNode(node_id=child_id, joint_intervals=[(0.5, 1.5), (0.5, 1.5)],
                         seed_config=np.array([1.0, 1.0]))
        manager.add_box(tid, child, parent_id=nid)

        assert manager.total_nodes == 2
        tree = manager.get_tree(tid)
        assert child_id in tree.nodes
        assert child.parent_id == nid

    def test_find_containing_box(self):
        manager = BoxTreeManager()
        nid = manager.allocate_node_id()
        root = BoxNode(node_id=nid, joint_intervals=[(0, 2), (0, 2)],
                        seed_config=np.array([1.0, 1.0]))
        manager.create_tree(root)

        found = manager.find_containing_box(np.array([1.0, 1.0]))
        assert found is not None
        assert found.node_id == nid

        not_found = manager.find_containing_box(np.array([5.0, 5.0]))
        assert not_found is None

    def test_find_nearest_box(self):
        manager = BoxTreeManager()
        nid0 = manager.allocate_node_id()
        box0 = BoxNode(node_id=nid0, joint_intervals=[(0, 1), (0, 1)],
                        seed_config=np.array([0.5, 0.5]))
        manager.create_tree(box0)

        nid1 = manager.allocate_node_id()
        box1 = BoxNode(node_id=nid1, joint_intervals=[(3, 4), (3, 4)],
                        seed_config=np.array([3.5, 3.5]))
        manager.create_tree(box1)

        # (2, 2) 离 box1 更近
        nearest = manager.find_nearest_box(np.array([2.5, 2.5]))
        assert nearest is not None
        assert nearest.node_id == nid1

    def test_boundary_samples(self):
        manager = BoxTreeManager()
        nid = manager.allocate_node_id()
        root = BoxNode(node_id=nid, joint_intervals=[(0, 2), (0, 2)],
                        seed_config=np.array([1.0, 1.0]))
        tid = manager.create_tree(root)

        rng = np.random.default_rng(42)
        samples = manager.get_boundary_samples(tid, n_samples=10, rng=rng)
        assert len(samples) == 10

        for s in samples:
            assert len(s) == 2
            # 至少一个维度应在边界上 (0 或 2)
            on_boundary = any(
                abs(s[i] - 0.0) < 1e-10 or abs(s[i] - 2.0) < 1e-10
                for i in range(2)
            )
            assert on_boundary, f"sample {s} not on boundary"

    def test_get_tree_for_box(self):
        manager = BoxTreeManager()
        nid = manager.allocate_node_id()
        root = BoxNode(node_id=nid, joint_intervals=[(0, 1), (0, 1)],
                        seed_config=np.array([0.5, 0.5]))
        tid = manager.create_tree(root)

        result = manager.get_tree_for_box(nid)
        assert result == tid

        result_none = manager.get_tree_for_box(999)
        assert result_none is None

    def test_multiple_trees(self):
        manager = BoxTreeManager()
        for i in range(3):
            nid = manager.allocate_node_id()
            box = BoxNode(node_id=nid, joint_intervals=[(i, i + 1), (i, i + 1)],
                           seed_config=np.array([i + 0.5, i + 0.5]))
            manager.create_tree(box)

        assert manager.n_trees == 3
        assert manager.total_nodes == 3
        assert len(manager.get_all_trees()) == 3
        assert len(manager.get_all_boxes()) == 3
