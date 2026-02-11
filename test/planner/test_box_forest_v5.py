"""test_box_forest.py - BoxForest 模块单元测试"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock

from planner.models import BoxNode, PlannerConfig
from planner.box_forest import BoxForest


# ---- helpers ----

def _make_box(intervals, node_id=0):
    return BoxNode(
        node_id=node_id,
        joint_intervals=intervals,
        seed_config=np.array([(lo + hi) / 2 for lo, hi in intervals]),
    )


def _mock_robot(n_joints=2):
    robot = MagicMock()
    robot.n_joints = n_joints
    robot.fingerprint.return_value = "test_robot_fp"
    robot.joint_limits = [(-3.14, 3.14)] * n_joints
    return robot


# ============================================================
# 基础操作
# ============================================================

class TestBoxForestBasic:
    """BoxForest 基本功能"""

    def test_create_empty(self):
        forest = BoxForest(
            robot_fingerprint="test",
            joint_limits=[(-1, 1), (-1, 1)],
        )
        assert forest.n_boxes == 0
        assert forest.total_volume == 0.0

    def test_allocate_id(self):
        forest = BoxForest("test", [(-1, 1)])
        ids = [forest.allocate_id() for _ in range(5)]
        assert ids == [0, 1, 2, 3, 4]

    def test_add_boxes_non_overlapping(self):
        """添加不重叠的 box"""
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(2, 3), (2, 3)], 1)
        forest.add_boxes([b1, b2])
        assert forest.n_boxes == 2

    def test_add_boxes_overlapping(self):
        """添加重叠的 box → deoverlap"""
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 3), (0, 3)], 0)
        b2 = _make_box([(2, 5), (2, 5)], 1)
        forest.add_boxes([b1, b2])
        # deoverlap 后无实质性重叠
        boxes = list(forest.boxes.values())
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ovlp = boxes[i].overlap_volume(boxes[j])
                assert ovlp < 1e-6

    def test_add_boxes_incremental(self):
        """增量添加"""
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 2), (0, 2)], forest.allocate_id())
        forest.add_boxes([b1])

        b2 = _make_box([(1, 3), (1, 3)], forest.allocate_id())
        added = forest.add_boxes_incremental([b2])
        assert len(added) > 0
        # 检查无重叠
        boxes = list(forest.boxes.values())
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert boxes[i].overlap_volume(boxes[j]) < 1e-6

    def test_remove_boxes(self):
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(1, 2), (0, 1)], 1)
        forest.add_boxes([b1, b2])
        n_before = forest.n_boxes
        forest.remove_boxes({b1.node_id for b1 in list(forest.boxes.values())[:1]})
        assert forest.n_boxes < n_before


# ============================================================
# 查询
# ============================================================

class TestBoxForestQuery:
    """find_containing / find_nearest"""

    def test_find_containing(self):
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b = _make_box([(0, 2), (0, 2)], 0)
        forest.add_boxes([b])
        found = forest.find_containing(np.array([1.0, 1.0]))
        assert found is not None

    def test_find_containing_miss(self):
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b = _make_box([(0, 1), (0, 1)], 0)
        forest.add_boxes([b])
        found = forest.find_containing(np.array([5.0, 5.0]))
        assert found is None

    def test_find_nearest(self):
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(5, 6), (5, 6)], 1)
        forest.add_boxes([b1, b2])
        nearest = forest.find_nearest(np.array([0.5, 0.5]))
        assert nearest is not None


# ============================================================
# 邻接
# ============================================================

class TestBoxForestAdjacency:
    """邻接关系"""

    def test_adjacent_boxes(self):
        """面接触的 box 有邻接边"""
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(1, 2), (0, 1)], 1)
        forest.add_boxes([b1, b2])
        # 应有邻接
        has_adj = False
        for bid, neighbors in forest.adjacency.items():
            if len(neighbors) > 0:
                has_adj = True
                break
        assert has_adj

    def test_no_adjacency_separated(self):
        """分离的 box 无邻接"""
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(3, 4), (3, 4)], 1)
        forest.add_boxes([b1, b2])
        for neighbors in forest.adjacency.values():
            assert len(neighbors) == 0


# ============================================================
# 持久化
# ============================================================

class TestBoxForestPersistence:
    """save / load"""

    def test_save_load_roundtrip(self):
        forest = BoxForest("test_fp", [(-3, 3), (-3, 3)])
        b1 = _make_box([(0, 1), (0, 1)], forest.allocate_id())
        b2 = _make_box([(1, 2), (0, 1)], forest.allocate_id())
        forest.add_boxes([b1, b2])

        robot = _mock_robot(2)
        robot.fingerprint.return_value = "test_fp"

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            forest.save(path)
            loaded = BoxForest.load(path, robot)
            assert loaded.n_boxes == forest.n_boxes
            assert len(loaded.adjacency) == len(forest.adjacency)
        finally:
            os.unlink(path)

    def test_load_fingerprint_mismatch(self):
        forest = BoxForest("fp_a", [(-1, 1)])
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            forest.save(path)
            robot = _mock_robot(1)
            robot.fingerprint.return_value = "fp_b"
            with pytest.raises(ValueError, match="指纹不匹配"):
                BoxForest.load(path, robot)
        finally:
            os.unlink(path)


# ============================================================
# validate_boxes
# ============================================================

class TestBoxForestValidation:
    """惰性碰撞验证"""

    def test_validate_no_collision(self):
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b = _make_box([(0, 1), (0, 1)], 0)
        forest.add_boxes([b])

        checker = MagicMock()
        checker.check_box_collision.return_value = False
        colliding = forest.validate_boxes(checker)
        assert len(colliding) == 0

    def test_validate_with_collision(self):
        forest = BoxForest("test", [(-3, 3), (-3, 3)])
        b = _make_box([(0, 1), (0, 1)], 0)
        forest.add_boxes([b])

        checker = MagicMock()
        checker.check_box_collision.return_value = True
        colliding = forest.validate_boxes(checker)
        assert len(colliding) == forest.n_boxes
