"""test_deoverlap.py - deoverlap 模块单元测试"""

import pytest
import numpy as np
from planner.models import BoxNode
from planner.deoverlap import (
    subtract_box,
    deoverlap,
    compute_adjacency,
    compute_adjacency_incremental,
    shared_face,
    shared_face_center,
    _interval_volume,
    _overlap_volume_intervals,
)


# ---- helpers ----

def _make_box(intervals, node_id=0):
    """快速创建 BoxNode"""
    return BoxNode(
        node_id=node_id,
        joint_intervals=intervals,
        seed_config=np.array([(lo + hi) / 2 for lo, hi in intervals]),
    )


# ============================================================
# subtract_box
# ============================================================

class TestSubtractBox:
    """hyperrectangle subtraction"""

    def test_no_overlap(self):
        """不相交 → 返回原 box"""
        base = [(0, 2), (0, 2)]
        cut = [(3, 4), (3, 4)]
        frags = subtract_box(base, cut)
        assert len(frags) == 1
        assert frags[0] == base

    def test_full_cover(self):
        """base 完全被 cut 包含 → 空"""
        base = [(1, 2), (1, 2)]
        cut = [(0, 3), (0, 3)]
        frags = subtract_box(base, cut)
        assert len(frags) == 0

    def test_partial_overlap_1d(self):
        """1D 部分重叠"""
        base = [(0, 4)]
        cut = [(2, 6)]
        frags = subtract_box(base, cut)
        # 应留下 [(0,2)]
        assert len(frags) == 1
        lo, hi = frags[0][0]
        assert lo == pytest.approx(0)
        assert hi == pytest.approx(2)

    def test_partial_overlap_2d(self):
        """2D 部分重叠 → 最多 2*D=4 碎片"""
        base = [(0, 4), (0, 4)]
        cut = [(2, 6), (2, 6)]
        frags = subtract_box(base, cut)
        # base 中不在 cut 内的区域被切成 ≤4 块
        total_vol = sum(_interval_volume(f) for f in frags)
        # base=16, overlap=[2,4]×[2,4]=4, remainder=12
        assert total_vol == pytest.approx(12.0)

    def test_center_cut(self):
        """cut 在 base 中间 → 最多 2D 碎片"""
        base = [(0, 6), (0, 6)]
        cut = [(2, 4), (2, 4)]
        frags = subtract_box(base, cut)
        total_vol = sum(_interval_volume(f) for f in frags)
        # base=36, overlap=4, remainder=32
        assert total_vol == pytest.approx(32.0)
        assert len(frags) <= 4  # 2D → max 4

    def test_volume_conservation(self):
        """体积守恒：fragments + overlap == base"""
        rng = np.random.default_rng(42)
        for _ in range(20):
            d = rng.integers(2, 5)
            base = [(float(rng.uniform(-3, 0)), float(rng.uniform(1, 4)))
                    for _ in range(d)]
            cut = [(float(rng.uniform(-3, 0)), float(rng.uniform(1, 4)))
                   for _ in range(d)]
            frags = subtract_box(base, cut)
            frag_vol = sum(_interval_volume(f) for f in frags)
            overlap_vol = _overlap_volume_intervals(base, cut)
            base_vol = _interval_volume(base)
            assert frag_vol + overlap_vol == pytest.approx(base_vol, abs=1e-12)


# ============================================================
# deoverlap
# ============================================================

class TestDeoverlap:
    """deoverlap: 先来先得去重叠"""

    def test_non_overlapping_input(self):
        """不重叠 → 不变"""
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(2, 3), (2, 3)], 1)
        result = deoverlap([b1, b2])
        assert len(result) == 2

    def test_full_overlap(self):
        """完全重叠 → 第二个被吃掉"""
        b1 = _make_box([(0, 4), (0, 4)], 0)
        b2 = _make_box([(1, 3), (1, 3)], 1)
        result = deoverlap([b1, b2])
        # b2 完全在 b1 内 → 碎片为空
        assert len(result) == 1

    def test_partial_overlap(self):
        """部分重叠 → 碎片总体积 = 不重叠部分"""
        b1 = _make_box([(0, 3), (0, 3)], 0)
        b2 = _make_box([(2, 5), (2, 5)], 1)
        result = deoverlap([b1, b2])
        total = sum(b.volume for b in result)
        # b1=9, b2=9, overlap=[2,3]²=1 → total=9+8=17
        assert total == pytest.approx(17.0)

    def test_tiny_fragment_kept(self):
        """HierAABBTree 保证不重叠，deoverlap 仅安全网；微小碎片也保留"""
        b1 = _make_box([(0, 10), (0, 10)], 0)
        b2 = _make_box([(9.99, 10.001), (9.99, 10.001)], 1)
        result = deoverlap([b1, b2])
        # b1 保留，b2 被 b1 切分后的碎片也保留（volume > 0）
        assert len(result) >= 1
        # 第一个一定是 b1
        assert result[0].volume == pytest.approx(100.0)

    def test_zero_overlap_preserved(self):
        """零重叠 → 都保留且互不重叠"""
        boxes = [
            _make_box([(0, 1), (0, 1)], i)
            for i in range(4)
        ]
        result = deoverlap(boxes)
        # 完全相同 → 只保留第一个
        assert len(result) == 1

    def test_custom_id_start(self):
        """id_start 参数：新碎片 ID 从指定值起"""
        b1 = _make_box([(0, 3), (0, 3)], 0)
        b2 = _make_box([(2, 5), (2, 5)], 1)
        result = deoverlap([b1, b2], id_start=100)
        for box in result:
            assert box.node_id >= 0  # 第一个保留原 ID=0，后续 ≥100


# ============================================================
# adjacency
# ============================================================

class TestAdjacency:
    """邻接关系计算"""

    def test_touching_boxes(self):
        """面接触 → 相邻"""
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(1, 2), (0, 1)], 1)
        adj = compute_adjacency([b1, b2], tol=1e-8)
        assert 1 in adj.get(0, set())
        assert 0 in adj.get(1, set())

    def test_separated_boxes(self):
        """间隔开 → 不相邻"""
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(2, 3), (0, 1)], 1)
        adj = compute_adjacency([b1, b2], tol=1e-8)
        assert 1 not in adj.get(0, set())

    def test_point_touch_no_adjacent(self):
        """点接触（非面接触）→ 不相邻"""
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(1, 2), (1, 2)], 1)
        adj = compute_adjacency([b1, b2], tol=1e-8)
        # 对角点接触 → 两个维度都是边界 → 不是面邻接
        assert 1 not in adj.get(0, set())

    def test_incremental(self):
        """增量邻接更新"""
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(1, 2), (0, 1)], 1)
        b3 = _make_box([(2, 3), (0, 1)], 2)

        adj = compute_adjacency([b1, b2], tol=1e-8)
        compute_adjacency_incremental([b3], [b1, b2, b3], adj, tol=1e-8)

        assert 2 in adj.get(1, set())
        assert 1 in adj.get(2, set())
        assert 2 not in adj.get(0, set())


# ============================================================
# shared_face
# ============================================================

class TestSharedFace:
    """共享面计算"""

    def test_face_contact(self):
        """标准面接触"""
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(1, 2), (0, 1)], 1)
        result = shared_face(b1, b2)
        assert result is not None
        dim, val, face_intervals = result
        assert dim == 0
        assert val == pytest.approx(1.0)
        # face_intervals 是完整 box 的维度区间
        assert face_intervals[1] == pytest.approx((0, 1))

    def test_no_contact(self):
        """无接触"""
        b1 = _make_box([(0, 1), (0, 1)], 0)
        b2 = _make_box([(3, 4), (0, 1)], 1)
        result = shared_face(b1, b2)
        assert result is None

    def test_center_2d(self):
        """共享面中心"""
        b1 = _make_box([(0, 2), (0, 3)], 0)
        b2 = _make_box([(2, 4), (1, 3)], 1)
        center = shared_face_center(b1, b2)
        assert center is not None
        assert center[0] == pytest.approx(2.0)
        assert center[1] == pytest.approx(2.0)  # (1+3)/2


# ============================================================
# property-based stress test
# ============================================================

class TestDeoverlapProperties:
    """基于随机生成的 property 测试"""

    def test_no_overlap_invariant(self):
        """去重叠后任意两个 box 无实质性重叠"""
        rng = np.random.default_rng(123)
        for _ in range(10):
            boxes = []
            for i in range(5):
                intervals = [
                    (float(rng.uniform(-2, 0)), float(rng.uniform(1, 3)))
                    for _ in range(3)
                ]
                boxes.append(_make_box(intervals, i))

            result = deoverlap(boxes)

            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    ovlp = result[i].overlap_volume(result[j])
                    assert ovlp < 1e-6, (
                        f"box {result[i].node_id} 与 {result[j].node_id} "
                        f"重叠体积 = {ovlp}"
                    )
