"""
test_models.py — Unit tests for models.py data classes.

Covers:
    - BoundaryConfig construction & format_joint_values
    - LinkAABBInfo volume / size properties
    - AABBEnvelopeResult aggregation helpers
    - AABBResult (legacy compat)
"""

import numpy as np
import pytest

from box_aabb.models import (
    BoundaryConfig,
    LinkAABBInfo,
    AABBEnvelopeResult,
    AABBResult,
)


class TestBoundaryConfig:

    def test_auto_ndarray(self):
        bc = BoundaryConfig(
            joint_values=[1.0, 2.0],
            boundary_value=0.5,
            boundary_type='x_min',
            link_index=1,
        )
        assert isinstance(bc.joint_values, np.ndarray)

    def test_format_joint_values(self):
        bc = BoundaryConfig(
            joint_values=np.array([0.0, 0.3, -0.5]),
            boundary_value=0.5,
            boundary_type='x_min',
            link_index=1,
            relevant_joints={0, 1, 2},
            boundary_joints={2},
        )
        intervals = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
        text = bc.format_joint_values(intervals)
        assert 'q0' in text
        assert '📍' in text  # q2 is at boundary (-0.5)


class TestLinkAABBInfo:

    def test_volume_unit_cube(self):
        info = LinkAABBInfo(
            link_index=1, link_name="L1",
            min_point=[0, 0, 0], max_point=[1, 1, 1],
        )
        assert info.volume == pytest.approx(1.0)

    def test_volume_flat(self):
        """Zero-thickness AABB → volume 0."""
        info = LinkAABBInfo(
            link_index=1, link_name="L1",
            min_point=[0, 0, 0], max_point=[1, 1, 0],
        )
        assert info.volume == pytest.approx(0.0)

    def test_size(self):
        info = LinkAABBInfo(
            link_index=1, link_name="L1",
            min_point=[-1, -2, -3], max_point=[1, 2, 3],
        )
        assert info.size == [pytest.approx(2), pytest.approx(4), pytest.approx(6)]

    def test_zero_length_flag(self):
        info = LinkAABBInfo(
            link_index=1, link_name="L1",
            min_point=[0, 0, 0], max_point=[0, 0, 0],
            is_zero_length=True,
        )
        assert info.is_zero_length
        assert info.volume == 0.0


class TestAABBEnvelopeResult:

    @pytest.fixture()
    def sample_result(self):
        aabbs = [
            LinkAABBInfo(1, "L1", [0, 0, 0], [1, 1, 1]),
            LinkAABBInfo(2, "L2", [0, 0, 0], [0, 0, 0], is_zero_length=True),
            LinkAABBInfo(3, "L3", [1, 1, 1], [2, 2, 2]),
        ]
        return AABBEnvelopeResult(
            robot_name="Test",
            n_joints=3,
            joint_intervals=[(-1, 1)] * 3,
            method="numerical_critical",
            link_aabbs=aabbs,
        )

    def test_total_volume_skips_zero(self, sample_result):
        """total_volume should skip zero-length links."""
        assert sample_result.total_volume() == pytest.approx(2.0)

    def test_get_robot_aabb(self, sample_result):
        mn, mx = sample_result.get_robot_aabb()
        assert mn == [pytest.approx(0)] * 3
        assert mx == [pytest.approx(2)] * 3

    def test_get_end_effector_aabb(self, sample_result):
        ee = sample_result.get_end_effector_aabb()
        assert ee is not None
        assert ee.link_index == 3

    def test_get_link_aabbs(self, sample_result):
        assert len(sample_result.get_link_aabbs(1)) == 1
        assert len(sample_result.get_link_aabbs(99)) == 0

    def test_generate_report_returns_string(self, sample_result):
        report = sample_result.generate_report()
        assert isinstance(report, str)
        assert "AABB" in report


class TestAABBResult:

    def test_volume(self):
        r = AABBResult(min_point=[0, 0, 0], max_point=[2, 3, 4],
                       method="test")
        assert r.volume == pytest.approx(24.0)

    def test_size(self):
        r = AABBResult(min_point=[1, 1, 1], max_point=[3, 4, 5],
                       method="test")
        assert r.size == [pytest.approx(2), pytest.approx(3), pytest.approx(4)]
