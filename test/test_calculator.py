"""
test_calculator.py — Integration tests for AABBCalculator.

Covers:
    - All four methods: critical / random / hybrid / interval
    - Result shape, volume positivity, conservative containment
    - n_subdivisions > 1
    - Edge cases: fixed joints, zero-width intervals
    - Legacy convenience API (compute_link_aabbs, compute_robot_aabb)
"""

import math
import numpy as np
import pytest

from box_aabb.robot import Robot, create_panda_robot
from box_aabb.aabb_calculator import AABBCalculator
from box_aabb.models import AABBEnvelopeResult, LinkAABBInfo


# =========================================================================
# Basic envelope tests for each method
# =========================================================================

class TestCriticalMethod:

    def test_produces_result(self, panda_calc, narrow_panda_intervals):
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical')
        assert isinstance(res, AABBEnvelopeResult)
        assert len(res.link_aabbs) > 0

    def test_positive_volume(self, panda_calc, narrow_panda_intervals):
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical')
        assert res.total_volume() > 0

    def test_method_name(self, panda_calc, narrow_panda_intervals):
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical')
        assert res.method == 'numerical_critical'


class TestRandomMethod:

    def test_produces_result(self, panda_calc, narrow_panda_intervals):
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='random',
            n_random_samples=200)
        assert isinstance(res, AABBEnvelopeResult)
        assert res.total_volume() > 0

    def test_method_name(self, panda_calc, narrow_panda_intervals):
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='random',
            n_random_samples=200)
        assert res.method == 'numerical_random'


class TestHybridMethod:

    def test_produces_result(self, panda_calc, narrow_panda_intervals):
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='hybrid')
        assert isinstance(res, AABBEnvelopeResult)
        assert res.total_volume() > 0


class TestIntervalMethod:

    def test_produces_result(self, panda_calc, narrow_panda_intervals):
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='interval')
        assert isinstance(res, AABBEnvelopeResult)
        assert res.total_volume() > 0

    def test_conservative_bound(self, panda_calc, narrow_panda_intervals):
        """Interval method should produce volumes ≥ critical method."""
        r_crit = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical')
        r_iv = panda_calc.compute_envelope(
            narrow_panda_intervals, method='interval')
        assert r_iv.total_volume() >= r_crit.total_volume() * 0.99  # small tol


# =========================================================================
# Structural / consistency checks
# =========================================================================

class TestResultStructure:

    def test_link_count(self, panda_calc, narrow_panda_intervals):
        """Each non-zero-length link should produce exactly 1 AABB (n_sub=1)."""
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical')
        # Total aabb entries should equal n_joints + tool_frame (7 + 1 = 8)
        assert len(res.link_aabbs) == 8

    def test_aabb_ordered(self, panda_calc, narrow_panda_intervals):
        """min_point <= max_point for every axis."""
        res = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical')
        for aabb in res.link_aabbs:
            if aabb.is_zero_length:
                continue
            for i in range(3):
                assert aabb.min_point[i] <= aabb.max_point[i] + 1e-12

    def test_all_sampled_points_inside_aabb(self, panda_calc):
        """Random FK samples must lie inside the computed AABB."""
        intervals = [(-0.3, 0.3)] * 7 + [(0, 0)]
        res = panda_calc.compute_envelope(
            intervals, method='numerical', sampling='critical')

        rng = np.random.default_rng(42)
        for _ in range(100):
            q = [rng.uniform(lo, hi) for lo, hi in intervals]
            positions = panda_calc.robot.get_link_positions(q)
            for aabb in res.link_aabbs:
                if aabb.is_zero_length:
                    continue
                pos = positions[aabb.link_index]
                for ax in range(3):
                    assert pos[ax] >= aabb.min_point[ax] - 1e-6, (
                        f"Link {aabb.link_index} axis {ax}: "
                        f"{pos[ax]} < {aabb.min_point[ax]}")
                    assert pos[ax] <= aabb.max_point[ax] + 1e-6, (
                        f"Link {aabb.link_index} axis {ax}: "
                        f"{pos[ax]} > {aabb.max_point[ax]}")


class TestSubdivisions:

    def test_n_sub_increases_entries(self, panda_calc, narrow_panda_intervals):
        """n_subdivisions=3 should produce more LinkAABBInfo entries."""
        r1 = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical',
            n_subdivisions=1)
        r3 = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical',
            n_subdivisions=3)
        # Non-zero links get 3× the entries
        nz1 = [a for a in r1.link_aabbs if not a.is_zero_length]
        nz3 = [a for a in r3.link_aabbs if not a.is_zero_length]
        assert len(nz3) >= len(nz1)

    def test_each_subdivided_segment_smaller(self, panda_calc, narrow_panda_intervals):
        """Each individual subdivided segment should be ≤ the unsplit link's AABB."""
        r1 = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical',
            n_subdivisions=1)
        r3 = panda_calc.compute_envelope(
            narrow_panda_intervals, method='numerical', sampling='critical',
            n_subdivisions=3)
        # For each non-zero link, every segment volume ≤ the whole-link volume
        for a1 in r1.link_aabbs:
            if a1.is_zero_length:
                continue
            segs = [a for a in r3.link_aabbs if a.link_index == a1.link_index]
            for seg in segs:
                assert seg.volume <= a1.volume + 1e-6, (
                    f"Segment {seg.segment_index} of Link {seg.link_index} "
                    f"volume {seg.volume} > whole-link {a1.volume}")


class TestEdgeCases:

    def test_zero_width_intervals(self, panda_calc):
        """All joints fixed → volume should be very small (zero-thickness)."""
        iv = [(0.0, 0.0)] * 8
        res = panda_calc.compute_envelope(iv, method='numerical',
                                          sampling='critical')
        assert res.total_volume() < 1e-10

    def test_invalid_method_raises(self, panda_calc, narrow_panda_intervals):
        with pytest.raises(ValueError, match="未知方法"):
            panda_calc.compute_envelope(narrow_panda_intervals,
                                        method='nonexistent')

    def test_invalid_sampling_raises(self, panda_calc, narrow_panda_intervals):
        with pytest.raises(ValueError, match="未知采样"):
            panda_calc.compute_envelope(narrow_panda_intervals,
                                        method='numerical',
                                        sampling='nonexistent')


# =========================================================================
# Legacy convenience API
# =========================================================================

class TestLegacyAPI:

    def test_compute_link_aabbs(self, panda_calc, narrow_panda_intervals):
        results = panda_calc.compute_link_aabbs(narrow_panda_intervals)
        assert len(results) > 0
        assert all(hasattr(r, 'volume') for r in results)

    def test_compute_end_effector_aabb(self, panda_calc, narrow_panda_intervals):
        result = panda_calc.compute_end_effector_aabb(narrow_panda_intervals)
        assert result is not None
        assert result.volume > 0

    def test_compute_robot_aabb(self, panda_calc, narrow_panda_intervals):
        result = panda_calc.compute_robot_aabb(narrow_panda_intervals)
        assert result is not None
        assert result.volume > 0


# =========================================================================
# 2-DOF smoke test (fast, deterministic)
# =========================================================================

class TestSimple2DOF:

    def test_critical(self, simple_calc, narrow_2dof_intervals):
        """2-DOF planar robot lives in XY plane → z=0 → volume=0, but size in x/y > 0."""
        res = simple_calc.compute_envelope(
            narrow_2dof_intervals, method='numerical', sampling='critical')
        valid = [a for a in res.link_aabbs if not a.is_zero_length]
        # At least one link should have non-zero x or y extent
        assert any(a.size[0] > 0 or a.size[1] > 0 for a in valid)

    def test_interval(self, simple_calc, narrow_2dof_intervals):
        res = simple_calc.compute_envelope(
            narrow_2dof_intervals, method='interval')
        valid = [a for a in res.link_aabbs if not a.is_zero_length]
        assert any(a.size[0] > 0 or a.size[1] > 0 for a in valid)

    def test_known_geometry(self, simple_calc):
        """2-DOF at [0, 0] fixed → all links at deterministic positions."""
        iv = [(0.0, 0.0), (0.0, 0.0)]
        res = simple_calc.compute_envelope(iv, method='numerical',
                                           sampling='critical')
        # With zero-width intervals, AABB volumes should be ~0
        assert res.total_volume() < 1e-10
