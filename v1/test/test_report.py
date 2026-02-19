"""
test_report.py — Tests for report.py (ReportGenerator).

Covers:
    - Report string structure (headings, tables)
    - Report includes key information
    - generate_report() on AABBEnvelopeResult
"""

import pytest

from box_aabb.models import AABBEnvelopeResult, LinkAABBInfo
from box_aabb.report import ReportGenerator


@pytest.fixture()
def minimal_result():
    aabbs = [
        LinkAABBInfo(1, "Link 1", [-0.1, -0.2, 0.0], [0.1, 0.2, 0.5]),
        LinkAABBInfo(2, "Link 2", [0, 0, 0], [0, 0, 0], is_zero_length=True),
        LinkAABBInfo(3, "Link 3", [0.2, -0.1, 0.1], [0.4, 0.1, 0.6]),
    ]
    return AABBEnvelopeResult(
        robot_name="TestBot",
        n_joints=3,
        joint_intervals=[(-0.5, 0.5), (-0.3, 0.3), (-0.1, 0.1)],
        method="numerical_critical",
        sampling_mode="critical",
        link_aabbs=aabbs,
        n_samples_evaluated=42,
        computation_time=0.123,
    )


class TestReportGenerator:

    def test_returns_string(self, minimal_result):
        report = ReportGenerator.generate(minimal_result)
        assert isinstance(report, str)

    def test_contains_title(self, minimal_result):
        report = ReportGenerator.generate(minimal_result)
        assert "AABB" in report

    def test_contains_robot_name(self, minimal_result):
        report = ReportGenerator.generate(minimal_result)
        assert "TestBot" in report

    def test_contains_joint_table(self, minimal_result):
        report = ReportGenerator.generate(minimal_result)
        assert "q0" in report
        assert "q1" in report
        assert "q2" in report

    def test_skipped_zero_length(self, minimal_result):
        report = ReportGenerator.generate(minimal_result)
        assert "零长度" in report or "跳过" in report or "SKIPPED" in report.upper()

    def test_contains_volume(self, minimal_result):
        report = ReportGenerator.generate(minimal_result)
        assert "体积" in report or "volume" in report.lower()

    def test_via_result_method(self, minimal_result):
        """AABBEnvelopeResult.generate_report() should also work."""
        report = minimal_result.generate_report()
        assert "AABB" in report
