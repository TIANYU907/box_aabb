"""v2 AABB package."""

from .robot import Robot, create_panda_robot, load_robot, PANDA_JOINT_LIMITS
from .models import AABBEnvelopeResult, LinkAABBInfo, BoundaryConfig
from .calculator import AABBCalculator
from .report import ReportGenerator
from .visualizer import Visualizer, visualize_envelope_result
from .interval_fk import compute_interval_aabb, compute_fk_full, compute_fk_incremental

__all__ = [
    "Robot",
    "create_panda_robot",
    "load_robot",
    "PANDA_JOINT_LIMITS",
    "AABBEnvelopeResult",
    "LinkAABBInfo",
    "BoundaryConfig",
    "AABBCalculator",
    "ReportGenerator",
    "Visualizer",
    "visualize_envelope_result",
    "compute_interval_aabb",
    "compute_fk_full",
    "compute_fk_incremental",
    "get_link_positions_batch",
]


def get_link_positions_batch(robot: Robot, joint_values_batch, link_idx: int):
    """便捷导出：调用 Robot.get_link_positions_batch。"""
    return robot.get_link_positions_batch(joint_values_batch, link_idx)
