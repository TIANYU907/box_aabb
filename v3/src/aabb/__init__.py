"""v3 AABB package."""

from .robot import Robot, load_robot
from .models import AABBEnvelopeResult, LinkAABBInfo, BoundaryConfig
from .calculator import AABBCalculator
from .interval_fk import compute_interval_aabb, compute_fk_full, compute_fk_incremental, _split_fk_pair

__all__ = [
    "Robot",
    "load_robot",
    "AABBEnvelopeResult",
    "LinkAABBInfo",
    "BoundaryConfig",
    "AABBCalculator",
    "compute_interval_aabb",
    "compute_fk_full",
    "compute_fk_incremental",
    "get_link_positions_batch",
]


def get_link_positions_batch(robot: Robot, joint_values_batch, link_idx: int):
    """便捷导出：调用 Robot.get_link_positions_batch。"""
    return robot.get_link_positions_batch(joint_values_batch, link_idx)
