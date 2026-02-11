"""
conftest.py — pytest fixtures shared across the test suite.

Provides pre-configured Robot, AABBCalculator, and standard joint intervals
so that individual test modules stay short and focused.
"""

import math
import pytest
import numpy as np

from box_aabb.robot import Robot, create_panda_robot, PANDA_JOINT_LIMITS
from box_aabb.aabb_calculator import AABBCalculator


# =========================================================================
# Robot fixtures
# =========================================================================

@pytest.fixture(scope="session")
def panda_robot() -> Robot:
    """Franka Emika Panda robot (7 DOF + tool_frame)."""
    return create_panda_robot()


@pytest.fixture(scope="session")
def simple_2dof_robot() -> Robot:
    """Minimal 2-DOF planar robot for fast unit tests."""
    dh = [
        {"alpha": 0, "a": 1.0, "d": 0, "theta": 0, "type": "revolute"},
        {"alpha": 0, "a": 1.0, "d": 0, "theta": 0, "type": "revolute"},
    ]
    return Robot(dh)


@pytest.fixture(scope="session")
def simple_3dof_robot() -> Robot:
    """3-DOF spatial robot (non-zero alpha) for broader coverage."""
    dh = [
        {"alpha": 0,            "a": 0.0, "d": 0.5, "theta": 0, "type": "revolute"},
        {"alpha": -math.pi / 2, "a": 0.0, "d": 0.0, "theta": 0, "type": "revolute"},
        {"alpha": 0,            "a": 0.4, "d": 0.0, "theta": 0, "type": "revolute"},
    ]
    return Robot(dh)


# =========================================================================
# Calculator fixtures
# =========================================================================

@pytest.fixture(scope="session")
def panda_calc(panda_robot) -> AABBCalculator:
    """AABBCalculator wrapping the Panda robot."""
    return AABBCalculator(panda_robot, robot_name="Panda")


@pytest.fixture(scope="session")
def simple_calc(simple_2dof_robot) -> AABBCalculator:
    """AABBCalculator wrapping the 2-DOF robot (skip_first_link=False)."""
    return AABBCalculator(simple_2dof_robot, robot_name="2DOF",
                          skip_first_link=False)


# =========================================================================
# Joint interval fixtures
# =========================================================================

@pytest.fixture()
def narrow_panda_intervals():
    """Small ±0.5 rad intervals for all 7 joints."""
    return [(-0.5, 0.5)] * 7


@pytest.fixture()
def narrow_2dof_intervals():
    """±0.3 rad for both joints of the 2-DOF robot."""
    return [(-0.3, 0.3), (-0.3, 0.3)]
