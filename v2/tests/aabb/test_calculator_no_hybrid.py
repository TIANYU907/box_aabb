import pytest

from aabb.calculator import AABBCalculator
from aabb.robot import load_robot


def test_calculator_rejects_hybrid_sampling() -> None:
    robot = load_robot("panda")
    calc = AABBCalculator(robot)

    with pytest.raises(ValueError):
        calc.compute_envelope([(-0.1, 0.1)] * robot.n_joints, sampling="hybrid")
