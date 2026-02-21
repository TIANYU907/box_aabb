import numpy as np

from aabb.robot import load_robot
from forest.scene import Scene
from planner.metrics import evaluate_result
from planner.models import SBFResult
from planner.report import PlannerReportGenerator
from planner.models import SBFConfig


def test_metrics_and_report_generate() -> None:
    robot = load_robot("2dof_planar")
    scene = Scene()

    result = SBFResult(success=True)
    result.path = [np.array([-1.0, 0.0]), np.array([1.0, 0.0])]
    result.compute_path_length()

    metrics = evaluate_result(result, robot, scene, robot.joint_limits)
    report = PlannerReportGenerator().generate(
        robot=robot,
        scene=scene,
        config=SBFConfig(),
        q_start=result.path[0],
        q_goal=result.path[-1],
        result=result,
        metrics=metrics,
    )

    assert "Box-RRT" in report
    assert metrics.path_length >= 0.0
