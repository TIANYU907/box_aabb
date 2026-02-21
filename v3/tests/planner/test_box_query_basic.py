import numpy as np

from aabb.robot import load_robot
from forest.safe_box_forest import SafeBoxForest
from forest.models import BoxNode
from forest.scene import Scene
from planner.sbf_query import SBFQuery


def test_sbf_query_plan_runs() -> None:
    robot = load_robot("2dof_planar")
    scene = Scene()
    forest = SafeBoxForest(
        robot_fingerprint=robot.fingerprint(),
        joint_limits=[(-3.14, 3.14), (-3.14, 3.14)],
    )
    forest.add_box_direct(BoxNode(0, [(-2.0, 0.0), (-1.0, 1.0)], np.array([-1.0, 0.0])))
    forest.add_box_direct(BoxNode(1, [(0.0, 2.0), (-1.0, 1.0)], np.array([1.0, 0.0])))

    query = SBFQuery(forest=forest, robot=robot, scene=scene)
    result = query.plan(np.array([-1.5, 0.0]), np.array([1.5, 0.0]), seed=1)

    assert isinstance(result.success, bool)
    assert result.path is not None
