import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from planner.models import PlannerConfig
from planner.box_rrt import BoxRRT
from common.output import make_output_dir


def main() -> None:
    robot = load_robot("2dof_planar")
    scene = Scene()
    scene.add_obstacle([0.6, -0.2], [0.9, 0.2], "obs")

    planner = BoxRRT(robot, scene, PlannerConfig(max_iterations=50, max_box_nodes=40))
    q_start = np.array([-1.0, 0.0], dtype=np.float64)
    q_goal = np.array([1.0, 0.0], dtype=np.float64)
    result = planner.plan(q_start, q_goal, seed=42)

    out = make_output_dir("plans", "planning_demo")
    print(f"success={result.success}, msg={result.message}")
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()
