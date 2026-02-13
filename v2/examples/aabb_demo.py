import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import load_robot
from aabb.calculator import AABBCalculator
from common.output import make_output_dir


def main() -> None:
    robot = load_robot("panda")
    calc = AABBCalculator(robot)
    intervals = [(-0.2, 0.2)] * robot.n_joints
    result = calc.compute_envelope(intervals, method="interval")

    out = make_output_dir("reports", "aabb_demo")
    report_path = out / "aabb_report.md"
    result.generate_report(str(report_path))
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
