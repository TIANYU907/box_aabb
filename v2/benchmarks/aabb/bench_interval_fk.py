import time

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import load_robot
from aabb.interval_fk import compute_interval_aabb
from common.output import make_output_dir


def main() -> None:
    robot = load_robot("panda")
    intervals = [(-0.2, 0.2)] * robot.n_joints
    t0 = time.perf_counter()
    for _ in range(100):
        compute_interval_aabb(robot, intervals, robot.zero_length_links)
    dt = time.perf_counter() - t0

    out = make_output_dir("benchmarks", "aabb_interval_fk")
    p = out / "result.txt"
    p.write_text(f"runs=100\ntime={dt:.6f}\n", encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main()
