import json
import time
import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import load_robot
import aabb.robot as robot_mod
from common.output import make_output_dir


def _timeit(fn, repeats: int = 5):
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(sum(vals) / len(vals)), vals


def main() -> None:
    robot = load_robot("panda")
    rng = np.random.default_rng(42)

    n = 5000
    link_idx = robot.n_joints
    lows = np.array([lo for lo, _ in robot.joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in robot.joint_limits], dtype=np.float64)
    q_batch = rng.uniform(lows, highs, size=(n, robot.n_joints))

    def run_python() -> None:
        for i in range(n):
            robot._get_link_position_python(q_batch[i].tolist(), link_idx)

    def run_dispatch() -> None:
        for i in range(n):
            robot.get_link_position(q_batch[i], link_idx)

    mean_py, raw_py = _timeit(run_python)
    mean_dispatch, raw_dispatch = _timeit(run_dispatch)

    result = {
        "n_samples": n,
        "link_idx": link_idx,
        "python_mean_s": mean_py,
        "python_raw_s": raw_py,
        "dispatch_mean_s": mean_dispatch,
        "dispatch_raw_s": raw_dispatch,
        "speedup_dispatch_over_python": (mean_py / mean_dispatch) if mean_dispatch > 0 else float("inf"),
        "cython_available": bool(getattr(robot_mod, "_link_position_core_cy", None) is not None),
    }

    out = make_output_dir("benchmarks", "aabb_fk_scalar_cython")
    out_file = out / "result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_file)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
