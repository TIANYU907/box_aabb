import json
import time
import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import load_robot
from common.output import make_output_dir


def _timeit(fn, repeats: int = 5) -> tuple[float, list[float]]:
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(sum(vals) / len(vals)), vals


def main() -> None:
    robot = load_robot("panda")
    rng = np.random.default_rng(42)

    n = 2000
    link_idx = robot.n_joints
    lows = np.array([lo for lo, _ in robot.joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in robot.joint_limits], dtype=np.float64)
    q_batch = rng.uniform(lows, highs, size=(n, robot.n_joints))

    def scalar_run() -> None:
        for i in range(n):
            robot.get_link_position(q_batch[i].tolist(), link_idx)

    def batch_run() -> None:
        robot.get_link_positions_batch(q_batch, link_idx)

    mean_scalar, raw_scalar = _timeit(scalar_run)
    mean_batch, raw_batch = _timeit(batch_run)

    result = {
        "robot": "panda",
        "n_samples": n,
        "link_idx": link_idx,
        "scalar_mean_s": mean_scalar,
        "scalar_raw_s": raw_scalar,
        "batch_mean_s": mean_batch,
        "batch_raw_s": raw_batch,
        "speedup": (mean_scalar / mean_batch) if mean_batch > 0 else float("inf"),
    }

    out = make_output_dir("benchmarks", "aabb_fk_batch")
    out_file = out / "result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(out_file)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
