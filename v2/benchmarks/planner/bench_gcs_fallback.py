import time
import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from forest.models import BoxNode
from planner.gcs_optimizer import GCSOptimizer
from common.output import make_output_dir


def main() -> None:
    optimizer = GCSOptimizer(fallback=True)
    boxes = {
        0: BoxNode(0, [(-1.0, 0.0), (-1.0, 1.0)], np.array([-0.5, 0.0])),
        1: BoxNode(1, [(0.0, 1.0), (-1.0, 1.0)], np.array([0.5, 0.0])),
    }
    graph = {
        "start": "start",
        "goal": "goal",
        "edges": {
            "start": [(0, 0.1, None)],
            0: [(1, 1.0, None)],
            1: [("goal", 0.1, None)],
            "goal": [],
        },
    }

    q_start = np.array([-0.2, 0.0])
    q_goal = np.array([0.2, 0.0])

    t0 = time.perf_counter()
    for _ in range(1000):
        optimizer.optimize(graph, boxes, q_start, q_goal)
    dt = time.perf_counter() - t0

    out = make_output_dir("benchmarks", "planner_gcs_fallback")
    p = out / "result.txt"
    p.write_text(f"runs=1000\ntime={dt:.6f}\n", encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main()
