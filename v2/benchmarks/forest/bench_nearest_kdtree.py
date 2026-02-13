import time
import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from forest.models import BoxNode
from forest.box_forest import BoxForest
from common.output import make_output_dir


def main() -> None:
    forest = BoxForest("demo", [(-1, 1), (-1, 1), (-1, 1)])
    for i in range(500):
        c = -1.0 + 2.0 * i / 500.0
        forest.add_box_direct(BoxNode(i, [(c - 0.01, c + 0.01), (-0.5, 0.5), (-0.5, 0.5)], np.array([c, 0, 0])))

    q = np.array([0.123, 0.0, 0.0])
    t0 = time.perf_counter()
    for _ in range(5000):
        forest.find_nearest(q)
    dt = time.perf_counter() - t0

    out = make_output_dir("benchmarks", "forest_kdtree")
    p = out / "result.txt"
    p.write_text(f"boxes=500\nqueries=5000\ntime={dt:.6f}\n", encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main()
