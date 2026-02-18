import json
import time
import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from forest.box_forest import BoxForest
from forest.models import BoxNode, PlannerConfig
from common.output import make_output_dir


def _make_box(node_id: int, lo: float, hi: float, dims: int = 6) -> BoxNode:
    intervals = [(lo, hi)] + [(-0.2, 0.2)] * (dims - 1)
    return BoxNode(
        node_id=node_id,
        joint_intervals=intervals,
        seed_config=np.zeros(dims, dtype=np.float64),
        tree_id=0,
    )


def _timeit(fn, repeats: int = 3) -> tuple[float, list[float]]:
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(sum(vals) / len(vals)), vals


def run_once(n_boxes: int = 1200, dims: int = 6) -> dict:
    forest = BoxForest(
        robot_fingerprint="bench_dummy",
        joint_limits=[(-2.0, 2.0)] * dims,
        config=PlannerConfig(adjacency_tolerance=1e-8),
    )

    # 按轴串接，保证相邻箱体面接触（邻接图非空）
    width = 0.01
    gap = 0.0

    def _work() -> None:
        forest.boxes.clear()
        forest.adjacency.clear()
        forest._intervals_arr = np.empty((0, 0, 2), dtype=np.float64)
        forest._interval_ids = []
        forest._interval_id_to_index = {}
        forest._next_id = 0

        for i in range(n_boxes):
            lo = i * (width + gap)
            hi = lo + width
            box = _make_box(i, lo, hi, dims=dims)
            forest.add_box_direct(box)

    mean_s, raw_s = _timeit(_work)

    # 执行一次获得最终结构统计
    _work()

    return {
        "n_boxes": n_boxes,
        "dims": dims,
        "mean_s": mean_s,
        "raw_s": raw_s,
        "boxes": forest.n_boxes,
        "adj_edges": sum(len(v) for v in forest.adjacency.values()) // 2,
        "interval_cache_shape": list(forest._intervals_arr.shape),
    }


def main() -> None:
    result = run_once()
    out = make_output_dir("benchmarks", "forest_incremental_add")
    out_file = out / "result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(out_file)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
