import json
import time
import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from forest.models import BoxNode
from forest.deoverlap import compute_adjacency, compute_adjacency_reference
from common.output import make_output_dir


def _make_boxes(n: int, dims: int = 6, seed: int = 42) -> list[BoxNode]:
    rng = np.random.default_rng(seed)
    boxes: list[BoxNode] = []
    for i in range(n):
        center = rng.uniform(-5.0, 5.0, size=dims)
        half = rng.uniform(0.02, 0.20, size=dims)
        ivs = [(float(c - h), float(c + h)) for c, h in zip(center, half)]
        boxes.append(
            BoxNode(
                node_id=i,
                joint_intervals=ivs,
                seed_config=np.array(center, dtype=np.float64),
                tree_id=0,
            )
        )
    return boxes


def _timeit(fn, *args, repeats: int = 3, **kwargs) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        samples.append(time.perf_counter() - t0)
    return sum(samples) / len(samples), samples


def main() -> None:
    boxes = _make_boxes(600, dims=6)

    mean_ref, raw_ref = _timeit(compute_adjacency_reference, boxes, tol=1e-8, repeats=3)
    mean_new, raw_new = _timeit(
        compute_adjacency,
        boxes,
        tol=1e-8,
        chunk_threshold=300,
        chunk_size=64,
        repeats=3,
    )

    speedup = (mean_ref / mean_new) if mean_new > 0 else float("inf")
    result = {
        "n_boxes": 600,
        "dims": 6,
        "reference_mean_s": mean_ref,
        "reference_raw_s": raw_ref,
        "vectorized_mean_s": mean_new,
        "vectorized_raw_s": raw_new,
        "speedup": speedup,
    }

    out = make_output_dir("benchmarks", "forest_adjacency_vectorized")
    out_file = out / "result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_file)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
