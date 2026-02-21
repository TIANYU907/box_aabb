import numpy as np

from forest.models import BoxNode
from forest.deoverlap import compute_adjacency


def _make_boxes(n: int, dims: int = 3) -> list[BoxNode]:
    rng = np.random.default_rng(123)
    boxes: list[BoxNode] = []
    for i in range(n):
        center = rng.uniform(-3.0, 3.0, size=dims)
        half = rng.uniform(0.05, 0.25, size=dims)
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


def test_compute_adjacency_matches_reference_small() -> None:
    """Non-chunked vs chunked should agree for small box sets."""
    boxes = _make_boxes(80, dims=4)
    # Force non-chunked (high threshold) vs chunked (low threshold)
    non_chunked = compute_adjacency(boxes, tol=1e-8, chunk_threshold=10000, chunk_size=64)
    chunked = compute_adjacency(boxes, tol=1e-8, chunk_threshold=1, chunk_size=32)
    assert chunked == non_chunked


def test_compute_adjacency_matches_reference_chunked() -> None:
    """Non-chunked vs chunked should agree for larger box sets."""
    boxes = _make_boxes(220, dims=3)
    non_chunked = compute_adjacency(boxes, tol=1e-8, chunk_threshold=10000, chunk_size=64)
    chunked = compute_adjacency(boxes, tol=1e-8, chunk_threshold=100, chunk_size=32)
    assert chunked == non_chunked
