import numpy as np

from forest.models import BoxNode
from planner.gcs_optimizer import GCSOptimizer


def test_gcs_fallback_optimize_runs() -> None:
    optimizer = GCSOptimizer(fallback=True)

    b0 = BoxNode(0, [(-1.0, 0.0), (-1.0, 1.0)], np.array([-0.5, 0.0]))
    b1 = BoxNode(1, [(0.0, 1.0), (-1.0, 1.0)], np.array([0.5, 0.0]))

    boxes = {0: b0, 1: b1}
    adjacency_graph = {
        "start": "start",
        "goal": "goal",
        "edges": {
            "start": [(0, 0.1, None)],
            0: [(1, 1.0, None)],
            1: [("goal", 0.1, None)],
            "goal": [],
        }
    }

    q_start = np.array([-0.2, 0.0])
    q_goal = np.array([0.2, 0.0])
    path = optimizer.optimize(adjacency_graph, boxes, q_start, q_goal)

    assert path is not None
    assert len(path) >= 2
