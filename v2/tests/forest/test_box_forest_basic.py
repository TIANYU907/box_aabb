import numpy as np

from forest.box_forest import BoxForest
from forest.models import BoxNode, PlannerConfig


def _mk_box(node_id: int, lo: float, hi: float) -> BoxNode:
    return BoxNode(
        node_id=node_id,
        joint_intervals=[(lo, hi), (lo, hi), (lo, hi)],
        seed_config=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tree_id=0,
    )


def test_box_forest_add_and_find() -> None:
    forest = BoxForest(
        robot_fingerprint="dummy",
        joint_limits=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        config=PlannerConfig(adjacency_tolerance=1e-8),
    )

    forest.add_box_direct(_mk_box(0, -0.5, 0.0))
    forest.add_box_direct(_mk_box(1, 0.0, 0.5))

    assert forest.n_boxes == 2
    assert forest.find_containing(np.array([-0.1, -0.1, -0.1], dtype=np.float64)) is not None
    assert forest.find_nearest(np.array([0.9, 0.9, 0.9], dtype=np.float64)) is not None
