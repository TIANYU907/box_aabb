import numpy as np

from forest.safe_box_forest import SafeBoxForest
from forest.models import BoxNode, SBFConfig


def _mk_box(node_id: int, lo: float, hi: float) -> BoxNode:
    return BoxNode(
        node_id=node_id,
        joint_intervals=[(lo, hi), (-0.25, 0.25), (-0.25, 0.25)],
        seed_config=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tree_id=0,
    )


def test_safe_box_forest_add_and_find() -> None:
    forest = SafeBoxForest(
        robot_fingerprint="dummy",
        joint_limits=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        config=SBFConfig(adjacency_tolerance=1e-8),
    )

    forest.add_box_direct(_mk_box(0, -0.5, 0.0))
    forest.add_box_direct(_mk_box(1, 0.0, 0.5))

    assert forest.n_boxes == 2
    assert 1 in forest.adjacency[0]
    assert 0 in forest.adjacency[1]
    assert forest._intervals_len == 2
    assert len(forest._interval_ids) == 2
    assert forest.find_containing(np.array([-0.1, -0.1, -0.1], dtype=np.float64)) is not None
    assert forest.find_nearest(np.array([0.9, 0.9, 0.9], dtype=np.float64)) is not None


def test_safe_box_forest_interval_cache_updates_on_remove() -> None:
    forest = SafeBoxForest(
        robot_fingerprint="dummy",
        joint_limits=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        config=SBFConfig(adjacency_tolerance=1e-8),
    )

    forest.add_box_direct(_mk_box(0, -0.5, 0.0))
    forest.add_box_direct(_mk_box(1, 0.0, 0.5))
    forest.add_box_direct(_mk_box(2, 0.5, 0.8))

    assert forest._intervals_len == 3
    forest.remove_boxes({1})

    assert forest.n_boxes == 2
    assert forest._intervals_len == 2
    assert 1 not in forest._interval_id_to_index
    assert 1 not in forest.adjacency
