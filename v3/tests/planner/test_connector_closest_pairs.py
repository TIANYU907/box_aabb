import numpy as np

from forest.models import BoxNode
from planner.box_tree import BoxTreeManager
from planner.connector import TreeConnector
from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from planner.models import BoxTree


def _mk_node(node_id: int, center: float, tree_id: int) -> BoxNode:
    return BoxNode(
        node_id=node_id,
        joint_intervals=[(center - 0.1, center + 0.1), (-0.1, 0.1)],
        seed_config=np.array([center, 0.0]),
        tree_id=tree_id,
    )


def test_connector_find_closest_pairs_runs() -> None:
    robot = load_robot("2dof_planar")
    checker = CollisionChecker(robot=robot, scene=Scene())
    manager = BoxTreeManager()

    ta = BoxTree(tree_id=0)
    tb = BoxTree(tree_id=1)
    ta.nodes[0] = _mk_node(0, -0.5, 0)
    tb.nodes[1] = _mk_node(1, 0.5, 1)

    manager._trees[0] = ta
    manager._trees[1] = tb

    connector = TreeConnector(manager, checker, max_attempts=5)
    pairs = connector._find_closest_box_pairs(ta, tb, k=1)

    assert len(pairs) == 1
