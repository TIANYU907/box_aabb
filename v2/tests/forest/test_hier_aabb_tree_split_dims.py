from aabb.robot import load_robot
from forest.hier_aabb_tree import HierAABBTree


def test_infer_relevant_split_dims_panda_excludes_joint7() -> None:
    robot = load_robot("panda")
    n_links = len(robot.dh_params) + (1 if robot.tool_frame is not None else 0)

    dims = HierAABBTree._infer_aabb_relevant_split_dims(
        robot,
        robot.n_joints,
        n_links,
    )

    assert dims == [0, 1, 2, 3, 4, 5]
    assert 6 not in dims


def test_resolve_active_split_dims_filters_irrelevant_dims() -> None:
    tree = HierAABBTree.__new__(HierAABBTree)
    tree.n_dims = 7
    tree._aabb_relevant_split_dim_set = {0, 1, 2, 3, 4, 5}

    resolved = tree._resolve_active_split_dims([0, 6, 5, 6])

    assert resolved == [0, 5]


def test_resolve_active_split_dims_default_uses_relevant_dims() -> None:
    tree = HierAABBTree.__new__(HierAABBTree)
    tree.n_dims = 7
    tree._aabb_relevant_split_dim_set = {0, 1, 2, 3, 4, 5}

    resolved = tree._resolve_active_split_dims(None)

    assert resolved == [0, 1, 2, 3, 4, 5]
