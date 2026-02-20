from aabb.interval_fk import compute_interval_aabb, compute_interval_aabb_fast
from aabb.robot import load_robot


def test_fast_alias_matches_default() -> None:
    robot = load_robot("panda")
    intervals = [(-0.1, 0.1)] * robot.n_joints
    zero_links = robot.zero_length_links

    aabbs_default, n_default = compute_interval_aabb(robot, intervals, zero_links)
    aabbs_fast, n_fast = compute_interval_aabb_fast(robot, intervals, zero_links)

    assert n_default == 0
    assert n_fast == 0
    assert len(aabbs_default) == len(aabbs_fast)
    assert aabbs_default[0].min_point == aabbs_fast[0].min_point
    assert aabbs_default[0].max_point == aabbs_fast[0].max_point
