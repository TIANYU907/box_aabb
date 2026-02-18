import numpy as np

from aabb.robot import load_robot


def test_get_link_positions_batch_matches_scalar() -> None:
    robot = load_robot("panda")
    rng = np.random.default_rng(42)

    n = 16
    lows = np.array([lo for lo, _ in robot.joint_limits], dtype=np.float64)
    highs = np.array([hi for _, hi in robot.joint_limits], dtype=np.float64)
    q_batch = rng.uniform(lows, highs, size=(n, robot.n_joints))

    link_candidates = [1, 3, robot.n_joints]
    if robot.tool_frame is not None:
        link_candidates.append(robot.n_joints + 1)

    for link_idx in link_candidates:
        batch_pos = robot.get_link_positions_batch(q_batch, link_idx)
        scalar_pos = np.vstack([
            robot.get_link_position(q_batch[i].tolist(), link_idx)
            for i in range(n)
        ])
        assert batch_pos.shape == (n, 3)
        assert np.allclose(batch_pos, scalar_pos, atol=1e-10)
