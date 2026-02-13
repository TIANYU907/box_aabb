#!/usr/bin/env python
"""
cProfile comparison: v5 (Python lists) vs old (HierAABBNode objects)

SAME random seed → same scene → comparable FK counts.
"""
import cProfile
import pstats
import time
import numpy as np

from box_aabb.robot import load_robot
from planner.obstacles import Scene

SEED = 12345
N_OBS = 5
MAX_BOXES = 100
MAX_DEPTH = 30


def random_scene_3d(robot, n_obs, rng):
    scene = Scene()
    for i in range(n_obs):
        cx, cy, cz = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.0, 1.0)
        sx, sy, sz = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25)
        scene.add_obstacle(
            min_point=[cx - sx/2, cy - sy/2, cz - sz/2],
            max_point=[cx + sx/2, cy + sy/2, cz + sz/2],
            name=f"obs_{i}",
        )
    return scene


def run_boxes(tree_cls, robot, n_obs, max_boxes, max_depth, seed):
    """Grow boxes on a fresh tree, return (n_boxes, n_fk, time)"""
    rng = np.random.default_rng(seed)
    scene = random_scene_3d(robot, n_obs, rng)
    obstacles = scene.get_obstacles()
    
    jl = list(robot.joint_limits)
    tree = tree_cls(robot, jl)
    
    n_boxes = 0
    t0 = time.perf_counter()
    for _ in range(max_boxes * 5):
        q = np.array([rng.uniform(lo, hi) for lo, hi in jl])
        # Quick collision reject
        from planner.collision import CollisionChecker
        cc = CollisionChecker(robot, scene)
        if cc.check_config_collision(q):
            continue
        if tree.is_occupied(q):
            continue
        
        result = tree.find_free_box(q, obstacles, max_depth=max_depth, mark_occupied=True)
        if result is None:
            continue
        ivs = result.intervals
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        n_boxes += 1
        if n_boxes >= max_boxes:
            break
    elapsed = time.perf_counter() - t0
    return n_boxes, tree.n_fk_calls, elapsed


def main():
    robot = load_robot("panda")
    
    # ── v5 (current) ──
    from planner.hier_aabb_tree import HierAABBTree
    print("=== v5 (Python lists + flat AABB) ===")
    
    # warmup
    run_boxes(HierAABBTree, robot, N_OBS, 10, MAX_DEPTH, SEED + 999)
    
    # timed
    nb, nfk, t = run_boxes(HierAABBTree, robot, N_OBS, MAX_BOXES, MAX_DEPTH, SEED)
    print(f"  boxes={nb}  FK={nfk}  time={t:.3f}s  per_fk={t/nfk*1000:.3f}ms")
    
    # cProfile
    print(f"\n  cProfile ({MAX_BOXES} boxes)...")
    pr = cProfile.Profile()
    pr.enable()
    nb2, nfk2, t2 = run_boxes(HierAABBTree, robot, N_OBS, MAX_BOXES, MAX_DEPTH, SEED)
    pr.disable()
    print(f"  boxes={nb2}  FK={nfk2}  time={t2:.3f}s")
    print("\n  Top 30 cumulative:")
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print("\n  Top 30 tottime:")
    stats.sort_stats('tottime')
    stats.print_stats(30)


if __name__ == "__main__":
    main()
