"""Detailed profiling: trace exactly where collision checks happen in v1 vs v2."""
import json, math, time, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))

def profile_variant(variant, seed=42):
    """Run one variant with detailed instrumentation."""
    if variant == "v1":
        sys.path.insert(0, ROOT + "/v1/src")
        from box_aabb.robot import load_robot
        from planner.obstacles import Scene
        from planner.models import PlannerConfig
        from planner.box_rrt import BoxRRT
    elif variant == "v2":
        sys.path.insert(0, ROOT + "/v2/src")
        from aabb.robot import load_robot
        from forest.scene import Scene
        from planner.models import PlannerConfig
        from planner.box_rrt import BoxRRT

    import numpy as np

    robot = load_robot("2dof_planar")
    scene = Scene()
    scene.add_obstacle([1.5, -0.3], [2.0, 0.3], "obs1")
    scene.add_obstacle([0.5, -1.8], [1.2, -1.2], "obs2")

    cfg = PlannerConfig(max_iterations=200, max_box_nodes=120, verbose=False)
    planner = BoxRRT(robot, scene, cfg)
    q_start = np.array([math.pi * 0.6, 0.3])
    q_goal = np.array([-math.pi * 0.4, -0.5])

    cc = planner.collision_checker

    # Phase tracking: count collision checks at each stage
    phases = {}

    # Phase 0: start/goal validation + direct connect
    cc.reset_counter()
    t0 = time.perf_counter()

    # Check start/goal
    c_start = cc.check_config_collision(q_start)
    c_goal = cc.check_config_collision(q_goal)
    phases["0_validation"] = cc.n_collision_checks

    # Direct connect
    cc.reset_counter()
    seg_col = cc.check_segment_collision(q_start, q_goal, cfg.segment_collision_resolution)
    phases["1_direct_connect"] = cc.n_collision_checks

    # Now run full plan with counter reset
    cc.reset_counter()
    result = planner.plan(q_start, q_goal, seed=seed)
    total_cc = cc.n_collision_checks
    dt = time.perf_counter() - t0

    # Collect HierAABBTree stats
    tree_stats = planner.hier_tree.get_stats()

    info = {
        "variant": variant,
        "seed": seed,
        "time_s": round(dt, 4),
        "success": result.success,
        "n_boxes": result.n_boxes_created,
        "total_cc": result.n_collision_checks,
        "tree_n_nodes": tree_stats["n_nodes"],
        "tree_n_fk_calls": tree_stats["n_fk_calls"],
        "tree_max_depth": tree_stats["max_depth"],
        "tree_avg_depth": round(tree_stats["avg_depth"], 2),
        "path_length": round(float(result.path_length), 4) if result.success else None,
    }

    return info


if __name__ == "__main__":
    # Need to run in separate processes to avoid import pollution
    import subprocess

    CHILD_CODE = r'''
import json, math, time, sys, os
import numpy as np

ROOT = sys.argv[1]
variant = sys.argv[2]
seed_val = int(sys.argv[3])

if variant == "v1":
    sys.path.insert(0, os.path.join(ROOT, "v1", "src"))
    from box_aabb.robot import load_robot
    from planner.obstacles import Scene
    from planner.models import PlannerConfig
    from planner.box_rrt import BoxRRT
elif variant == "v2":
    sys.path.insert(0, os.path.join(ROOT, "v2", "src"))
    from aabb.robot import load_robot
    from forest.scene import Scene
    from planner.models import PlannerConfig
    from planner.box_rrt import BoxRRT

robot = load_robot("2dof_planar")
scene = Scene()
scene.add_obstacle([1.5, -0.3], [2.0, 0.3], "obs1")
scene.add_obstacle([0.5, -1.8], [1.2, -1.2], "obs2")

cfg = PlannerConfig(max_iterations=200, max_box_nodes=120, verbose=False)
planner = BoxRRT(robot, scene, cfg)
q_start = np.array([math.pi * 0.6, 0.3])
q_goal = np.array([-math.pi * 0.4, -0.5])

cc = planner.collision_checker

# ---- Phase 0: start/goal check ----
cc.reset_counter()
_ = cc.check_config_collision(q_start)
_ = cc.check_config_collision(q_goal)
cc_phase0 = cc.n_collision_checks

# ---- Phase 1: direct connect ----
cc.reset_counter()
_ = cc.check_segment_collision(q_start, q_goal, cfg.segment_collision_resolution)
cc_phase1 = cc.n_collision_checks

# ---- Full plan (includes phases 0-1 again inside plan()) ----
cc.reset_counter()
t0 = time.perf_counter()
result = planner.plan(q_start, q_goal, seed=seed_val)
dt = time.perf_counter() - t0
cc_total = cc.n_collision_checks

# Tree stats
tree_stats = planner.hier_tree.get_stats()

# ---- Count validate_boxes CC ----
# After plan, forest has boxes. Count how many validate_boxes would cost.
forest = result.forest
n_forest_boxes = len(forest.boxes) if forest else 0

# For v2, check if _sample_seed uses batch (counts N per call)
# vs v1 that uses individual checks
sample_seed_mode = "batch" if variant == "v2" else "individual"

info = {
    "variant": variant,
    "seed": seed_val,
    "time_s": round(dt, 4),
    "success": result.success,
    "n_boxes": result.n_boxes_created,
    "total_cc": cc_total,
    "tree_n_nodes": tree_stats["n_nodes"],
    "tree_n_fk_calls": tree_stats["n_fk_calls"],
    "tree_max_depth": tree_stats["max_depth"],
    "tree_avg_depth": round(tree_stats["avg_depth"], 2),
    "n_forest_boxes": n_forest_boxes,
    "sample_seed_mode": sample_seed_mode,
    "path_length": round(float(result.path_length), 4) if result.success else None,
}

print(json.dumps(info))
'''

    print("=" * 70)
    print("V1 vs V2 Detailed Profiling")
    print("=" * 70)

    for variant in ["v1", "v2"]:
        print(f"\n[{variant.upper()}]")
        for seed in [42, 43, 44]:
            proc = subprocess.run(
                [sys.executable, "-c", CHILD_CODE, ROOT, variant, str(seed)],
                capture_output=True, text=True, cwd=ROOT,
            )
            if proc.returncode != 0:
                print(f"  seed={seed} FAILED:")
                stderr = proc.stderr.strip()
                print(stderr[-800:] if len(stderr) > 800 else stderr)
                continue
            line = proc.stdout.strip().splitlines()[-1]
            d = json.loads(line)
            print(f"  seed={seed}: time={d['time_s']:.3f}s  cc={d['total_cc']}  "
                  f"boxes={d['n_boxes']}  forest_boxes={d['n_forest_boxes']}  "
                  f"tree_nodes={d['tree_n_nodes']}  fk_calls={d['tree_n_fk_calls']}  "
                  f"max_depth={d['tree_max_depth']}  avg_depth={d['tree_avg_depth']:.1f}  "
                  f"seed_mode={d['sample_seed_mode']}")
