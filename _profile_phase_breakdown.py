"""Phase-by-phase collision check breakdown for v1 vs v2."""
import json, math, time, sys, os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

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

# ---- Instrument _sample_seed ----
original_sample_seed = planner._sample_seed
sample_seed_calls = [0]
sample_seed_cc = [0]

def instrumented_sample_seed(*args, **kwargs):
    sample_seed_calls[0] += 1
    before = cc.n_collision_checks
    result = original_sample_seed(*args, **kwargs)
    sample_seed_cc[0] += cc.n_collision_checks - before
    return result

planner._sample_seed = instrumented_sample_seed

# ---- Instrument validate_boxes ----
original_validate = type(planner)._plan_impl
validate_cc = [0]

# Instead, wrap check_box_collision to count validate phase
box_cc_total = [0]
original_check_box = cc.check_box_collision
def instrumented_check_box(*args, **kwargs):
    box_cc_total[0] += 1
    return original_check_box(*args, **kwargs)
cc.check_box_collision = instrumented_check_box

# ---- Run plan ----
cc.reset_counter()
t0 = time.perf_counter()
result = planner.plan(q_start, q_goal, seed=seed_val)
dt = time.perf_counter() - t0

total_cc = result.n_collision_checks
tree_stats = planner.hier_tree.get_stats()

info = {
    "variant": variant,
    "seed": seed_val,
    "time_s": round(dt, 4),
    "success": result.success,
    "n_boxes": result.n_boxes_created,
    "total_cc": total_cc,
    "sample_seed_calls": sample_seed_calls[0],
    "sample_seed_cc": sample_seed_cc[0],
    "box_collision_cc": box_cc_total[0],
    "non_seed_cc": total_cc - sample_seed_cc[0],
    "cc_per_sample_seed": round(sample_seed_cc[0] / max(1, sample_seed_calls[0]), 1),
    "tree_n_nodes": tree_stats["n_nodes"],
    "tree_n_fk_calls": tree_stats["n_fk_calls"],
}

print(json.dumps(info))
'''

if __name__ == "__main__":
    print("=" * 80)
    print("Phase-by-Phase Collision Check Breakdown")
    print("=" * 80)

    for variant in ["v1", "v2"]:
        print(f"\n[{variant.upper()}]")
        for seed in [42, 43, 44]:
            proc = subprocess.run(
                [sys.executable, "-c", CHILD_CODE, ROOT, variant, str(seed)],
                capture_output=True, text=True, cwd=ROOT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            if proc.returncode != 0:
                print(f"  seed={seed} FAILED:")
                stderr = proc.stderr.strip()
                print(stderr[-600:] if len(stderr) > 600 else stderr)
                continue
            line = proc.stdout.strip().splitlines()[-1]
            d = json.loads(line)
            print(f"  seed={seed}:")
            print(f"    total_cc={d['total_cc']}  time={d['time_s']:.3f}s  boxes={d['n_boxes']}")
            print(f"    _sample_seed: {d['sample_seed_calls']} calls, {d['sample_seed_cc']} CC "
                  f"({d['cc_per_sample_seed']:.1f} CC/call)")
            print(f"    validate_boxes (box_cc): {d['box_collision_cc']}")
            print(f"    non-seed CC: {d['non_seed_cc']}  "
                  f"(validate+endpoint+bridge+smooth)")
            print(f"    tree: {d['tree_n_nodes']} nodes, {d['tree_n_fk_calls']} FK calls")
