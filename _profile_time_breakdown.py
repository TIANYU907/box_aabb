"""Time breakdown: where does v2 spend extra time vs v1?"""
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

# ---- Instrument _sample_seed for time ----
original_sample_seed = planner._sample_seed
sample_seed_time = [0.0]
sample_seed_calls = [0]

def timed_sample_seed(*args, **kwargs):
    sample_seed_calls[0] += 1
    t = time.perf_counter()
    result = original_sample_seed(*args, **kwargs)
    sample_seed_time[0] += time.perf_counter() - t
    return result
planner._sample_seed = timed_sample_seed

# ---- Instrument add_box_direct ----
forest_ref = [None]
add_box_time = [0.0]
add_box_calls = [0]

# We'll patch after forest is created, via plan wrapping
original_plan_impl = planner._plan_impl

def patched_plan_impl(q_start, q_goal, rng, t0, result):
    # Call original, which creates forest internally
    return original_plan_impl(q_start, q_goal, rng, t0, result)

# ---- Run plan ----
cc = planner.collision_checker
cc.reset_counter()
t0 = time.perf_counter()
result = planner.plan(q_start, q_goal, seed=seed_val)
total_time = time.perf_counter() - t0

info = {
    "variant": variant,
    "seed": seed_val,
    "total_time": round(total_time, 4),
    "sample_seed_time": round(sample_seed_time[0], 4),
    "sample_seed_calls": sample_seed_calls[0],
    "sample_seed_pct": round(sample_seed_time[0] / max(total_time, 1e-9) * 100, 1),
    "non_seed_time": round(total_time - sample_seed_time[0], 4),
    "total_cc": result.n_collision_checks,
    "n_boxes": result.n_boxes_created,
    "success": result.success,
}
print(json.dumps(info))
'''

if __name__ == "__main__":
    print("=" * 80)
    print("Time Breakdown: _sample_seed vs rest")
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
            print(f"  seed={seed}: total={d['total_time']:.3f}s  "
                  f"seed_sampling={d['sample_seed_time']:.3f}s ({d['sample_seed_pct']:.0f}%)  "
                  f"non_seed={d['non_seed_time']:.3f}s  "
                  f"cc={d['total_cc']}  boxes={d['n_boxes']}")
