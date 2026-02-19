"""Quick v1 vs v2 forest performance comparison (isolated subprocess per run)."""
import subprocess
import sys
import json
import statistics
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

CHILD_CODE = r'''
import json, math, time, sys
import numpy as np

root = sys.argv[1]
variant = sys.argv[2]
seed = int(sys.argv[3])

if variant == "v1":
    sys.path.insert(0, root + "/v1/src")
    from box_aabb.robot import load_robot
    from planner.obstacles import Scene
    from planner.models import PlannerConfig
    from planner.box_rrt import BoxRRT
elif variant == "v2":
    sys.path.insert(0, root + "/v2/src")
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

t0 = time.perf_counter()
result = planner.plan(q_start, q_goal, seed=seed)
dt = time.perf_counter() - t0

print(json.dumps({
    "time_s": dt,
    "success": result.success,
    "n_boxes": result.n_boxes_created,
    "n_cc": result.n_collision_checks,
    "path_length": float(result.path_length),
}))
'''

REPEATS = 5

def run_variant(variant):
    times = []
    successes = []
    boxes = []
    ccs = []
    for i in range(REPEATS):
        proc = subprocess.run(
            [sys.executable, "-c", CHILD_CODE, ROOT, variant, str(42 + i)],
            capture_output=True, text=True, cwd=ROOT,
        )
        if proc.returncode != 0:
            print(f"  {variant} run {i} FAILED:")
            stderr = proc.stderr.strip()
            print(stderr[-500:] if len(stderr) > 500 else stderr)
            continue
        line = proc.stdout.strip().splitlines()[-1]
        d = json.loads(line)
        times.append(d["time_s"])
        successes.append(d["success"])
        boxes.append(d["n_boxes"])
        ccs.append(d["n_cc"])
        status = "OK" if d["success"] else "FAIL"
        print(f"  {variant} run {i}: {d['time_s']:.4f}s  {status}  boxes={d['n_boxes']}  cc={d['n_cc']}")

    if times:
        print(f"\n  {variant} mean: {statistics.mean(times):.4f}s, median: {statistics.median(times):.4f}s")
        print(f"  {variant} success_rate: {sum(successes)/len(successes)*100:.0f}%")
        print(f"  {variant} mean_boxes: {statistics.mean(boxes):.1f}, mean_cc: {statistics.mean(ccs):.1f}")
    print()
    return {
        "times": times,
        "successes": successes,
        "boxes": boxes,
        "ccs": ccs,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("V1 vs V2 Forest Performance Benchmark")
    print("Scenario: 2dof_planar, 2 obstacles, 200 iters, 120 boxes")
    print("=" * 60)
    print()

    print("[V1]")
    v1 = run_variant("v1")

    print("[V2]")
    v2 = run_variant("v2")

    if v1["times"] and v2["times"]:
        v1_mean = statistics.mean(v1["times"])
        v2_mean = statistics.mean(v2["times"])
        speedup = v1_mean / v2_mean if v2_mean > 0 else float("inf")
        print("=" * 60)
        print(f"V1 mean: {v1_mean:.4f}s  |  V2 mean: {v2_mean:.4f}s")
        print(f"Speedup (V2 over V1): {speedup:.2f}x")
        print(f"V1 success: {sum(v1['successes'])}/{REPEATS}  |  V2 success: {sum(v2['successes'])}/{REPEATS}")
        print(f"V1 mean_cc: {statistics.mean(v1['ccs']):.0f}  |  V2 mean_cc: {statistics.mean(v2['ccs']):.0f}")
        print("=" * 60)
