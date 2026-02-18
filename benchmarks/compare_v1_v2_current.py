"""End-to-end performance regression: v1 vs v2 (current state).

Runs the same planning scenario in isolated subprocesses for v1 and v2,
then writes a JSON + TXT report under comparison_reports/.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "comparison_reports"


CHILD_CODE = r'''
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
variant = sys.argv[2]
repeats = int(sys.argv[3])

if variant == "v1":
    sys.path.insert(0, str(root / "src"))
    from box_aabb.robot import load_robot
    from planner.obstacles import Scene
    from planner.models import PlannerConfig
    from planner.box_rrt import BoxRRT
elif variant == "v2":
    sys.path.insert(0, str(root / "v2" / "src"))
    from aabb.robot import load_robot
    from forest.scene import Scene
    from planner.models import PlannerConfig
    from planner.box_rrt import BoxRRT
else:
    raise ValueError(f"Unknown variant: {variant}")


def run_once(seed: int) -> dict:
    robot = load_robot("2dof_planar")
    scene = Scene()
    scene.add_obstacle([1.5, -0.3], [2.0, 0.3], "obs1")
    scene.add_obstacle([0.5, -1.8], [1.2, -1.2], "obs2")

    cfg = PlannerConfig(
        max_iterations=200,
        max_box_nodes=120,
        seed_batch_size=5,
        expansion_resolution=0.03,
        max_expansion_rounds=3,
        goal_bias=0.15,
        connection_radius=3.0,
        connection_max_attempts=50,
        path_shortcut_iters=100,
        segment_collision_resolution=0.03,
        verbose=False,
    )

    planner = BoxRRT(robot, scene, cfg)
    q_start = np.array([math.pi * 0.6, 0.3], dtype=np.float64)
    q_goal = np.array([-math.pi * 0.4, -0.5], dtype=np.float64)

    t0 = time.perf_counter()
    result = planner.plan(q_start, q_goal, seed=seed)
    dt = time.perf_counter() - t0

    return {
        "time_s": float(dt),
        "success": bool(result.success),
        "path_length": float(result.path_length),
        "n_boxes_created": int(result.n_boxes_created),
        "n_collision_checks": int(result.n_collision_checks),
    }


# warmup
_ = run_once(12345)

samples = [run_once(42 + i) for i in range(repeats)]
times = [s["time_s"] for s in samples]

payload = {
    "variant": variant,
    "repeats": repeats,
    "mean_s": float(statistics.mean(times)),
    "median_s": float(statistics.median(times)),
    "min_s": float(min(times)),
    "max_s": float(max(times)),
    "success_rate": float(sum(1 for s in samples if s["success"]) / repeats),
    "mean_boxes": float(statistics.mean(s["n_boxes_created"] for s in samples)),
    "mean_collision_checks": float(statistics.mean(s["n_collision_checks"] for s in samples)),
    "samples": samples,
}
print(json.dumps(payload, ensure_ascii=False))
'''


def run_variant(variant: str, repeats: int = 5) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD_CODE,
            str(ROOT),
            variant,
            str(repeats),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=dict(os.environ),
    )
    # 取最后一行 JSON
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v1 = run_variant("v1", repeats=5)
    v2 = run_variant("v2", repeats=5)

    speedup = (v1["mean_s"] / v2["mean_s"]) if v2["mean_s"] > 0 else float("inf")

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "scenario": "2dof_planar fixed obstacles, PlannerConfig(max_iterations=200,max_box_nodes=120)",
        "v1": v1,
        "v2": v2,
        "speedup_v2_over_v1": speedup,
    }

    ts = summary["timestamp"]
    json_path = OUT_DIR / f"v1_v2_regression_{ts}.json"
    txt_path = OUT_DIR / f"v1_v2_regression_{ts}.txt"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "v1 vs v2 current regression",
        f"timestamp: {ts}",
        f"scenario: {summary['scenario']}",
        "",
        f"v1 mean: {v1['mean_s']:.6f}s, median: {v1['median_s']:.6f}s, success_rate: {v1['success_rate']:.2%}",
        f"v2 mean: {v2['mean_s']:.6f}s, median: {v2['median_s']:.6f}s, success_rate: {v2['success_rate']:.2%}",
        f"speedup (v2/v1): {speedup:.4f}x",
        "",
        f"v1 mean_boxes: {v1['mean_boxes']:.2f}, mean_collision_checks: {v1['mean_collision_checks']:.2f}",
        f"v2 mean_boxes: {v2['mean_boxes']:.2f}, mean_collision_checks: {v2['mean_collision_checks']:.2f}",
        "",
        f"json: {json_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(txt_path)
    print(json_path)
    print(json.dumps({
        "v1_mean_s": v1["mean_s"],
        "v2_mean_s": v2["mean_s"],
        "speedup_v2_over_v1": speedup,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
