"""
Performance regression guard for SafeBoxForest (SBF) planner.

Runs standard planning scenarios, compares timing against v2 baselines.
Any metric exceeding the allowed tolerance (default: +15%) triggers FAIL.

Usage:
    python -m pytest tests/perf_regression.py -v
    python tests/perf_regression.py             # standalone mode

Baselines are stored in tests/v2_baseline.json. To regenerate:
    python tests/perf_regression.py --record
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap imports (make sure src/ is on sys.path)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aabb.robot import load_robot
from forest.scene import Scene
from planner.sbf_planner import SBFPlanner
from planner.models import SBFConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINE_FILE = _HERE / "v2_baseline.json"

# ---------------------------------------------------------------------------
# Default tolerance: actual <= baseline * (1 + TOLERANCE)
# Set generous 15% to absorb CI/machine variance; tighten after stabilising.
# ---------------------------------------------------------------------------
DEFAULT_TOLERANCE = 0.15

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    robot_name: str
    obstacles: List[Dict[str, Any]]
    q_start: List[float]
    q_goal: List[float]
    seed: int
    config_overrides: Dict[str, Any]


SCENARIOS: List[Scenario] = [
    Scenario(
        name="2dof_simple",
        robot_name="2dof_planar",
        obstacles=[
            {"min": [0.3, -0.4], "max": [0.8, 0.1]},
        ],
        q_start=[2.0, 0.5],
        q_goal=[-2.0, -0.5],
        seed=42,
        config_overrides={
            "max_iterations": 1000,
            "max_box_nodes": 200,
            "min_box_size": 0.03,
        },
    ),
    Scenario(
        name="2dof_gcs_style",
        robot_name="2dof_planar",
        obstacles=[
            {"min": [0.3, -0.4], "max": [0.8, 0.1]},
        ],
        # From gcs_planner_2dof.py default config
        q_start=[2.5133, 0.2],   # 0.8*pi, 0.2
        q_goal=[-2.1991, -0.4],  # -0.7*pi, -0.4
        seed=20260219,
        config_overrides={
            "max_iterations": 3000,
            "max_box_nodes": 500,
            "min_box_size": 0.03,
            "goal_bias": 0.15,
            "guided_sample_ratio": 0.6,
        },
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _make_scene(obstacles: List[Dict[str, Any]]) -> Scene:
    scene = Scene()
    for obs in obstacles:
        scene.add_obstacle(obs["min"], obs["max"], obs.get("name", ""))
    return scene


def run_scenario(scenario: Scenario) -> Dict[str, float]:
    """Run a single scenario and return timing metrics (in ms)."""
    robot = load_robot(scenario.robot_name)
    scene = _make_scene(scenario.obstacles)
    config = SBFConfig(**scenario.config_overrides)

    q_start = np.array(scenario.q_start, dtype=np.float64)
    q_goal = np.array(scenario.q_goal, dtype=np.float64)

    # Warm-up: create planner (includes HierAABBTree init)
    planner = SBFPlanner(robot, scene, config=config, no_cache=True)

    # Time the actual planning
    t0 = time.perf_counter()
    result = planner.plan(q_start, q_goal, seed=scenario.seed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "total_ms": elapsed_ms,
        "success": result.success,
        "n_boxes": result.n_boxes_created,
        "computation_time_ms": result.computation_time * 1000.0,
    }


def load_baselines() -> Dict[str, Dict[str, float]]:
    """Load v2 baseline JSON. Returns empty dict if file missing."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def record_baselines() -> None:
    """Run all scenarios and save results as the new baseline."""
    baselines: Dict[str, Dict[str, float]] = {}

    # Run each scenario multiple times and take the median
    n_trials = 5
    for scenario in SCENARIOS:
        trial_results: List[Dict[str, float]] = []
        for _ in range(n_trials):
            trial_results.append(run_scenario(scenario))

        median_total = sorted(r["total_ms"] for r in trial_results)[n_trials // 2]

        baselines[scenario.name] = {
            "total_ms": round(median_total, 1),
            "success": trial_results[0]["success"],
            "n_boxes": trial_results[0]["n_boxes"],
        }
        status = "OK" if trial_results[0]["success"] else "FAIL"
        print(f"  {scenario.name}: {median_total:.1f} ms [{status}]")

    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, ensure_ascii=False)
    print(f"\nBaseline saved to {BASELINE_FILE}")


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------

def _check_scenario(scenario: Scenario, baselines: Dict, tolerance: float):
    """Run scenario and compare against baselines. Returns (pass, message)."""
    metrics = run_scenario(scenario)
    baseline = baselines.get(scenario.name)

    if baseline is None:
        return True, f"{scenario.name}: no baseline (skip comparison)"

    total_ms = metrics["total_ms"]
    baseline_ms = baseline["total_ms"]
    limit_ms = baseline_ms * (1 + tolerance)

    passed = total_ms <= limit_ms
    msg = (
        f"{scenario.name}: {total_ms:.1f}ms "
        f"(baseline={baseline_ms:.1f}ms, limit={limit_ms:.1f}ms, "
        f"{'OK' if passed else 'REGRESSION'})"
    )
    return passed, msg


def test_perf_regression_2dof_simple():
    """Performance regression: 2dof_simple scenario."""
    baselines = load_baselines()
    scenario = next(s for s in SCENARIOS if s.name == "2dof_simple")
    passed, msg = _check_scenario(scenario, baselines, DEFAULT_TOLERANCE)
    print(f"  {msg}")
    if baselines.get(scenario.name):
        assert passed, msg


def test_perf_regression_2dof_gcs_style():
    """Performance regression: 2dof_gcs_style scenario."""
    baselines = load_baselines()
    scenario = next(s for s in SCENARIOS if s.name == "2dof_gcs_style")
    passed, msg = _check_scenario(scenario, baselines, DEFAULT_TOLERANCE)
    print(f"  {msg}")
    if baselines.get(scenario.name):
        assert passed, msg


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--record" in sys.argv:
        print("Recording v2 baselines...\n")
        record_baselines()
    else:
        baselines = load_baselines()
        if not baselines:
            print("No baseline file found. Run with --record first.\n")
            print("Recording baselines now...\n")
            record_baselines()
            baselines = load_baselines()

        print("\nPerformance regression check")
        print("=" * 60)
        all_pass = True
        for scenario in SCENARIOS:
            passed, msg = _check_scenario(scenario, baselines, DEFAULT_TOLERANCE)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {msg}")
            if not passed:
                all_pass = False

        print("=" * 60)
        if all_pass:
            print("All scenarios within tolerance.")
        else:
            print("REGRESSION DETECTED!")
            sys.exit(1)
