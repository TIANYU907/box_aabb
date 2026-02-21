"""
experiments/exp5_ablation.py — 实验 5: 消融实验

量化各管线组件的贡献:
- Full pipeline (default)
- No coarsen
- No bridge
- Serial grow
- No GCS refine (Dijkstra only)
- No shortcut
- No wavefront (boundary_expand=False)

用法:
    python -m experiments.exp5_ablation [--quick]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.runner import (ExperimentRunner, ExperimentResults,
                                 SingleTrialResult, load_scene_from_config)
from experiments.scenes import load_scenes

logger = logging.getLogger(__name__)

OUTPUT_DIR = _ROOT / "experiments" / "output" / "raw"

# 消融变体
ABLATION_VARIANTS = {
    "full_pipeline": {
        "type": "SBF", "method": "dijkstra", "no_cache": True,
        "boundary_expand": True, "max_boxes": 500,
    },
    "no_wavefront": {
        "type": "SBF", "method": "dijkstra", "no_cache": True,
        "boundary_expand": False, "max_boxes": 500,
    },
    "no_gcs_refine": {
        "type": "SBF", "method": "dijkstra", "no_cache": True,
        "boundary_expand": True, "max_boxes": 500,
        "corridor_hops": 0,
    },
    "small_forest": {
        "type": "SBF", "method": "dijkstra", "no_cache": True,
        "boundary_expand": True, "max_boxes": 100,
    },
    "large_forest": {
        "type": "SBF", "method": "dijkstra", "no_cache": True,
        "boundary_expand": True, "max_boxes": 1000,
    },
}


def run(quick: bool = False) -> Path:
    n_seeds = 2 if quick else 20
    n_trials = 1 if quick else 3
    timeout = 15.0 if quick else 30.0

    scene_names = ["panda_8obs_open"]
    if not quick:
        scene_names.append("panda_15obs_moderate")

    scenes = load_scenes(scene_names)

    print("=== Experiment 5: Ablation Study ===")
    print(f"  Variants: {list(ABLATION_VARIANTS.keys())}")
    print(f"  Scenes:   {scene_names}")
    print(f"  Seeds:    {n_seeds}")
    print()

    config = {
        "name": "exp5_ablation",
        "scenes": scenes,
        "planners": list(ABLATION_VARIANTS.values()),
        "seeds": list(range(n_seeds)),
        "n_trials": n_trials,
        "timeout": timeout,
    }

    runner = ExperimentRunner(config)
    results = runner.run()

    out_path = OUTPUT_DIR / "exp5_ablation.json"
    results.metadata["variants"] = list(ABLATION_VARIANTS.keys())
    results.save(out_path)

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
