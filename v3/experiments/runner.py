"""
experiments/runner.py — 统一实验运行器

ExperimentRunner:  (scene × planner × seed × trial) 组合自动运行
SingleTrialResult: 单次运行结果
ExperimentResults: 全部运行结果 + 聚合

用法:
    runner = ExperimentRunner(config)
    results = runner.run()
    results.summary_df()   # pandas DataFrame
    results.save("output/raw/exp1.json")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from baselines.base import BasePlanner, PlanningResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SingleTrialResult:
    """单次 (scene, planner, seed, trial) 运行结果."""
    scene_name: str
    planner_name: str
    seed: int
    trial: int
    result: Dict[str, Any]  # PlanningResult.to_dict()
    wall_clock: float       # 总 wall-clock 秒

    def to_dict(self) -> dict:
        return {
            "scene": self.scene_name,
            "planner": self.planner_name,
            "seed": self.seed,
            "trial": self.trial,
            "wall_clock": self.wall_clock,
            **self.result,
        }


@dataclass
class ExperimentResults:
    """全部运行结果容器."""
    experiment_name: str
    trials: List[SingleTrialResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, trial: SingleTrialResult):
        self.trials.append(trial)

    def to_list(self) -> List[dict]:
        return [t.to_dict() for t in self.trials]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiment": self.experiment_name,
            "metadata": self.metadata,
            "n_trials": len(self.trials),
            "results": self.to_list(),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False,
                      default=_json_default)
        logger.info("Saved %d trials → %s", len(self.trials), p)

    @staticmethod
    def load(path: str | Path) -> "ExperimentResults":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        er = ExperimentResults(
            experiment_name=data["experiment"],
            metadata=data.get("metadata", {}),
        )
        for r in data["results"]:
            er.trials.append(SingleTrialResult(
                scene_name=r["scene"],
                planner_name=r["planner"],
                seed=r["seed"],
                trial=r["trial"],
                result={k: v for k, v in r.items()
                        if k not in ("scene", "planner", "seed", "trial",
                                     "wall_clock")},
                wall_clock=r["wall_clock"],
            ))
        return er

    def summary_df(self):
        """返回 pandas DataFrame 汇总 (需要 pandas)."""
        import pandas as pd
        return pd.DataFrame(self.to_list())

    def grouped_stats(self, group_keys: Sequence[str] = ("scene", "planner"),
                      value_key: str = "planning_time") -> Dict[str, Dict]:
        """按 group_keys 分组计算统计量."""
        from collections import defaultdict
        groups: Dict[str, list] = defaultdict(list)
        for t in self.trials:
            d = t.to_dict()
            key = tuple(d.get(k, "") for k in group_keys)
            val = d.get(value_key)
            if val is not None and not (isinstance(val, float) and
                                         np.isnan(val)):
                groups[str(key)].append(val)

        stats = {}
        for key, vals in groups.items():
            arr = np.array(vals)
            stats[key] = {
                "n": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
            }
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# Scene & Planner factories
# ═══════════════════════════════════════════════════════════════════════════

def load_scene_from_config(cfg: dict):
    """从配置字典创建 Scene + Robot + query pairs.

    Returns:
        (robot, scene, query_pairs)
        query_pairs: List[(q_start, q_goal)]
    """
    from aabb.robot import load_robot
    from forest.scene import Scene
    from forest.collision import CollisionChecker

    robot = load_robot(cfg.get("robot", "panda"))
    scene = Scene()

    # 固定障碍物
    for obs in cfg.get("obstacles", []):
        scene.add_obstacle(obs["min"], obs["max"],
                           name=obs.get("name", ""))

    # 随机障碍物
    if cfg.get("random_obstacles"):
        rc = cfg["random_obstacles"]
        rng = np.random.default_rng(rc.get("seed", 0))
        checker = CollisionChecker(robot=robot, scene=scene)
        for i in range(rc.get("count", 6)):
            for _ in range(200):
                cx = rng.uniform(*rc.get("x_range", [-0.8, 0.8]))
                cy = rng.uniform(*rc.get("y_range", [-0.8, 0.8]))
                cz = rng.uniform(*rc.get("z_range", [0.0, 1.0]))
                h = rng.uniform(*rc.get("half_size_range", [0.05, 0.15]))
                scene.add_obstacle(
                    [cx - h, cy - h, cz - h],
                    [cx + h, cy + h, cz + h],
                    name=f"rand_obs_{i}")
                break

    # query pairs
    query_pairs = []
    for qp in cfg.get("query_pairs", []):
        query_pairs.append((
            np.array(qp["start"], dtype=np.float64),
            np.array(qp["goal"], dtype=np.float64),
        ))

    # 默认 query pair
    if not query_pairs:
        q_start = np.array(cfg.get("q_start",
                                    [0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5]))
        q_goal = np.array(cfg.get("q_goal",
                                   [-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8]))
        query_pairs.append((q_start, q_goal))

    return robot, scene, query_pairs


def create_planner(cfg: dict) -> BasePlanner:
    """从配置字典创建 BasePlanner 实例."""
    from baselines import (SBFAdapter, RRTPlanner, OMPLPlanner,
                            IRISGCSPlanner)

    ptype = cfg["type"]
    params = {k: v for k, v in cfg.items() if k != "type"}

    if ptype == "SBF":
        return SBFAdapter(method=params.pop("method", "dijkstra"))
    elif ptype == "RRT":
        return RRTPlanner(algorithm=params.pop("algorithm", "RRTConnect"))
    elif ptype == "OMPL":
        return OMPLPlanner(algorithm=params.pop("algorithm", "RRTConnect"))
    elif ptype == "IRIS-GCS":
        return IRISGCSPlanner()
    else:
        raise ValueError(f"Unknown planner type: {ptype}")


# ═══════════════════════════════════════════════════════════════════════════
# ExperimentRunner
# ═══════════════════════════════════════════════════════════════════════════

class ExperimentRunner:
    """(scene × planner × seed × trial) 全组合实验运行器."""

    def __init__(self, config: dict):
        """
        config 结构:
        {
            "name": "exp1_main_comparison",
            "scenes": [...],          # scene configs
            "planners": [...],        # planner configs
            "seeds": [0,1,2,...],     # random seeds
            "n_trials": 3,           # trials per seed
            "timeout": 30.0,         # seconds
        }
        """
        self.config = config
        self.name = config.get("name", "unnamed")
        self.scene_cfgs = config["scenes"]
        self.planner_cfgs = config["planners"]
        self.seeds = config.get("seeds", list(range(10)))
        self.n_trials = config.get("n_trials", 1)
        self.timeout = config.get("timeout", 30.0)
        self._results = ExperimentResults(experiment_name=self.name)

    def run(self, progress_callback=None) -> ExperimentResults:
        """运行全部组合."""
        total = (len(self.scene_cfgs) * len(self.planner_cfgs)
                 * len(self.seeds) * self.n_trials)
        done = 0

        for scene_cfg in self.scene_cfgs:
            scene_name = scene_cfg.get("name", "scene")
            robot, scene, query_pairs = load_scene_from_config(scene_cfg)

            for planner_cfg in self.planner_cfgs:
                planner = create_planner(planner_cfg)
                planner_params = {k: v for k, v in planner_cfg.items()
                                   if k != "type"}

                for seed in self.seeds:
                    planner_params_with_seed = {**planner_params, "seed": seed}

                    for trial in range(self.n_trials):
                        # setup planner fresh each trial
                        planner.reset()
                        planner.setup(robot, scene, planner_params_with_seed)

                        # pick query pair (cycle if multiple)
                        qidx = (seed + trial) % len(query_pairs)
                        q_start, q_goal = query_pairs[qidx]

                        t0 = time.perf_counter()
                        result = planner.plan(q_start, q_goal,
                                              timeout=self.timeout)
                        wall = time.perf_counter() - t0

                        trial_result = SingleTrialResult(
                            scene_name=scene_name,
                            planner_name=planner.name,
                            seed=seed,
                            trial=trial,
                            result=result.to_dict(),
                            wall_clock=wall,
                        )
                        self._results.add(trial_result)

                        done += 1
                        if progress_callback:
                            progress_callback(done, total, trial_result)
                        elif done % 10 == 0 or done == total:
                            logger.info("[%d/%d] %s / %s / seed=%d / t=%d "
                                        "→ %s (%.3fs)",
                                        done, total, scene_name,
                                        planner.name, seed, trial,
                                        "OK" if result.success else "FAIL",
                                        wall)

        self._results.metadata = {
            "config": self.config,
            "total_trials": total,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        return self._results

    def run_reuse(self, progress_callback=None) -> ExperimentResults:
        """运行复用模式: 同一 planner 实例连续 plan() 多个 query.

        专用于实验 2 (forest reuse).  每个 planner 只 setup() 一次.
        """
        total_queries = 0
        for scene_cfg in self.scene_cfgs:
            nq = len(scene_cfg.get("query_pairs", []))
            total_queries += nq * len(self.planner_cfgs) * len(self.seeds)

        done = 0
        for scene_cfg in self.scene_cfgs:
            scene_name = scene_cfg.get("name", "scene")
            robot, scene, query_pairs = load_scene_from_config(scene_cfg)

            for planner_cfg in self.planner_cfgs:
                for seed in self.seeds:
                    planner = create_planner(planner_cfg)
                    planner_params = {k: v for k, v in planner_cfg.items()
                                       if k != "type"}
                    planner_params["seed"] = seed
                    planner.setup(robot, scene, planner_params)

                    for qi, (q_start, q_goal) in enumerate(query_pairs):
                        t0 = time.perf_counter()
                        result = planner.plan(q_start, q_goal,
                                              timeout=self.timeout)
                        wall = time.perf_counter() - t0

                        trial_result = SingleTrialResult(
                            scene_name=scene_name,
                            planner_name=planner.name,
                            seed=seed,
                            trial=qi,
                            result={**result.to_dict(),
                                    "query_index": qi,
                                    "is_first_query": (qi == 0)},
                            wall_clock=wall,
                        )
                        self._results.add(trial_result)
                        done += 1
                        if progress_callback:
                            progress_callback(done, total_queries,
                                              trial_result)

        self._results.metadata = {
            "config": self.config,
            "mode": "reuse",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        return self._results


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _json_default(obj):
    """JSON serialization fallback."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
