"""
baselines/ompl_adapter.py — OMPL C++ 规划器适配器 (WSL subprocess)

通过 WSL subprocess 调用 ompl_bridge.py, 使用相同碰撞检测后端.
支持: RRT, RRTConnect, RRTstar, InformedRRTstar, BITstar, ABITstar.

需要 WSL + OMPL Python bindings 安装在 WSL 环境中.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import BasePlanner, PlanningResult

logger = logging.getLogger(__name__)

# 默认支持的 OMPL 算法
OMPL_ALGORITHMS = [
    "RRT", "RRTConnect", "RRTstar",
    "InformedRRTstar", "BITstar", "ABITstar",
]

# OMPL bridge 脚本的 WSL 路径
_DEFAULT_BRIDGE_PATH = (
    "/mnt/c/Users/TIAN/Documents/box_aabb/v2/examples/ompl_bridge.py"
)


class OMPLPlanner(BasePlanner):
    """OMPL C++ planner via WSL subprocess bridge.

    Args:
        algorithm: OMPL planner name (e.g. 'RRTConnect', 'RRTstar')
        bridge_path: WSL path to ompl_bridge.py
    """

    def __init__(self, algorithm: str = "RRTConnect",
                 bridge_path: str = _DEFAULT_BRIDGE_PATH):
        self._algorithm = algorithm
        self._bridge_path = bridge_path
        self._robot = None
        self._scene = None
        self._config: dict = {}

    @property
    def name(self) -> str:
        return f"OMPL-{self._algorithm}"

    @property
    def supports_reuse(self) -> bool:
        return False

    def setup(self, robot, scene, config: dict) -> None:
        self._robot = robot
        self._scene = scene
        self._config = config

    def plan(self, start: np.ndarray, goal: np.ndarray,
             timeout: float = 30.0) -> PlanningResult:
        q_start = np.asarray(start, dtype=np.float64)
        q_goal = np.asarray(goal, dtype=np.float64)

        # serialize obstacles
        obs_list = []
        for obs in self._scene.get_obstacles():
            obs_list.append({
                "min_point": obs.min_point.tolist(),
                "max_point": obs.max_point.tolist(),
                "name": obs.name,
            })

        problem = {
            "q_start": q_start.tolist(),
            "q_goal": q_goal.tolist(),
            "obstacles": obs_list,
            "algorithms": [self._algorithm],
            "timeout": timeout,
            "trials": 1,
            "seed": self._config.get("seed", 42),
            "step_size": self._config.get("step_size", 0.5),
        }

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["wsl", "-e", "bash", "-c",
                 f"python3 {self._bridge_path} 2>/dev/null"],
                input=json.dumps(problem),
                capture_output=True, text=True,
                timeout=timeout + 30,
            )
        except subprocess.TimeoutExpired:
            dt = time.perf_counter() - t0
            logger.warning("OMPL bridge timed out for %s", self._algorithm)
            return PlanningResult.failure(planning_time=dt)
        except FileNotFoundError:
            dt = time.perf_counter() - t0
            logger.error("WSL not available")
            return PlanningResult.failure(planning_time=dt)

        dt = time.perf_counter() - t0

        if proc.returncode != 0:
            logger.warning("OMPL bridge exit code %d for %s",
                           proc.returncode, self._algorithm)
            return PlanningResult.failure(planning_time=dt)

        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            logger.warning("OMPL bridge JSON parse error for %s",
                           self._algorithm)
            return PlanningResult.failure(planning_time=dt)

        data = raw.get(self._algorithm, {})
        if "error" in data:
            return PlanningResult.failure(planning_time=dt)

        # extract best trial result
        trials = data.get("trials", [])
        if not trials:
            s = data.get("summary", {})
            success = s.get("n_success", 0) > 0
            if not success:
                return PlanningResult.failure(planning_time=dt)
            wps_raw = data.get("best_waypoints", [])
            path = (np.array(wps_raw, dtype=np.float64)
                    if wps_raw else None)
            return PlanningResult(
                success=True, path=path,
                cost=s.get("avg_path_length", float("inf")),
                planning_time=dt,
                first_solution_time=s.get("avg_first_solution_time", dt),
                collision_checks=int(s.get("avg_collision_checks", 0)),
                nodes_explored=0,
                metadata={"algorithm": self.name},
            )

        # single trial mode (trials=1)
        t = trials[0]
        success = t.get("success", False)
        if not success:
            return PlanningResult.failure(
                planning_time=dt,
                collision_checks=t.get("n_collision_checks", 0))

        wps_raw = data.get("best_waypoints", [])
        path = (np.array(wps_raw, dtype=np.float64)
                if wps_raw else None)

        return PlanningResult(
            success=True,
            path=path,
            cost=t.get("path_length", float("inf")),
            planning_time=dt,
            first_solution_time=t.get("first_solution_time", dt),
            collision_checks=t.get("n_collision_checks", 0),
            nodes_explored=t.get("n_nodes", 0),
            metadata={
                "algorithm": self.name,
                "raw_path_length": t.get("raw_path_length"),
            },
        )

    def reset(self) -> None:
        pass
