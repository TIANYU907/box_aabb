"""
baselines/base.py — 统一规划器接口

BasePlanner ABC  ：所有规划器的统一接口
PlanningResult   ：所有方法的统一结果格式
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# PlanningResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlanningResult:
    """所有规划方法的统一结果格式."""

    success: bool
    path: Optional[np.ndarray]       # (N, DOF) waypoints, None if failed
    cost: float                      # path length (L2 in joint space)
    planning_time: float             # wall-clock seconds (total)
    first_solution_time: float       # seconds until first valid solution
    collision_checks: int            # number of collision check calls
    nodes_explored: int              # tree nodes / box count / etc.
    phase_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────

    @property
    def n_waypoints(self) -> int:
        if self.path is None:
            return 0
        return self.path.shape[0]

    def to_dict(self) -> dict:
        d = {
            "success": self.success,
            "cost": self.cost,
            "planning_time": self.planning_time,
            "first_solution_time": self.first_solution_time,
            "collision_checks": self.collision_checks,
            "nodes_explored": self.nodes_explored,
            "n_waypoints": self.n_waypoints,
            "phase_times": dict(self.phase_times),
        }
        d.update(self.metadata)
        return d

    @staticmethod
    def failure(planning_time: float = 0.0,
                collision_checks: int = 0,
                nodes_explored: int = 0,
                **metadata) -> "PlanningResult":
        """快捷构造失败结果."""
        return PlanningResult(
            success=False, path=None, cost=float("inf"),
            planning_time=planning_time,
            first_solution_time=float("nan"),
            collision_checks=collision_checks,
            nodes_explored=nodes_explored,
            metadata=metadata,
        )


# ═══════════════════════════════════════════════════════════════════════════
# BasePlanner ABC
# ═══════════════════════════════════════════════════════════════════════════

class BasePlanner(abc.ABC):
    """所有规划器的统一接口.

    生命周期::

        planner = SomePlanner()
        planner.setup(robot, scene, config)
        r1 = planner.plan(start, goal)
        r2 = planner.plan(start2, goal2)   # 若 supports_reuse, 可复用
        planner.reset()                     # 清除内部状态
    """

    @abc.abstractmethod
    def setup(self, robot, scene, config: dict) -> None:
        """设置机器人、场景、配置.

        Args:
            robot: ``aabb.robot.Robot`` 实例
            scene: ``forest.scene.Scene`` 实例
            config: 方法特定配置字典
        """

    @abc.abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray,
             timeout: float = 30.0) -> PlanningResult:
        """执行规划.

        Args:
            start: 起始配置 (ndim,)
            goal:  目标配置 (ndim,)
            timeout: 最大规划时间 (秒)

        Returns:
            PlanningResult 统一结果
        """

    def reset(self) -> None:
        """重置内部状态 (用于公平计时对比).

        默认空操作; 子类可覆写以清除缓存/树/forest 等.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """该规划器的名称 (用于报告/图表)."""

    @property
    def supports_reuse(self) -> bool:
        """是否支持跨查询复用已有数据结构.

        返回 True 时, 连续 ``plan()`` 调用可复用 setup 阶段
        构建的数据结构 (如 box forest / roadmap).
        """
        return False
