"""
aabb_calculator.py - AABB包围盒计算器

作为瘦调度层，将计算委托给提取后的模块：
- strategies/   — 采样策略 (critical / random)
- optimization.py — L-BFGS-B 优化
- interval_fk.py  — 区间/仿射算术
- models.py       — 数据类
- report.py       — 报告生成

计算方法：
1. numerical/critical  - 关键点枚举 + 约束优化 + 流形随机采样
2. numerical/random    - 随机采样 + 局部优化
3. interval            - 区间算术保守估计
"""

import logging
import time
from typing import List, Tuple, Dict, Set, Optional

from .robot import Robot
from .models import (
    BoundaryConfig,
    LinkAABBInfo,
    AABBEnvelopeResult,
)
from .report import ReportGenerator
from .strategies import CriticalStrategy, RandomStrategy
from .interval_fk import compute_interval_aabb

logger = logging.getLogger(__name__)


class AABBCalculator:
    """AABB包围盒计算器

    连杆等分策略 (n_subdivisions > 1):
      - 每根连杆被等分为 n 段，每段返回独立的 LinkAABBInfo
      - 段上位置 p(t) = (1-t)*FK(start) + t*FK(end)，FK只算两次
      - 所有采样点在各段间完全复用
      - 每段拥有独立的极值跟踪、优化目标和边界臂形
    """

    def __init__(self, robot: Robot, robot_name: Optional[str] = None,
                 skip_first_link: bool = True):
        self.robot = robot
        self.robot_name = robot_name or robot.name

        self._zero_length_links: Set[int] = robot.zero_length_links.copy()
        if skip_first_link:
            self._zero_length_links.add(1)

        # 预计算各连杆的相关关节
        self._link_relevant: Dict[int, Set[int]] = {}
        for link_idx in range(1, robot.n_joints + 1):
            rel_end = robot.compute_relevant_joints(link_idx)
            rel_start = (robot.compute_relevant_joints(link_idx - 1)
                         if link_idx > 1 else set())
            self._link_relevant[link_idx] = rel_end | rel_start

    # ------------------------------------------------------------------
    #  内部：创建策略实例
    # ------------------------------------------------------------------

    def _make_strategy(self, name: str):
        """根据名称创建策略实例"""
        cls_map = {
            'critical': CriticalStrategy,
            'random': RandomStrategy,
        }
        if name not in cls_map:
            raise ValueError(f"未知采样方式: {name}")
        return cls_map[name](
            robot=self.robot,
            robot_name=self.robot_name,
            zero_length_links=self._zero_length_links,
            link_relevant=self._link_relevant,
        )

    # ------------------------------------------------------------------
    #  主 API
    # ------------------------------------------------------------------

    def compute_envelope(
        self,
        joint_intervals: List[Tuple[float, float]],
        method: str = 'numerical',
        sampling: str = 'critical',
        n_random_samples: int = 10000,
        critical_proximity_threshold: float = 0.05,
        n_subdivisions: int = 1,
        skip_zero_length: bool = True,
        visualize: bool = False,
        save_report: Optional[str] = None,
        viz_kwargs: Optional[dict] = None,
    ) -> AABBEnvelopeResult:
        """计算AABB包络

        Args:
            joint_intervals: 关节区间列表
            method: 'numerical' 或 'interval'
            sampling: 'critical' | 'random'
            n_random_samples: random 模式每连杆采样数
            critical_proximity_threshold: 随机采样避开关键点邻域半径
            n_subdivisions: 每根连杆等分段数(≥1)
            skip_zero_length: 是否跳过零长度连杆
            visualize: 是否启动可视化
            save_report: 报告保存路径
            viz_kwargs: 额外可视化参数
        """
        t0 = time.time()
        n_sub = max(1, n_subdivisions)
        intervals = self._pad_intervals(joint_intervals)
        method_name = (f"{method}_{sampling}"
                       if method == 'numerical' else method)

        if method == 'numerical':
            strategy = self._make_strategy(sampling)
            # 构建策略特定的关键字参数
            strategy_kwargs: Dict = {}
            if sampling == 'random':
                strategy_kwargs['n_samples'] = n_random_samples
                strategy_kwargs['critical_proximity_threshold'] = \
                    critical_proximity_threshold

            link_aabbs, n_samples = strategy.execute(
                intervals, skip_zero_length, n_sub, **strategy_kwargs)

        elif method == 'interval':
            link_aabbs, n_samples = compute_interval_aabb(
                self.robot, intervals, self._zero_length_links,
                skip_zero_length, n_sub)
        else:
            raise ValueError(f"未知方法: {method}")

        result = AABBEnvelopeResult(
            robot_name=self.robot_name,
            n_joints=self.robot.n_joints,
            joint_intervals=intervals,
            method=method_name,
            sampling_mode=sampling if method == 'numerical' else 'N/A',
            link_aabbs=link_aabbs,
            n_samples_evaluated=n_samples,
            n_subdivisions=n_sub,
            computation_time=time.time() - t0,
        )

        if save_report:
            result.generate_report(save_path=save_report)
            logger.info("报告已保存至: %s", save_report)

        if visualize:
            from .visualizer import visualize_envelope_result
            kwargs = dict(result=result, robot=self.robot,
                          show_boundary_configs=True, show_samples=True,
                          show_aabbs=True, interactive=True)
            if viz_kwargs:
                kwargs.update(viz_kwargs)
            visualize_envelope_result(**kwargs).show()

        return result

    # ------------------------------------------------------------------
    #  内部工具
    # ------------------------------------------------------------------

    def _pad_intervals(
        self,
        intervals: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """将区间列表补齐到关节数"""
        result = list(intervals)
        while len(result) < self.robot.n_joints:
            result.append((0.0, 0.0))
        return result
