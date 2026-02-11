"""
planner - Box-RRT 机械臂避障路径规划

独立于 box_aabb 的路径规划包，调用 box_aabb 提供的区间 AABB 计算
和机器人运动学模型进行碰撞检测与区间拓展。

核心思路：
1. 在 C-space 中随机采样无碰撞 seed 点
2. 启发式拓展 box（优先拓展雅可比范数小的方向）
3. 在 box 边界上再采样 seed 点，继续拓展
4. 构建 box tree
5. 用 RRT 线段连接多棵树，构建 Graph of Convex Sets
6. 通过 GCS 求解器优化路径
7. 路径平滑后处理

v4.0 新增：
- BoxForest: 可复用的 box 森林，始末点变化时无需重建
- BoxForestQuery: 基于已有森林的快速查询规划
- AABBCache: AABB 包络缓存系统，避免重复计算
- FreeSpaceTiler: 自由空间瓦片化器，反选无碰撞区间
- animate_robot_path: 动态可视化机械臂运动

参考论文:
    Marcucci et al., "Motion planning around obstacles with convex optimization",
    Science Robotics, 2023. DOI: 10.1126/scirobotics.adf7843
"""

from .models import (
    BoxNode,
    BoxTree,
    Obstacle,
    PlannerConfig,
    PlannerResult,
    Edge,
)
from .obstacles import Scene
from .collision import CollisionChecker
from .box_expansion import BoxExpander, ExpansionLog
from .box_tree import BoxTreeManager
from .box_rrt import BoxRRT
from .connector import TreeConnector
from .path_smoother import PathSmoother
from .metrics import PathMetrics, evaluate_result, compare_results, format_comparison_table
from .report import PlannerReportGenerator
from .aabb_cache import AABBCache, IntervalStore, CacheEntry
from .box_forest import BoxForest
from .deoverlap import deoverlap, compute_adjacency, compute_adjacency_incremental
from .box_query import BoxForestQuery
from .free_space_tiler import FreeSpaceTiler, FreeSpaceTile
from .dynamic_visualizer import animate_robot_path, resample_path
from .interactive_viewer import launch_viewer

__all__ = [
    # 数据模型
    'BoxNode',
    'BoxTree',
    'Obstacle',
    'PlannerConfig',
    'PlannerResult',
    'Edge',
    # 场景与碰撞
    'Scene',
    'CollisionChecker',
    # 核心算法
    'BoxExpander',
    'ExpansionLog',
    'BoxTreeManager',
    'BoxRRT',
    'TreeConnector',
    'PathSmoother',
    # 评价指标
    'PathMetrics',
    'evaluate_result',
    'compare_results',
    'format_comparison_table',
    # 报告生成
    'PlannerReportGenerator',
    # v4.0 新增
    'AABBCache',
    'IntervalStore',
    'CacheEntry',
    'BoxForest',
    'deoverlap',
    'compute_adjacency',
    'compute_adjacency_incremental',
    'BoxForestQuery',
    'FreeSpaceTiler',
    'FreeSpaceTile',
    'animate_robot_path',
    'resample_path',
    'launch_viewer',
]
