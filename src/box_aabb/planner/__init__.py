"""
box_aabb.planner - Box-RRT 机械臂避障路径规划

基于关节区间(box)拓展的 RRT 路径规划算法。核心思路：
1. 在 C-space 中随机采样无碰撞 seed 点
2. 启发式拓展 box（优先拓展雅可比范数小的方向）
3. 在 box 边界上再采样 seed 点，继续拓展
4. 构建 box tree
5. 用 RRT 线段连接多棵树，构建 Graph of Convex Sets
6. 通过 GCS 求解器优化路径
7. 路径平滑后处理

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
from .box_expansion import BoxExpander
from .box_tree import BoxTreeManager
from .box_rrt import BoxRRT
from .connector import TreeConnector
from .path_smoother import PathSmoother
from .metrics import PathMetrics, evaluate_result, compare_results, format_comparison_table

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
    'BoxTreeManager',
    'BoxRRT',
    'TreeConnector',
    'PathSmoother',
    # 评价指标
    'PathMetrics',
    'evaluate_result',
    'compare_results',
    'format_comparison_table',
]
