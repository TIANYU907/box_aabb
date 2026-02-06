"""
BOX-AABB: 机器人连杆包围盒计算库

提供四种AABB计算策略：
1. numerical/critical - 关键点枚举（梯度为零 + 约束点优化 + 流形随机采样）
2. numerical/random   - 随机采样 + 局部优化（可避开关键点邻域）
3. numerical/hybrid   - 混合策略（关键枚举 + 约束优化 + 流形随机 + 随机补充）
4. interval           - 区间/仿射算术保守估计

v2.0 重构亮点：
- AABBCalculator 瘦身为调度层，策略 / 优化 / 区间 FK / 报告各自独立模块
- 采样策略使用 Strategy 模式，消除 critical / random / hybrid 的代码重复
- 数据模型独立为 models.py，各模块共享
- 耦合关节约束参数化到 Robot 类，不再硬编码
- 所有 print 替换为 logging
- 添加 pyproject.toml 项目打包配置
"""

from .robot import Robot, create_panda_robot, load_robot, PANDA_JOINT_LIMITS
from .models import (
    AABBResult,
    AABBEnvelopeResult,
    LinkAABBInfo,
    BoundaryConfig,
)
from .aabb_calculator import AABBCalculator
from .report import ReportGenerator
from .visualizer import Visualizer, visualize_envelope_result

__version__ = "3.0.0"
__all__ = [
    # 核心
    'Robot',
    'load_robot',
    'create_panda_robot',
    'PANDA_JOINT_LIMITS',
    'AABBCalculator',
    # 数据模型
    'AABBResult',
    'AABBEnvelopeResult',
    'LinkAABBInfo',
    'BoundaryConfig',
    # 报告
    'ReportGenerator',
    # 可视化
    'Visualizer',
    'visualize_envelope_result',
]
