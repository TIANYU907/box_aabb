"""
strategies/ - 采样策略模块

提供 AABB 计算的不同采样策略：
- CriticalStrategy: 关键点枚举 + 约束优化 + 流形随机采样
- RandomStrategy:   随机采样 + 局部优化
- HybridStrategy:   混合策略（关键枚举 + 流形随机 + 随机补充）
"""

from .base import SamplingStrategy
from .critical import CriticalStrategy
from .random import RandomStrategy
from .hybrid import HybridStrategy

__all__ = [
    'SamplingStrategy',
    'CriticalStrategy',
    'RandomStrategy',
    'HybridStrategy',
]
