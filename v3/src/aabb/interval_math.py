"""
interval_math.py - 区间算术

提供区间运算，用于保守的AABB计算
"""

import math
from typing import Tuple


# ============================================================================
# 区间类
# ============================================================================

class Interval:
    """
    区间 [min, max]
    
    支持基本的区间算术运算
    """
    
    def __init__(self, lo: float, hi: float):
        self.min = float(lo)
        self.max = float(hi)
        if self.min > self.max:
            self.min, self.max = self.max, self.min
    
    @property
    def center(self) -> float:
        return (self.min + self.max) / 2
    
    @property
    def radius(self) -> float:
        return (self.max - self.min) / 2
    
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.min + other.min, self.max + other.max)
        elif isinstance(other, (int, float)):
            return Interval(self.min + other, self.max + other)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.min - other.max, self.max - other.min)
        elif isinstance(other, (int, float)):
            return Interval(self.min - other, self.max - other)
        return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Interval(other - self.max, other - self.min)
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [
                self.min * other.min, self.min * other.max,
                self.max * other.min, self.max * other.max
            ]
            return Interval(min(products), max(products))
        elif isinstance(other, (int, float)):
            p1, p2 = self.min * other, self.max * other
            return Interval(min(p1, p2), max(p1, p2))
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        return Interval(-self.max, -self.min)
    
    def __repr__(self):
        return f"[{self.min:.4f}, {self.max:.4f}]"


# ============================================================================
# 区间三角函数
# ============================================================================

def I_sin(x) -> Interval:
    """
    计算sin的区间扩展
    
    Args:
        x: Interval 或 float
    """
    if isinstance(x, (int, float)):
        v = math.sin(x)
        return Interval(v, v)
    
    lo, hi = x.min, x.max
    
    # 处理宽度超过2π的情况
    if hi - lo >= 2 * math.pi:
        return Interval(-1.0, 1.0)
    
    # 标准化到 [0, 2π)
    two_pi = 2 * math.pi
    lo_norm = lo % two_pi
    hi_norm = lo_norm + (hi - lo)
    
    # 计算端点值
    sin_lo = math.sin(lo)
    sin_hi = math.sin(hi)
    
    result_min = min(sin_lo, sin_hi)
    result_max = max(sin_lo, sin_hi)
    
    # 检查是否包含极值点
    half_pi = math.pi / 2
    
    # 检查 π/2 + 2kπ (最大值点)
    k_start = math.ceil((lo_norm - half_pi) / two_pi)
    k_end = math.floor((hi_norm - half_pi) / two_pi)
    if k_start <= k_end:
        result_max = 1.0
    
    # 检查 3π/2 + 2kπ (最小值点)
    three_half_pi = 3 * math.pi / 2
    k_start = math.ceil((lo_norm - three_half_pi) / two_pi)
    k_end = math.floor((hi_norm - three_half_pi) / two_pi)
    if k_start <= k_end:
        result_min = -1.0
    
    return Interval(result_min, result_max)


def I_cos(x) -> Interval:
    """
    计算cos的区间扩展
    
    Args:
        x: Interval 或 float
    """
    if isinstance(x, (int, float)):
        v = math.cos(x)
        return Interval(v, v)
    
    lo, hi = x.min, x.max
    
    if hi - lo >= 2 * math.pi:
        return Interval(-1.0, 1.0)
    
    two_pi = 2 * math.pi
    lo_norm = lo % two_pi
    hi_norm = lo_norm + (hi - lo)
    
    cos_lo = math.cos(lo)
    cos_hi = math.cos(hi)
    
    result_min = min(cos_lo, cos_hi)
    result_max = max(cos_lo, cos_hi)
    
    # 检查 0 + 2kπ (最大值点)
    k_start = math.ceil(lo_norm / two_pi)
    k_end = math.floor(hi_norm / two_pi)
    if k_start <= k_end:
        result_max = 1.0
    
    # 检查 π + 2kπ (最小值点)
    k_start = math.ceil((lo_norm - math.pi) / two_pi)
    k_end = math.floor((hi_norm - math.pi) / two_pi)
    if k_start <= k_end:
        result_min = -1.0
    
    return Interval(result_min, result_max)
