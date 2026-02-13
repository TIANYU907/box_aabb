"""
interval_math.py - 区间算术和仿射算术

提供区间运算和仿射形式运算，用于保守的AABB计算
"""

import math
from typing import Dict, Tuple


# ============================================================================
# 全局噪声符号计数器 (用于仿射算术)
# ============================================================================

_AFFINE_NOISE_COUNTER = 0


def _get_new_noise_index() -> int:
    """获取新的噪声符号索引"""
    global _AFFINE_NOISE_COUNTER
    _AFFINE_NOISE_COUNTER += 1
    return _AFFINE_NOISE_COUNTER


def reset_affine_noise_counter():
    """重置噪声计数器 (每次新的FK计算前调用)"""
    global _AFFINE_NOISE_COUNTER
    _AFFINE_NOISE_COUNTER = 0


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
# 仿射形式类
# ============================================================================

class AffineForm:
    """
    仿射形式：x = x0 + Σ(x_i * ε_i)
    
    其中 ε_i ∈ [-1, 1] 是噪声符号
    
    优势：
    - 跟踪变量间的线性相关性
    - x - x = [0, 0] (区间算术会得到 [-2r, 2r])
    """
    
    def __init__(self, center: float, terms: Dict[int, float] = None):
        """
        Args:
            center: 中心值 x0
            terms: {noise_index: coefficient} 字典
        """
        self.x0 = float(center)
        self.terms = terms.copy() if terms else {}
    
    @classmethod
    def from_interval(cls, lo: float, hi: float, noise_idx: int = None) -> 'AffineForm':
        """
        从区间创建仿射形式
        
        Args:
            lo, hi: 区间边界
            noise_idx: 噪声符号索引（可选，默认自动分配）
        """
        center = (lo + hi) / 2.0
        radius = (hi - lo) / 2.0
        
        if noise_idx is None:
            noise_idx = _get_new_noise_index()
        
        terms = {noise_idx: radius} if radius > 1e-15 else {}
        return cls(center, terms)
    
    def to_interval(self) -> Interval:
        """转换为标准区间"""
        radius = sum(abs(c) for c in self.terms.values())
        return Interval(self.x0 - radius, self.x0 + radius)
    
    @property
    def min(self) -> float:
        radius = sum(abs(c) for c in self.terms.values())
        return self.x0 - radius
    
    @property
    def max(self) -> float:
        radius = sum(abs(c) for c in self.terms.values())
        return self.x0 + radius
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return AffineForm(self.x0 + other, self.terms.copy())
        if isinstance(other, AffineForm):
            new_center = self.x0 + other.x0
            new_terms = self.terms.copy()
            for idx, coeff in other.terms.items():
                new_terms[idx] = new_terms.get(idx, 0.0) + coeff
            # 清理零项
            new_terms = {k: v for k, v in new_terms.items() if abs(v) > 1e-15}
            return AffineForm(new_center, new_terms)
        return self.to_interval() + other
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return AffineForm(self.x0 - other, self.terms.copy())
        if isinstance(other, AffineForm):
            new_center = self.x0 - other.x0
            new_terms = self.terms.copy()
            for idx, coeff in other.terms.items():
                new_terms[idx] = new_terms.get(idx, 0.0) - coeff
            new_terms = {k: v for k, v in new_terms.items() if abs(v) > 1e-15}
            return AffineForm(new_center, new_terms)
        return self.to_interval() - other
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            new_terms = {k: -v for k, v in self.terms.items()}
            return AffineForm(other - self.x0, new_terms)
        return other - self.to_interval()
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # 标量乘法：保持线性
            c = float(other)
            new_terms = {k: v * c for k, v in self.terms.items()}
            return AffineForm(self.x0 * c, new_terms)
        if isinstance(other, AffineForm):
            # 仿射乘法: x*y = x0*y0 + x0*Σ(yi*εi) + y0*Σ(xi*εi) + δ
            # 其中 δ = Σ(xi*εi)*Σ(yi*εi) 的范围用单个新噪声符号保守包围
            new_center = self.x0 * other.x0
            new_terms = {}

            # x0 * other 的线性部分
            for idx, coeff in other.terms.items():
                new_terms[idx] = new_terms.get(idx, 0.0) + self.x0 * coeff

            # y0 * self 的线性部分
            for idx, coeff in self.terms.items():
                new_terms[idx] = new_terms.get(idx, 0.0) + other.x0 * coeff

            # 二次余项: |Σ|xi|| * |Σ|yi||
            self_radius = sum(abs(c) for c in self.terms.values())
            other_radius = sum(abs(c) for c in other.terms.values())
            remainder = self_radius * other_radius

            if remainder > 1e-15:
                noise_idx = _get_new_noise_index()
                new_terms[noise_idx] = remainder

            new_terms = {k: v for k, v in new_terms.items() if abs(v) > 1e-15}
            return AffineForm(new_center, new_terms)
        return self.to_interval() * other
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        new_terms = {k: -v for k, v in self.terms.items()}
        return AffineForm(-self.x0, new_terms)
    
    def __repr__(self):
        return f"AF({self.x0:.4f}, [{self.min:.4f}, {self.max:.4f}])"


# ============================================================================
# 区间三角函数
# ============================================================================

def I_sin(x) -> Interval:
    """
    计算sin的区间扩展
    
    Args:
        x: Interval, AffineForm, 或 float
    """
    if isinstance(x, (int, float)):
        v = math.sin(x)
        return Interval(v, v)
    
    if isinstance(x, AffineForm):
        x = x.to_interval()
    
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
        x: Interval, AffineForm, 或 float
    """
    if isinstance(x, (int, float)):
        v = math.cos(x)
        return Interval(v, v)
    
    if isinstance(x, AffineForm):
        x = x.to_interval()
    
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


def smart_sin(x) -> AffineForm:
    """sin运算：Chebyshev 线性化保留仿射相关性

    对窄区间使用一阶泰勒展开 + 二阶余项，
    对宽区间回退到区间求值。这使得 sin(θ) 保留对 θ 的
    噪声符号依赖，大幅减少后续矩阵运算的过估计。
    """
    if not isinstance(x, AffineForm):
        interval = I_sin(x)
        return AffineForm.from_interval(interval.min, interval.max)

    x_int = x.to_interval()
    width = x_int.max - x_int.min

    # 宽区间 (>π/2)：回退为区间
    if width > math.pi / 2:
        interval = I_sin(x_int)
        return AffineForm.from_interval(interval.min, interval.max)

    # 窄区间: 使用 Chebyshev 线性近似
    # sin(x) ≈ sin(x0) + cos(x0)*(x-x0) + δ
    # |δ| ≤ width² / 8  (二阶余项上界)
    x0 = x.x0
    sin_x0 = math.sin(x0)
    cos_x0 = math.cos(x0)

    # 线性部分: sin(x0) + cos(x0)*(x - x0) = sin(x0) - cos(x0)*x0 + cos(x0)*x
    result = x * cos_x0  # AffineForm * scalar => 保留所有噪声符号
    result = result + (sin_x0 - cos_x0 * x0)

    # 二阶余项: |sin''(ξ)/2| * radius² ≤ radius² / 2
    radius = (x_int.max - x_int.min) / 2.0
    remainder = radius * radius / 2.0
    if remainder > 1e-15:
        noise_idx = _get_new_noise_index()
        result.terms[noise_idx] = remainder

    # 安全裁剪到 [-1, 1]
    iv = result.to_interval()
    if iv.min < -1.0 or iv.max > 1.0:
        interval = I_sin(x_int)
        return AffineForm.from_interval(interval.min, interval.max)

    return result


def smart_cos(x) -> AffineForm:
    """cos运算：Chebyshev 线性化保留仿射相关性

    类似 smart_sin，使用一阶线性化减少过估计。
    """
    if not isinstance(x, AffineForm):
        interval = I_cos(x)
        return AffineForm.from_interval(interval.min, interval.max)

    x_int = x.to_interval()
    width = x_int.max - x_int.min

    if width > math.pi / 2:
        interval = I_cos(x_int)
        return AffineForm.from_interval(interval.min, interval.max)

    x0 = x.x0
    cos_x0 = math.cos(x0)
    sin_x0 = math.sin(x0)

    # cos(x) ≈ cos(x0) - sin(x0)*(x - x0) + δ
    result = x * (-sin_x0)
    result = result + (cos_x0 + sin_x0 * x0)

    radius = (x_int.max - x_int.min) / 2.0
    remainder = radius * radius / 2.0
    if remainder > 1e-15:
        noise_idx = _get_new_noise_index()
        result.terms[noise_idx] = remainder

    iv = result.to_interval()
    if iv.min < -1.0 or iv.max > 1.0:
        interval = I_cos(x_int)
        return AffineForm.from_interval(interval.min, interval.max)

    return result
