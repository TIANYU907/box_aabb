"""
models.py - 数据模型

定义 AABB 计算结果的数据类，包括 BoundaryConfig、LinkAABBInfo 和 AABBEnvelopeResult。
从 aabb_calculator.py 中提取，作为各模块共享的数据结构。
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ==================== 常量 ====================

BOUNDARY_TYPES = ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')
DIM_MAP = {'x': 0, 'y': 1, 'z': 2}


# ==================== 数据类 ====================

@dataclass
class BoundaryConfig:
    """边界配置信息

    记录使 AABB 某个面达到极值的关节配置及相关分析信息。

    Attributes:
        joint_values: 完整的关节角数组
        boundary_value: 该边界面上的极值 (m)
        boundary_type: 边界类型 ('x_min' | 'x_max' | 'y_min' | 'y_max' | 'z_min' | 'z_max')
        link_index: 对应的连杆索引 (1-based)
        relevant_joints: 影响该连杆位置的关节索引集合 (0-based)
        boundary_joints: 处于区间边界的关节索引集合
        is_aabb_vertex: 该配置是否同时是多个 AABB 面的极值点
        angle_constraints: 检测到的角度约束描述列表
    """
    joint_values: np.ndarray
    boundary_value: float
    boundary_type: str
    link_index: int
    relevant_joints: Set[int] = field(default_factory=set)
    boundary_joints: Set[int] = field(default_factory=set)
    is_aabb_vertex: bool = False
    angle_constraints: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.joint_values, np.ndarray):
            self.joint_values = np.array(self.joint_values, dtype=np.float64)

    def format_joint_values(self, joint_intervals: List[Tuple[float, float]]) -> str:
        """格式化关节值为可读字符串，标注边界和零值"""
        parts: List[str] = []
        for idx in sorted(self.relevant_joints):
            if idx >= len(self.joint_values):
                continue
            val = self.joint_values[idx]
            symbols: List[str] = []
            if idx in self.boundary_joints:
                symbols.append('📍')
            if abs(val) < 1e-6:
                symbols.append('◯')
            parts.append(f"q{idx}={val:7.4f}{''.join(symbols)}")
        return '[' + ', '.join(parts) + ']'


@dataclass
class LinkAABBInfo:
    """单个连杆（或连杆子段）的 AABB 信息

    当 n_subdivisions=1 时: segment_index=0, t_start=0, t_end=1
    当 n_subdivisions>1 时: 同一连杆产生 n 个 LinkAABBInfo，各有不同 segment_index

    Attributes:
        link_index: 连杆索引 (1-based)
        link_name: 连杆显示名称
        min_point: AABB 最小角点 [x, y, z]
        max_point: AABB 最大角点 [x, y, z]
        is_zero_length: 是否为零长度连杆
        boundary_configs: 各边界面的配置信息
        segment_index: 段索引 (0-based)
        n_segments: 该连杆的总段数
        t_start: 段在连杆上的参数起点
        t_end: 段在连杆上的参数终点
    """
    link_index: int
    link_name: str
    min_point: List[float]
    max_point: List[float]
    is_zero_length: bool = False
    boundary_configs: Dict[str, BoundaryConfig] = field(default_factory=dict)
    segment_index: int = 0
    n_segments: int = 1
    t_start: float = 0.0
    t_end: float = 1.0

    @property
    def volume(self) -> float:
        """AABB 体积 (m³)"""
        s = self.size
        return s[0] * s[1] * s[2]

    @property
    def size(self) -> List[float]:
        """AABB 各轴尺寸 [dx, dy, dz]"""
        return [max(0.0, self.max_point[i] - self.min_point[i]) for i in range(3)]


@dataclass
class AABBEnvelopeResult:
    """AABB 计算完整结果

    Attributes:
        robot_name: 机器人名称
        n_joints: 关节数
        joint_intervals: 关节区间列表
        method: 计算方法标识
        sampling_mode: 采样模式
        link_aabbs: 所有连杆（含子段）的 AABB 列表
        computation_time: 计算耗时 (s)
        n_samples_evaluated: 采样点总数
        n_subdivisions: 连杆等分段数
        timestamp: 时间戳
    """
    robot_name: str
    n_joints: int
    joint_intervals: List[Tuple[float, float]]
    method: str
    sampling_mode: str = 'critical'
    link_aabbs: List[LinkAABBInfo] = field(default_factory=list)
    computation_time: float = 0.0
    n_samples_evaluated: int = 0
    n_subdivisions: int = 1
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))

    def get_robot_aabb(self) -> Tuple[List[float], List[float]]:
        """获取包含所有连杆的整体 AABB"""
        valid = [a for a in self.link_aabbs if not a.is_zero_length]
        if not valid:
            return [0, 0, 0], [0, 0, 0]
        return (
            [min(a.min_point[i] for a in valid) for i in range(3)],
            [max(a.max_point[i] for a in valid) for i in range(3)],
        )

    def get_end_effector_aabb(self) -> Optional[LinkAABBInfo]:
        """获取末端执行器的 AABB"""
        return self.link_aabbs[-1] if self.link_aabbs else None

    def total_volume(self) -> float:
        """所有非零长度连杆 AABB 的总体积"""
        return sum(a.volume for a in self.link_aabbs if not a.is_zero_length)

    def get_link_aabbs(self, link_idx: int) -> List[LinkAABBInfo]:
        """获取某个连杆的所有段 AABB"""
        return [a for a in self.link_aabbs if a.link_index == link_idx]

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成 Markdown 分析报告
        
        Args:
            save_path: 可选的保存路径
            
        Returns:
            Markdown 格式报告字符串
        """
        from .report import ReportGenerator
        report = ReportGenerator.generate(self)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        return report


# ==================== 旧兼容结果类 ====================

@dataclass
class AABBResult:
    """旧版兼容 AABB 结果
    
    .. deprecated:: 2.0.0
        使用 :class:`LinkAABBInfo` 代替。
    """
    min_point: List[float]
    max_point: List[float]
    method: str

    @property
    def volume(self) -> float:
        s = self.size
        return s[0] * s[1] * s[2]

    @property
    def size(self) -> List[float]:
        return [self.max_point[i] - self.min_point[i] for i in range(3)]
