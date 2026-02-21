"""
planner/metrics.py - 路径质量评价指标

提供多维度路径质量评估：
- 路径长度 (L2)
- 平滑度 (曲率 / 角度变化)
- 安全裕度 (最小障碍物距离)
- 效率指标 (路径长度 / 直线距离)
- box 覆盖率
- 计算统计

所有指标支持从 SBFResult 或路径点列表直接计算。
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

from aabb.robot import Robot
from .models import SBFResult
from forest.models import BoxNode
from forest.scene import Scene
from forest.collision import CollisionChecker

logger = logging.getLogger(__name__)


@dataclass
class PathMetrics:
    """路径质量指标汇总

    Attributes:
        path_length: L2 路径总长度
        direct_distance: 起终点直线距离
        length_ratio: 路径长度 / 直线距离 (≥1.0, 越接近1越高效)
        smoothness: 平滑度指标 (角度变化均值, 越小越平滑)
        max_curvature: 最大曲率 (最大角度变化, rad)
        min_clearance: 最小安全裕度 (工作空间中离障碍物最近距离)
        avg_clearance: 平均安全裕度
        n_waypoints: 路径点数量
        joint_range_usage: 各关节使用范围占比
        computation_time: 规划计算时间 (s)
        n_collision_checks: 碰撞检测调用次数
        box_coverage: box 总体积 (C-space)
        n_boxes: box 数量
    """
    path_length: float = 0.0
    direct_distance: float = 0.0
    length_ratio: float = float('inf')
    smoothness: float = 0.0
    max_curvature: float = 0.0
    min_clearance: float = float('inf')
    avg_clearance: float = 0.0
    n_waypoints: int = 0
    joint_range_usage: Optional[np.ndarray] = None
    computation_time: float = 0.0
    n_collision_checks: int = 0
    box_coverage: float = 0.0
    n_boxes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        d = {
            'path_length': self.path_length,
            'direct_distance': self.direct_distance,
            'length_ratio': self.length_ratio,
            'smoothness': self.smoothness,
            'max_curvature': self.max_curvature,
            'min_clearance': self.min_clearance,
            'avg_clearance': self.avg_clearance,
            'n_waypoints': self.n_waypoints,
            'computation_time': self.computation_time,
            'n_collision_checks': self.n_collision_checks,
            'box_coverage': self.box_coverage,
            'n_boxes': self.n_boxes,
        }
        if self.joint_range_usage is not None:
            d['joint_range_usage'] = self.joint_range_usage.tolist()
        return d

    def summary(self) -> str:
        """返回可读的指标摘要"""
        lines = [
            "=" * 50,
            "路径质量指标",
            "=" * 50,
            f"路径长度 (L2):      {self.path_length:.4f}",
            f"直线距离:           {self.direct_distance:.4f}",
            f"路径效率 (比值):    {self.length_ratio:.4f}",
            f"平滑度 (均值角变):  {self.smoothness:.4f} rad",
            f"最大曲率:           {self.max_curvature:.4f} rad",
            f"最小安全裕度:       {self.min_clearance:.6f}",
            f"平均安全裕度:       {self.avg_clearance:.6f}",
            f"路径点数:           {self.n_waypoints}",
            f"计算时间:           {self.computation_time:.3f} s",
            f"碰撞检测次数:       {self.n_collision_checks}",
            f"Box 总体积:         {self.box_coverage:.6f}",
            f"Box 数量:           {self.n_boxes}",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_path_length(path: List[np.ndarray]) -> float:
    """计算 L2 路径总长度"""
    if len(path) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path[i + 1] - path[i]))
        for i in range(len(path) - 1)
    )


def compute_smoothness(path: List[np.ndarray]) -> Tuple[float, float]:
    """计算路径平滑度

    以相邻线段之间的角度变化衡量。

    Returns:
        (mean_angle_change, max_angle_change) in radians
    """
    if len(path) < 3:
        return 0.0, 0.0

    angles = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)

    if not angles:
        return 0.0, 0.0

    return float(np.mean(angles)), float(np.max(angles))


def compute_clearance(
    path: List[np.ndarray],
    robot: Robot,
    scene: Scene,
    n_samples_per_segment: int = 5,
) -> Tuple[float, float]:
    """计算路径的安全裕度（工作空间中离障碍物的最小距离）

    对路径上的每个采样点：
    1. FK 得到连杆端点
    2. 计算每个端点到每个障碍物 AABB 表面的最小距离

    Returns:
        (min_clearance, avg_clearance)
    """
    obstacles = scene.get_obstacles()
    if not obstacles or len(path) < 1:
        return float('inf'), float('inf')

    clearances: List[float] = []

    for i in range(len(path)):
        if i < len(path) - 1:
            # 在当前段上插值
            for t in np.linspace(0, 1, n_samples_per_segment, endpoint=False):
                q = path[i] + t * (path[i + 1] - path[i])
                c = _config_clearance(q, robot, obstacles)
                clearances.append(c)
        else:
            c = _config_clearance(path[i], robot, obstacles)
            clearances.append(c)

    if not clearances:
        return float('inf'), float('inf')

    return float(np.min(clearances)), float(np.mean(clearances))


def _config_clearance(
    q: np.ndarray,
    robot: Robot,
    obstacles: list,
) -> float:
    """计算单配置到所有障碍物的最小距离"""
    positions = robot.get_link_positions(q)
    min_dist = float('inf')

    for li in range(1, len(positions)):
        p = positions[li]
        for obs in obstacles:
            dist = _point_aabb_distance(p[:3], obs.min_point[:3], obs.max_point[:3])
            min_dist = min(min_dist, dist)

    return min_dist


def _point_aabb_distance(
    point: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> float:
    """计算点到 AABB 的最小距离 (0 表示在内部)"""
    ndim = min(len(point), len(aabb_min), len(aabb_max))
    clamped = np.clip(point[:ndim], aabb_min[:ndim], aabb_max[:ndim])
    return float(np.linalg.norm(point[:ndim] - clamped))


def compute_joint_range_usage(
    path: List[np.ndarray],
    joint_limits: List[Tuple[float, float]],
) -> np.ndarray:
    """计算路径中各关节使用的范围占总限制范围的比例

    Returns:
        (n_joints,) 数组，每个元素 ∈ [0, 1]
    """
    if len(path) < 2:
        return np.zeros(len(joint_limits))

    path_arr = np.array(path)
    n_joints = min(path_arr.shape[1], len(joint_limits))
    usage = np.zeros(n_joints)

    for j in range(n_joints):
        lo, hi = joint_limits[j]
        total_range = hi - lo
        if total_range < 1e-10:
            usage[j] = 0.0
        else:
            used_range = np.ptp(path_arr[:, j])
            usage[j] = min(used_range / total_range, 1.0)

    return usage


def evaluate_result(
    result: SBFResult,
    robot: Robot,
    scene: Scene,
    joint_limits: Optional[List[Tuple[float, float]]] = None,
) -> PathMetrics:
    """从 SBFResult 计算完整的路径质量指标

    Args:
        result: Box-RRT 规划结果
        robot: 机器人模型
        scene: 障碍物场景
        joint_limits: 关节限制 (默认从 robot 获取)

    Returns:
        PathMetrics 指标对象
    """
    metrics = PathMetrics()
    metrics.computation_time = result.computation_time
    metrics.n_collision_checks = result.n_collision_checks

    # box 统计
    if result.box_trees:
        metrics.n_boxes = result.n_boxes_created
        metrics.box_coverage = sum(t.total_volume for t in result.box_trees)

    if not result.success or not result.path or len(result.path) < 2:
        return metrics

    path = result.path
    metrics.n_waypoints = len(path)

    # 路径长度
    metrics.path_length = compute_path_length(path)
    metrics.direct_distance = float(np.linalg.norm(path[-1] - path[0]))
    if metrics.direct_distance > 1e-10:
        metrics.length_ratio = metrics.path_length / metrics.direct_distance
    else:
        metrics.length_ratio = 1.0

    # 平滑度
    metrics.smoothness, metrics.max_curvature = compute_smoothness(path)

    # 安全裕度
    metrics.min_clearance, metrics.avg_clearance = compute_clearance(
        path, robot, scene,
    )

    # 关节范围使用
    limits = joint_limits or robot.joint_limits
    if limits:
        metrics.joint_range_usage = compute_joint_range_usage(path, limits)

    return metrics


def compare_results(
    results: Dict[str, Tuple[SBFResult, Robot, Scene]],
    joint_limits: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, PathMetrics]:
    """比较多个规划结果的质量指标

    Args:
        results: {名称: (SBFResult, Robot, Scene)} 字典
        joint_limits: 共用关节限制

    Returns:
        {名称: PathMetrics} 字典
    """
    metrics_dict: Dict[str, PathMetrics] = {}
    for name, (result, robot, scene) in results.items():
        metrics_dict[name] = evaluate_result(result, robot, scene, joint_limits)
    return metrics_dict


def format_comparison_table(metrics_dict: Dict[str, PathMetrics]) -> str:
    """将多组指标格式化为对比表"""
    if not metrics_dict:
        return "无数据"

    names = list(metrics_dict.keys())
    header = f"{'指标':<20}" + "".join(f"{n:>16}" for n in names)

    rows = [
        ("路径长度", "path_length", ".4f"),
        ("直线距离", "direct_distance", ".4f"),
        ("路径效率", "length_ratio", ".4f"),
        ("平滑度", "smoothness", ".4f"),
        ("最大曲率", "max_curvature", ".4f"),
        ("最小安全裕度", "min_clearance", ".6f"),
        ("平均安全裕度", "avg_clearance", ".6f"),
        ("路径点数", "n_waypoints", "d"),
        ("计算时间(s)", "computation_time", ".3f"),
        ("碰撞检测次数", "n_collision_checks", "d"),
        ("Box 数量", "n_boxes", "d"),
        ("Box 体积", "box_coverage", ".4f"),
    ]

    lines = ["=" * (20 + 16 * len(names)), header, "-" * (20 + 16 * len(names))]
    for label, attr, fmt in rows:
        row = f"{label:<20}"
        for n in names:
            val = getattr(metrics_dict[n], attr)
            row += f"{val:>16{fmt}}"
        lines.append(row)
    lines.append("=" * (20 + 16 * len(names)))

    return "\n".join(lines)
