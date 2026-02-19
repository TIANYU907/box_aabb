"""
visualizer.py - 3D可视化工具

用于可视化机器人、AABB包围盒和障碍物。
支持边界臂形和采样臂形的可视化。

配色方案参考 visualize_methods_comparison.py:
- 中点臂形: line_style='-', linewidth=3.0, alpha=1.0
- 采样臂形: line_style='--', linewidth=1.0, alpha=0.2
- 采样步长: 0.5 radians
- 最大采样数: 100
- AABB透明度: 0.1
- 连杆颜色: 彩虹色
"""

import math
import random
import itertools
import logging
from typing import List, Dict, Any, Optional, Tuple

import warnings

logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    # 配置中文字体
    for font in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        try:
            matplotlib.rcParams['font.sans-serif'] = (
                [font] + matplotlib.rcParams['font.sans-serif'])
            break
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', category=UserWarning,
                            module='matplotlib')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Visualizer:
    """3D可视化器

    配色方案（参考 visualize_methods_comparison.py）:
    - 中点臂形: 粗实线 (linewidth=3.0, alpha=1.0)
    - 采样臂形: 细虚线 (linewidth=1.0, alpha=0.2)
    - AABB: 半透明方框 (alpha=0.1)
    """

    AABB_COLORS = ['#FF0000', '#00FF00', '#0000FF']
    LINK_COLORS = [
        '#FF6B6B', '#FFA500', '#FFFF00', '#32CD32',
        '#00FFFF', '#4169E1', '#9370DB', '#FF69B4',
    ]
    BOUNDARY_COLORS = {
        'x_max': '#FF0000', 'x_min': '#8B0000',
        'y_max': '#00FF00', 'y_min': '#006400',
        'z_max': '#0000FF', 'z_min': '#00008B',
    }

    SAMPLING_STEP = 0.5
    MAX_SAMPLES = 100

    MIDPOINT_LINEWIDTH = 3.0
    MIDPOINT_ALPHA = 1.0
    MIDPOINT_LINESTYLE = '-'

    SAMPLE_LINEWIDTH = 1.0
    SAMPLE_ALPHA = 0.2
    SAMPLE_LINESTYLE = '--'

    AABB_ALPHA = 0.1

    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        if not HAS_MATPLOTLIB:
            raise ImportError("需要安装matplotlib才能使用可视化功能")
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._setup_axes()

    def _setup_axes(self) -> None:
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.grid(True, alpha=0.3)

    def plot_robot(self, robot, joint_values: List[float],
                   color: Optional[str] = None, linewidth: float = 2.0,
                   alpha: float = 1.0, linestyle: str = '-',
                   show_joints: bool = False, joint_size: float = 40,
                   use_link_colors: bool = True,
                   end_effector_color: str = 'red',
                   end_effector_size: float = 40,
                   show_end_effector: bool = True) -> None:
        """绘制机器人当前姿态"""
        positions = robot.get_link_positions(joint_values)
        for i in range(len(positions) - 1):
            if use_link_colors and color is None:
                link_color = self.LINK_COLORS[i % len(self.LINK_COLORS)]
            else:
                link_color = color if color else 'black'
            self.ax.plot(
                [positions[i][0], positions[i+1][0]],
                [positions[i][1], positions[i+1][1]],
                [positions[i][2], positions[i+1][2]],
                linestyle=linestyle, color=link_color,
                linewidth=linewidth, alpha=alpha)

        base_pt = positions[0]
        self.ax.scatter([base_pt[0]], [base_pt[1]], [base_pt[2]],
                        c='k', s=40, marker='s', edgecolors='none', alpha=0.5)

        if show_end_effector:
            ee_pt = positions[-1]
            self.ax.scatter([ee_pt[0]], [ee_pt[1]], [ee_pt[2]],
                            c=end_effector_color, s=end_effector_size,
                            marker='o', edgecolors='none', alpha=alpha)

    def plot_aabbs(self, aabbs: List, alpha: Optional[float] = None,
                   colors: Optional[List[str]] = None,
                   show_edges: bool = True) -> None:
        """绘制AABB包围盒列表"""
        if alpha is None:
            alpha = self.AABB_ALPHA
        if colors is None:
            colors = self.AABB_COLORS
        for i, aabb in enumerate(aabbs):
            if hasattr(aabb, 'min_point'):
                min_pt, max_pt = aabb.min_point, aabb.max_point
            else:
                min_pt, max_pt = aabb
            color = colors[i % len(colors)]
            self._draw_box(min_pt, max_pt, color=color, alpha=alpha,
                           show_edges=show_edges)

    def plot_obstacles(self, obstacles: List[Dict],
                       color: str = 'red', alpha: float = 0.5) -> None:
        """绘制障碍物"""
        for obs in obstacles:
            self._draw_box(obs['min'], obs['max'], color=color, alpha=alpha)

    def _draw_box(self, min_pt: List[float], max_pt: List[float],
                  color: str = 'blue', alpha: float = 0.3,
                  show_edges: bool = True) -> None:
        """绘制3D盒子"""
        x0, y0, z0 = min_pt
        x1, y1, z1 = max_pt
        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
        ]
        collection = Poly3DCollection(
            faces, alpha=alpha, facecolor=color,
            edgecolor='black' if show_edges else None,
            linewidth=0.5 if show_edges else 0)
        self.ax.add_collection3d(collection)

    def plot_point(self, point: List[float], color: str = 'red',
                   size: float = 50, marker: str = 'o',
                   label: Optional[str] = None) -> None:
        """绘制单个点"""
        self.ax.scatter([point[0]], [point[1]], [point[2]],
                        c=color, s=size, marker=marker, label=label)

    def plot_frame(self, position: List[float],
                   rotation: Any = None, scale: float = 0.1) -> None:
        """绘制坐标系"""
        origin = np.array(position)
        if rotation is None:
            rotation = np.eye(3)
        colors = ['r', 'g', 'b']
        for i in range(3):
            direction = rotation[:, i] * scale
            end = origin + direction
            self.ax.plot([origin[0], end[0]], [origin[1], end[1]],
                         [origin[2], end[2]], color=colors[i], linewidth=2)

    def set_limits(self, x_lim: Optional[Tuple[float, float]] = None,
                   y_lim: Optional[Tuple[float, float]] = None,
                   z_lim: Optional[Tuple[float, float]] = None) -> None:
        if x_lim: self.ax.set_xlim(x_lim)
        if y_lim: self.ax.set_ylim(y_lim)
        if z_lim: self.ax.set_zlim(z_lim)

    def auto_scale(self, margin: float = 0.1) -> None:
        x_data, y_data, z_data = [], [], []
        for line in self.ax.lines:
            x_data.extend(line.get_xdata())
            y_data.extend(line.get_ydata())
            z_data.extend(line.get_zdata())
        if x_data and y_data and z_data:
            for data, setter in [(x_data, self.ax.set_xlim),
                                 (y_data, self.ax.set_ylim),
                                 (z_data, self.ax.set_zlim)]:
                lo, hi = min(data), max(data)
                m = (hi - lo) * margin
                setter(lo - m, hi + m)

    def set_title(self, title: str) -> None:
        self.ax.set_title(title)

    def add_legend(self) -> None:
        self.ax.legend()

    def clear(self) -> None:
        self.ax.clear()
        self._setup_axes()

    def show(self) -> None:
        plt.tight_layout()
        plt.show()

    def save(self, filepath: str, dpi: int = 150) -> None:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')


# ==================== 便捷函数 ====================

BOUNDARY_ARM_COLOR = '#6699FF'
BOUNDARY_MARKER_COLOR = '#FF6666'


def visualize_envelope_result(
    result,
    robot,
    show_boundary_configs: bool = True,
    show_samples: bool = True,
    show_aabbs: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    interactive: bool = True,
    boundary_link_index: Optional[int] = None,
) -> Visualizer:
    """可视化 AABBEnvelopeResult

    Args:
        result: AABBEnvelopeResult 实例
        robot: Robot 实例
        show_boundary_configs: 是否显示边界臂形
        show_samples: 是否显示采样臂形
        show_aabbs: 是否显示 AABB
        title: 标题
        figsize: 图形大小
        save_path: 保存路径
        interactive: 是否显示交互提示
        boundary_link_index: 边界臂形要显示的连杆索引（默认末端连杆）

    Returns:
        Visualizer 实例
    """
    viz = Visualizer(figsize=figsize)

    if show_aabbs:
        valid_aabbs = [a for a in result.link_aabbs if not a.is_zero_length]
        for i, aabb in enumerate(valid_aabbs):
            color = viz.AABB_COLORS[i % len(viz.AABB_COLORS)]
            viz._draw_box(aabb.min_point, aabb.max_point,
                          color=color, alpha=viz.AABB_ALPHA, show_edges=True)

    # 中点臂形
    mid_values = [(lo + hi) / 2 for lo, hi in result.joint_intervals]
    viz.plot_robot(robot, mid_values,
                   linewidth=viz.SAMPLE_LINEWIDTH,
                   alpha=viz.SAMPLE_ALPHA,
                   linestyle=viz.SAMPLE_LINESTYLE,
                   use_link_colors=True,
                   end_effector_color='#66CC66',
                   end_effector_size=15,
                   show_end_effector=True)

    if show_boundary_configs:
        target_idx = boundary_link_index or robot.n_joints
        _plot_boundary_configurations(viz, result, robot, target_idx)

    if show_samples:
        joint_intervals = result.joint_intervals
        sampled_combinations = _generate_interval_samples(joint_intervals)
        logger.info("可视化 %d 个采样组合", len(sampled_combinations))
        for sample_values in sampled_combinations:
            viz.plot_robot(robot, sample_values,
                           linewidth=viz.SAMPLE_LINEWIDTH,
                           alpha=viz.SAMPLE_ALPHA,
                           linestyle=viz.SAMPLE_LINESTYLE,
                           use_link_colors=True,
                           end_effector_color='#66CC66',
                           end_effector_size=15,
                           show_end_effector=True)

    _add_envelope_legend(viz, show_boundary_configs, show_samples,
                         boundary_link_index)

    if title:
        viz.set_title(title)
    else:
        viz.set_title(
            f'{result.robot_name} - Robot with Boundary and Sampled Configs')

    if save_path:
        viz.save(save_path)

    if interactive:
        logger.info(
            '交互式 3D 查看器:\n'
            '  • 左键拖动 - 旋转\n'
            '  • 滚轮缩放\n'
            '  • 右键拖动 - 平移\n'
            '关闭窗口以继续...')

    return viz


def _generate_interval_samples(
    joint_intervals: List[Tuple[float, float]],
) -> List[List[float]]:
    """生成区间采样点"""
    sampling_step = Visualizer.SAMPLING_STEP
    max_samples = Visualizer.MAX_SAMPLES

    joint_samples: List[List[float]] = []
    for min_v, max_v in joint_intervals:
        interval_length = max_v - min_v
        if interval_length <= 0:
            joint_samples.append([min_v])
        else:
            num_steps = math.ceil(interval_length / sampling_step)
            actual_step = interval_length / num_steps if num_steps > 0 else 0
            samples = [min_v + i * actual_step for i in range(num_steps + 1)]
            joint_samples.append(samples)

    logger.debug("关节采样数: %s", [len(s) for s in joint_samples])
    total_combinations = 1
    for s in joint_samples:
        total_combinations *= len(s)
    logger.debug("总组合数: %d", total_combinations)

    all_combinations = list(itertools.product(*joint_samples))
    if len(all_combinations) > max_samples:
        sampled_combinations = random.sample(all_combinations, max_samples)
        logger.debug("随机采样了 %d / %d 个组合",
                      max_samples, len(all_combinations))
    else:
        sampled_combinations = all_combinations

    return [list(c) for c in sampled_combinations]


def _plot_boundary_configurations(
    viz: Visualizer,
    result,
    robot,
    target_link_index: int,
) -> None:
    """绘制指定连杆的边界臂形"""
    target_aabb = None
    for aabb in result.link_aabbs:
        if aabb.link_index == target_link_index and not aabb.is_zero_length:
            target_aabb = aabb
            break

    if target_aabb is None or not target_aabb.boundary_configs:
        logger.debug("未找到 Link %d 的边界配置", target_link_index)
        return

    plotted_count = 0
    plotted_configs: set = set()

    for boundary_type in ['x_max', 'x_min', 'y_max', 'y_min',
                           'z_max', 'z_min']:
        if boundary_type not in target_aabb.boundary_configs:
            continue
        config = target_aabb.boundary_configs[boundary_type]
        joint_values = list(config.joint_values)
        config_key = tuple(joint_values)
        if config_key in plotted_configs:
            continue
        plotted_configs.add(config_key)

        positions = robot.get_link_positions(joint_values)
        idx_start = max(0, target_link_index - 1)
        idx_end = min(target_link_index, len(positions) - 1)
        pos_start = positions[idx_start]
        pos_end = positions[idx_end]

        dim_idx = {'x': 0, 'y': 1, 'z': 2}[boundary_type[0]]
        boundary_value = config.boundary_value
        dist_start = abs(pos_start[dim_idx] - boundary_value)
        dist_end = abs(pos_end[dim_idx] - boundary_value)
        marker_position = pos_start if dist_start < dist_end else pos_end

        viz.plot_robot(robot, joint_values,
                       color=BOUNDARY_ARM_COLOR, linewidth=2.5,
                       alpha=0.9, linestyle='-',
                       use_link_colors=False, show_end_effector=False)
        viz.ax.scatter(
            [marker_position[0]], [marker_position[1]],
            [marker_position[2]],
            c=BOUNDARY_MARKER_COLOR, s=80, marker='o',
            edgecolors='none', alpha=0.9)
        plotted_count += 1

    logger.debug("绘制了 %d 个 Link %d 边界臂形",
                  plotted_count, target_link_index)


def _add_envelope_legend(
    viz: Visualizer,
    show_boundary: bool,
    show_samples: bool,
    boundary_link_index: Optional[int] = None,
) -> None:
    """添加图例"""
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='blue', markersize=10,
                   alpha=0.3, label='Link AABBs'),
        plt.Line2D([0], [0], color='gray', linewidth=1.0,
                   linestyle='--', alpha=0.5, marker='o',
                   markersize=4, markerfacecolor='#66CC66',
                   label='Sampled Configs'),
    ]
    if show_boundary:
        link_label = (f'Link{boundary_link_index}'
                      if boundary_link_index else 'End-Effector')
        legend_elements.append(
            plt.Line2D([0], [0], color=BOUNDARY_ARM_COLOR,
                       linewidth=2.5, linestyle='-', marker='o',
                       markersize=6,
                       markerfacecolor=BOUNDARY_MARKER_COLOR,
                       label=f'{link_label} Boundary Configs'))
    viz.ax.legend(handles=legend_elements, loc='upper left', fontsize=9)


def visualize_robot_with_aabbs(
    robot,
    joint_values: Optional[List[float]] = None,
    joint_intervals: Optional[List[Tuple[float, float]]] = None,
    method: str = 'numerical',
    obstacles: Optional[List[Dict]] = None,
    title: Optional[str] = None,
) -> Visualizer:
    """一站式可视化函数"""
    from .aabb_calculator import AABBCalculator

    viz = Visualizer()
    if obstacles:
        viz.plot_obstacles(obstacles)
    if joint_intervals:
        calc = AABBCalculator(robot)
        aabbs = calc.compute_link_aabbs(joint_intervals, method=method)
        viz.plot_aabbs(aabbs)
    if joint_values:
        viz.plot_robot(robot, joint_values)
    elif joint_intervals:
        mid_values = [(lo + hi) / 2 for lo, hi in joint_intervals]
        while len(mid_values) < robot.n_joints:
            mid_values.append(0.0)
        viz.plot_robot(robot, mid_values)
    if title:
        viz.set_title(title)
    viz.set_limits((-1, 1), (-1, 1), (0, 1.5))
    viz.show()
    return viz
