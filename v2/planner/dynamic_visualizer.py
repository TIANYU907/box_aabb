"""
planner/dynamic_visualizer.py - 动态可视化：机械臂路径动画

给定机器人 DH 参数和关节空间路径，在环境中动态展现
机械臂从头到尾的动作。

核心功能：
- animate_robot_path(): 生成机械臂运动动画
- 自动检测 2D 平面 / 3D 空间机器人
- 支持障碍物显示、末端轨迹跟踪、帧率控制
- 可保存为 GIF/MP4

使用方式：
    from planner.dynamic_visualizer import animate_robot_path
    anim = animate_robot_path(robot, path, scene=scene)
    anim.save("robot_motion.gif", writer='pillow', fps=20)
"""

import logging
from typing import List, Optional, Tuple, Any

import numpy as np

from aabb.robot import Robot
from forest.scene import Scene

logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for font in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        try:
            matplotlib.rcParams['font.sans-serif'] = (
                [font] + matplotlib.rcParams['font.sans-serif'])
            break
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def animate_robot_path(
    robot: Robot,
    path: List[np.ndarray],
    scene: Optional[Scene] = None,
    fps: int = 20,
    trail_length: int = 30,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "Robot Motion Animation",
    show_ee_trail: bool = True,
    ghost_interval: int = 0,
    repeat: bool = True,
) -> Any:
    """生成机械臂运动路径的动画

    自动检测机器人是 2D 平面型还是 3D 空间型，选择对应的可视化方式。

    Args:
        robot: 机器人模型（含 DH 参数）
        path: 关节空间路径点序列
        scene: 障碍物场景（可选）
        fps: 帧率
        trail_length: 末端轨迹尾巴长度（帧数）
        figsize: 图形尺寸
        title: 动画标题
        show_ee_trail: 是否显示末端执行器轨迹
        ghost_interval: 残影间隔（0 表示不显示残影）
        repeat: 是否循环播放

    Returns:
        matplotlib.animation.FuncAnimation 对象
        可调用 anim.save("out.gif", writer='pillow', fps=fps) 保存

    Raises:
        ImportError: 若 matplotlib 不可用
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib 未安装，无法生成动画")

    if not path or len(path) < 2:
        raise ValueError("路径至少需要 2 个路径点")

    # 检测是否为平面机器人
    is_planar = all(
        abs(p['alpha']) < 1e-10 and abs(p['d']) < 1e-10
        for p in robot.dh_params
    )

    # 预计算所有帧的连杆位置和末端轨迹
    all_positions = []
    ee_trail = []
    for q in path:
        positions = robot.get_link_positions(np.asarray(q, dtype=np.float64))
        all_positions.append(positions)
        ee_trail.append(positions[-1].copy())

    if is_planar:
        return _animate_2d(
            robot, all_positions, ee_trail, scene,
            fps, trail_length, figsize, title, show_ee_trail,
            ghost_interval, repeat)
    else:
        return _animate_3d(
            robot, all_positions, ee_trail, scene,
            fps, trail_length, figsize, title, show_ee_trail,
            ghost_interval, repeat)


def _animate_2d(
    robot, all_positions, ee_trail, scene,
    fps, trail_length, figsize, title, show_ee_trail,
    ghost_interval, repeat,
):
    """2D 平面机械臂动画"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 计算坐标范围
    all_xs = [p[i][0] for p in all_positions for i in range(len(p))]
    all_ys = [p[i][1] for p in all_positions for i in range(len(p))]
    margin = 0.3
    x_range = (min(all_xs) - margin, max(all_xs) + margin)
    y_range = (min(all_ys) - margin, max(all_ys) + margin)

    # 画障碍物（静态）
    if scene is not None:
        for obs in scene.get_obstacles():
            rect = Rectangle(
                (obs.min_point[0], obs.min_point[1]),
                obs.size[0], obs.size[1],
                linewidth=1, edgecolor='red', facecolor='red', alpha=0.4,
            )
            ax.add_patch(rect)

    # 初始化绘图元素
    arm_line, = ax.plot([], [], 'o-', color='#2196F3', linewidth=3,
                         markersize=6, zorder=5)
    ee_dot, = ax.plot([], [], 'o', color='#FF5722', markersize=8, zorder=6)
    trail_line, = ax.plot([], [], '-', color='#FF5722', linewidth=1.5,
                           alpha=0.6, zorder=4)
    ghost_lines = []
    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                          fontsize=10, va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    n_frames = len(all_positions)

    def init():
        arm_line.set_data([], [])
        ee_dot.set_data([], [])
        trail_line.set_data([], [])
        frame_text.set_text('')
        return arm_line, ee_dot, trail_line, frame_text

    def update(frame):
        positions = all_positions[frame]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        arm_line.set_data(xs, ys)

        # 末端位置
        ee_dot.set_data([xs[-1]], [ys[-1]])

        # 末端轨迹
        if show_ee_trail:
            start = max(0, frame - trail_length)
            trail_xs = [ee_trail[i][0] for i in range(start, frame + 1)]
            trail_ys = [ee_trail[i][1] for i in range(start, frame + 1)]
            trail_line.set_data(trail_xs, trail_ys)

        # 残影
        if ghost_interval > 0:
            # 清除旧残影
            for gl in ghost_lines:
                gl.remove()
            ghost_lines.clear()

            for g in range(0, frame, ghost_interval):
                g_pos = all_positions[g]
                g_xs = [p[0] for p in g_pos]
                g_ys = [p[1] for p in g_pos]
                gl, = ax.plot(g_xs, g_ys, 'o-', color='gray',
                              linewidth=1, markersize=2, alpha=0.3, zorder=2)
                ghost_lines.append(gl)

        frame_text.set_text(f'Frame {frame + 1}/{n_frames}')
        return arm_line, ee_dot, trail_line, frame_text

    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=False, repeat=repeat)

    return anim


def _animate_3d(
    robot, all_positions, ee_trail, scene,
    fps, trail_length, figsize, title, show_ee_trail,
    ghost_interval, repeat,
):
    """3D 空间机械臂动画"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 计算坐标范围
    all_xs = [p[i][0] for p in all_positions for i in range(len(p))]
    all_ys = [p[i][1] for p in all_positions for i in range(len(p))]
    all_zs = [p[i][2] for p in all_positions for i in range(len(p))]
    margin = 0.2
    x_range = (min(all_xs) - margin, max(all_xs) + margin)
    y_range = (min(all_ys) - margin, max(all_ys) + margin)
    z_range = (min(all_zs) - margin, max(all_zs) + margin)

    # 画障碍物（静态）
    if scene is not None:
        for obs in scene.get_obstacles():
            _draw_3d_box(ax, obs.min_point, obs.max_point,
                         color='red', alpha=0.25)

    # 初始化绘图元素
    arm_line, = ax.plot([], [], [], 'o-', color='#2196F3', linewidth=3,
                         markersize=5, zorder=5)
    ee_dot, = ax.plot([], [], [], 'o', color='#FF5722', markersize=8, zorder=6)
    trail_line, = ax.plot([], [], [], '-', color='#FF5722', linewidth=1.5,
                           alpha=0.6, zorder=4)

    frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                           fontsize=10, va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    ghost_lines = []
    n_frames = len(all_positions)

    def init():
        arm_line.set_data_3d([], [], [])
        ee_dot.set_data_3d([], [], [])
        trail_line.set_data_3d([], [], [])
        frame_text.set_text('')
        return arm_line, ee_dot, trail_line, frame_text

    def update(frame):
        positions = all_positions[frame]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        arm_line.set_data_3d(xs, ys, zs)

        ee_dot.set_data_3d([xs[-1]], [ys[-1]], [zs[-1]])

        if show_ee_trail:
            start = max(0, frame - trail_length)
            trail_xs = [ee_trail[i][0] for i in range(start, frame + 1)]
            trail_ys = [ee_trail[i][1] for i in range(start, frame + 1)]
            trail_zs = [ee_trail[i][2] for i in range(start, frame + 1)]
            trail_line.set_data_3d(trail_xs, trail_ys, trail_zs)

        # 残影
        if ghost_interval > 0:
            for gl in ghost_lines:
                gl.remove()
            ghost_lines.clear()
            for g in range(0, frame, ghost_interval):
                g_pos = all_positions[g]
                g_xs = [p[0] for p in g_pos]
                g_ys = [p[1] for p in g_pos]
                g_zs = [p[2] for p in g_pos]
                gl, = ax.plot(g_xs, g_ys, g_zs, 'o-', color='gray',
                              linewidth=1, markersize=2, alpha=0.2, zorder=2)
                ghost_lines.append(gl)

        frame_text.set_text(f'Frame {frame + 1}/{n_frames}')
        return arm_line, ee_dot, trail_line, frame_text

    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=False, repeat=repeat)

    return anim


def _draw_3d_box(ax, min_pt, max_pt, color='red', alpha=0.3):
    """绘制 3D 方框"""
    x0, y0, z0 = min_pt[:3]
    x1, y1, z1 = max_pt[:3]
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
        edgecolor='black', linewidth=0.5)
    ax.add_collection3d(collection)


def resample_path(
    path: List[np.ndarray],
    n_frames: int = 100,
) -> List[np.ndarray]:
    """将路径重采样为指定帧数（等弧长参数化）

    Args:
        path: 原始路径点序列
        n_frames: 目标帧数

    Returns:
        重采样后的路径
    """
    if len(path) < 2:
        return list(path)

    # 计算累计弧长
    cum_lengths = [0.0]
    for i in range(1, len(path)):
        seg_len = float(np.linalg.norm(
            np.asarray(path[i]) - np.asarray(path[i - 1])))
        cum_lengths.append(cum_lengths[-1] + seg_len)

    total = cum_lengths[-1]
    if total < 1e-10:
        return [path[0].copy()] * n_frames

    # 等弧长采样
    resampled = []
    for k in range(n_frames):
        t = k / (n_frames - 1) * total
        # 找到 t 所在的线段
        for i in range(1, len(cum_lengths)):
            if cum_lengths[i] >= t - 1e-10:
                seg_start = cum_lengths[i - 1]
                seg_end = cum_lengths[i]
                seg_len = seg_end - seg_start
                if seg_len < 1e-10:
                    alpha = 0.0
                else:
                    alpha = (t - seg_start) / seg_len
                q = (1 - alpha) * np.asarray(path[i - 1]) + \
                    alpha * np.asarray(path[i])
                resampled.append(q)
                break
        else:
            resampled.append(np.asarray(path[-1]).copy())

    return resampled
