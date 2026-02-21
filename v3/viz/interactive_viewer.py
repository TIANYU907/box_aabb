"""
viz/interactive_viewer.py - 交互式 3D 路径回放查看器

基于 matplotlib TkAgg 后端，在可交互窗口中循环播放规划路径。
支持：
  - 鼠标拖拽旋转 3D 视角
  - Space 暂停/播放
  - ←/→ 单步前进/后退
  - +/- 加速/减速
  - R 重置到第 0 帧
  - Q / Esc 关闭

使用方式：
    from viz.interactive_viewer import launch_viewer
    launch_viewer(robot, path, scene=scene)

也可通过 CLI 入口：
    python -m examples.view_path <path.json>
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from aabb.robot import Robot
from forest.scene import Scene

logger = logging.getLogger(__name__)

try:
    import matplotlib
    # 尝试使用可交互后端 (TkAgg)
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass  # 在某些环境中可能已锁定后端
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Rectangle

    for _font in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        try:
            matplotlib.rcParams['font.sans-serif'] = (
                [_font] + matplotlib.rcParams['font.sans-serif'])
            break
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _draw_3d_box(ax, min_pt, max_pt, color='red', alpha=0.3):
    """绘制 3D AABB 方块"""
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


def launch_viewer(
    robot: Robot,
    path: List[np.ndarray],
    scene: Optional[Scene] = None,
    fps: int = 20,
    trail_length: int = 40,
    figsize: Tuple[float, float] = (11, 9),
    title: str = "",
    ghost_interval: int = 0,
) -> None:
    """启动交互式路径回放窗口

    Args:
        robot: 机器人模型
        path: 关节空间路径点序列 (已重采样或原始)
        scene: 障碍物场景（可选）
        fps: 初始帧率
        trail_length: 末端轨迹尾长度（帧数）
        figsize: 窗口尺寸
        title: 窗口标题
        ghost_interval: 残影间隔（0=不显示）
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib 未安装，无法启动交互式查看器")
    if not path or len(path) < 2:
        raise ValueError("路径至少需要 2 个路径点")

    # ── 检测 2D / 3D ──
    is_planar = all(
        abs(p['alpha']) < 1e-10 and abs(p['d']) < 1e-10
        for p in robot.dh_params
    )

    # ── 预计算连杆位置 ──
    all_positions = []
    ee_trail = []
    for q in path:
        positions = robot.get_link_positions(np.asarray(q, dtype=np.float64))
        all_positions.append(positions)
        ee_trail.append(positions[-1].copy())

    n_frames = len(all_positions)
    if not title:
        title = f"Interactive Viewer — {robot.name} ({n_frames} frames)"

    # ── 播放状态 ──
    state = {
        'playing': True,
        'frame': 0,
        'interval': 1000 // fps,   # ms per frame
        'fps': fps,
    }

    if is_planar:
        _run_2d(robot, all_positions, ee_trail, scene,
                trail_length, figsize, title, ghost_interval, state, n_frames)
    else:
        _run_3d(robot, all_positions, ee_trail, scene,
                trail_length, figsize, title, ghost_interval, state, n_frames)


# ─────────────────────────────────────────────────────────────
#  3D 交互式回放
# ─────────────────────────────────────────────────────────────

def _run_3d(robot, all_positions, ee_trail, scene,
            trail_length, figsize, title, ghost_interval, state, n_frames):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # ── 坐标范围 ──
    all_xs = [p[i][0] for p in all_positions for i in range(len(p))]
    all_ys = [p[i][1] for p in all_positions for i in range(len(p))]
    all_zs = [p[i][2] for p in all_positions for i in range(len(p))]
    margin = 0.2
    ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    ax.set_zlim(min(all_zs) - margin, max(all_zs) + margin)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # ── 障碍物 ──
    if scene is not None:
        for obs in scene.get_obstacles():
            _draw_3d_box(ax, obs.min_point, obs.max_point,
                         color='red', alpha=0.25)

    # ── 起终点标记 ──
    start_pos = all_positions[0]
    goal_pos = all_positions[-1]
    ax.plot([p[0] for p in start_pos], [p[1] for p in start_pos],
            [p[2] for p in start_pos], 'o-', color='green',
            linewidth=2, markersize=4, alpha=0.5, label='Start')
    ax.plot([p[0] for p in goal_pos], [p[1] for p in goal_pos],
            [p[2] for p in goal_pos], 'o-', color='purple',
            linewidth=2, markersize=4, alpha=0.5, label='Goal')

    # ── 动态元素 ──
    arm_line, = ax.plot([], [], [], 'o-', color='#2196F3', linewidth=3,
                         markersize=5, zorder=5)
    ee_dot, = ax.plot([], [], [], 'o', color='#FF5722', markersize=8, zorder=6)
    trail_line, = ax.plot([], [], [], '-', color='#FF5722', linewidth=1.5,
                           alpha=0.6, zorder=4)

    info_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                          fontsize=10, va='top', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    help_text = ax.text2D(0.98, 0.02, 'Space=Play/Pause  Left/Right=Step  +/-=Speed  R=Reset  Q=Quit',
                          transform=ax.transAxes, fontsize=8, va='bottom',
                          ha='right', color='gray')

    ax.legend(loc='upper right', fontsize=8)
    ghost_lines = []

    def _update_display():
        frame = state['frame']
        positions = all_positions[frame]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        arm_line.set_data_3d(xs, ys, zs)
        ee_dot.set_data_3d([xs[-1]], [ys[-1]], [zs[-1]])

        # 末端轨迹
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

        status = "[PLAY]" if state['playing'] else "[PAUSE]"
        info_text.set_text(
            f"{status}  Frame {frame + 1}/{n_frames}  "
            f"FPS={state['fps']}")
        fig.canvas.draw_idle()

    def _on_timer(frame_arg=None):
        if state['playing']:
            state['frame'] = (state['frame'] + 1) % n_frames
            _update_display()
        return (arm_line, ee_dot, trail_line, info_text)

    def _on_key(event):
        if event.key == ' ':
            state['playing'] = not state['playing']
            _update_display()
        elif event.key == 'right':
            state['playing'] = False
            state['frame'] = (state['frame'] + 1) % n_frames
            _update_display()
        elif event.key == 'left':
            state['playing'] = False
            state['frame'] = (state['frame'] - 1) % n_frames
            _update_display()
        elif event.key in ('+', '='):
            state['fps'] = min(state['fps'] + 5, 120)
            state['interval'] = max(1000 // state['fps'], 8)
            anim.event_source.interval = state['interval']
            _update_display()
        elif event.key in ('-', '_'):
            state['fps'] = max(state['fps'] - 5, 1)
            state['interval'] = 1000 // state['fps']
            anim.event_source.interval = state['interval']
            _update_display()
        elif event.key in ('r', 'R'):
            state['frame'] = 0
            state['playing'] = True
            _update_display()
        elif event.key in ('q', 'Q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', _on_key)

    anim = FuncAnimation(
        fig, _on_timer, frames=n_frames,
        interval=state['interval'], blit=False, repeat=True)

    # 初始显示
    _update_display()

    logger.info("交互式查看器已启动 — Space=暂停/播放, 左右=单步, +/-=变速, R=重置, Q=退出")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  2D 交互式回放
# ─────────────────────────────────────────────────────────────

def _run_2d(robot, all_positions, ee_trail, scene,
            trail_length, figsize, title, ghost_interval, state, n_frames):

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ── 坐标范围 ──
    all_xs = [p[i][0] for p in all_positions for i in range(len(p))]
    all_ys = [p[i][1] for p in all_positions for i in range(len(p))]
    margin = 0.3
    ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # ── 障碍物 ──
    if scene is not None:
        for obs in scene.get_obstacles():
            rect = Rectangle(
                (obs.min_point[0], obs.min_point[1]),
                obs.size[0], obs.size[1],
                linewidth=1, edgecolor='red', facecolor='red', alpha=0.4)
            ax.add_patch(rect)

    # ── 起终点 ──
    start_pos = all_positions[0]
    goal_pos = all_positions[-1]
    ax.plot([p[0] for p in start_pos], [p[1] for p in start_pos],
            'o-', color='green', linewidth=2, markersize=4, alpha=0.5, label='Start')
    ax.plot([p[0] for p in goal_pos], [p[1] for p in goal_pos],
            'o-', color='purple', linewidth=2, markersize=4, alpha=0.5, label='Goal')

    # ── 动态元素 ──
    arm_line, = ax.plot([], [], 'o-', color='#2196F3', linewidth=3,
                         markersize=6, zorder=5)
    ee_dot, = ax.plot([], [], 'o', color='#FF5722', markersize=8, zorder=6)
    trail_line, = ax.plot([], [], '-', color='#FF5722', linewidth=1.5,
                           alpha=0.6, zorder=4)

    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=10, va='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    help_text = ax.text(0.98, 0.02, 'Space=Play/Pause  Left/Right=Step  +/-=Speed  R=Reset  Q=Quit',
                        transform=ax.transAxes, fontsize=8, va='bottom',
                        ha='right', color='gray')

    ax.legend(loc='upper right', fontsize=8)
    ghost_lines = []

    def _update_display():
        frame = state['frame']
        positions = all_positions[frame]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        arm_line.set_data(xs, ys)
        ee_dot.set_data([xs[-1]], [ys[-1]])

        start = max(0, frame - trail_length)
        trail_xs = [ee_trail[i][0] for i in range(start, frame + 1)]
        trail_ys = [ee_trail[i][1] for i in range(start, frame + 1)]
        trail_line.set_data(trail_xs, trail_ys)

        if ghost_interval > 0:
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

        status = "[PLAY]" if state['playing'] else "[PAUSE]"
        info_text.set_text(
            f"{status}  Frame {frame + 1}/{n_frames}  "
            f"FPS={state['fps']}")
        fig.canvas.draw_idle()

    def _on_timer(frame_arg=None):
        if state['playing']:
            state['frame'] = (state['frame'] + 1) % n_frames
            _update_display()
        return (arm_line, ee_dot, trail_line, info_text)

    def _on_key(event):
        if event.key == ' ':
            state['playing'] = not state['playing']
            _update_display()
        elif event.key == 'right':
            state['playing'] = False
            state['frame'] = (state['frame'] + 1) % n_frames
            _update_display()
        elif event.key == 'left':
            state['playing'] = False
            state['frame'] = (state['frame'] - 1) % n_frames
            _update_display()
        elif event.key in ('+', '='):
            state['fps'] = min(state['fps'] + 5, 120)
            state['interval'] = max(1000 // state['fps'], 8)
            anim.event_source.interval = state['interval']
            _update_display()
        elif event.key in ('-', '_'):
            state['fps'] = max(state['fps'] - 5, 1)
            state['interval'] = 1000 // state['fps']
            anim.event_source.interval = state['interval']
            _update_display()
        elif event.key in ('r', 'R'):
            state['frame'] = 0
            state['playing'] = True
            _update_display()
        elif event.key in ('q', 'Q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', _on_key)

    anim = FuncAnimation(
        fig, _on_timer, frames=n_frames,
        interval=state['interval'], blit=False, repeat=True)

    _update_display()

    logger.info("交互式查看器已启动 — Space=暂停/播放, 左右=单步, +/-=变速, R=重置, Q=退出")
    plt.tight_layout()
    plt.show()
