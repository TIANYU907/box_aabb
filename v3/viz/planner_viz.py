"""
viz/planner_viz.py - 规划过程可视化

提供 C-space 和工作空间两种视角的可视化：
- C-space: box 区域、路径、碰撞区域（2DOF 时直接可视化）
- 工作空间: 机械臂姿态、障碍物、AABB、路径动画

复用现有 ``box_aabb.Visualizer`` 的绘图能力。
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

from aabb.robot import Robot
from forest.models import BoxNode, Obstacle
from planner.models import BoxTree, SBFResult
from forest.scene import Scene
from forest.collision import CollisionChecker

logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    import matplotlib.cm as cm

    # 中文字体
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


# ============================================================
# SafeBoxForest v5 可视化
# ============================================================


def plot_cspace_forest(
    result: SBFResult,
    joint_limits: Optional[List[Tuple[float, float]]] = None,
    dim_x: int = 0,
    dim_y: int = 1,
    ax: Optional[Any] = None,
    show_path: bool = True,
    show_adjacency: bool = True,
    title: str = "C-Space SafeBoxForest（无重叠）",
    figsize: Tuple[float, float] = (10, 8),
) -> Any:
    """绘制 SafeBoxForest 无重叠 box 集合 + 邻接边 + 路径

    每个 box 按邻接度着色，邻接 box 之间用虚线连接共享面中心。

    Args:
        result: SBFResult（需含 forest 属性）
        joint_limits: 关节限制
        dim_x, dim_y: 投影维度
        ax: matplotlib Axes
        show_path: 绘制路径
        show_adjacency: 绘制邻接边（共享面中心连线）
        title: 标题
        figsize: 图形尺寸

    Returns:
        matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib 不可用")
        return None

    forest = getattr(result, 'forest', None)
    if forest is None:
        logger.warning("SBFResult 无 forest 属性，回退到 plot_cspace_boxes")
        return plot_cspace_boxes(result, joint_limits, dim_x, dim_y, ax,
                                 show_path, title=title, figsize=figsize)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    boxes = forest.boxes
    adjacency = forest.adjacency

    # 邻接度 → 颜色映射
    degrees = {bid: len(adjacency.get(bid, set())) for bid in boxes}
    max_deg = max(degrees.values()) if degrees else 1
    cmap = plt.cm.viridis

    for bid, box in boxes.items():
        lo_x = box.joint_intervals[dim_x][0]
        hi_x = box.joint_intervals[dim_x][1]
        lo_y = box.joint_intervals[dim_y][0]
        hi_y = box.joint_intervals[dim_y][1]

        deg = degrees.get(bid, 0)
        color = cmap(deg / max(max_deg, 1))

        rect = Rectangle(
            (lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
            linewidth=0.6, edgecolor=color,
            facecolor=color, alpha=0.25,
        )
        ax.add_patch(rect)

    # 邻接边：共享面中心连线
    if show_adjacency:
        from forest.deoverlap import shared_face_center
        seen = set()
        for bid, neighbors in adjacency.items():
            if bid not in boxes:
                continue
            for nb in neighbors:
                if nb not in boxes:
                    continue
                key = (min(bid, nb), max(bid, nb))
                if key in seen:
                    continue
                seen.add(key)
                wp = shared_face_center(boxes[bid], boxes[nb])
                if wp is None:
                    continue
                ca = np.array(boxes[bid].center)
                cb = np.array(boxes[nb].center)
                # 画邻接虚线
                ax.plot(
                    [ca[dim_x], wp[dim_x], cb[dim_x]],
                    [ca[dim_y], wp[dim_y], cb[dim_y]],
                    'k--', linewidth=0.3, alpha=0.25,
                )
                # 共享面中心点
                ax.plot(wp[dim_x], wp[dim_y], 's',
                        color='orange', markersize=2.5, alpha=0.6, zorder=4)

    # 路径
    if show_path and result.path:
        path_x = [p[dim_x] for p in result.path]
        path_y = [p[dim_y] for p in result.path]
        ax.plot(path_x, path_y, 'b-', linewidth=2.0, label='Path', zorder=5)
        ax.plot(path_x[0], path_y[0], 'go', markersize=10,
                label='Start', zorder=6)
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=12,
                label='Goal', zorder=6)

    if joint_limits:
        lim_x = joint_limits[dim_x]
        lim_y = joint_limits[dim_y]
        ax.set_xlim(lim_x[0] - 0.1, lim_x[1] + 0.1)
        ax.set_ylim(lim_y[0] - 0.1, lim_y[1] + 0.1)

    ax.set_xlabel(f'q{dim_x} (rad)')
    ax.set_ylabel(f'q{dim_y} (rad)')
    ax.set_title(title)
    ax.set_aspect('equal')

    # 色条
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_deg))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label='邻接度')

    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig


def plot_cspace_boxes(
    result: SBFResult,
    joint_limits: Optional[List[Tuple[float, float]]] = None,
    dim_x: int = 0,
    dim_y: int = 1,
    ax: Optional[Any] = None,
    show_path: bool = True,
    show_seeds: bool = True,
    title: str = "C-Space Box Tree",
    figsize: Tuple[float, float] = (10, 8),
) -> Any:
    """在 C-space 中绘制 box tree 和路径

    对于 2DOF 机械臂，直接绘制完整的 C-space。
    对于高维，投影到指定的两个维度。

    Args:
        result: 规划结果
        joint_limits: 关节限制（用于画边界）
        dim_x: X 轴对应的关节维度
        dim_y: Y 轴对应的关节维度
        ax: matplotlib Axes（可选）
        show_path: 是否绘制路径
        show_seeds: 是否标注 seed 点
        title: 标题
        figsize: 图形尺寸

    Returns:
        matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib 不可用")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # 配色：不同树用不同颜色
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(result.box_trees), 1)))

    for tree_idx, tree in enumerate(result.box_trees):
        color = colors[tree_idx % len(colors)]

        for box in tree.nodes.values():
            lo_x = box.joint_intervals[dim_x][0]
            hi_x = box.joint_intervals[dim_x][1]
            lo_y = box.joint_intervals[dim_y][0]
            hi_y = box.joint_intervals[dim_y][1]

            w = hi_x - lo_x
            h = hi_y - lo_y

            rect = Rectangle(
                (lo_x, lo_y), w, h,
                linewidth=0.8, edgecolor=color,
                facecolor=color, alpha=0.25,
            )
            ax.add_patch(rect)

            if show_seeds:
                ax.plot(
                    box.seed_config[dim_x], box.seed_config[dim_y],
                    '.', color=color, markersize=3, alpha=0.7,
                )

    # 绘制路径
    if show_path and result.path:
        path_x = [p[dim_x] for p in result.path]
        path_y = [p[dim_y] for p in result.path]
        ax.plot(path_x, path_y, 'b-', linewidth=2.0, label='Path', zorder=5)
        ax.plot(path_x[0], path_y[0], 'go', markersize=10,
                label='Start', zorder=6)
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=12,
                label='Goal', zorder=6)

    # 画关节限制
    if joint_limits:
        lim_x = joint_limits[dim_x]
        lim_y = joint_limits[dim_y]
        ax.set_xlim(lim_x[0] - 0.1, lim_x[1] + 0.1)
        ax.set_ylim(lim_y[0] - 0.1, lim_y[1] + 0.1)

    ax.set_xlabel(f'q{dim_x} (rad)')
    ax.set_ylabel(f'q{dim_y} (rad)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig


def plot_cspace_with_collision(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    result: Optional[SBFResult] = None,
    resolution: float = 0.05,
    dim_x: int = 0,
    dim_y: int = 1,
    figsize: Tuple[float, float] = (10, 8),
) -> Any:
    """绘制 C-space 碰撞区域（仅适用于 2DOF）

    扫描整个 C-space 并标记碰撞区域和安全区域，
    叠加显示 box tree 和规划路径。

    Args:
        robot: 机器人模型
        scene: 障碍物场景
        joint_limits: 关节限制
        result: 规划结果（可选）
        resolution: C-space 扫描分辨率
        dim_x, dim_y: 投影维度
        figsize: 图形尺寸
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib 不可用")
        return None

    checker = CollisionChecker(robot, scene)

    lo_x, hi_x = joint_limits[dim_x]
    lo_y, hi_y = joint_limits[dim_y]

    xs = np.arange(lo_x, hi_x, resolution)
    ys = np.arange(lo_y, hi_y, resolution)
    collision_map = np.zeros((len(ys), len(xs)))

    n_joints = robot.n_joints
    base_config = np.zeros(n_joints)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            q = base_config.copy()
            q[dim_x] = x
            q[dim_y] = y
            if checker.check_config_collision(q):
                collision_map[i, j] = 1.0

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(
        collision_map, origin='lower',
        extent=[lo_x, hi_x, lo_y, hi_y],
        cmap='Reds', alpha=0.4, aspect='auto',
    )

    # 叠加 box tree 和路径
    if result is not None:
        plot_cspace_boxes(result, joint_limits, dim_x, dim_y, ax=ax,
                          title="C-Space: 碰撞区域 + Box Tree + 路径")
    else:
        ax.set_xlabel(f'q{dim_x} (rad)')
        ax.set_ylabel(f'q{dim_y} (rad)')
        ax.set_title("C-Space 碰撞地图")
        ax.grid(True, alpha=0.3)

    return fig


def plot_workspace_result(
    robot: Robot,
    scene: Scene,
    result: SBFResult,
    n_poses: int = 10,
    figsize: Tuple[float, float] = (10, 8),
) -> Any:
    """在工作空间中可视化规划结果

    绘制机械臂沿路径的多个姿态、障碍物和末端轨迹。

    Args:
        robot: 机器人模型
        scene: 障碍物场景
        result: 规划结果
        n_poses: 绘制的中间姿态数量
        figsize: 图形尺寸
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib 不可用")
        return None

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)

    # 检测机械臂是否为平面机械臂（所有 alpha=0, d=0）
    is_planar = all(
        abs(p['alpha']) < 1e-10 and abs(p['d']) < 1e-10
        for p in robot.dh_params
    )

    if is_planar:
        ax = fig.add_subplot(111)
        _plot_2d_workspace(ax, robot, scene, result, n_poses)
    else:
        ax = fig.add_subplot(111, projection='3d')
        _plot_3d_workspace(ax, robot, scene, result, n_poses)

    return fig


def _plot_2d_workspace(ax, robot, scene, result, n_poses):
    """2D 工作空间可视化"""
    # 画障碍物
    for obs in scene.get_obstacles():
        rect = Rectangle(
            (obs.min_point[0], obs.min_point[1]),
            obs.size[0], obs.size[1],
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.4,
        )
        ax.add_patch(rect)

    if not result.path:
        ax.set_title("工作空间（无路径）")
        return

    # 选择要绘制的姿态
    indices = np.linspace(0, len(result.path) - 1, n_poses, dtype=int)

    cmap = plt.cm.viridis
    for idx_i, idx in enumerate(indices):
        q = result.path[idx]
        positions = robot.get_link_positions(q)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        alpha = 0.3 + 0.7 * (idx_i / max(len(indices) - 1, 1))
        color = cmap(idx_i / max(len(indices) - 1, 1))

        ax.plot(xs, ys, 'o-', color=color, linewidth=2.0,
                markersize=4, alpha=alpha)

    # 画末端轨迹
    ee_trajectory = []
    for q in result.path:
        pos = robot.get_link_positions(q)[-1]
        ee_trajectory.append(pos)
    ee_x = [p[0] for p in ee_trajectory]
    ee_y = [p[1] for p in ee_trajectory]
    ax.plot(ee_x, ee_y, 'b--', linewidth=1.5, alpha=0.8, label='EE 轨迹')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('工作空间: 机械臂姿态 + 障碍物')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_3d_workspace(ax, robot, scene, result, n_poses):
    """3D 工作空间可视化"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 画障碍物
    for obs in scene.get_obstacles():
        _draw_3d_box(ax, obs.min_point, obs.max_point,
                     color='red', alpha=0.3)

    if not result.path:
        return

    indices = np.linspace(0, len(result.path) - 1, n_poses, dtype=int)
    cmap = plt.cm.viridis

    for idx_i, idx in enumerate(indices):
        q = result.path[idx]
        positions = robot.get_link_positions(q)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]

        alpha = 0.2 + 0.8 * (idx_i / max(len(indices) - 1, 1))
        color = cmap(idx_i / max(len(indices) - 1, 1))

        ax.plot(xs, ys, zs, 'o-', color=color, linewidth=2.0,
                markersize=3, alpha=alpha)

    # 末端轨迹
    ee_trajectory = []
    for q in result.path:
        pos = robot.get_link_positions(q)[-1]
        ee_trajectory.append(pos)
    ee_x = [p[0] for p in ee_trajectory]
    ee_y = [p[1] for p in ee_trajectory]
    ee_z = [p[2] for p in ee_trajectory]
    ax.plot(ee_x, ee_y, ee_z, 'b--', linewidth=1.5, alpha=0.8, label='EE 轨迹')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('工作空间: 机械臂姿态 + 障碍物')
    ax.legend()


def _draw_3d_box(ax, min_pt, max_pt, color='red', alpha=0.3):
    """绘制 3D 方框"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
        edgecolor='black', linewidth=0.5,
    )
    ax.add_collection3d(collection)
