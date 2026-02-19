"""
examples/planner_demo_animation.py - 动态可视化 demo

展示机械臂沿规划路径运动的动画。

使用方式：
    python examples/planner_demo_animation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用交互后端

from box_aabb.robot import load_robot
from planner import BoxRRT, Scene, PlannerConfig
from planner.dynamic_visualizer import animate_robot_path, resample_path


def demo_2dof():
    """2DOF 平面机器人动画 demo"""
    print("=== 2DOF 平面机器人动画 ===")

    robot = load_robot('2dof_planar')
    scene = Scene()
    scene.add_obstacle([0.8, -0.3], [1.2, 0.3], name="block")

    config = PlannerConfig(
        max_iterations=200,
        max_box_nodes=50,
        path_shortcut_iters=30,
    )

    q_start = np.array([0.5, 0.5])
    q_goal = np.array([-1.0, -0.5])

    planner = BoxRRT(robot, scene, config)
    result = planner.plan(q_start, q_goal, seed=42)

    if not result.success:
        print(f"规划失败: {result.message}")
        return

    print(f"路径点数: {len(result.path)}")
    print(f"路径长度: {result.path_length:.4f}")

    # 重采样使动画更平滑
    smooth_path = resample_path(result.path, n_frames=60)

    # 生成动画
    anim = animate_robot_path(
        robot, smooth_path, scene=scene,
        fps=20, trail_length=20,
        title="2DOF Robot Motion",
        ghost_interval=10,
    )

    # 保存为 GIF
    output_path = os.path.join(os.path.dirname(__file__), 'output', 'robot_2dof.gif')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        anim.save(output_path, writer='pillow', fps=20)
        print(f"动画已保存至 {output_path}")
    except Exception as e:
        print(f"保存失败: {e}")

    import matplotlib.pyplot as plt
    plt.show()


def demo_3dof():
    """3DOF 平面机器人动画 demo"""
    print("\n=== 3DOF 平面机器人动画 ===")

    robot = load_robot('3dof_planar')
    scene = Scene()
    scene.add_obstacle([1.0, -0.5], [1.5, 0.5], name="wall")

    config = PlannerConfig(
        max_iterations=300,
        max_box_nodes=80,
        path_shortcut_iters=50,
    )

    q_start = np.array([0.3, 0.3, 0.3])
    q_goal = np.array([-0.8, -0.5, -0.3])

    planner = BoxRRT(robot, scene, config)
    result = planner.plan(q_start, q_goal, seed=42)

    if not result.success:
        print(f"规划失败: {result.message}")
        return

    smooth_path = resample_path(result.path, n_frames=80)

    anim = animate_robot_path(
        robot, smooth_path, scene=scene,
        fps=25, trail_length=30,
        title="3DOF Robot Motion",
        show_ee_trail=True,
    )

    output_path = os.path.join(os.path.dirname(__file__), 'output', 'robot_3dof.gif')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        anim.save(output_path, writer='pillow', fps=25)
        print(f"动画已保存至 {output_path}")
    except Exception as e:
        print(f"保存失败: {e}")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    demo_2dof()
    demo_3dof()
