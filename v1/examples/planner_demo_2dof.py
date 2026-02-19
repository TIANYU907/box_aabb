"""
examples/planner_demo_2dof.py - 2DOF 平面机械臂 Box-RRT 规划演示

演示 Box-RRT 规划器的完整工作流程：
1. 加载 2DOF 平面机械臂
2. 设置障碍物场景
3. 执行 Box-RRT 路径规划
4. 可视化结果（C-space boxes + 工作空间路径）

运行：
    python examples/planner_demo_2dof.py
"""
import math
import numpy as np

from box_aabb.robot import load_robot
from planner import (
    BoxRRT, Scene, PlannerConfig, PlannerResult,
)
from planner.visualizer import (
    plot_cspace_boxes,
    plot_cspace_with_collision,
    plot_workspace_result,
)


def main():
    # ---- 1. 加载机器人 ----
    robot = load_robot('2dof_planar')
    print(f"机器人: {robot.name}, {robot.n_joints} 自由度")
    print(f"关节限制: {robot.joint_limits}")

    # ---- 2. 设置障碍物场景 ----
    scene = Scene()
    # 障碍物放在 x=1.5~2.0 附近，挡住直伸手臂
    scene.add_obstacle(
        min_point=[1.5, -0.3],
        max_point=[2.0, 0.3],
        name="obstacle_1",
    )
    # 第二个障碍物在负 y 方向
    scene.add_obstacle(
        min_point=[0.5, -1.8],
        max_point=[1.2, -1.2],
        name="obstacle_2",
    )
    print(f"\n场景: {scene.n_obstacles} 个障碍物")
    for obs in scene.get_obstacles():
        print(f"  {obs.name}: min={obs.min_point[:2]}, max={obs.max_point[:2]}")

    # ---- 3. 配置规划器 ----
    config = PlannerConfig(
        max_iterations=500,
        max_box_nodes=150,
        seed_batch_size=5,
        expansion_resolution=0.03,
        max_expansion_rounds=3,
        goal_bias=0.15,
        connection_radius=3.0,
        connection_max_attempts=50,
        path_shortcut_iters=200,
        segment_collision_resolution=0.03,
        verbose=True,
    )

    planner = BoxRRT(robot, scene, config)

    # ---- 4. 定义起终点 ----
    q_start = np.array([math.pi * 0.6, 0.3])    # 手臂偏上方
    q_goal = np.array([-math.pi * 0.4, -0.5])   # 手臂偏另一侧

    print(f"\n起点: q = [{q_start[0]:.3f}, {q_start[1]:.3f}]")
    print(f"终点: q = [{q_goal[0]:.3f}, {q_goal[1]:.3f}]")

    # ---- 5. 执行规划 ----
    print("\n开始规划...")
    result = planner.plan(q_start, q_goal, seed=42)

    # ---- 6. 输出结果 ----
    print(f"\n{'='*50}")
    print(f"规划结果: {'成功' if result.success else '失败'}")
    print(f"信息: {result.message}")
    print(f"计算时间: {result.computation_time:.3f} s")
    print(f"box 数量: {result.n_boxes_created}")
    print(f"碰撞检测次数: {result.n_collision_checks}")
    print(f"路径点数: {len(result.path)}")
    if result.success:
        print(f"路径长度 (L2): {result.path_length:.4f}")
    print(f"box tree 数量: {len(result.box_trees)}")
    for i, tree in enumerate(result.box_trees):
        print(f"  树 {tree.tree_id}: {tree.n_nodes} 节点, 体积 {tree.total_volume:.4f}")
    print(f"{'='*50}")

    # ---- 7. 可视化 ----
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # 或 'Agg' 仅保存文件

        # 7a. C-space box 可视化
        print("\n生成 C-space 可视化...")
        fig1 = plot_cspace_boxes(result, q_start, q_goal,
                                  title="Box-RRT C-space (2DOF)")
        fig1.savefig("examples/cspace_boxes_2dof.png", dpi=150, bbox_inches='tight')
        print("  保存到 examples/cspace_boxes_2dof.png")

        # 7b. C-space 碰撞地图叠加
        print("生成碰撞地图可视化...")
        fig2 = plot_cspace_with_collision(
            robot, scene, result, q_start, q_goal,
            resolution=100,
            title="Box-RRT + Collision Map (2DOF)",
        )
        fig2.savefig("examples/cspace_collision_2dof.png", dpi=150, bbox_inches='tight')
        print("  保存到 examples/cspace_collision_2dof.png")

        # 7c. 工作空间可视化
        print("生成工作空间可视化...")
        fig3 = plot_workspace_result(robot, scene, result, q_start, q_goal)
        fig3.savefig("examples/workspace_2dof.png", dpi=150, bbox_inches='tight')
        print("  保存到 examples/workspace_2dof.png")

        print("\n完成！可通过以下命令查看图片：")
        print("  start examples/cspace_boxes_2dof.png")
        print("  start examples/cspace_collision_2dof.png")
        print("  start examples/workspace_2dof.png")

    except ImportError:
        print("\n(matplotlib 未安装，跳过可视化)")
    except Exception as e:
        print(f"\n可视化失败: {e}")


if __name__ == '__main__':
    main()
