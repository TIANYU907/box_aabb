"""
examples/planner_demo_3dof.py - 3DOF 平面机械臂 Box-RRT 规划演示

演示 Box-RRT 规划器的完整工作流程：
1. 加载 3DOF 平面机械臂（Modified DH，连杆长度 1.0, 1.0, 0.5）
2. 设置障碍物场景（obstacle 恰好超出 link2 可达范围，仅 link3 碰撞）
3. 执行 Box-RRT 路径规划
4. 可视化结果（C-space boxes + 工作空间路径）

为什么用 3DOF 而非 2DOF:
    Modified DH 约定下，2DOF 机器人的所有 link 端点位置仅依赖 q0，
    q1 不影响碰撞检测，导致 C-space 退化为一维问题。
    3DOF 中 link3 的位置依赖 (q0, q1)，使碰撞区域在 C-space 中
    形成真正的二维障碍物，展示 Box-RRT 绕障碍物规划路径的能力。

运行：
    python examples/planner_demo_3dof.py
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
    robot = load_robot('3dof_planar')
    print(f"机器人: {robot.name}, {robot.n_joints} 自由度")
    print(f"关节限制: {robot.joint_limits}")
    print(f"DH 连杆长度: {[p['a'] for p in robot.dh_params]}")

    # ---- 2. 设置障碍物场景 ----
    scene = Scene()
    # 障碍物放在 x=2.05~2.4, y=-0.3~0.3
    # link2 的最大 x 范围是 1+cos(q0) ≤ 2.0，永远不会碰到 x=2.05
    # link3 可以到达 x=2.5 (当 q0=0, q1=0)，所以碰撞仅来自 link3
    # 碰撞区域在 C-space 中形成 (q0, q1) 相关的二维区域
    scene.add_obstacle(
        min_point=[2.05, -0.3],
        max_point=[2.4, 0.3],
        name="wall",
    )
    print(f"\n场景: {scene.n_obstacles} 个障碍物")
    for obs in scene.get_obstacles():
        print(f"  {obs.name}: min={obs.min_point[:2].tolist()}, "
              f"max={obs.max_point[:2].tolist()}")

    # ---- 3. 配置规划器 ----
    config = PlannerConfig(
        max_iterations=600,
        max_box_nodes=200,
        seed_batch_size=5,
        expansion_resolution=0.03,
        max_expansion_rounds=3,
        goal_bias=0.15,
        connection_radius=3.0,
        connection_max_attempts=60,
        path_shortcut_iters=200,
        segment_collision_resolution=0.03,
        verbose=True,
    )

    planner = BoxRRT(robot, scene, config)

    # ---- 4. 定义起终点 ----
    # start: arm extended, link3 folded back (q1=2.0, away from obstacle)
    # goal:  arm extended, link3 folded opposite way (q1=-2.0)
    # 直连路径经过 q1≈0 处碰撞区域，需绕行 q0 方向
    q_start = np.array([0.0, 2.0, 0.0])
    q_goal = np.array([0.0, -2.0, 0.0])

    print(f"\n起点: q = [{q_start[0]:.3f}, {q_start[1]:.3f}, {q_start[2]:.3f}]")
    print(f"终点: q = [{q_goal[0]:.3f}, {q_goal[1]:.3f}, {q_goal[2]:.3f}]")

    # 验证起终点
    from planner.collision import CollisionChecker
    checker = CollisionChecker(robot, scene)
    assert not checker.check_config_collision(q_start), "起点有碰撞！"
    assert not checker.check_config_collision(q_goal), "终点有碰撞！"
    print("起终点碰撞检测: 通过")

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
        print(f"  树 {tree.tree_id}: {tree.n_nodes} 节点, "
              f"体积 {tree.total_volume:.4f}")
    print(f"{'='*50}")

    # ---- 7. 可视化 ----
    try:
        import matplotlib
        matplotlib.use('TkAgg')

        joint_limits = robot.joint_limits

        # 7a. C-space box 可视化 (q0 vs q1 投影)
        print("\n生成 C-space 可视化...")
        fig1 = plot_cspace_boxes(
            result, joint_limits=joint_limits,
            dim_x=0, dim_y=1,
            title="Box-RRT C-space (q0 vs q1 投影)",
        )
        fig1.savefig("examples/cspace_boxes_3dof.png",
                     dpi=150, bbox_inches='tight')
        print("  保存到 examples/cspace_boxes_3dof.png")

        # 7b. C-space 碰撞地图叠加
        print("生成碰撞地图可视化...")
        fig2 = plot_cspace_with_collision(
            robot, scene, joint_limits=joint_limits,
            result=result,
            resolution=0.05,
            dim_x=0, dim_y=1,
        )
        fig2.savefig("examples/cspace_collision_3dof.png",
                     dpi=150, bbox_inches='tight')
        print("  保存到 examples/cspace_collision_3dof.png")

        # 7c. 工作空间可视化
        print("生成工作空间可视化...")
        fig3 = plot_workspace_result(robot, scene, result, n_poses=10)
        fig3.savefig("examples/workspace_3dof.png",
                     dpi=150, bbox_inches='tight')
        print("  保存到 examples/workspace_3dof.png")

        print("\n完成！可查看生成的图片：")
        print("  examples/cspace_boxes_3dof.png")
        print("  examples/cspace_collision_3dof.png")
        print("  examples/workspace_3dof.png")

    except ImportError:
        print("\n(matplotlib 未安装，跳过可视化)")
    except Exception as e:
        print(f"\n可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
