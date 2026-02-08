"""
examples/planner_demo_panda.py - 7DOF Panda Box-RRT 规划演示

演示 Box-RRT 规划器在 7DOF Franka Emika Panda 机器人上的工作流程：
1. 加载 Panda 机器人 (8 DH 参数: 7 revolute + 1 固定指关节)
2. 设置工作空间障碍物
3. 执行 Box-RRT 路径规划
4. 输出路径质量评价指标

Panda 特性:
    - 所有 a=0 (Modified DH), 连杆长度由 d 值定义
    - 第 8 关节 (指关节) 限制为 [0,0], 在规划中固定为 0
    - 工作空间近似为半径 ~1.14m 的球体

运行: python examples/planner_demo_panda.py
"""
import numpy as np

from box_aabb.robot import load_robot
from box_aabb.planner import (
    BoxRRT, Scene, PlannerConfig, PlannerResult,
    evaluate_result,
)


def main():
    # ---- 1. 加载 Panda ----
    robot = load_robot('panda')
    print(f"机器人: {robot.name}, {robot.n_joints} 个 DH 参数")
    print(f"活跃关节: 7 (第 8 个指关节固定)")
    print(f"连杆 d 值: {[p['d'] for p in robot.dh_params]}")
    print(f"零长度连杆: {robot.zero_length_links}")

    # ---- 2. 设置障碍物场景 ----
    scene = Scene()

    # 障碍物 1: 前方桌面上的盒子
    scene.add_obstacle(
        min_point=[0.3, -0.15, 0.2],
        max_point=[0.55, 0.15, 0.5],
        name="front_box",
    )

    # 障碍物 2: 侧方小立柱
    scene.add_obstacle(
        min_point=[-0.2, 0.3, 0.0],
        max_point=[0.0, 0.5, 0.6],
        name="side_pillar",
    )

    print(f"\n场景: {scene.n_obstacles} 个障碍物")
    for obs in scene.get_obstacles():
        print(f"  {obs.name}: min={obs.min_point.tolist()}, "
              f"max={obs.max_point.tolist()}")

    # ---- 3. 配置规划器 ----
    # Panda 是 8D (含固定指关节), 需要更多迭代和更大搜索空间
    config = PlannerConfig(
        max_iterations=400,
        max_box_nodes=150,
        seed_batch_size=4,
        expansion_resolution=0.05,
        max_expansion_rounds=2,
        goal_bias=0.15,
        connection_radius=4.0,
        connection_max_attempts=60,
        path_shortcut_iters=100,
        segment_collision_resolution=0.05,
        verbose=True,
    )

    planner = BoxRRT(robot, scene, config)

    # ---- 4. 定义起终点 ----
    # 起点: Panda "就绪" 姿态
    q_start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0])

    # 终点: 手臂向左伸展
    q_goal = np.array([-1.5, -0.5, 1.0, -1.2, -0.5, 2.0, -0.3, 0.0])

    print(f"\n起点: q = {np.array2string(q_start[:7], precision=3)}")
    print(f"终点: q = {np.array2string(q_goal[:7], precision=3)}")

    # 验证起终点无碰撞
    from box_aabb.planner.collision import CollisionChecker
    checker = CollisionChecker(robot, scene)
    start_collision = checker.check_config_collision(q_start)
    goal_collision = checker.check_config_collision(q_goal)
    print(f"起点碰撞: {'是' if start_collision else '否'}")
    print(f"终点碰撞: {'是' if goal_collision else '否'}")

    if start_collision or goal_collision:
        print("起终点存在碰撞，尝试微调...")
        # 如果碰撞，随机微调找无碰撞配置
        rng = np.random.default_rng(42)
        if start_collision:
            for _ in range(100):
                q_start[:7] += rng.normal(0, 0.1, size=7)
                q_start[:7] = np.clip(
                    q_start[:7],
                    [l[0] for l in robot.joint_limits[:7]],
                    [l[1] for l in robot.joint_limits[:7]],
                )
                if not checker.check_config_collision(q_start):
                    print(f"  新起点: {np.array2string(q_start[:7], precision=3)}")
                    break
        if goal_collision:
            for _ in range(100):
                q_goal[:7] += rng.normal(0, 0.1, size=7)
                q_goal[:7] = np.clip(
                    q_goal[:7],
                    [l[0] for l in robot.joint_limits[:7]],
                    [l[1] for l in robot.joint_limits[:7]],
                )
                if not checker.check_config_collision(q_goal):
                    print(f"  新终点: {np.array2string(q_goal[:7], precision=3)}")
                    break

    # ---- 5. 执行规划 ----
    print("\n开始规划...")
    result = planner.plan(q_start, q_goal, seed=42)

    # ---- 6. 输出结果 ----
    print(f"\n{'='*60}")
    print(f"规划结果: {'成功' if result.success else '失败'}")
    print(f"消息: {result.message}")
    print(f"计算时间: {result.computation_time:.3f} s")
    print(f"Box 数量: {result.n_boxes_created}")
    print(f"碰撞检测次数: {result.n_collision_checks}")
    if result.path:
        print(f"路径点数: {len(result.path)}")
        print(f"路径长度 (L2): {result.path_length:.4f}")

    if result.box_trees:
        print(f"Box tree 数量: {len(result.box_trees)}")
        for tree in result.box_trees:
            print(f"  树 {tree.tree_id}: {tree.n_nodes} 节点, "
                  f"体积 {tree.total_volume:.6f}")

    # ---- 7. 路径质量评价 ----
    if result.success:
        metrics = evaluate_result(result, robot, scene)
        print(f"\n{metrics.summary()}")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
