"""
examples/planner_scenarios.py - 多障碍物场景示例

演示 Box-RRT 规划器在不同难度场景下的表现：
1. 单障碍物（简单绕行）
2. 窄通道（两侧障碍物之间的缝隙）
3. 密集杂乱环境（多个随机障碍物）
4. L 形障碍物（需拐弯）
5. 围栏场景（包围式障碍物）

使用 3DOF 平面机械臂演示。
运行: python examples/planner_scenarios.py
"""
import time
import numpy as np

from box_aabb.robot import load_robot
from box_aabb.planner import (
    BoxRRT, Scene, PlannerConfig, PlannerResult,
    evaluate_result, PathMetrics, format_comparison_table,
)


def _make_config(difficulty: str = 'normal') -> PlannerConfig:
    """根据难度级别创建规划配置"""
    if difficulty == 'easy':
        return PlannerConfig(
            max_iterations=300, max_box_nodes=100,
            seed_batch_size=5, expansion_resolution=0.03,
            goal_bias=0.15, connection_radius=3.0,
            path_shortcut_iters=100, verbose=False,
        )
    elif difficulty == 'hard':
        return PlannerConfig(
            max_iterations=800, max_box_nodes=300,
            seed_batch_size=8, expansion_resolution=0.02,
            max_expansion_rounds=4, goal_bias=0.2,
            connection_radius=4.0, connection_max_attempts=80,
            path_shortcut_iters=200, verbose=False,
        )
    else:  # normal
        return PlannerConfig(
            max_iterations=500, max_box_nodes=200,
            seed_batch_size=5, expansion_resolution=0.03,
            max_expansion_rounds=3, goal_bias=0.15,
            connection_radius=3.0, connection_max_attempts=60,
            path_shortcut_iters=150, verbose=False,
        )


# ==================== 场景定义 ====================

def scenario_single_obstacle() -> dict:
    """场景1: 单障碍物 - 简单绕行"""
    scene = Scene()
    scene.add_obstacle([2.05, -0.3], [2.4, 0.3], name="wall")
    return {
        'name': '单障碍物',
        'scene': scene,
        'q_start': np.array([0.0, 2.0, 0.0]),
        'q_goal': np.array([0.0, -2.0, 0.0]),
        'difficulty': 'easy',
    }


def scenario_narrow_corridor() -> dict:
    """场景2: 窄通道 - 两面墙之间"""
    scene = Scene()
    # 上方墙
    scene.add_obstacle([1.8, 0.15], [2.5, 0.8], name="upper_wall")
    # 下方墙
    scene.add_obstacle([1.8, -0.8], [2.5, -0.15], name="lower_wall")
    # 通道宽度: 0.3 (y=-0.15 到 y=0.15)
    return {
        'name': '窄通道',
        'scene': scene,
        'q_start': np.array([0.0, 2.0, 0.0]),
        'q_goal': np.array([0.0, -2.0, 0.0]),
        'difficulty': 'hard',
    }


def scenario_cluttered() -> dict:
    """场景3: 密集杂乱环境 - 多个小障碍物"""
    scene = Scene()
    rng = np.random.default_rng(42)
    # 在工作空间中撒 6 个小障碍物
    centers = [
        [1.8, 0.3], [2.0, -0.2], [1.5, 0.6],
        [1.3, -0.5], [0.8, 0.8], [2.2, 0.0],
    ]
    for i, c in enumerate(centers):
        half = rng.uniform(0.08, 0.15, size=2)
        scene.add_obstacle(
            [c[0] - half[0], c[1] - half[1]],
            [c[0] + half[0], c[1] + half[1]],
            name=f"clutter_{i}",
        )
    return {
        'name': '密集杂乱',
        'scene': scene,
        'q_start': np.array([0.0, 1.5, 0.0]),
        'q_goal': np.array([0.0, -1.5, 0.0]),
        'difficulty': 'hard',
    }


def scenario_l_shape() -> dict:
    """场景4: L 形障碍物 - 需要拐弯"""
    scene = Scene()
    # L 形由两个矩形组成
    scene.add_obstacle([1.5, -0.5], [2.5, 0.1], name="l_bottom")
    scene.add_obstacle([1.5, 0.1], [1.8, 0.8], name="l_left")
    return {
        'name': 'L形障碍物',
        'scene': scene,
        'q_start': np.array([0.0, 1.5, 0.0]),
        'q_goal': np.array([0.3, -1.5, 0.0]),
        'difficulty': 'normal',
    }


def scenario_enclosed() -> dict:
    """场景5: 围栏 - 三面包围，仅留一个出口"""
    scene = Scene()
    # 上墙
    scene.add_obstacle([1.6, 0.3], [2.5, 0.5], name="fence_top")
    # 下墙
    scene.add_obstacle([1.6, -0.5], [2.5, -0.3], name="fence_bottom")
    # 右墙
    scene.add_obstacle([2.3, -0.3], [2.5, 0.3], name="fence_right")
    # 左侧开口: x=1.6, y=-0.3~0.3
    return {
        'name': '围栏',
        'scene': scene,
        'q_start': np.array([0.0, 1.8, 0.0]),
        'q_goal': np.array([0.0, -1.8, 0.0]),
        'difficulty': 'hard',
    }


# ==================== 主函数 ====================

ALL_SCENARIOS = [
    scenario_single_obstacle,
    scenario_narrow_corridor,
    scenario_cluttered,
    scenario_l_shape,
    scenario_enclosed,
]


def run_scenario(scenario_func, robot, seed=42):
    """运行单个场景"""
    info = scenario_func()
    config = _make_config(info['difficulty'])
    planner = BoxRRT(robot, info['scene'], config)

    t0 = time.time()
    result = planner.plan(info['q_start'], info['q_goal'], seed=seed)
    elapsed = time.time() - t0

    metrics = evaluate_result(result, robot, info['scene'])

    return info['name'], result, metrics


def main():
    robot = load_robot('3dof_planar')
    print(f"机器人: {robot.name}, {robot.n_joints} 自由度")
    print(f"连杆长度: {[p['a'] for p in robot.dh_params]}")
    print()

    all_metrics = {}

    for scenario_func in ALL_SCENARIOS:
        name, result, metrics = run_scenario(scenario_func, robot)
        all_metrics[name] = metrics

        status = "✓ 成功" if result.success else "✗ 失败"
        print(f"[{status}] {name}")
        print(f"  路径点: {len(result.path) if result.path else 0}, "
              f"长度: {result.path_length:.4f}, "
              f"时间: {result.computation_time:.3f}s, "
              f"box: {result.n_boxes_created}")
        if result.success:
            print(f"  效率: {metrics.length_ratio:.2f}x, "
                  f"平滑度: {metrics.smoothness:.4f} rad, "
                  f"最小安全裕度: {metrics.min_clearance:.4f}")
        print()

    # 汇总对比
    print("\n" + format_comparison_table(all_metrics))

    # 可视化（可选）
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        from box_aabb.planner.visualizer import (
            plot_cspace_boxes, plot_workspace_result,
        )

        for scenario_func in ALL_SCENARIOS:
            info = scenario_func()
            config = _make_config(info['difficulty'])
            planner = BoxRRT(robot, info['scene'], config)
            result = planner.plan(info['q_start'], info['q_goal'], seed=42)
            if result.success:
                safe_name = info['name'].replace(' ', '_')
                fig = plot_cspace_boxes(
                    result, joint_limits=robot.joint_limits,
                    dim_x=0, dim_y=1,
                    title=f"C-space: {info['name']}",
                )
                fig.savefig(f"examples/scenario_{safe_name}_cspace.png",
                            dpi=150, bbox_inches='tight')
                print(f"保存: examples/scenario_{safe_name}_cspace.png")

    except ImportError:
        print("(matplotlib 未安装，跳过可视化)")
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == '__main__':
    main()
