"""
benchmarks/planner_benchmark.py - Box-RRT 规划器性能基准测试

测量项目:
1. 碰撞检测吞吐量 (点/box/线段)
2. Box 拓展速度
3. 端到端规划时间 (3DOF vs Panda)
4. 并行碰撞检测加速比
5. 不同参数配置的影响

运行: python benchmarks/planner_benchmark.py
"""
import time
import sys
import os
import statistics
from typing import List, Tuple

import numpy as np

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from box_aabb.robot import load_robot, Robot
from box_aabb.planner import (
    BoxRRT, Scene, PlannerConfig, PlannerResult,
    CollisionChecker, BoxExpander, BoxTreeManager,
    evaluate_result, PathMetrics, format_comparison_table,
)
from box_aabb.planner.parallel_collision import ParallelCollisionChecker


# ==================== 配置 ====================

N_RUNS = 5            # 每组测试重复次数
N_WARMUP = 1          # 预热次数
SEED_BASE = 42        # 随机种子基准


def timer(func, *args, n_runs=N_RUNS, **kwargs):
    """计时辅助: 返回 (结果, 平均时间, 标准差)"""
    times = []
    result = None
    for i in range(N_WARMUP + n_runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if i >= N_WARMUP:
            times.append(elapsed)
    return result, statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


# ==================== 1. 碰撞检测基准 ====================

def benchmark_collision_detection():
    print_header("碰撞检测吞吐量")

    for robot_name, dof in [('3dof_planar', 3), ('panda', 8)]:
        robot = load_robot(robot_name)
        scene = Scene()
        if dof == 3:
            scene.add_obstacle([2.05, -0.3], [2.4, 0.3], name="wall")
        else:
            scene.add_obstacle([0.3, -0.15, 0.3], [0.6, 0.15, 0.6], name="box")
        checker = CollisionChecker(robot, scene)
        limits = robot.joint_limits

        # 生成测试配置
        rng = np.random.default_rng(SEED_BASE)
        n_configs = 1000
        configs = []
        for _ in range(n_configs):
            q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
            configs.append(q)

        # 点碰撞检测
        def run_point_checks():
            for q in configs:
                checker.check_config_collision(q)

        _, avg, std = timer(run_point_checks, n_runs=3)
        rate = n_configs / avg
        print(f"\n  [{robot_name}] 点碰撞检测: {rate:.0f} checks/s "
              f"({avg*1000:.1f} ± {std*1000:.1f} ms / {n_configs})")

        # 线段碰撞检测
        segments = [(configs[i], configs[i+1]) for i in range(0, 200, 2)]

        def run_segment_checks():
            for q_s, q_e in segments:
                checker.check_segment_collision(q_s, q_e, 0.05)

        _, avg, std = timer(run_segment_checks, n_runs=3)
        rate = len(segments) / avg
        print(f"  [{robot_name}] 线段碰撞检测: {rate:.0f} checks/s "
              f"({avg*1000:.1f} ± {std*1000:.1f} ms / {len(segments)})")

        # Box 碰撞检测
        intervals_list = []
        for q in configs[:200]:
            iv = [(q[i] - 0.05, q[i] + 0.05) for i in range(len(q))]
            # 裁剪到关节限制
            iv = [(max(lo, limits[i][0]), min(hi, limits[i][1]))
                  for i, (lo, hi) in enumerate(iv)]
            intervals_list.append(iv)

        def run_box_checks():
            for iv in intervals_list:
                checker.check_box_collision(iv)

        _, avg, std = timer(run_box_checks, n_runs=3)
        rate = len(intervals_list) / avg
        print(f"  [{robot_name}] Box 碰撞检测:  {rate:.0f} checks/s "
              f"({avg*1000:.1f} ± {std*1000:.1f} ms / {len(intervals_list)})")


# ==================== 2. Box 拓展基准 ====================

def benchmark_box_expansion():
    print_header("Box 拓展速度")

    for robot_name, dof in [('3dof_planar', 3), ('panda', 8)]:
        robot = load_robot(robot_name)
        scene = Scene()  # 空场景，测量纯拓展速度
        checker = CollisionChecker(robot, scene)
        limits = robot.joint_limits

        expander = BoxExpander(
            robot, checker, limits,
            expansion_resolution=0.03,
            max_rounds=2,
        )

        rng = np.random.default_rng(SEED_BASE)
        seeds = []
        for _ in range(20):
            q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
            seeds.append(q)

        def run_expansions():
            volumes = []
            for i, q in enumerate(seeds):
                box = expander.expand(q, node_id=i)
                if box is not None:
                    volumes.append(box.volume)
            return volumes

        volumes, avg, std = timer(run_expansions, n_runs=3)
        rate = len(seeds) / avg
        avg_vol = statistics.mean(volumes) if volumes else 0
        print(f"\n  [{robot_name}] Box 拓展: {rate:.1f} box/s "
              f"({avg*1000:.1f} ± {std*1000:.1f} ms / {len(seeds)} seeds)")
        print(f"  [{robot_name}] 成功率: {len(volumes)}/{len(seeds)}, "
              f"平均体积: {avg_vol:.6f}")


# ==================== 3. 端到端规划基准 ====================

def benchmark_planning():
    print_header("端到端规划时间")

    scenarios = []

    # 3DOF 简单场景
    robot_3dof = load_robot('3dof_planar')
    scene_3dof = Scene()
    scene_3dof.add_obstacle([2.05, -0.3], [2.4, 0.3], name="wall")
    scenarios.append({
        'name': '3DOF-单障碍物',
        'robot': robot_3dof,
        'scene': scene_3dof,
        'q_start': np.array([0.0, 2.0, 0.0]),
        'q_goal': np.array([0.0, -2.0, 0.0]),
        'config': PlannerConfig(
            max_iterations=400, max_box_nodes=150,
            seed_batch_size=5, expansion_resolution=0.03,
            goal_bias=0.15, connection_radius=3.0,
            path_shortcut_iters=100, verbose=False,
        ),
    })

    # 3DOF 多障碍物
    scene_3dof_multi = Scene()
    scene_3dof_multi.add_obstacle([2.05, -0.3], [2.4, 0.3], name="wall1")
    scene_3dof_multi.add_obstacle([1.5, 0.5], [1.8, 0.8], name="wall2")
    scenarios.append({
        'name': '3DOF-多障碍物',
        'robot': robot_3dof,
        'scene': scene_3dof_multi,
        'q_start': np.array([0.0, 1.5, 0.0]),
        'q_goal': np.array([0.0, -1.5, 0.0]),
        'config': PlannerConfig(
            max_iterations=500, max_box_nodes=200,
            seed_batch_size=5, expansion_resolution=0.03,
            goal_bias=0.15, connection_radius=3.0,
            path_shortcut_iters=100, verbose=False,
        ),
    })

    # Panda 空场景（直连）
    robot_panda = load_robot('panda')
    scene_panda_empty = Scene()
    scenarios.append({
        'name': 'Panda-空场景',
        'robot': robot_panda,
        'scene': scene_panda_empty,
        'q_start': np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0]),
        'q_goal': np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3, 0.0]),
        'config': PlannerConfig(
            max_iterations=200, max_box_nodes=80,
            verbose=False,
        ),
    })

    # Panda 单障碍物
    scene_panda_obs = Scene()
    scene_panda_obs.add_obstacle([0.3, -0.15, 0.3], [0.55, 0.15, 0.5], name="box")
    scenarios.append({
        'name': 'Panda-单障碍物',
        'robot': robot_panda,
        'scene': scene_panda_obs,
        'q_start': np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0]),
        'q_goal': np.array([-1.0, -0.5, 0.8, -1.5, -0.3, 2.0, -0.2, 0.0]),
        'config': PlannerConfig(
            max_iterations=300, max_box_nodes=120,
            expansion_resolution=0.05,
            max_expansion_rounds=2,
            connection_radius=4.0,
            path_shortcut_iters=80,
            verbose=False,
        ),
    })

    # 汇总表
    all_metrics = {}

    for sc in scenarios:
        successes = 0
        times = []
        metrics_list = []

        for run_i in range(N_RUNS):
            planner = BoxRRT(sc['robot'], sc['scene'], sc['config'])
            t0 = time.perf_counter()
            result = planner.plan(
                sc['q_start'], sc['q_goal'],
                seed=SEED_BASE + run_i,
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if result.success:
                successes += 1
                metrics_list.append(
                    evaluate_result(result, sc['robot'], sc['scene'])
                )

        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        success_rate = successes / N_RUNS * 100

        print(f"\n  [{sc['name']}]")
        print(f"    成功率:   {success_rate:.0f}% ({successes}/{N_RUNS})")
        print(f"    平均时间: {avg_time:.3f} ± {std_time:.3f} s")

        if metrics_list:
            avg_length = statistics.mean(m.path_length for m in metrics_list)
            avg_ratio = statistics.mean(m.length_ratio for m in metrics_list)
            avg_boxes = statistics.mean(m.n_boxes for m in metrics_list)
            print(f"    平均路径长度: {avg_length:.4f}")
            print(f"    平均路径效率: {avg_ratio:.2f}x")
            print(f"    平均 Box 数:  {avg_boxes:.1f}")

            # 用最后一次的 metrics 作为代表
            all_metrics[sc['name']] = metrics_list[-1]

    if len(all_metrics) > 1:
        print(f"\n{format_comparison_table(all_metrics)}")


# ==================== 4. 并行碰撞检测基准 ====================

def benchmark_parallel_collision():
    print_header("并行碰撞检测加速比")

    robot = load_robot('3dof_planar')
    scene = Scene()
    scene.add_obstacle([2.05, -0.3], [2.4, 0.3], name="wall")
    checker = CollisionChecker(robot, scene)
    limits = robot.joint_limits

    rng = np.random.default_rng(SEED_BASE)
    n_configs = 500
    configs = [np.array([rng.uniform(lo, hi) for lo, hi in limits])
               for _ in range(n_configs)]

    # 串行
    def run_serial():
        return [checker.check_config_collision(q) for q in configs]

    _, serial_time, _ = timer(run_serial, n_runs=3)
    print(f"\n  串行 ({n_configs} configs): {serial_time*1000:.1f} ms")

    # 并行 (不同 worker 数)
    for n_workers in [2, 4]:
        par = ParallelCollisionChecker(
            checker, n_workers=n_workers, batch_threshold=1,
        )

        def run_parallel():
            return par.batch_check_configs(configs)

        _, par_time, _ = timer(run_parallel, n_runs=3)
        speedup = serial_time / par_time if par_time > 0 else 0
        print(f"  并行 ({n_workers} workers): {par_time*1000:.1f} ms, "
              f"加速比: {speedup:.2f}x")


# ==================== 5. 参数敏感度 ====================

def benchmark_parameter_sensitivity():
    print_header("参数敏感度 (3DOF)")

    robot = load_robot('3dof_planar')
    scene = Scene()
    scene.add_obstacle([2.05, -0.3], [2.4, 0.3], name="wall")

    q_start = np.array([0.0, 2.0, 0.0])
    q_goal = np.array([0.0, -2.0, 0.0])

    param_sets = {
        '默认': PlannerConfig(
            max_iterations=400, max_box_nodes=150, verbose=False,
        ),
        '高迭代': PlannerConfig(
            max_iterations=800, max_box_nodes=300, verbose=False,
        ),
        '大batch': PlannerConfig(
            max_iterations=400, max_box_nodes=150,
            seed_batch_size=10, verbose=False,
        ),
        '细分辨率': PlannerConfig(
            max_iterations=400, max_box_nodes=150,
            expansion_resolution=0.01, verbose=False,
        ),
        '高goal_bias': PlannerConfig(
            max_iterations=400, max_box_nodes=150,
            goal_bias=0.3, verbose=False,
        ),
    }

    for name, config in param_sets.items():
        times = []
        successes = 0
        for run_i in range(N_RUNS):
            planner = BoxRRT(robot, scene, config)
            t0 = time.perf_counter()
            result = planner.plan(q_start, q_goal, seed=SEED_BASE + run_i)
            times.append(time.perf_counter() - t0)
            if result.success:
                successes += 1

        avg_t = statistics.mean(times)
        print(f"  [{name}] 成功率: {successes}/{N_RUNS}, "
              f"平均时间: {avg_t:.3f}s")


# ==================== 主入口 ====================

def main():
    print("Box-RRT 规划器性能基准测试")
    print(f"重复次数: {N_RUNS}, 预热: {N_WARMUP}")

    benchmark_collision_detection()
    benchmark_box_expansion()
    benchmark_planning()
    benchmark_parallel_collision()
    benchmark_parameter_sensitivity()

    print(f"\n{'='*60}")
    print(" 基准测试完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
