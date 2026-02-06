"""
examples/basic_usage.py - 基本使用示例

演示如何使用box_aabb库计算和可视化机器人AABB
"""

import sys
import os

# 添加 src/ 目录到路径，使 box_aabb 包可导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, 'src'))

from box_aabb import (
    Robot, AABBCalculator, Visualizer,
    create_panda_robot, PANDA_JOINT_LIMITS,
    AABBEnvelopeResult
)


def example_basic():
    """基本用法示例"""
    print("=" * 60)
    print("示例1: 基本AABB计算")
    print("=" * 60)
    
    # 创建Panda机器人
    robot = create_panda_robot()
    print(f"机器人关节数: {robot.n_joints}")
    
    # 定义关节区间
    joint_intervals = [
        (-0.5, 0.5),    # q0: 基座旋转
        (-0.3, 0.3),    # q1: 肩部
        (-0.5, 0.5),    # q2
        (-2.0, -1.0),   # q3: 肘部
        (-0.5, 0.5),    # q4
        (0.5, 1.5),     # q5
        (-0.5, 0.5),    # q6: 腕部
    ]
    
    # 创建计算器
    calc = AABBCalculator(robot, robot_name="Panda")
    
    # 使用关键点采样（默认方法）
    print("\n--- 数值方法 - 关键点采样 (Critical) ---")
    result_critical = calc.compute_envelope(joint_intervals, method='numerical', sampling='critical')
    print(f"采样点数: {result_critical.n_samples_evaluated}")
    print(f"计算耗时: {result_critical.computation_time:.4f} 秒")
    print(f"总体积: {result_critical.total_volume():.6f} m^3")
    
    print("\n--- 区间方法 (Interval) ---")
    result_interval = calc.compute_envelope(joint_intervals, method='interval')
    print(f"计算耗时: {result_interval.computation_time:.4f} 秒")
    print(f"总体积: {result_interval.total_volume():.6f} m^3")
    
    # 紧凑度比较
    print(f"\n紧凑度比较:")
    print(f"  Critical: {result_critical.total_volume():.6f} (基准)")
    print(f"  Interval: {result_interval.total_volume():.6f} ({result_interval.total_volume()/result_critical.total_volume():.1%})")
    
    return result_critical


def example_sampling_comparison():
    """采样方法比较示例"""
    print("\n" + "=" * 60)
    print("示例2: 采样方法比较")
    print("=" * 60)
    
    robot = create_panda_robot()
    calc = AABBCalculator(robot, robot_name="Panda")
    
    # 使用更宽的区间来测试
    joint_intervals = [
        (-1.0, 1.0),    # q0
        (-0.5, 0.5),    # q1
        (-1.0, 1.0),    # q2
        (-2.5, -0.5),   # q3
        (-1.0, 1.0),    # q4
        (0.0, 2.0),     # q5
        (-1.0, 1.0),    # q6
    ]
    
    print("\n--- 关键点采样 (Critical) ---")
    result_critical = calc.compute_envelope(joint_intervals, method='numerical', sampling='critical')
    print(f"采样点数: {result_critical.n_samples_evaluated}")
    print(f"计算耗时: {result_critical.computation_time:.4f} 秒")
    
    print("\n--- 随机采样 (Random) ---")
    result_random = calc.compute_envelope(
        joint_intervals, 
        method='numerical', 
        sampling='random',
        n_random_samples=5000
    )
    print(f"采样点数: {result_random.n_samples_evaluated}")
    print(f"计算耗时: {result_random.computation_time:.4f} 秒")
    
    # 比较末端执行器AABB
    ee_critical = result_critical.get_end_effector_aabb()
    ee_random = result_random.get_end_effector_aabb()
    
    print("\n末端执行器AABB比较:")
    print(f"  Critical - 体积: {ee_critical.volume:.6f} m^3")
    print(f"  Random   - 体积: {ee_random.volume:.6f} m^3")
    
    return result_critical, result_random


def example_report_generation():
    """报告生成示例"""
    print("\n" + "=" * 60)
    print("示例3: 报告生成")
    print("=" * 60)
    
    robot = create_panda_robot()
    calc = AABBCalculator(robot, robot_name="Panda")
    
    # 小区间，容易看到边界配置
    joint_intervals = [
        (-0.2, 0.2),    # q0
        (-0.1, 0.1),    # q1
        (-0.2, 0.2),    # q2
        (-2.0, -1.5),   # q3
        (-0.2, 0.2),    # q4
        (0.5, 1.0),     # q5
        (-0.1, 0.1),    # q6
    ]
    
    result = calc.compute_envelope(joint_intervals, method='numerical', sampling='critical')
    
    # 生成并打印报告
    report = result.generate_report()
    print("\n" + "-" * 40)
    print("生成的报告预览 (前60行):")
    print("-" * 40)
    lines = report.split('\n')
    for line in lines[:60]:
        print(line)
    
    if len(lines) > 60:
        print(f"... (还有 {len(lines) - 60} 行)")
    
    # 保存报告
    report_path = os.path.join(os.path.dirname(__file__), 'aabb_report.md')
    result.generate_report(save_path=report_path)
    print(f"\n完整报告已保存到: {report_path}")
    
    return result


def example_zero_length_link():
    """零长度连杆处理示例"""
    print("\n" + "=" * 60)
    print("示例4: 零长度连杆处理")
    print("=" * 60)
    
    robot = create_panda_robot()
    calc = AABBCalculator(robot, robot_name="Panda")
    
    print(f"识别的零长度连杆索引: {calc._zero_length_links}")
    
    joint_intervals = [
        (-0.5, 0.5),
        (-0.3, 0.3),
        (-0.5, 0.5),
        (-2.0, -1.0),
        (-0.5, 0.5),
        (0.5, 1.5),
        (-0.5, 0.5),
    ]
    
    # 跳过零长度连杆
    result_skip = calc.compute_envelope(joint_intervals, skip_zero_length=True)
    
    # 不跳过零长度连杆
    result_no_skip = calc.compute_envelope(joint_intervals, skip_zero_length=False)
    
    print("\n各连杆状态:")
    for aabb in result_skip.link_aabbs:
        status = "(零长度，已跳过)" if aabb.is_zero_length else ""
        print(f"  {aabb.link_name}: 体积={aabb.volume:.6f} m^3 {status}")


def example_boundary_configs():
    """边界配置分析示例"""
    print("\n" + "=" * 60)
    print("示例5: 边界配置分析")
    print("=" * 60)
    
    robot = create_panda_robot()
    calc = AABBCalculator(robot, robot_name="Panda")
    
    # 窄区间，便于观察边界配置
    joint_intervals = [
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-1.8, -1.5),
        (-0.1, 0.1),
        (0.8, 1.0),
        (-0.1, 0.1),
    ]
    
    result = calc.compute_envelope(joint_intervals, method='numerical', sampling='critical')
    
    # 分析末端执行器的边界配置
    ee = result.get_end_effector_aabb()
    print(f"\n末端执行器 ({ee.link_name}) 边界配置分析:")
    
    for boundary_type, config in sorted(ee.boundary_configs.items()):
        print(f"\n  {boundary_type.upper()}:")
        print(f"    边界值: {config.boundary_value:.4f} m")
        print(f"    关节配置: {config.format_joint_values(joint_intervals)}")
        if config.angle_constraints:
            print(f"    角度约束: {', '.join(config.angle_constraints)}")
        if config.is_aabb_vertex:
            print(f"    ⭐ 这是AABB顶点")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("BOX-AABB 库使用示例")
    print("=" * 60)
    
    # 基本使用
    result = example_basic()
    
    # 采样方法比较
    example_sampling_comparison()
    
    # 报告生成
    example_report_generation()
    
    # 零长度连杆
    example_zero_length_link()
    
    # 边界配置分析
    example_boundary_configs()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
