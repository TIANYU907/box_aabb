"""
示例：可视化AABB包络和边界臂形

展示如何使用 visualize_envelope_result 函数
可视化机器人的AABB包围盒、边界臂形和采样臂形
"""
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, 'src'))

from box_aabb import (
    Robot, 
    AABBCalculator, 
    visualize_envelope_result,
    load_robot,
)
import math


def main():
    # 创建Panda机器人
    print("=" * 60)
    print("AABB Envelope Visualization Demo")
    print("=" * 60)
    
    robot = load_robot('panda')
    robot_name = robot.name
    print(f"机器人名称: {robot_name}")
    print(f"关节数: {robot.n_joints}")
    print()
    
    # 定义关节区间 (较小区间便于可视化)
    joint_intervals = [
        (-0.3, 0.3),      # Joint 1
        (-0.4, 0.2),      # Joint 2
        (-0.2, 0.2),      # Joint 3
        (-1.8, -1.2),     # Joint 4
        (-0.2, 0.2),      # Joint 5
        (1.2, 1.8),       # Joint 6
        (-0.3, 0.3),      # Joint 7
    ]
    
    print("关节区间:")
    for i, (lo, hi) in enumerate(joint_intervals):
        print(f"  Joint {i+1}: [{lo:.3f}, {hi:.3f}] rad")
    print()
    
    # 创建计算器并计算AABB
    print("计算AABB...")
    calc = AABBCalculator(robot, robot_name=robot_name, skip_first_link=True)
    result = calc.compute_envelope(joint_intervals, method='numerical', sampling='critical')
    
    print(f"计算方法: {result.method}")
    print(f"有效AABB数: {len([a for a in result.link_aabbs if not a.is_zero_length])}")
    print()
    
    # 打印AABB信息
    print("各连杆AABB:")
    for aabb in result.link_aabbs:
        if aabb.is_zero_length:
            print(f"  Link {aabb.link_index}: [SKIPPED - zero length]")
        else:
            size = [aabb.max_point[i] - aabb.min_point[i] for i in range(3)]
            print(f"  Link {aabb.link_index}: size = ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})")
    print()
    
    # 可视化（参考 visualize_methods_comparison.py 的设置）
    print("启动可视化...")
    print("  - 采样臂形（含中点）: 细虚线 (linewidth=1.0, alpha=0.2), 彩虹色")
    print("  - 边界臂形（仅Link8）: 实线 (linewidth=2.5, alpha=0.9), 蓝色")
    print("  - 边界标记球: 红色，显示在真正达到边界的位置（始端或末端）")
    print("  - 采样步长: 0.5 radians")
    print("  - 最大采样: 100 个")
    print()
    
    viz = visualize_envelope_result(
        result=result,
        robot=robot,
        show_boundary_configs=True,   # 显示边界臂形（仅Link8）
        show_samples=True,
        show_aabbs=True,
        title=f'{robot_name} - AABB Envelope with Link8 Boundaries',
        figsize=(8, 6),  # 参考原实现
        interactive=True
    )
    
    viz.show()
    
    print("可视化完成!")


if __name__ == '__main__':
    main()
