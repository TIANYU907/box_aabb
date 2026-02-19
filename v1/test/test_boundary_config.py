"""
验证边界配置和报告的正确性
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from box_aabb import Robot, AABBCalculator, load_robot
import numpy as np


def main():
    robot = load_robot('panda')
    
    # 定义关节区间
    joint_intervals = [
        (-0.3, 0.3),      # Joint 0
        (-0.4, 0.2),      # Joint 1
        (-0.2, 0.2),      # Joint 2
        (-1.8, -1.2),     # Joint 3
        (-0.2, 0.2),      # Joint 4
        (1.2, 1.8),       # Joint 5
        (-0.3, 0.3),      # Joint 6
        (0.04, 0.04),     # Joint 7 (fixed)
    ]
    
    calc = AABBCalculator(robot, skip_first_link=True)
    result = calc.compute_envelope(joint_intervals, method='numerical', sampling='critical')
    
    print("=" * 70)
    print("验证边界配置")
    print("=" * 70)
    
    for aabb in result.link_aabbs:
        if aabb.is_zero_length:
            continue
        
        print(f"\n{aabb.link_name}:")
        print(f"  相关关节检测:")
        
        for boundary_type, config in aabb.boundary_configs.items():
            print(f"\n  {boundary_type}:")
            print(f"    joint_values type: {type(config.joint_values)}")
            print(f"    joint_values: {config.joint_values}")
            print(f"    relevant_joints: {sorted(config.relevant_joints)}")
            print(f"    boundary_joints (📍): {sorted(config.boundary_joints)}")
            print(f"    boundary_value: {config.boundary_value:.4f}")
            
            # 验证边界标记
            for idx in sorted(config.relevant_joints):
                if idx < len(joint_intervals):
                    lo, hi = joint_intervals[idx]
                    val = config.joint_values[idx]
                    is_at_boundary = abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6
                    mark = "📍" if idx in config.boundary_joints else ""
                    actual = "边界" if is_at_boundary else "内部"
                    status = "✓" if (is_at_boundary == (idx in config.boundary_joints)) else "✗ ERROR"
                    print(f"      q{idx}={val:.4f}{mark} [{actual}] {status}")
    
    print("\n" + "=" * 70)
    print("生成报告预览")
    print("=" * 70)
    
    report = result.generate_report()
    # 只显示边界配置部分
    lines = report.split('\n')
    in_boundary_section = False
    for line in lines:
        if '边界配置' in line:
            in_boundary_section = True
        if in_boundary_section:
            print(line)
            if line.strip() == '' and in_boundary_section:
                break


if __name__ == '__main__':
    main()
