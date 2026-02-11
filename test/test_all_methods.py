"""
测试所有AABB计算方法是否正确处理始末端点
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from box_aabb import Robot, AABBCalculator, create_panda_robot


def test_all_methods():
    """测试所有AABB计算方法"""
    robot = create_panda_robot()
    
    # 定义测试区间
    joint_intervals = [
        (-0.3, 0.3),      # Joint 1
        (-0.4, 0.2),      # Joint 2
        (-0.2, 0.2),      # Joint 3
        (-1.8, -1.2),     # Joint 4
        (-0.2, 0.2),      # Joint 5
        (1.2, 1.8),       # Joint 6
        (-0.3, 0.3),      # Joint 7
    ]
    
    calc = AABBCalculator(robot, skip_first_link=True)
    
    print("=" * 60)
    print("测试所有AABB计算方法")
    print("=" * 60)
    
    # 测试方法列表
    test_cases = [
        ('numerical_critical', 'critical'),
        ('numerical_random', 'random'),
        ('interval', None),
    ]
    
    for method, sampling in test_cases:
        print(f"\n方法: {method}" + (f" (sampling={sampling})" if sampling else ""))
        print("-" * 60)
        
        try:
            result = calc.compute_envelope(joint_intervals, method=method.split('_')[0], 
                                          sampling=sampling if sampling else 'critical')
            
            # 打印有效AABB数量
            valid_aabbs = [a for a in result.link_aabbs if not a.is_zero_length]
            print(f"有效AABB数: {len(valid_aabbs)}")
            
            # 打印各连杆AABB尺寸
            for aabb in result.link_aabbs:
                if aabb.is_zero_length:
                    print(f"  Link {aabb.link_index}: [SKIPPED]")
                else:
                    size = aabb.size
                    volume = aabb.volume
                    print(f"  Link {aabb.link_index}: size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}), "
                          f"volume={volume:.6f}")
                    
                    # 检查边界配置
                    if aabb.boundary_configs:
                        print(f"    边界配置: {len(aabb.boundary_configs)} 个")
            
        except Exception as e:
            print(f"  错误: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    test_all_methods()
