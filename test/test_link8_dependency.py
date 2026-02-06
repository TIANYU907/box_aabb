"""
验证Link8位置是否与q6, q7无关
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from box_aabb import Robot
import numpy as np


def main():
    config_path = os.path.join(os.path.dirname(__file__), 
                               '..', 'src', 'robot', 'panda_config.json')
    robot = Robot.from_json(config_path)
    
    print("=" * 70)
    print("验证 Link8 位置对各关节的依赖")
    print("=" * 70)
    
    # 基准配置
    base_q = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.0]
    
    # 获取Link8位置
    positions = robot.get_link_positions(base_q)
    link8_base = np.array(positions[-1])
    print(f"\n基准配置 q = {base_q}")
    print(f"Link8 位置: {link8_base}")
    
    # 测试每个关节的影响
    delta = 0.5  # 关节变化量
    
    print(f"\n各关节变化 {delta} rad 后 Link8 位置变化:")
    print("-" * 50)
    
    for i in range(8):
        q_test = list(base_q)
        q_test[i] += delta
        positions = robot.get_link_positions(q_test)
        link8_test = np.array(positions[-1])
        diff = np.linalg.norm(link8_test - link8_base)
        
        status = "有影响" if diff > 1e-6 else "无影响"
        print(f"  q{i}: 位置变化 = {diff:.6f} m  [{status}]")
    
    print("\n" + "=" * 70)
    print("DH参数分析")
    print("=" * 70)
    
    for i, param in enumerate(robot.dh_params):
        a = param['a']
        d = param['d']
        alpha = param['alpha']
        theta = param['theta']
        print(f"Joint {i}: a={a:.4f}, d={d:.4f}, alpha={alpha:.4f}, theta={theta:.4f}")


if __name__ == '__main__':
    main()
