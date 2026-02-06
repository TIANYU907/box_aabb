"""
对比关键点采样(critical) vs 随机采样(random)，找出关键点采样遗漏的极值点

目标：
1. 多次随机生成关节区间，分别用 critical 和 random 两种策略计算AABB包络
2. 比较两种方法的结果，找出 random 比 critical 更大的维度 (gap)
3. 详细记录 gap 处两种方法的包络信息和边界臂形，用于改进 critical 策略

使用方式：
    cd box_aabb
    python compare_critical_vs_random.py
"""

import os
import sys
import random
import math
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# 确保可以导入 box_aabb
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from box_aabb import (
    Robot,
    AABBCalculator,
    AABBEnvelopeResult,
    LinkAABBInfo,
    BoundaryConfig,
    create_panda_robot,
)

# 强制行缓冲，避免跨环境调用时 stdout 块缓冲导致无输出
sys.stdout.reconfigure(line_buffering=True)


# ==================== 配置 ====================

N_RUNS = 30               # 测试轮数
N_RANDOM_SAMPLES = 5000   # 随机采样数量（设大一些以充分探索）
MIN_GAP = 0.005           # gap 阈值，小于此值视为数值误差
N_ACTIVE_JOINTS = 7       # 活跃关节数（Panda前7个，第8个finger固定）
INTERVAL_WIDTH_MIN = 0.3  # 关节区间最小宽度
INTERVAL_WIDTH_MAX = 1.5  # 关节区间最大宽度


# ==================== 辅助函数 ====================

def generate_random_intervals(n_joints: int = N_ACTIVE_JOINTS, seed: int = None) -> List[Tuple[float, float]]:
    """生成随机关节区间"""
    if seed is not None:
        random.seed(seed)
    
    intervals = []
    for _ in range(n_joints):
        center = random.uniform(-1.5, 1.5)
        width = random.uniform(INTERVAL_WIDTH_MIN, INTERVAL_WIDTH_MAX)
        lo = center - width / 2
        hi = center + width / 2
        intervals.append((lo, hi))
    
    # 第8关节（finger）固定
    intervals.append((0.04, 0.04))
    return intervals


def format_boundary_config(config: Optional[BoundaryConfig], 
                           intervals: List[Tuple[float, float]]) -> str:
    """格式化边界配置为可读字符串"""
    if config is None:
        return "N/A"
    return config.format_joint_values(intervals)


def format_joint_values_raw(config: Optional[BoundaryConfig]) -> str:
    """输出原始关节值"""
    if config is None:
        return "N/A"
    vals = config.joint_values
    return "[" + ", ".join(f"{v:.6f}" for v in vals) + "]"


def analyze_config_features(config: Optional[BoundaryConfig], 
                            intervals: List[Tuple[float, float]]) -> List[str]:
    """分析边界配置的特征（角度和等）"""
    if config is None:
        return []
    
    features = []
    q = config.joint_values
    n = min(len(q), len(intervals))
    
    # 检查各关节是否在边界
    for i in range(n):
        lo, hi = intervals[i]
        if abs(q[i] - lo) < 1e-4:
            features.append(f"q{i}=LOW({lo:.4f})")
        elif abs(q[i] - hi) < 1e-4:
            features.append(f"q{i}=HIGH({hi:.4f})")
    
    # 检查常见角度和
    combos = [(0, 2), (1, 3), (0, 2, 4), (0, 2, 4, 6)]
    for combo in combos:
        if all(j < n for j in combo):
            angle_sum = sum(q[j] for j in combo)
            half_pi = math.pi / 2
            k = round(angle_sum / half_pi)
            residual = abs(angle_sum - k * half_pi)
            if residual < 0.05:
                joint_str = "+".join(f"q{j}" for j in combo)
                features.append(f"Σ({joint_str})≈{k}π/2 (res={residual:.4f})")
    
    return features


def compare_envelopes(result_random: AABBEnvelopeResult, 
                      result_critical: AABBEnvelopeResult,
                      intervals: List[Tuple[float, float]],
                      min_gap: float = MIN_GAP) -> List[Dict]:
    """比较两个包络结果，找出 random 更大的维度
    
    返回 gap 列表，每个 gap 包含详细的两种方法包络和边界信息
    """
    gaps = []
    axes = ['x', 'y', 'z']
    
    # 按 link_index 建立索引
    aabbs_r = {a.link_index: a for a in result_random.link_aabbs}
    aabbs_c = {a.link_index: a for a in result_critical.link_aabbs}
    
    for link_idx in sorted(set(aabbs_r.keys()) | set(aabbs_c.keys())):
        aabb_r = aabbs_r.get(link_idx)
        aabb_c = aabbs_c.get(link_idx)
        
        if aabb_r is None or aabb_c is None:
            continue
        if aabb_r.is_zero_length or aabb_c.is_zero_length:
            continue
        
        for i, axis in enumerate(axes):
            # random 找到更小的 min（更大的包络）
            gap_min = aabb_c.min_point[i] - aabb_r.min_point[i]
            if gap_min > min_gap:
                boundary_type = f"{axis}_min"
                cfg_r = aabb_r.boundary_configs.get(boundary_type) if aabb_r.boundary_configs else None
                cfg_c = aabb_c.boundary_configs.get(boundary_type) if aabb_c.boundary_configs else None
                
                gaps.append({
                    'link_index': link_idx,
                    'link_name': aabb_r.link_name,
                    'axis': axis,
                    'direction': 'min',
                    'boundary_type': boundary_type,
                    'gap': gap_min,
                    'random_value': aabb_r.min_point[i],
                    'critical_value': aabb_c.min_point[i],
                    # 完整AABB信息
                    'random_aabb_min': list(aabb_r.min_point),
                    'random_aabb_max': list(aabb_r.max_point),
                    'random_aabb_size': aabb_r.size,
                    'critical_aabb_min': list(aabb_c.min_point),
                    'critical_aabb_max': list(aabb_c.max_point),
                    'critical_aabb_size': aabb_c.size,
                    # 边界配置
                    'random_config': cfg_r,
                    'critical_config': cfg_c,
                    'random_config_str': format_boundary_config(cfg_r, intervals),
                    'critical_config_str': format_boundary_config(cfg_c, intervals),
                    'random_config_raw': format_joint_values_raw(cfg_r),
                    'critical_config_raw': format_joint_values_raw(cfg_c),
                    # 特征分析
                    'random_features': analyze_config_features(cfg_r, intervals),
                    'critical_features': analyze_config_features(cfg_c, intervals),
                    # 区间
                    'intervals': intervals,
                })
            
            # random 找到更大的 max（更大的包络）
            gap_max = aabb_r.max_point[i] - aabb_c.max_point[i]
            if gap_max > min_gap:
                boundary_type = f"{axis}_max"
                cfg_r = aabb_r.boundary_configs.get(boundary_type) if aabb_r.boundary_configs else None
                cfg_c = aabb_c.boundary_configs.get(boundary_type) if aabb_c.boundary_configs else None
                
                gaps.append({
                    'link_index': link_idx,
                    'link_name': aabb_r.link_name,
                    'axis': axis,
                    'direction': 'max',
                    'boundary_type': boundary_type,
                    'gap': gap_max,
                    'random_value': aabb_r.max_point[i],
                    'critical_value': aabb_c.max_point[i],
                    # 完整AABB信息
                    'random_aabb_min': list(aabb_r.min_point),
                    'random_aabb_max': list(aabb_r.max_point),
                    'random_aabb_size': aabb_r.size,
                    'critical_aabb_min': list(aabb_c.min_point),
                    'critical_aabb_max': list(aabb_c.max_point),
                    'critical_aabb_size': aabb_c.size,
                    # 边界配置
                    'random_config': cfg_r,
                    'critical_config': cfg_c,
                    'random_config_str': format_boundary_config(cfg_r, intervals),
                    'critical_config_str': format_boundary_config(cfg_c, intervals),
                    'random_config_raw': format_joint_values_raw(cfg_r),
                    'critical_config_raw': format_joint_values_raw(cfg_c),
                    # 特征分析
                    'random_features': analyze_config_features(cfg_r, intervals),
                    'critical_features': analyze_config_features(cfg_c, intervals),
                    # 区间
                    'intervals': intervals,
                })
    
    return gaps


# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("对比 Critical vs Random 采样 — 寻找关键点采样遗漏的极值点")
    print("=" * 70)
    print(f"测试轮数: {N_RUNS}")
    print(f"随机采样数: {N_RANDOM_SAMPLES}")
    print(f"Gap 阈值: {MIN_GAP}")
    print()
    
    # 创建机器人
    robot = create_panda_robot()
    calc = AABBCalculator(robot, robot_name="Panda", skip_first_link=True)
    
    all_gaps = []
    run_results = []
    total_start = time.time()
    
    for run_idx in range(N_RUNS):
        seed = random.randint(0, 1_000_000)
        intervals = generate_random_intervals(seed=seed)
        
        print(f"\n--- Run {run_idx + 1}/{N_RUNS} (seed={seed}) ---")
        print(f"  区间: {[(f'{l:.3f}', f'{h:.3f}') for l, h in intervals[:N_ACTIVE_JOINTS]]}")
        
        # Critical 采样
        t0 = time.time()
        result_critical = calc.compute_envelope(
            intervals, method='numerical', sampling='critical')
        time_critical = time.time() - t0
        
        # Random 采样
        t0 = time.time()
        result_random = calc.compute_envelope(
            intervals, method='numerical', sampling='random', 
            n_random_samples=N_RANDOM_SAMPLES)
        time_random = time.time() - t0
        
        vol_critical = result_critical.total_volume()
        vol_random = result_random.total_volume()
        
        print(f"  Critical: vol={vol_critical:.6f}, time={time_critical:.3f}s, "
              f"samples={result_critical.n_samples_evaluated}")
        print(f"  Random:   vol={vol_random:.6f}, time={time_random:.3f}s, "
              f"samples={result_random.n_samples_evaluated}")
        
        # 找出 random 更大的地方
        gaps = compare_envelopes(result_random, result_critical, intervals)
        
        if gaps:
            print(f"  ⚠️  发现 {len(gaps)} 处 random 更大!")
            for gap in gaps:
                print(f"    {gap['link_name']} {gap['axis']}-{gap['direction']}: "
                      f"gap={gap['gap']:.6f} "
                      f"(random={gap['random_value']:.6f}, critical={gap['critical_value']:.6f})")
            all_gaps.extend(gaps)
        else:
            print(f"  ✅ Critical >= Random")
        
        run_results.append({
            'run_idx': run_idx,
            'seed': seed,
            'intervals': intervals,
            'vol_critical': vol_critical,
            'vol_random': vol_random,
            'time_critical': time_critical,
            'time_random': time_random,
            'n_gaps': len(gaps),
        })
    
    total_time = time.time() - total_start
    
    # ==================== 汇总报告 ====================
    
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    
    n_wins_critical = sum(1 for r in run_results if r['vol_critical'] >= r['vol_random'])
    n_wins_random = N_RUNS - n_wins_critical
    avg_time_critical = np.mean([r['time_critical'] for r in run_results])
    avg_time_random = np.mean([r['time_random'] for r in run_results])
    
    print(f"Critical 胜出: {n_wins_critical}/{N_RUNS}")
    print(f"Random 胜出:   {n_wins_random}/{N_RUNS}")
    print(f"发现 gap 总数: {len(all_gaps)} (阈值>{MIN_GAP})")
    print(f"平均耗时 - Critical: {avg_time_critical:.3f}s, Random: {avg_time_random:.3f}s")
    print(f"总耗时: {total_time:.1f}s")
    
    if all_gaps:
        print(f"\n--- 遗漏的极值点详细分析 ---")
        for i, gap in enumerate(all_gaps):
            print(f"\n[{i+1}] {gap['link_name']} {gap['axis']}-{gap['direction']}  gap={gap['gap']:.6f}")
            print(f"    Random   AABB: min={_fmt3(gap['random_aabb_min'])} max={_fmt3(gap['random_aabb_max'])} "
                  f"size={_fmt3(gap['random_aabb_size'])}")
            print(f"    Critical AABB: min={_fmt3(gap['critical_aabb_min'])} max={_fmt3(gap['critical_aabb_max'])} "
                  f"size={_fmt3(gap['critical_aabb_size'])}")
            print(f"    Random   臂形: {gap['random_config_str']}")
            print(f"    Critical 臂形: {gap['critical_config_str']}")
            if gap['random_features']:
                print(f"    Random 特征: {', '.join(gap['random_features'])}")
    
    # ==================== 保存详细报告 ====================
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_reports")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"critical_vs_random_{timestamp}.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Critical vs Random 采样对比 — 遗漏分析\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("## 配置\n\n")
        f.write(f"  测试轮数:     {N_RUNS}\n")
        f.write(f"  随机采样数:   {N_RANDOM_SAMPLES}\n")
        f.write(f"  Gap 阈值:     {MIN_GAP}\n")
        f.write(f"  活跃关节数:   {N_ACTIVE_JOINTS}\n")
        f.write(f"  区间宽度范围: [{INTERVAL_WIDTH_MIN}, {INTERVAL_WIDTH_MAX}]\n")
        f.write(f"  总耗时:       {total_time:.1f}s\n\n")
        
        f.write("## 汇总\n\n")
        f.write(f"  Critical 胜出: {n_wins_critical}/{N_RUNS}\n")
        f.write(f"  Random 胜出:   {n_wins_random}/{N_RUNS}\n")
        f.write(f"  Gap 总数:      {len(all_gaps)}\n")
        f.write(f"  平均耗时 Critical: {avg_time_critical:.3f}s\n")
        f.write(f"  平均耗时 Random:   {avg_time_random:.3f}s\n\n")
        
        # 逐轮结果
        f.write("## 逐轮结果\n\n")
        f.write(f"{'Run':>4} {'Seed':>6} {'Vol_Critical':>14} {'Vol_Random':>14} "
                f"{'T_C(s)':>8} {'T_R(s)':>8} {'Gaps':>5}\n")
        f.write("-" * 70 + "\n")
        for r in run_results:
            f.write(f"{r['run_idx']+1:>4} {r['seed']:>6} {r['vol_critical']:>14.6f} "
                    f"{r['vol_random']:>14.6f} {r['time_critical']:>8.3f} "
                    f"{r['time_random']:>8.3f} {r['n_gaps']:>5}\n")
        f.write("\n")
        
        # 详细 gap 信息
        if all_gaps:
            f.write("## 遗漏的极值点详情\n\n")
            f.write(f"共 {len(all_gaps)} 处 random 比 critical 更大 (gap > {MIN_GAP})\n\n")
            
            for i, gap in enumerate(all_gaps):
                f.write(f"### [{i+1}] {gap['link_name']} {gap['axis']}-{gap['direction']}\n\n")
                f.write(f"  Gap 大小:      {gap['gap']:.6f}\n")
                f.write(f"  Random 值:     {gap['random_value']:.6f}\n")
                f.write(f"  Critical 值:   {gap['critical_value']:.6f}\n\n")
                
                f.write(f"  Random   AABB: min={_fmt3(gap['random_aabb_min'])} "
                        f"max={_fmt3(gap['random_aabb_max'])} "
                        f"size={_fmt3(gap['random_aabb_size'])}\n")
                f.write(f"  Critical AABB: min={_fmt3(gap['critical_aabb_min'])} "
                        f"max={_fmt3(gap['critical_aabb_max'])} "
                        f"size={_fmt3(gap['critical_aabb_size'])}\n\n")
                
                f.write(f"  Random   臂形: {gap['random_config_str']}\n")
                f.write(f"  Random   原始: {gap['random_config_raw']}\n")
                f.write(f"  Critical 臂形: {gap['critical_config_str']}\n")
                f.write(f"  Critical 原始: {gap['critical_config_raw']}\n\n")
                
                if gap['random_features']:
                    f.write(f"  Random 特征分析:\n")
                    for feat in gap['random_features']:
                        f.write(f"    - {feat}\n")
                    f.write("\n")
                
                if gap['critical_features']:
                    f.write(f"  Critical 特征分析:\n")
                    for feat in gap['critical_features']:
                        f.write(f"    - {feat}\n")
                    f.write("\n")
                
                f.write(f"  关节区间:\n")
                for j, (lo, hi) in enumerate(gap['intervals'][:N_ACTIVE_JOINTS]):
                    f.write(f"    q{j}: [{lo:.4f}, {hi:.4f}] (width={hi-lo:.4f})\n")
                f.write("\n")
                f.write("-" * 70 + "\n\n")
        else:
            f.write("## 未发现遗漏\n\n")
            f.write("Critical 采样在所有测试中均 >= Random 采样\n")
    
    print(f"\n详细报告已保存至: {output_path}")


def _fmt3(vals) -> str:
    """格式化3个浮点数"""
    return f"({vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f})"


if __name__ == "__main__":
    main()
