"""
benchmarks/threshold_experiment.py - 区间 vs 数值方法阈值实验

实验目标：确定何时使用区间算术方法、何时使用数值采样方法的合适阈值。

核心思想：
- interval 方法：安全但保守（过估计），区间越宽过估计越严重
- numerical 方法：紧致但不安全，通过多采样逼近真实值
- 阈值：找到两种方法 AABB 体积相近（交叉点）的区间宽度

实验流程：
1. 对每个关节维度，固定其他关节在某个配置
2. 逐步增大该维度的区间宽度 [q-w, q+w]
3. 分别计算 interval FK 和 numerical sampling 的 AABB 体积
4. 画出体积-宽度曲线，找交叉点

运行方式：
    python -m benchmarks.threshold_experiment
"""

import time
import logging
import sys
import os
from typing import List, Tuple, Optional, Dict

import numpy as np

# 确保项目路径在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from box_aabb.robot import Robot, load_robot
from box_aabb.interval_fk import compute_interval_aabb

logger = logging.getLogger(__name__)


def compute_numerical_aabb(
    robot: Robot,
    intervals: List[Tuple[float, float]],
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[dict], float]:
    """数值采样计算 AABB

    在区间内均匀采样多个配置，对每个配置做 FK 得到连杆位置，
    取所有采样点的 min/max 作为 AABB。

    Args:
        robot: 机器人
        intervals: 关节区间
        n_samples: 采样数
        rng: 随机数生成器

    Returns:
        (link_aabbs_dict_list, total_volume)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_joints = len(intervals)

    # 采样
    configs = []
    for _ in range(n_samples):
        q = np.array([rng.uniform(lo, hi) if hi > lo else lo
                       for lo, hi in intervals])
        configs.append(q)

    # 加入端点和中心
    center = np.array([(lo + hi) / 2.0 for lo, hi in intervals])
    configs.append(center)
    for i in range(n_joints):
        lo, hi = intervals[i]
        for val in [lo, hi]:
            q = center.copy()
            q[i] = val
            configs.append(q)

    # FK 并收集所有连杆位置
    all_positions = {}  # link_idx → list of positions
    for q in configs:
        positions = robot.get_link_positions(q)
        for li in range(len(positions)):
            if li not in all_positions:
                all_positions[li] = []
            all_positions[li].append(positions[li].copy())

    # 计算每个连杆的 AABB
    link_aabbs = []
    total_volume = 0.0
    for li in sorted(all_positions.keys()):
        pts = np.array(all_positions[li])
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        size = max_pt - min_pt
        vol = float(np.prod(np.maximum(size, 0.0)))
        link_aabbs.append({
            'link_index': li,
            'min_point': min_pt.tolist(),
            'max_point': max_pt.tolist(),
            'volume': vol,
        })
        total_volume += vol

    return link_aabbs, total_volume


def sweep_interval_width(
    robot: Robot,
    base_config: np.ndarray,
    dim: int,
    widths: np.ndarray,
    n_samples_numerical: int = 2000,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """对指定维度扫描不同区间宽度

    Args:
        robot: 机器人
        base_config: 基准配置
        dim: 扫描的关节维度
        widths: 半宽序列 [w1, w2, ...]
        n_samples_numerical: 数值采样数
        seed: 随机种子

    Returns:
        {'widths': [...], 'interval_volumes': [...], 'numerical_volumes': [...]}
    """
    rng = np.random.default_rng(seed)
    joint_limits = robot.joint_limits or [(-np.pi, np.pi)] * robot.n_joints

    result = {
        'widths': [],
        'interval_volumes': [],
        'numerical_volumes': [],
        'interval_times': [],
        'numerical_times': [],
    }

    for w in widths:
        # 构造区间：仅 dim 维度有宽度，其他维度为点区间
        intervals = []
        for d in range(robot.n_joints):
            if d == dim:
                lo = max(joint_limits[d][0], base_config[d] - w)
                hi = min(joint_limits[d][1], base_config[d] + w)
                intervals.append((lo, hi))
            else:
                intervals.append((base_config[d], base_config[d]))

        # Interval FK
        t0 = time.time()
        link_aabbs_iv, _ = compute_interval_aabb(
            robot=robot, intervals=intervals,
            zero_length_links=robot.zero_length_links,
            skip_zero_length=True, n_sub=1)
        t_iv = time.time() - t0

        iv_vol = sum(la.volume for la in link_aabbs_iv if not la.is_zero_length)

        # Numerical sampling
        t0 = time.time()
        _, num_vol = compute_numerical_aabb(
            robot, intervals, n_samples=n_samples_numerical, rng=rng)
        t_num = time.time() - t0

        result['widths'].append(float(2 * w))
        result['interval_volumes'].append(iv_vol)
        result['numerical_volumes'].append(num_vol)
        result['interval_times'].append(t_iv)
        result['numerical_times'].append(t_num)

    return result


def sweep_all_dims(
    robot: Robot,
    base_config: Optional[np.ndarray] = None,
    widths: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    seed: int = 42,
) -> Dict[int, Dict]:
    """对所有关节维度进行宽度扫描

    Returns:
        {dim: sweep_result, ...}
    """
    if base_config is None:
        base_config = np.zeros(robot.n_joints)

    if widths is None:
        widths = np.linspace(0.01, 1.5, 30)

    results = {}
    for d in range(robot.n_joints):
        print(f"扫描关节 {d}...")
        results[d] = sweep_interval_width(
            robot, base_config, d, widths,
            n_samples_numerical=n_samples, seed=seed)

    return results


def find_threshold(sweep_result: Dict) -> Optional[float]:
    """从扫描结果中找到体积交叉点（interval 体积首次超过 numerical 的 2 倍）

    Returns:
        阈值宽度，或 None
    """
    widths = sweep_result['widths']
    iv_vols = sweep_result['interval_volumes']
    num_vols = sweep_result['numerical_volumes']

    for i in range(len(widths)):
        if num_vols[i] > 0 and iv_vols[i] > 2 * num_vols[i]:
            return widths[i]
    return None


def plot_results(all_results: Dict[int, Dict], robot_name: str = "Robot"):
    """绘制所有维度的扫描结果"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    n_dims = len(all_results)
    ncols = min(4, n_dims)
    nrows = (n_dims + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_dims == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for d, sweep in all_results.items():
        ax = axes[d]
        ax.plot(sweep['widths'], sweep['interval_volumes'],
                'b-o', markersize=3, label='Interval FK')
        ax.plot(sweep['widths'], sweep['numerical_volumes'],
                'r-s', markersize=3, label='Numerical')

        threshold = find_threshold(sweep)
        if threshold is not None:
            ax.axvline(threshold, color='green', linestyle='--',
                       alpha=0.7, label=f'Threshold={threshold:.2f}')

        ax.set_xlabel('Interval width (rad)')
        ax.set_ylabel('Total AABB volume')
        ax.set_title(f'Joint {d}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'{robot_name}: Interval vs Numerical AABB Volume', fontsize=14)
    fig.tight_layout()
    return fig


def run_experiment(robot_name: str = 'panda', save_plot: bool = True):
    """运行完整实验"""
    robot = load_robot(robot_name)
    print(f"机器人: {robot.name}, {robot.n_joints} DOF")

    base_config = np.zeros(robot.n_joints)
    widths = np.linspace(0.01, 1.5, 25)

    print("开始扫描...")
    t0 = time.time()
    all_results = sweep_all_dims(robot, base_config, widths, n_samples=1000)
    elapsed = time.time() - t0
    print(f"扫描完成，耗时 {elapsed:.1f}s")

    # 输出阈值
    print("\n=== 阈值结果 ===")
    thresholds = []
    for d in range(robot.n_joints):
        th = find_threshold(all_results[d])
        thresholds.append(th)
        if th is not None:
            print(f"  Joint {d}: threshold = {th:.3f} rad")
        else:
            print(f"  Joint {d}: 未找到阈值 (interval 始终 < 2x numerical)")

    valid = [t for t in thresholds if t is not None]
    if valid:
        print(f"\n推荐全局阈值: {np.median(valid):.3f} rad (中位数)")
    else:
        print("\n未找到有效阈值")

    # 绘图
    if save_plot:
        fig = plot_results(all_results, robot.name)
        if fig is not None:
            output_path = f"threshold_{robot_name}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n图表已保存至 {output_path}")

    return all_results, thresholds


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 对 2DOF 和 3DOF 进行快速实验
    for name in ['2dof_planar', '3dof_planar']:
        try:
            print(f"\n{'=' * 50}")
            run_experiment(name)
        except Exception as e:
            print(f"跳过 {name}: {e}")

    # 对 Panda 7DOF 进行完整实验
    try:
        print(f"\n{'=' * 50}")
        run_experiment('panda')
    except Exception as e:
        print(f"Panda 实验失败: {e}")
