#!/usr/bin/env python
"""
examples/random_scene_planning.py - 随机场景规划完整演示

随机化一个 2DOF 场景（包括障碍物和始末点），运行 Box-RRT 规划，
并对每一步的中间结果做详细记录。最终生成四张可视化图片：
  1. C-space box tree + 路径
  2. C-space 碰撞地图叠加
  3. 工作空间多姿态
  4. 动态动画 GIF

输出：
  - 终端详细日志
  - examples/output/random_scene_<timestamp>/  目录下所有产物

运行：
    python examples/random_scene_planning.py
    python examples/random_scene_planning.py --seed 123
    python examples/random_scene_planning.py --robot 3dof_planar --n-obs 5
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── 项目导入 ──────────────────────────────────────────────
from box_aabb.robot import Robot, load_robot
from planner import (
    BoxRRT,
    PlannerConfig,
    PlannerResult,
    Scene,
    AABBCache,
)
from planner.metrics import evaluate_result, PathMetrics
from planner.visualizer import (
    plot_cspace_boxes,
    plot_cspace_with_collision,
    plot_workspace_result,
)
from planner.dynamic_visualizer import animate_robot_path, resample_path

# ── 日志配置 ──────────────────────────────────────────────
LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("random_scene_planning")


# =====================================================================
# 1. 随机场景生成
# =====================================================================

def random_scene_2d(
    robot: Robot,
    n_obstacles: int,
    rng: np.random.Generator,
    q_start: np.ndarray | None = None,
    q_goal: np.ndarray | None = None,
) -> Scene:
    """为 2D 平面机器人随机生成障碍物场景

    如果提供了 q_start/q_goal，至少一个障碍物会放在路径中间的
    工作空间位置附近，迫使规划器必须绕行。

    Returns:
        Scene 对象
    """
    # 估算可达半径：所有连杆长度之和
    reach = sum(p['a'] for p in robot.dh_params)
    if reach < 1e-6:
        reach = 2.0  # fallback

    scene = Scene()

    # 若提供了始末点，先在路径中间工作空间位置放置"阻挡"障碍物
    n_blocking = 0
    if q_start is not None and q_goal is not None and n_obstacles >= 2:
        n_blocking = max(1, n_obstacles // 3)
        for i in range(n_blocking):
            # 在 C-space 路径上均匀插值
            t = (i + 1) / (n_blocking + 1)
            q_mid = (1 - t) * q_start + t * q_goal
            pos_mid = robot.get_link_positions(q_mid)
            # 选末端执行器位置附近放障碍物
            ee = pos_mid[-1]
            # 小偏移 + 随机扰动
            dx = rng.uniform(-0.1 * reach, 0.1 * reach)
            dy = rng.uniform(-0.1 * reach, 0.1 * reach)
            cx = ee[0] + dx
            cy = ee[1] + dy
            hw = rng.uniform(0.04 * reach, 0.10 * reach)
            hh = rng.uniform(0.04 * reach, 0.10 * reach)
            scene.add_obstacle(
                min_point=[cx - hw, cy - hh],
                max_point=[cx + hw, cy + hh],
                name=f"block_{i}",
            )

    # 剩余障碍物随机散布
    for i in range(n_obstacles - n_blocking):
        r = rng.uniform(0.3 * reach, 0.9 * reach)
        theta = rng.uniform(-math.pi, math.pi)
        cx = r * math.cos(theta)
        cy = r * math.sin(theta)

        hw = rng.uniform(0.03 * reach, 0.12 * reach)
        hh = rng.uniform(0.03 * reach, 0.12 * reach)

        scene.add_obstacle(
            min_point=[cx - hw, cy - hh],
            max_point=[cx + hw, cy + hh],
            name=f"obs_{i}",
        )
    return scene


def random_scene_3d(
    robot: Robot,
    n_obstacles: int,
    rng: np.random.Generator,
) -> Scene:
    """为 3D 空间机器人随机生成障碍物场景"""
    reach = sum(p.get('a', 0) for p in robot.dh_params)
    reach += sum(abs(p.get('d', 0)) for p in robot.dh_params)
    if reach < 1e-6:
        reach = 1.0

    scene = Scene()
    for i in range(n_obstacles):
        r = rng.uniform(0.15 * reach, 0.85 * reach)
        phi = rng.uniform(0, 2 * math.pi)
        costh = rng.uniform(-1, 1)
        sinth = math.sqrt(1 - costh ** 2)
        cx = r * sinth * math.cos(phi)
        cy = r * sinth * math.sin(phi)
        cz = r * costh

        hw = rng.uniform(0.03 * reach, 0.12 * reach, size=3)
        scene.add_obstacle(
            min_point=[cx - hw[0], cy - hw[1], cz - hw[2]],
            max_point=[cx + hw[0], cy + hw[1], cz + hw[2]],
            name=f"obs_{i}",
        )
    return scene


def random_collision_free_config(
    robot: Robot,
    scene: Scene,
    rng: np.random.Generator,
    max_attempts: int = 500,
) -> np.ndarray:
    """在关节限制范围内随机采样一个无碰撞配置

    Raises:
        RuntimeError: 若 max_attempts 次尝试都失败
    """
    from planner.collision import CollisionChecker
    checker = CollisionChecker(robot, scene)

    limits = robot.joint_limits or [(-math.pi, math.pi)] * robot.n_joints
    for _ in range(max_attempts):
        q = np.array([
            rng.uniform(lo, hi) for lo, hi in limits
        ])
        if not checker.check_config_collision(q):
            return q
    raise RuntimeError(
        f"在 {max_attempts} 次尝试内未找到无碰撞配置，"
        "场景可能过于拥挤"
    )


# =====================================================================
# 2. 详细日志记录
# =====================================================================

def log_robot_info(robot: Robot) -> str:
    """记录机器人信息"""
    lines = [
        "=" * 60,
        "  机器人信息",
        "=" * 60,
        f"  名称        : {robot.name}",
        f"  自由度      : {robot.n_joints}",
        f"  连杆数      : {len(robot.dh_params)}",
    ]
    if robot.joint_limits:
        lines.append("  关节限制    :")
        for i, (lo, hi) in enumerate(robot.joint_limits):
            lines.append(f"    q{i}: [{lo:+.4f}, {hi:+.4f}]  "
                         f"(范围 {hi - lo:.4f} rad)")
    lines.append("  DH 参数     :")
    for i, p in enumerate(robot.dh_params):
        lines.append(
            f"    Link {i}: α={p['alpha']:+.4f}  a={p['a']:.4f}  "
            f"d={p['d']:.4f}  type={p.get('type', 'revolute')}"
        )
    lines.append("=" * 60)
    text = "\n".join(lines)
    logger.info("\n%s", text)
    return text


def log_scene_info(scene: Scene) -> str:
    """记录场景信息"""
    lines = [
        "-" * 60,
        "  场景信息",
        "-" * 60,
        f"  障碍物数量  : {scene.n_obstacles}",
    ]
    total_vol = 0.0
    for obs in scene.get_obstacles():
        vol = obs.volume
        total_vol += vol
        sz = obs.size
        lines.append(
            f"  [{obs.name}]  "
            f"min=({obs.min_point[0]:+.3f}, {obs.min_point[1]:+.3f})  "
            f"max=({obs.max_point[0]:+.3f}, {obs.max_point[1]:+.3f})  "
            f"size=({sz[0]:.3f}×{sz[1]:.3f})  "
        )
    lines.append(f"  障碍物总 '面积' (2D): {total_vol:.4f}")
    lines.append("-" * 60)
    text = "\n".join(lines)
    logger.info("\n%s", text)
    return text


def log_endpoints(q_start: np.ndarray, q_goal: np.ndarray) -> str:
    """记录始末点"""
    def fmt(q):
        return ", ".join(f"{v:+.4f}" for v in q)

    dist = float(np.linalg.norm(q_goal - q_start))
    lines = [
        "-" * 60,
        "  始末点配置",
        "-" * 60,
        f"  起点  : [{fmt(q_start)}]",
        f"  终点  : [{fmt(q_goal)}]",
        f"  C-space 直线距离 : {dist:.4f} rad",
        "-" * 60,
    ]
    text = "\n".join(lines)
    logger.info("\n%s", text)
    return text


def log_planner_config(config: PlannerConfig) -> str:
    """记录规划器参数"""
    lines = [
        "-" * 60,
        "  规划器参数",
        "-" * 60,
        f"  max_iterations          : {config.max_iterations}",
        f"  max_box_nodes           : {config.max_box_nodes}",
        f"  seed_batch_size         : {config.seed_batch_size}",
        f"  min_box_volume          : {config.min_box_volume}",
        f"  goal_bias               : {config.goal_bias}",
        f"  expansion_resolution    : {config.expansion_resolution}",
        f"  max_expansion_rounds    : {config.max_expansion_rounds}",
        f"  connection_radius       : {config.connection_radius}",
        f"  connection_max_attempts : {config.connection_max_attempts}",
        f"  path_shortcut_iters     : {config.path_shortcut_iters}",
        f"  segment_collision_res   : {config.segment_collision_resolution}",
        f"  use_aabb_cache          : {config.use_aabb_cache}",
        f"  verbose                 : {config.verbose}",
        "-" * 60,
    ]
    text = "\n".join(lines)
    logger.info("\n%s", text)
    return text


def log_planner_result(result: PlannerResult) -> str:
    """记录规划结果"""
    lines = [
        "=" * 60,
        "  规划结果",
        "=" * 60,
        f"  状态        : {'✓ 成功' if result.success else '✗ 失败'}",
        f"  消息        : {result.message}",
        f"  计算时间    : {result.computation_time:.3f} s",
        f"  box 总数    : {result.n_boxes_created}",
        f"  碰撞检测数  : {result.n_collision_checks}",
        f"  路径点数    : {len(result.path)}",
    ]
    if result.success:
        lines.append(f"  路径长度    : {result.path_length:.4f} rad")
    lines.append(f"  box tree 数 : {len(result.box_trees)}")
    for tree in result.box_trees:
        lines.append(
            f"    树 {tree.tree_id}: {tree.n_nodes} 节点, "
            f"总体积 {tree.total_volume:.4f}"
        )
    lines.append("=" * 60)
    text = "\n".join(lines)
    logger.info("\n%s", text)
    return text


def log_metrics(metrics: PathMetrics) -> str:
    """记录路径质量指标"""
    lines = [
        "-" * 60,
        "  路径质量指标",
        "-" * 60,
        f"  路径长度 (L2)      : {metrics.path_length:.4f}",
        f"  直线距离           : {metrics.direct_distance:.4f}",
        f"  路径效率 (比值)    : {metrics.length_ratio:.4f}",
        f"  平滑度 (均值角变)  : {metrics.smoothness:.4f} rad",
        f"  最大曲率           : {metrics.max_curvature:.4f} rad",
        f"  最小安全裕度       : {metrics.min_clearance:.6f}",
        f"  平均安全裕度       : {metrics.avg_clearance:.6f}",
        f"  路径点数           : {metrics.n_waypoints}",
        f"  Box 总体积         : {metrics.box_coverage:.4f}",
        f"  Box 数量           : {metrics.n_boxes}",
    ]
    if metrics.joint_range_usage is not None:
        lines.append("  关节范围使用率     :")
        for i, u in enumerate(metrics.joint_range_usage):
            bar = "█" * int(u * 20) + "░" * (20 - int(u * 20))
            lines.append(f"    q{i}: {bar} {u * 100:.1f}%")
    lines.append("-" * 60)
    text = "\n".join(lines)
    logger.info("\n%s", text)
    return text


# =====================================================================
# 3. 可视化（保存文件）
# =====================================================================

def save_visualizations(
    robot: Robot,
    scene: Scene,
    result: PlannerResult,
    output_dir: Path,
) -> List[str]:
    """生成并保存所有可视化产物

    Returns:
        生成的文件路径列表
    """
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端
    import matplotlib.pyplot as plt

    saved: List[str] = []

    joint_limits = robot.joint_limits or [(-math.pi, math.pi)] * robot.n_joints

    # ---- 图1: C-space box tree ----
    logger.info("▶ 生成图1: C-space box tree ...")
    t0 = time.time()
    fig1 = plot_cspace_boxes(
        result,
        joint_limits=joint_limits,
        title=f"C-Space Box Tree ({robot.name})",
    )
    if fig1 is not None:
        p = str(output_dir / "01_cspace_boxes.png")
        fig1.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        saved.append(p)
        logger.info("  ✓ 保存到 %s (%.2fs)", p, time.time() - t0)

    # ---- 图2: C-space 碰撞地图叠加 ----
    is_2dof = (robot.n_joints <= 3)
    if is_2dof:
        logger.info("▶ 生成图2: C-space 碰撞地图叠加 ...")
        t0 = time.time()
        fig2 = plot_cspace_with_collision(
            robot, scene, joint_limits,
            result=result,
            resolution=0.05,
        )
        if fig2 is not None:
            p = str(output_dir / "02_cspace_collision.png")
            fig2.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            saved.append(p)
            logger.info("  ✓ 保存到 %s (%.2fs)", p, time.time() - t0)
    else:
        logger.info("  ⊘ 跳过碰撞地图（>2DOF 太慢）")

    # ---- 图3: 工作空间多姿态 ----
    if result.success:
        logger.info("▶ 生成图3: 工作空间多姿态 ...")
        t0 = time.time()
        fig3 = plot_workspace_result(
            robot, scene, result,
            n_poses=min(12, len(result.path)),
        )
        if fig3 is not None:
            p = str(output_dir / "03_workspace.png")
            fig3.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            saved.append(p)
            logger.info("  ✓ 保存到 %s (%.2fs)", p, time.time() - t0)

    # ---- 图4: 动态动画 GIF ----
    if result.success and len(result.path) >= 2:
        logger.info("▶ 生成图4: 动态动画 GIF ...")
        t0 = time.time()
        # 重采样到 80 帧，保证动画流畅
        smooth_path = resample_path(result.path, n_frames=80)
        try:
            anim = animate_robot_path(
                robot, smooth_path,
                scene=scene,
                fps=20,
                trail_length=30,
                title=f"Robot Motion ({robot.name})",
                show_ee_trail=True,
                ghost_interval=10,
            )
            p = str(output_dir / "04_animation.gif")
            anim.save(p, writer="pillow", fps=20)
            plt.close("all")
            saved.append(p)
            logger.info("  ✓ 保存到 %s (%.2fs)", p, time.time() - t0)
        except Exception as e:
            logger.warning("  ✗ 动画生成失败: %s", e)

    return saved


# =====================================================================
# 4. 报告生成
# =====================================================================

def write_report(
    output_dir: Path,
    robot: Robot,
    scene: Scene,
    config: PlannerConfig,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    result: PlannerResult,
    metrics: PathMetrics | None,
    saved_files: List[str],
    rng_seed: int | None,
) -> str:
    """生成 Markdown 报告文件"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    lines.append(f"# 随机场景规划报告")
    lines.append(f"")
    lines.append(f"> 生成时间: {ts}")
    lines.append(f"> 随机种子: {rng_seed}")
    lines.append(f"")

    # ── 1. 机器人 ──
    lines.append(f"## 1. 机器人")
    lines.append(f"")
    lines.append(f"| 属性 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 名称 | {robot.name} |")
    lines.append(f"| 自由度 | {robot.n_joints} |")
    if robot.joint_limits:
        for i, (lo, hi) in enumerate(robot.joint_limits):
            lines.append(f"| q{i} 范围 | [{lo:+.4f}, {hi:+.4f}] |")
    lines.append(f"")

    # ── 2. 场景 ──
    lines.append(f"## 2. 场景 ({scene.n_obstacles} 个障碍物)")
    lines.append(f"")
    lines.append(f"| 名称 | min | max | 尺寸 |")
    lines.append(f"|------|-----|-----|------|")
    for obs in scene.get_obstacles():
        mn = f"({obs.min_point[0]:+.3f}, {obs.min_point[1]:+.3f})"
        mx = f"({obs.max_point[0]:+.3f}, {obs.max_point[1]:+.3f})"
        sz = f"{obs.size[0]:.3f} × {obs.size[1]:.3f}"
        lines.append(f"| {obs.name} | {mn} | {mx} | {sz} |")
    lines.append(f"")

    # ── 3. 始末点 ──
    def qfmt(q):
        return ", ".join(f"{v:+.4f}" for v in q)
    dist_cspace = float(np.linalg.norm(q_goal - q_start))
    lines.append(f"## 3. 始末点")
    lines.append(f"")
    lines.append(f"- **起点**: [{qfmt(q_start)}]")
    lines.append(f"- **终点**: [{qfmt(q_goal)}]")
    lines.append(f"- **C-space 直线距离**: {dist_cspace:.4f} rad")
    lines.append(f"")

    # ── 4. 规划器参数 ──
    lines.append(f"## 4. 规划器参数")
    lines.append(f"")
    lines.append(f"| 参数 | 值 |")
    lines.append(f"|------|----|")
    for k, v in vars(config).items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"")

    # ── 5. 规划结果 ──
    lines.append(f"## 5. 规划结果")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 状态 | {'✓ 成功' if result.success else '✗ 失败'} |")
    lines.append(f"| 消息 | {result.message} |")
    lines.append(f"| 计算时间 | {result.computation_time:.3f} s |")
    lines.append(f"| Box 总数 | {result.n_boxes_created} |")
    lines.append(f"| 碰撞检测次数 | {result.n_collision_checks} |")
    lines.append(f"| 路径点数 | {len(result.path)} |")
    if result.success:
        lines.append(f"| 路径长度 (L2) | {result.path_length:.4f} rad |")
    lines.append(f"| Box tree 数 | {len(result.box_trees)} |")
    lines.append(f"")

    # Box tree 明细
    if result.box_trees:
        lines.append(f"### Box Tree 明细")
        lines.append(f"")
        lines.append(f"| 树 ID | 节点数 | 总体积 |")
        lines.append(f"|-------|--------|--------|")
        for tree in result.box_trees:
            lines.append(
                f"| {tree.tree_id} | {tree.n_nodes} "
                f"| {tree.total_volume:.4f} |"
            )
        lines.append(f"")

    # ── 6. 质量指标 ──
    if metrics is not None and result.success:
        lines.append(f"## 6. 路径质量指标")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 路径长度 (L2) | {metrics.path_length:.4f} |")
        lines.append(f"| 直线距离 | {metrics.direct_distance:.4f} |")
        lines.append(f"| 路径效率 | {metrics.length_ratio:.4f} |")
        lines.append(f"| 平滑度 | {metrics.smoothness:.4f} rad |")
        lines.append(f"| 最大曲率 | {metrics.max_curvature:.4f} rad |")
        lines.append(f"| 最小安全裕度 | {metrics.min_clearance:.6f} |")
        lines.append(f"| 平均安全裕度 | {metrics.avg_clearance:.6f} |")
        lines.append(f"| Box 总体积 | {metrics.box_coverage:.4f} |")
        lines.append(f"")

        if metrics.joint_range_usage is not None:
            lines.append(f"### 关节使用率")
            lines.append(f"")
            lines.append(f"| 关节 | 使用率 |")
            lines.append(f"|------|--------|")
            for i, u in enumerate(metrics.joint_range_usage):
                bar = "█" * int(u * 20) + "░" * (20 - int(u * 20))
                lines.append(f"| q{i} | {bar} {u * 100:.1f}% |")
            lines.append(f"")

    # ── 7. 路径点 ──
    if result.success and result.path:
        lines.append(f"## 7. 路径详情 ({len(result.path)} 点)")
        lines.append(f"")
        lines.append("| # | " + " | ".join(
            f"q{j}" for j in range(len(result.path[0]))) + " |")
        lines.append("|-" + "|-".join(
            "-" for _ in range(len(result.path[0]) + 1)) + "|")
        for idx, q in enumerate(result.path):
            vals = " | ".join(f"{v:+.4f}" for v in q)
            lines.append(f"| {idx} | {vals} |")
        lines.append(f"")

    # ── 8. 产物文件 ──
    lines.append(f"## 8. 产物文件")
    lines.append(f"")
    for f in saved_files:
        name = os.path.basename(f)
        lines.append(f"- `{name}`")
    lines.append(f"")

    report_text = "\n".join(lines)
    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("报告已保存到 %s", report_path)
    return str(report_path)


# =====================================================================
# 5. 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="随机场景 Box-RRT 规划演示")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (默认: 随机)")
    parser.add_argument("--robot", type=str, default="2dof_planar",
                        help="机器人配置名 (默认: 2dof_planar)")
    parser.add_argument("--n-obs", type=int, default=5,
                        help="障碍物数量 (默认: 5)")
    parser.add_argument("--max-iter", type=int, default=500,
                        help="最大迭代数 (默认: 500)")
    parser.add_argument("--max-boxes", type=int, default=200,
                        help="最大 box 数 (默认: 200)")
    parser.add_argument("--no-viz", action="store_true",
                        help="跳过可视化")
    args = parser.parse_args()

    rng_seed = args.seed if args.seed is not None else int(time.time()) % 100000
    rng = np.random.default_rng(rng_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("examples/output") / f"random_scene_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  随机场景 Box-RRT 规划演示")
    logger.info("  随机种子: %d", rng_seed)
    logger.info("  输出目录: %s", output_dir)
    logger.info("=" * 60)

    # ────────────────────────────────────────────────────────
    # Step 1: 加载机器人
    # ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Step 1/7: 加载机器人 '%s'", args.robot)
    t_step = time.time()
    robot = load_robot(args.robot)
    robot_text = log_robot_info(robot)
    logger.info("  完成 (%.3fs)", time.time() - t_step)

    # 判断是否为平面机器人
    is_planar = all(
        abs(p['alpha']) < 1e-10 and abs(p['d']) < 1e-10
        for p in robot.dh_params
    )

    # ────────────────────────────────────────────────────────
    # Step 2: 随机采样始末点（在空场景中）
    # ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Step 2/7: 随机采样始末点")
    t_step = time.time()
    empty_scene = Scene()  # 空场景：保证无碰撞
    q_start = random_collision_free_config(robot, empty_scene, rng)
    q_goal = random_collision_free_config(robot, empty_scene, rng)
    # 确保始末点距离足够远（至少 2.0 rad），使场景具有实际规划难度
    min_dist = 2.0
    attempts = 0
    while float(np.linalg.norm(q_goal - q_start)) < min_dist and attempts < 200:
        q_goal = random_collision_free_config(robot, empty_scene, rng)
        attempts += 1
    endpoint_text = log_endpoints(q_start, q_goal)
    logger.info("  完成 (%.3fs, %d 次重采样)", time.time() - t_step, attempts)

    # ────────────────────────────────────────────────────────
    # Step 3: 随机生成场景（利用始末点信息放置阻挡障碍物）
    # ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Step 3/7: 随机生成场景 (%d 个障碍物)", args.n_obs)
    t_step = time.time()
    if is_planar:
        scene = random_scene_2d(robot, args.n_obs, rng,
                                q_start=q_start, q_goal=q_goal)
    else:
        scene = random_scene_3d(robot, args.n_obs, rng)
    # 验证始末点在新场景中仍然无碰撞；若碰撞则移除冲突障碍物
    from planner.collision import CollisionChecker as _CC
    _checker = _CC(robot, scene)
    removed = []
    while _checker.check_config_collision(q_start) or \
          _checker.check_config_collision(q_goal):
        # 移除最后添加的障碍物并重试
        obs_list = scene.get_obstacles()
        if not obs_list:
            break
        last = obs_list[-1]
        scene.remove_obstacle(last.name)
        removed.append(last.name)
        _checker = _CC(robot, scene)
    if removed:
        logger.info("  移除了 %d 个与始末点冲突的障碍物: %s",
                     len(removed), ", ".join(removed))
    scene_text = log_scene_info(scene)
    # 保存场景 JSON
    scene_json = str(output_dir / "scene.json")
    scene.to_json(scene_json)
    logger.info("  场景 JSON 已保存到 %s", scene_json)
    logger.info("  完成 (%.3fs)", time.time() - t_step)

    # ────────────────────────────────────────────────────────
    # Step 4: 配置规划器
    # ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Step 4/7: 配置规划器")
    config = PlannerConfig(
        max_iterations=args.max_iter,
        max_box_nodes=args.max_boxes,
        seed_batch_size=5,
        expansion_resolution=0.03,
        max_expansion_rounds=3,
        goal_bias=0.20,
        connection_radius=4.0,
        connection_max_attempts=80,
        path_shortcut_iters=200,
        segment_collision_resolution=0.03,
        use_aabb_cache=False,
        verbose=True,
    )
    config_text = log_planner_config(config)

    planner = BoxRRT(robot, scene, config)

    # ────────────────────────────────────────────────────────
    # Step 5: 执行规划
    # ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Step 5/7: 执行 Box-RRT 规划")
    logger.info("  (计时开始 ...)")
    t_plan = time.time()
    result = planner.plan(q_start, q_goal, seed=rng_seed)
    dt_plan = time.time() - t_plan
    result_text = log_planner_result(result)
    logger.info("  规划耗时: %.3f s", dt_plan)

    # ────────────────────────────────────────────────────────
    # Step 6: 评估质量指标
    # ────────────────────────────────────────────────────────
    metrics: PathMetrics | None = None
    if result.success:
        logger.info("")
        logger.info("▶ Step 6/7: 评估路径质量指标")
        t_step = time.time()
        metrics = evaluate_result(result, robot, scene)
        metrics_text = log_metrics(metrics)
        logger.info("  完成 (%.3fs)", time.time() - t_step)
    else:
        logger.warning("  ⊘ 规划失败，跳过质量指标")

    # ────────────────────────────────────────────────────────
    # Step 7: 可视化
    # ────────────────────────────────────────────────────────
    saved_files: List[str] = []
    if not args.no_viz:
        logger.info("")
        logger.info("▶ Step 7/7: 生成可视化")
        t_step = time.time()
        try:
            saved_files = save_visualizations(robot, scene, result, output_dir)
            logger.info("  共生成 %d 个文件 (%.2fs)",
                        len(saved_files), time.time() - t_step)
        except Exception as e:
            logger.error("  可视化失败: %s", e, exc_info=True)
    else:
        logger.info("  ⊘ 跳过可视化 (--no-viz)")

    # ────────────────────────────────────────────────────────
    # 报告
    # ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ 生成 Markdown 报告")
    report_path = write_report(
        output_dir, robot, scene, config,
        q_start, q_goal, result, metrics,
        saved_files, rng_seed,
    )

    # ── 最终总结 ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("  完成！")
    logger.info("=" * 60)
    logger.info("  结果    : %s", "成功 ✓" if result.success else "失败 ✗")
    if result.success:
        logger.info("  路径长度 : %.4f rad", result.path_length)
    logger.info("  总耗时  : %.3f s", dt_plan)
    logger.info("  输出目录 : %s", output_dir)
    logger.info("  报告    : %s", report_path)
    for f in saved_files:
        logger.info("  图片    : %s", os.path.basename(f))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
