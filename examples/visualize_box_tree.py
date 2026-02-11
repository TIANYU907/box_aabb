#!/usr/bin/env python
"""
examples/visualize_box_tree.py - 2DOF Box Tree 可视化实验

在随机 2D 障碍物场景中，用 2DOF 平面机械臂尽可能多地生成
free box，然后在 C-space 中可视化 box tree + 碰撞地图。

目的：直观拆解 Box-RRT 的核心步骤——box 拓展的质量和覆盖效果。

输出（保存到 examples/output/box_tree_viz_<ts>/）：
  - cspace_collision.png  : 纯 C-space 碰撞地图
  - cspace_boxes.png      : C-space box tree 叠加碰撞地图
  - workspace.png         : 工作空间中机械臂 + 障碍物
  - stats.txt             : box 统计摘要

用法：
    python -m examples.visualize_box_tree
    python -m examples.visualize_box_tree --seed 42 --n-obs 4 --max-boxes 300
    python -m examples.visualize_box_tree --resolution 0.02 --max-rounds 5
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from box_aabb.robot import Robot, load_robot
from planner.models import BoxNode, PlannerConfig
from planner.obstacles import Scene
from planner.collision import CollisionChecker
from planner.box_expansion import BoxExpander
from planner.box_tree import BoxTreeManager

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("box_tree_viz")


# ─────────────────────────────────────────────────────────
#  随机场景生成（2D）
# ─────────────────────────────────────────────────────────

def random_scene_2d(robot: Robot, n_obs: int, rng: np.random.Generator) -> Scene:
    """为 2DOF 平面机器人随机生成障碍物"""
    reach = sum(p['a'] for p in robot.dh_params)
    if robot.tool_frame:
        reach += robot.tool_frame.get('a', 0)
    reach = reach or 2.0
    scene = Scene()
    for i in range(n_obs):
        r = rng.uniform(0.3 * reach, 0.9 * reach)
        theta = rng.uniform(-math.pi, math.pi)
        cx, cy = r * math.cos(theta), r * math.sin(theta)
        hw = rng.uniform(0.05 * reach, 0.15 * reach)
        hh = rng.uniform(0.05 * reach, 0.15 * reach)
        scene.add_obstacle(
            min_point=[cx - hw, cy - hh],
            max_point=[cx + hw, cy + hh],
            name=f"obs_{i}",
        )
    return scene


# ─────────────────────────────────────────────────────────
#  核心：大规模 box 生成
# ─────────────────────────────────────────────────────────

def grow_box_forest(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    max_boxes: int = 300,
    max_iters: int = 2000,
    expansion_resolution: float = 0.02,
    max_rounds: int = 3,
    seed_batch: int = 8,
    rng_seed: int = 0,
    farthest_k: int = 12,
) -> Tuple[BoxTreeManager, int]:
    """从 C-space 中大量采样，尽可能多地拓展 free box

    v2: 使用最远点采样策略，优先探索远离已有 box 的区域。

    Returns:
        (tree_manager, n_collision_checks)
    """
    rng = np.random.default_rng(rng_seed)

    checker = CollisionChecker(robot, scene)
    expander = BoxExpander(
        robot=robot,
        collision_checker=checker,
        joint_limits=joint_limits,
        expansion_resolution=expansion_resolution,
        max_rounds=max_rounds,
    )
    manager = BoxTreeManager()

    t0 = time.time()
    n_sampled = 0
    n_farthest_fails = 0

    for iteration in range(max_iters):
        if manager.total_nodes >= max_boxes:
            break

        # ── 最远点采样: 批量候选 → 选最远 ──
        candidates = []
        for _ in range(farthest_k):
            q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
            n_sampled += 1
            if checker.check_config_collision(q):
                continue
            if manager.find_containing_box(q) is not None:
                continue
            nearest = manager.find_nearest_box(q)
            dist = nearest.distance_to_config(q) if nearest else float('inf')
            candidates.append((q, dist))

        if not candidates:
            n_farthest_fails += 1
            if n_farthest_fails > 50:
                break
            continue
        n_farthest_fails = 0

        q_seed, _ = max(candidates, key=lambda x: x[1])

        # ── 拓展 box ──
        node_id = manager.allocate_node_id()
        box = expander.expand(q_seed, node_id=node_id, rng=rng)
        if box is None or box.volume < 1e-6:
            continue

        # ── 加入最近树或建新树 ──
        nearest = manager.find_nearest_box(q_seed)
        if nearest is not None and nearest.distance_to_config(q_seed) < 2.0:
            manager.add_box(nearest.tree_id, box, parent_id=nearest.node_id)
        else:
            manager.create_tree(box)

        # ── 边界再采样拓展（让树生长） ──
        if box.tree_id >= 0 and manager.total_nodes < max_boxes:
            samples = manager.get_boundary_samples(
                box.tree_id, n_samples=seed_batch, rng=rng)
            for qs in samples:
                if manager.total_nodes >= max_boxes:
                    break
                if checker.check_config_collision(qs):
                    continue
                if manager.find_containing_box(qs) is not None:
                    continue
                nid = manager.allocate_node_id()
                child = expander.expand(qs, node_id=nid, rng=rng)
                if child is None or child.volume < 1e-6:
                    continue
                nearest_in = manager.find_nearest_box_in_tree(box.tree_id, qs)
                if nearest_in is not None:
                    manager.add_box(box.tree_id, child,
                                    parent_id=nearest_in.node_id)

        # 进度日志
        if (iteration + 1) % 200 == 0:
            logger.info(
                "iter %d/%d: %d trees, %d boxes, volume=%.3f (sampled %d seeds)",
                iteration + 1, max_iters, manager.n_trees,
                manager.total_nodes, manager.get_total_volume(), n_sampled,
            )

    dt = time.time() - t0
    logger.info(
        "Box 生成完成: %d trees, %d boxes, total_volume=%.4f, "
        "%.2fs, %d seeds sampled, %d collision checks",
        manager.n_trees, manager.total_nodes, manager.get_total_volume(),
        dt, n_sampled, checker.n_collision_checks,
    )
    return manager, checker.n_collision_checks


# ─────────────────────────────────────────────────────────
#  可视化
# ─────────────────────────────────────────────────────────

def visualize(
    robot: Robot,
    scene: Scene,
    manager: BoxTreeManager,
    joint_limits: List[Tuple[float, float]],
    output_dir: Path,
    resolution: float = 0.03,
):
    """生成三张可视化图

    1. C-space 碰撞地图（纯底图）
    2. C-space box tree 叠加碰撞地图
    3. 工作空间障碍物 + 几个 random 配置的机械臂姿态
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm

    checker = CollisionChecker(robot, scene)
    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]

    # ── 扫描 C-space 碰撞地图 ──
    logger.info("扫描 C-space 碰撞地图 (resolution=%.3f) ...", resolution)
    xs = np.arange(lo_x, hi_x, resolution)
    ys = np.arange(lo_y, hi_y, resolution)
    collision_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if checker.check_config_collision(np.array([x, y])):
                collision_map[i, j] = 1.0
    logger.info("  C-space 大小: %d x %d", len(xs), len(ys))

    # 计算自由空间比例
    n_free = int(np.sum(collision_map == 0))
    n_total = collision_map.size
    free_ratio = n_free / n_total
    logger.info("  自由空间比例: %.1f%%", free_ratio * 100)

    # ──── 图 1: 纯碰撞地图 ────
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(collision_map, origin='lower',
               extent=[lo_x, hi_x, lo_y, hi_y],
               cmap='Reds', alpha=0.5, aspect='auto')
    ax1.set_xlabel('q0 (rad)')
    ax1.set_ylabel('q1 (rad)')
    ax1.set_title(f'C-Space Collision Map  (free={free_ratio*100:.1f}%)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    p1 = output_dir / "cspace_collision.png"
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    logger.info("  保存: %s", p1)

    # ──── 图 2: Box tree + 碰撞地图 ────
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(collision_map, origin='lower',
               extent=[lo_x, hi_x, lo_y, hi_y],
               cmap='Reds', alpha=0.35, aspect='auto')

    trees = manager.get_all_trees()
    tree_colors = cm.Set3(np.linspace(0, 1, max(len(trees), 1)))

    # ── 用栅格法计算 box 真实 union 覆盖面积（去重）──
    box_mask = np.zeros_like(collision_map, dtype=bool)   # same grid as collision_map
    cell_area = resolution * resolution

    total_box_area_raw = 0.0   # 未去重的 box 面积之和（仅用于信息展示）
    for t_idx, tree in enumerate(trees):
        color = tree_colors[t_idx % len(tree_colors)]
        for box in tree.nodes.values():
            lo0, hi0 = box.joint_intervals[0]
            lo1, hi1 = box.joint_intervals[1]
            w, h = hi0 - lo0, hi1 - lo1
            total_box_area_raw += w * h
            # 在 grid 上标记此 box 覆盖的像素
            j0 = max(int((lo0 - lo_x) / resolution), 0)
            j1 = min(int(np.ceil((hi0 - lo_x) / resolution)), len(xs))
            i0 = max(int((lo1 - lo_y) / resolution), 0)
            i1 = min(int(np.ceil((hi1 - lo_y) / resolution)), len(ys))
            box_mask[i0:i1, j0:j1] = True
            rect = Rectangle(
                (lo0, lo1), w, h,
                linewidth=0.6, edgecolor=color, facecolor=color, alpha=0.30)
            ax2.add_patch(rect)
            # seed 标记
            ax2.plot(box.seed_config[0], box.seed_config[1],
                     '.', color=color, markersize=2, alpha=0.7)

    cspace_area = (hi_x - lo_x) * (hi_y - lo_y)
    union_pixels = int(np.sum(box_mask))
    coverage = union_pixels / n_total * 100        # 去重后的真实覆盖率
    coverage_raw = total_box_area_raw / cspace_area * 100  # 未去重（含重叠）

    ax2.set_xlim(lo_x - 0.1, hi_x + 0.1)
    ax2.set_ylim(lo_y - 0.1, hi_y + 0.1)
    ax2.set_xlabel('q0 (rad)')
    ax2.set_ylabel('q1 (rad)')
    ax2.set_title(
        f'C-Space Box Tree  ({manager.total_nodes} boxes, '
        f'{manager.n_trees} trees, coverage={coverage:.1f}%, '
        f'raw={coverage_raw:.1f}%)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    p2 = output_dir / "cspace_boxes.png"
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info("  保存: %s", p2)

    # ──── 图 3: 工作空间 ────
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    # 障碍物
    for obs in scene.get_obstacles():
        rect = Rectangle(
            (obs.min_point[0], obs.min_point[1]),
            obs.size[0], obs.size[1],
            linewidth=1.2, edgecolor='red', facecolor='red', alpha=0.4)
        ax3.add_patch(rect)

    # 从各 box 中心取几个配置画机械臂
    all_boxes = manager.get_all_boxes()
    # 按体积降序，取前 20 个大 box
    all_boxes.sort(key=lambda b: b.volume, reverse=True)
    n_show = min(25, len(all_boxes))
    cmap = cm.viridis
    for idx in range(n_show):
        q = all_boxes[idx].center
        positions = robot.get_link_positions(q)
        xs = [p[0] for p in positions]
        ys_pos = [p[1] for p in positions]
        alpha = 0.2 + 0.6 * (1 - idx / max(n_show - 1, 1))
        color = cmap(idx / max(n_show - 1, 1))
        ax3.plot(xs, ys_pos, 'o-', color=color, linewidth=1.5,
                 markersize=3, alpha=alpha)

    reach = sum(p['a'] for p in robot.dh_params) or 2.0
    ax3.set_xlim(-reach * 1.3, reach * 1.3)
    ax3.set_ylim(-reach * 1.3, reach * 1.3)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Workspace: {n_show} arm poses from largest boxes')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    p3 = output_dir / "workspace.png"
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    logger.info("  保存: %s", p3)

    return [str(p1), str(p2), str(p3)], coverage, coverage_raw, free_ratio


# ─────────────────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2DOF Box Tree 可视化实验")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-obs", type=int, default=4,
                        help="障碍物数量")
    parser.add_argument("--max-boxes", type=int, default=300,
                        help="最大 box 数")
    parser.add_argument("--max-iters", type=int, default=3000,
                        help="最大采样迭代")
    parser.add_argument("--expansion-res", type=float, default=0.02,
                        help="box 拓展二分精度 (rad)")
    parser.add_argument("--max-rounds", type=int, default=4,
                        help="box 拓展迭代轮数")
    parser.add_argument("--resolution", type=float, default=0.03,
                        help="C-space 碰撞地图扫描精度 (rad)")
    parser.add_argument("--boundary-batch", type=int, default=8,
                        help="边界再采样 batch size")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("examples/output") / f"box_tree_viz_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  2DOF Box Tree 可视化实验")
    logger.info("  seed=%d  n_obs=%d  max_boxes=%d  max_iters=%d",
                args.seed, args.n_obs, args.max_boxes, args.max_iters)
    logger.info("  expansion_res=%.3f  max_rounds=%d  resolution=%.3f",
                args.expansion_res, args.max_rounds, args.resolution)
    logger.info("  output: %s", output_dir)
    logger.info("=" * 60)

    # 1. 加载机器人
    robot = load_robot("2dof_planar")
    joint_limits = list(robot.joint_limits)
    logger.info("Robot: %s, %dDOF, limits=%s",
                robot.name, robot.n_joints,
                [(f"{lo:.2f}", f"{hi:.2f}") for lo, hi in joint_limits])

    # 2. 随机场景
    rng = np.random.default_rng(args.seed)
    scene = random_scene_2d(robot, args.n_obs, rng)
    logger.info("Scene: %d obstacles", scene.n_obstacles)
    for obs in scene.get_obstacles():
        logger.info("  [%s] min=(%.3f,%.3f) max=(%.3f,%.3f) size=%.3f x %.3f",
                     obs.name,
                     obs.min_point[0], obs.min_point[1],
                     obs.max_point[0], obs.max_point[1],
                     obs.size[0], obs.size[1])

    # 保存 scene.json
    scene.to_json(str(output_dir / "scene.json"))

    # 3. 生成 box forest
    logger.info("")
    logger.info("▶ 生成 box forest ...")
    manager, n_checks = grow_box_forest(
        robot=robot,
        scene=scene,
        joint_limits=joint_limits,
        max_boxes=args.max_boxes,
        max_iters=args.max_iters,
        expansion_resolution=args.expansion_res,
        max_rounds=args.max_rounds,
        seed_batch=args.boundary_batch,
        rng_seed=args.seed,
    )

    # 4. 可视化
    logger.info("")
    logger.info("▶ 生成可视化 ...")
    saved_files, coverage, coverage_raw, free_ratio = visualize(
        robot=robot,
        scene=scene,
        manager=manager,
        joint_limits=joint_limits,
        output_dir=output_dir,
        resolution=args.resolution,
    )

    # 5. 统计摘要
    all_boxes = manager.get_all_boxes()
    volumes = [b.volume for b in all_boxes]
    widths_0 = [b.widths[0] for b in all_boxes]
    widths_1 = [b.widths[1] for b in all_boxes]

    stats_lines = [
        "=" * 50,
        "  Box Tree 统计摘要",
        "=" * 50,
        f"  树数量       : {manager.n_trees}",
        f"  Box 总数     : {manager.total_nodes}",
        f"  总体积       : {manager.get_total_volume():.4f}",
        f"  C-space 覆盖 : {coverage:.1f}%  (去重 union)",
        f"  C-space 覆盖(raw): {coverage_raw:.1f}%  (未去重，含 box 重叠)",
        f"  自由空间比例 : {free_ratio*100:.1f}%  (grid 扫描)",
        f"  碰撞检测次数 : {n_checks}",
        "",
        f"  Box 体积  — mean={np.mean(volumes):.4f}  "
        f"median={np.median(volumes):.4f}  "
        f"max={np.max(volumes):.4f}  min={np.min(volumes):.6f}",
        f"  q0 宽度   — mean={np.mean(widths_0):.3f}  "
        f"max={np.max(widths_0):.3f}  min={np.min(widths_0):.4f}",
        f"  q1 宽度   — mean={np.mean(widths_1):.3f}  "
        f"max={np.max(widths_1):.3f}  min={np.min(widths_1):.4f}",
        "",
        "  前 10 大 Box:",
    ]
    all_boxes.sort(key=lambda b: b.volume, reverse=True)
    for i, b in enumerate(all_boxes[:10]):
        stats_lines.append(
            f"    #{i}: tree={b.tree_id} id={b.node_id} "
            f"vol={b.volume:.4f} widths=[{b.widths[0]:.3f}, {b.widths[1]:.3f}]"
        )
    stats_lines.append("=" * 50)

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    stats_path = output_dir / "stats.txt"
    stats_path.write_text(stats_text, encoding="utf-8")

    logger.info("")
    logger.info("完成！输出目录: %s", output_dir)
    for f in saved_files:
        logger.info("  %s", f)


if __name__ == "__main__":
    main()
