#!/usr/bin/env python
"""
examples/visualize_box_forest.py - BoxForest 增量拓展过程可视化

在 2DOF 随机障碍物场景中，逐步采样 → 拓展 → 去重叠 → 邻接更新，
每一步保存一帧快照，最终合成 GIF/MP4 动画，直观展示 Box Forest
从零开始生长的全过程。

**不做路径规划**——纯粹展示 box 拓展质量与覆盖效果。

输出（保存到 examples/output/forest_viz_<ts>/）：
  - frames/step_NNN.png : 每步快照
  - forest_growth.gif   : 动画
  - forest_growth.mp4   : 动画 (MP4)
  - final_forest.png    : 最终 BoxForest（邻接度着色）
  - collision_map.png   : 纯 C-space 碰撞地图
  - overlay.png         : BoxForest 叠加碰撞地图
  - workspace.png       : 工作空间 arm 姿态
  - stats.txt           : 统计摘要

用法：
    python -m examples.visualize_box_forest
    python -m examples.visualize_box_forest --seed 42 --n-obs 5 --max-boxes 150
    python -m examples.visualize_box_forest --step-interval 5 --fps 4
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

import numpy as np

from box_aabb.robot import Robot, load_robot
from planner.aabb_cache import AABBCache
from planner.models import BoxNode, PlannerConfig
from planner.obstacles import Scene
from planner.collision import CollisionChecker
from planner.box_expansion import BoxExpander
from planner.box_forest import BoxForest
from planner.deoverlap import shared_face_center, compute_adjacency
from planner.hier_aabb_tree import HierAABBTree

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("forest_viz")


# ─────────────────────────────────────────────────────────
#  随机场景生成（2D）
# ─────────────────────────────────────────────────────────

def random_scene_2d(robot: Robot, n_obs: int, rng: np.random.Generator) -> Scene:
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
#  帧渲染
# ─────────────────────────────────────────────────────────

def render_frame(
    ax,
    forest: BoxForest,
    joint_limits: List[Tuple[float, float]],
    collision_map: np.ndarray,
    step: int,
    new_seed: Optional[np.ndarray] = None,
    new_box_ids: Optional[List[int]] = None,
    cmap_boxes=None,
):
    """在 ax 上绘制当前 BoxForest 状态的一帧"""
    ax.clear()

    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]

    # 碰撞地图底图
    ax.imshow(
        collision_map, origin='lower',
        extent=[lo_x, hi_x, lo_y, hi_y],
        cmap='Reds', alpha=0.30, aspect='auto',
    )

    from matplotlib.patches import Rectangle

    boxes = forest.boxes
    adjacency = forest.adjacency
    degrees = {bid: len(adjacency.get(bid, set())) for bid in boxes}
    max_deg = max(degrees.values()) if degrees else 1

    if cmap_boxes is None:
        import matplotlib.cm as cm
        cmap_boxes = cm.viridis

    new_set = set(new_box_ids) if new_box_ids else set()

    # 所有 box
    for bid, box in boxes.items():
        lo0 = box.joint_intervals[0][0]
        hi0 = box.joint_intervals[0][1]
        lo1 = box.joint_intervals[1][0]
        hi1 = box.joint_intervals[1][1]

        deg = degrees.get(bid, 0)
        color = cmap_boxes(deg / max(max_deg, 1))
        alpha = 0.50 if bid in new_set else 0.25
        lw = 1.2 if bid in new_set else 0.5

        rect = Rectangle(
            (lo0, lo1), hi0 - lo0, hi1 - lo1,
            linewidth=lw, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)

    # 邻接边
    seen = set()
    for bid, neighbors in adjacency.items():
        if bid not in boxes:
            continue
        for nb in neighbors:
            if nb not in boxes:
                continue
            key = (min(bid, nb), max(bid, nb))
            if key in seen:
                continue
            seen.add(key)
            wp = shared_face_center(boxes[bid], boxes[nb])
            if wp is None:
                continue
            ca = np.array(boxes[bid].center)
            cb = np.array(boxes[nb].center)
            ax.plot(
                [ca[0], wp[0], cb[0]],
                [ca[1], wp[1], cb[1]],
                'k--', linewidth=0.3, alpha=0.20,
            )

    # 当前 seed 点高亮
    if new_seed is not None:
        ax.plot(new_seed[0], new_seed[1], 'r*', markersize=12,
                zorder=10, label='seed')

    n_adj = sum(len(v) for v in adjacency.values()) // 2
    ax.set_xlim(lo_x - 0.1, hi_x + 0.1)
    ax.set_ylim(lo_y - 0.1, hi_y + 0.1)
    ax.set_xlabel('q0 (rad)')
    ax.set_ylabel('q1 (rad)')
    ax.set_title(
        f'Step {step}  |  {forest.n_boxes} boxes  |  '
        f'{n_adj} adj edges  |  vol={forest.total_volume:.3f}',
        fontsize=11,
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)


# ─────────────────────────────────────────────────────────
#  碰撞地图扫描
# ─────────────────────────────────────────────────────────

def scan_collision_map(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    resolution: float = 0.03,
) -> np.ndarray:
    checker = CollisionChecker(robot, scene)
    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]
    xs = np.arange(lo_x, hi_x, resolution)
    ys = np.arange(lo_y, hi_y, resolution)
    n_rows, n_cols = len(ys), len(xs)
    cmap = np.zeros((n_rows, n_cols), dtype=np.float32)
    t0 = time.time()
    for i, y in enumerate(ys):
        row = np.column_stack([xs, np.full(n_cols, y)])
        cmap[i, :] = checker.check_config_collision_batch(row).astype(np.float32)
        if (i + 1) % 50 == 0 or i == n_rows - 1:
            logger.info("  碰撞地图扫描: %d/%d (%.1f%%) %.1fs",
                        i + 1, n_rows, (i + 1) / n_rows * 100,
                        time.time() - t0)
    return cmap


# ─────────────────────────────────────────────────────────
#  核心：逐步拓展 BoxForest + 录制帧
# ─────────────────────────────────────────────────────────

def grow_forest_animated(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    collision_map: np.ndarray,
    output_dir: Path,
    max_boxes: int = 150,
    max_seeds: int = 1500,
    expansion_resolution: float = 0.02,
    max_rounds: int = 3,
    step_interval: int = 3,
    farthest_k: int = 12,
    rng_seed: int = 42,
    fps: int = 4,
):
    """逐步拓展 BoxForest，每隔 step_interval 步保存帧"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    rng = np.random.default_rng(rng_seed)

    aabb_cache = AABBCache.auto_load(robot)
    checker = CollisionChecker(robot, scene, aabb_cache=aabb_cache)

    config = PlannerConfig(
        expansion_resolution=expansion_resolution,
        max_expansion_rounds=max_rounds,
        hard_overlap_reject=True,
        verbose=False,
    )

    # 自动采样模式
    use_sampling = robot.n_joints > 4
    expander = BoxExpander(
        robot=robot,
        collision_checker=checker,
        joint_limits=joint_limits,
        expansion_resolution=config.expansion_resolution,
        max_rounds=config.max_expansion_rounds,
        hard_overlap_reject=config.hard_overlap_reject,
    )

    forest = BoxForest(robot.fingerprint(), joint_limits, config)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmap_boxes = cm.viridis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    frame_paths: List[str] = []
    step = 0
    n_farthest_fails = 0
    t0 = time.time()

    # ── 初始帧（空 forest）──
    render_frame(ax, forest, joint_limits, collision_map, step,
                 cmap_boxes=cmap_boxes)
    p = str(frames_dir / f"step_{step:04d}.png")
    fig.savefig(p, dpi=100, bbox_inches='tight')
    frame_paths.append(p)

    for seed_iter in range(max_seeds):
        if forest.n_boxes >= max_boxes:
            logger.info("达到 max_boxes=%d，停止", max_boxes)
            break

        # 最远点采样
        candidates = []
        for _ in range(farthest_k):
            q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
            if checker.check_config_collision(q):
                continue
            if forest.find_containing(q) is not None:
                continue
            nearest = forest.find_nearest(q)
            dist = nearest.distance_to_config(q) if nearest else float('inf')
            candidates.append((q, dist))

        if not candidates:
            n_farthest_fails += 1
            if n_farthest_fails > 80:
                logger.info("连续 80 次无候选，C-free 已基本覆盖 (iter=%d)", seed_iter)
                break
            continue
        n_farthest_fails = 0

        q_seed, _ = max(candidates, key=lambda x: x[1])

        # 拓展
        nid = forest.allocate_id()
        box = expander.expand(
            q_seed, node_id=nid, rng=rng,
            existing_boxes=list(forest.boxes.values()),
        )
        if box is None or box.volume < 1e-6:
            continue

        # 增量添加 → deoverlap + 邻接更新
        added = forest.add_boxes_incremental([box])
        step += 1

        if step % step_interval == 0 or forest.n_boxes >= max_boxes:
            new_ids = [b.node_id for b in added]
            render_frame(ax, forest, joint_limits, collision_map, step,
                         new_seed=q_seed, new_box_ids=new_ids,
                         cmap_boxes=cmap_boxes)
            p = str(frames_dir / f"step_{step:04d}.png")
            fig.savefig(p, dpi=100, bbox_inches='tight')
            frame_paths.append(p)

        if step % 20 == 0:
            n_adj = sum(len(v) for v in forest.adjacency.values()) // 2
            logger.info(
                "step %4d | boxes=%3d | adj=%d | vol=%.3f | %.1fs",
                step, forest.n_boxes, n_adj, forest.total_volume,
                time.time() - t0,
            )

    plt.close(fig)

    dt = time.time() - t0
    n_adj = sum(len(v) for v in forest.adjacency.values()) // 2
    logger.info(
        "拓展完成: %d 步, %d boxes, %d adj edges, vol=%.4f, %.2fs",
        step, forest.n_boxes, n_adj, forest.total_volume, dt,
    )

    # ── 合成 GIF ──
    logger.info("合成 GIF (%d 帧, fps=%d) ...", len(frame_paths), fps)
    try:
        from PIL import Image
        images = [Image.open(fp) for fp in frame_paths]
        # 最后一帧驻留更久 (repeat 4x)
        for _ in range(3):
            images.append(images[-1])
        gif_path = str(output_dir / "forest_growth.gif")
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=int(1000 / fps), loop=0,
        )
        logger.info("  GIF 已保存: %s", gif_path)
    except Exception as e:
        logger.warning("  GIF 生成失败: %s", e)
        gif_path = None

    # ── 合成 MP4 ──
    try:
        import imageio
        mp4_path = str(output_dir / "forest_growth.mp4")
        writer = imageio.get_writer(mp4_path, fps=fps)
        for fp in frame_paths:
            writer.append_data(imageio.imread(fp))
        # 尾帧驻留
        last_frame = imageio.imread(frame_paths[-1])
        for _ in range(fps * 2):
            writer.append_data(last_frame)
        writer.close()
        logger.info("  MP4 已保存: %s", mp4_path)
    except Exception as e:
        logger.warning("  MP4 生成失败: %s", e)

    # 保存 AABB 缓存
    try:
        aabb_cache.auto_save(robot)
    except Exception:
        pass

    return forest, step, dt, frame_paths


# ─────────────────────────────────────────────────────────
#  核心（HierAABBTree 模式）：层级切分拓展 + 录制帧
# ─────────────────────────────────────────────────────────

def grow_forest_hier_animated(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    collision_map: np.ndarray,
    output_dir: Path,
    max_boxes: int = 150,
    max_seeds: int = 2000,
    boundary_batch: int = 6,
    max_tree_stalls: int = 15,
    step_interval: int = 3,
    farthest_k: int = 12,
    rng_seed: int = 42,
    fps: int = 4,
    max_depth: int = 40,
    min_edge_length: float = 0.01,
    early_stop_window: int = 30,
    early_stop_min_vol: float = 1e-4,
):
    """用 HierAABBTree 逐步拓展 BoxForest

    改进策略：
    1. 使用 HierAABBTree 做 box 拓展（自顶向下切分，缓存精化）
    2. DFS 边界优先：先从上一个 box 边缘采样继续拓展
    3. 当边界采样连续失败时，切换到远处新 seed
    4. 全局采样连续失败时停止
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    rng = np.random.default_rng(rng_seed)
    checker = CollisionChecker(robot, scene)
    obstacles = scene.get_obstacles()

    hier_tree = HierAABBTree.auto_load(robot, joint_limits)
    config = PlannerConfig(hard_overlap_reject=True, verbose=False)
    forest = BoxForest(robot.fingerprint(), joint_limits, config)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmap_boxes = cm.viridis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    frame_paths: List[str] = []
    step = 0
    t0 = time.time()

    # ── 分环节计时器 ──
    t_collision_check = 0.0   # seed 碰撞检测
    t_find_free_box = 0.0     # HierAABBTree.find_free_box
    t_deoverlap = 0.0         # forest.add_boxes_incremental (含 deoverlap)
    t_sampling = 0.0          # 采样 (边界/远处)
    t_render = 0.0            # 渲染帧
    n_seed_collision = 0      # seed 碰撞拒绝次数
    n_seed_inside = 0         # seed 已在 box 内拒绝
    n_find_none = 0           # find_free_box 返回 None
    n_find_tiny = 0           # find_free_box 返回体积太小
    n_boundary_attempts = 0   # 边界采样尝试总数
    n_farthest_attempts = 0   # 远处采样尝试总数
    expansion_log: List[dict] = []  # 每步详细记录
    recent_vols: List[float] = []   # 滑动窗口体积记录

    # 初始帧
    render_frame(ax, forest, joint_limits, collision_map, step,
                 cmap_boxes=cmap_boxes)
    p = str(frames_dir / f"step_{step:04d}.png")
    fig.savefig(p, dpi=100, bbox_inches='tight')
    frame_paths.append(p)

    # ── 辅助函数 ──
    def _sample_boundary(box_intervals, n, rng_):
        """从 box 各面上均匀采样 seed"""
        seeds = []
        ndim = len(box_intervals)
        for _ in range(n):
            dim = rng_.integers(0, ndim)
            side = rng_.integers(0, 2)  # 0=低面, 1=高面
            q = np.array([
                rng_.uniform(lo, hi) for lo, hi in box_intervals
            ])
            q[dim] = box_intervals[dim][side]
            # 稍微偏移到 box 外侧
            offset = 0.01 if side == 1 else -0.01
            q[dim] = np.clip(
                q[dim] + offset,
                joint_limits[dim][0], joint_limits[dim][1],
            )
            seeds.append(q)
        return seeds

    def _try_add_box(seed_q, source=""):
        """尝试用 HierAABBTree 拓展一个 box 并加入 forest"""
        nonlocal t_collision_check, t_find_free_box, t_deoverlap
        nonlocal n_seed_collision, n_seed_inside, n_find_none, n_find_tiny

        tc0 = time.time()
        in_collision = checker.check_config_collision(seed_q)
        t_collision_check += time.time() - tc0
        if in_collision:
            n_seed_collision += 1
            return None

        tc0 = time.time()
        containing = forest.find_containing(seed_q)
        t_collision_check += time.time() - tc0
        if containing is not None:
            n_seed_inside += 1
            return None

        tf0 = time.time()
        ivs = hier_tree.find_free_box(
            seed_q, obstacles, max_depth=max_depth,
            min_edge_length=min_edge_length)
        t_find_free_box += time.time() - tf0
        if ivs is None:
            n_find_none += 1
            return None
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        if vol < 1e-6:
            n_find_tiny += 1
            return None

        nid = forest.allocate_id()
        box = BoxNode(
            node_id=nid,
            joint_intervals=ivs,
            seed_config=seed_q.copy(),
            volume=vol,
        )
        td0 = time.time()
        added = forest.add_boxes_incremental([box])
        t_deoverlap += time.time() - td0

        if added:
            widths = [hi - lo for lo, hi in ivs]
            expansion_log.append({
                'step': step + 1,
                'id': nid,
                'vol': vol,
                'widths': widths,
                'source': source,
                'depth': hier_tree.get_stats()['max_depth'] if vol < 0.001 else 0,
                'n_boxes': forest.n_boxes,
                'elapsed': time.time() - t0,
            })
            recent_vols.append(vol)
        return added

    # ── 主循环 ──
    global_stalls = 0
    last_box_ivs = None  # 上一个成功 box，用于 DFS 边界采样
    stop_reason = "max_seeds"

    for seed_iter in range(max_seeds):
        if forest.n_boxes >= max_boxes:
            stop_reason = f"达到 max_boxes={max_boxes}"
            logger.info("%s，停止", stop_reason)
            break
        if global_stalls > max_tree_stalls * 3:
            stop_reason = f"全局连续 {global_stalls} 次无新 box"
            logger.info("%s，停止", stop_reason)
            break

        # ── 早停：最近 N 个 box 体积都很小 ──
        if (early_stop_window > 0
                and len(recent_vols) >= early_stop_window
                and all(v < early_stop_min_vol
                        for v in recent_vols[-early_stop_window:])):
            stop_reason = (f"早停: 最近 {early_stop_window} 个 box "
                           f"体积均 < {early_stop_min_vol:.1e}")
            logger.info("%s，停止", stop_reason)
            break

        added = None

        # ── DFS 模式：从上一个 box 边界采样 ──
        if last_box_ivs is not None:
            ts0 = time.time()
            boundary_seeds = _sample_boundary(last_box_ivs, boundary_batch, rng)
            t_sampling += time.time() - ts0
            n_boundary_attempts += len(boundary_seeds)
            for bs in boundary_seeds:
                if forest.n_boxes >= max_boxes:
                    break
                added = _try_add_box(bs, source="boundary")
                if added:
                    step += 1
                    last_box_ivs = list(added[-1].joint_intervals)
                    global_stalls = 0

                    if step % step_interval == 0 or forest.n_boxes >= max_boxes:
                        tr0 = time.time()
                        new_ids = [b.node_id for b in added]
                        render_frame(ax, forest, joint_limits, collision_map,
                                     step, new_seed=bs, new_box_ids=new_ids,
                                     cmap_boxes=cmap_boxes)
                        p = str(frames_dir / f"step_{step:04d}.png")
                        fig.savefig(p, dpi=100, bbox_inches='tight')
                        frame_paths.append(p)
                        t_render += time.time() - tr0
                    break  # DFS：成功后立即用新 box 继续
            else:
                # 边界采样全部失败 → 切换到远处新 seed
                last_box_ivs = None

        # ── 远处新 seed（最远点采样） ──
        if added is None:
            ts0 = time.time()
            candidates = []
            for _ in range(farthest_k):
                q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])
                if checker.check_config_collision(q):
                    continue
                if forest.find_containing(q) is not None:
                    continue
                nearest = forest.find_nearest(q)
                dist = nearest.distance_to_config(q) if nearest else float('inf')
                candidates.append((q, dist))
            t_sampling += time.time() - ts0
            n_farthest_attempts += 1

            if not candidates:
                global_stalls += 1
                continue

            q_seed, _ = max(candidates, key=lambda x: x[1])
            added = _try_add_box(q_seed, source="farthest")

            if added:
                step += 1
                last_box_ivs = list(added[-1].joint_intervals)
                global_stalls = 0

                if step % step_interval == 0 or forest.n_boxes >= max_boxes:
                    tr0 = time.time()
                    new_ids = [b.node_id for b in added]
                    render_frame(ax, forest, joint_limits, collision_map,
                                 step, new_seed=q_seed, new_box_ids=new_ids,
                                 cmap_boxes=cmap_boxes)
                    p = str(frames_dir / f"step_{step:04d}.png")
                    fig.savefig(p, dpi=100, bbox_inches='tight')
                    frame_paths.append(p)
                    t_render += time.time() - tr0
            else:
                global_stalls += 1

        if step % 20 == 0 and step > 0:
            n_adj = sum(len(v) for v in forest.adjacency.values()) // 2
            ht_stats = hier_tree.get_stats()
            logger.info(
                "step %4d | boxes=%3d | adj=%d | vol=%.3f | "
                "tree_nodes=%d fk=%d | %.1fs",
                step, forest.n_boxes, n_adj, forest.total_volume,
                ht_stats['n_nodes'], ht_stats['n_fk_calls'],
                time.time() - t0,
            )

    plt.close(fig)

    dt = time.time() - t0
    n_adj = sum(len(v) for v in forest.adjacency.values()) // 2
    ht_stats = hier_tree.get_stats()
    logger.info(
        "拓展完成: %d 步, %d boxes, %d adj edges, vol=%.4f, %.2fs",
        step, forest.n_boxes, n_adj, forest.total_volume, dt,
    )
    logger.info(
        "HierAABBTree: %d nodes, max_depth=%d, %d FK calls",
        ht_stats['n_nodes'], ht_stats['max_depth'], ht_stats['n_fk_calls'],
    )
    logger.info("停止原因: %s", stop_reason)

    # 保存到全局缓存 + 本次输出目录
    global_path = hier_tree.auto_save()
    local_path = str(output_dir / "hier_cache.pkl")
    hier_tree.save(local_path)
    logger.info("全局缓存: %s", global_path)

    # 合成动画（复用逻辑）
    _save_animations(frame_paths, output_dir, fps)

    # 构建 timing/profiling 报告
    timing_report = {
        'total': dt,
        'find_free_box': t_find_free_box,
        'collision_check': t_collision_check,
        'deoverlap': t_deoverlap,
        'sampling': t_sampling,
        'render': t_render,
        'other': max(0, dt - t_find_free_box - t_collision_check
                     - t_deoverlap - t_sampling - t_render),
        'n_seed_collision': n_seed_collision,
        'n_seed_inside': n_seed_inside,
        'n_find_none': n_find_none,
        'n_find_tiny': n_find_tiny,
        'n_boundary_attempts': n_boundary_attempts,
        'n_farthest_attempts': n_farthest_attempts,
        'stop_reason': stop_reason,
    }

    return forest, step, dt, frame_paths, hier_tree, timing_report, expansion_log


def _save_animations(frame_paths: List[str], output_dir: Path, fps: int):
    """GIF + MP4 合成"""
    logger.info("合成 GIF (%d 帧, fps=%d) ...", len(frame_paths), fps)
    try:
        from PIL import Image
        images = [Image.open(fp) for fp in frame_paths]
        for _ in range(3):
            images.append(images[-1])
        gif_path = str(output_dir / "forest_growth.gif")
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=int(1000 / fps), loop=0,
        )
        logger.info("  GIF 已保存: %s", gif_path)
    except Exception as e:
        logger.warning("  GIF 生成失败: %s", e)

    try:
        import imageio
        mp4_path = str(output_dir / "forest_growth.mp4")
        writer = imageio.get_writer(mp4_path, fps=fps)
        for fp in frame_paths:
            writer.append_data(imageio.imread(fp))
        last_frame = imageio.imread(frame_paths[-1])
        for _ in range(fps * 2):
            writer.append_data(last_frame)
        writer.close()
        logger.info("  MP4 已保存: %s", mp4_path)
    except Exception as e:
        logger.warning("  MP4 生成失败: %s", e)


# ─────────────────────────────────────────────────────────
#  最终静态图
# ─────────────────────────────────────────────────────────

def save_final_figures(
    robot: Robot,
    scene: Scene,
    forest: BoxForest,
    joint_limits: List[Tuple[float, float]],
    collision_map: np.ndarray,
    output_dir: Path,
    resolution: float = 0.03,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm

    saved: List[str] = []
    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]

    # ── 1. 纯碰撞地图 ──
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    n_free = int(np.sum(collision_map == 0))
    free_ratio = n_free / collision_map.size
    ax1.imshow(collision_map, origin='lower',
               extent=[lo_x, hi_x, lo_y, hi_y],
               cmap='Reds', alpha=0.5, aspect='auto')
    ax1.set_xlabel('q0 (rad)')
    ax1.set_ylabel('q1 (rad)')
    ax1.set_title(f'C-Space Collision Map  (free={free_ratio*100:.1f}%)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    p1 = str(output_dir / "collision_map.png")
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    saved.append(p1)
    logger.info("  已保存: %s", p1)

    # ── 2. 最终 BoxForest（邻接度着色 + 邻接边）──
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(collision_map, origin='lower',
               extent=[lo_x, hi_x, lo_y, hi_y],
               cmap='Reds', alpha=0.25, aspect='auto')

    boxes = forest.boxes
    adjacency = forest.adjacency
    degrees = {bid: len(adjacency.get(bid, set())) for bid in boxes}
    max_deg = max(degrees.values()) if degrees else 1
    cmap_v = cm.viridis

    for bid, box in boxes.items():
        lo0 = box.joint_intervals[0][0]
        hi0 = box.joint_intervals[0][1]
        lo1 = box.joint_intervals[1][0]
        hi1 = box.joint_intervals[1][1]
        deg = degrees.get(bid, 0)
        color = cmap_v(deg / max(max_deg, 1))
        rect = Rectangle(
            (lo0, lo1), hi0 - lo0, hi1 - lo1,
            linewidth=0.6, edgecolor=color, facecolor=color, alpha=0.30,
        )
        ax2.add_patch(rect)

    # 邻接边
    seen = set()
    for bid, neighbors in adjacency.items():
        if bid not in boxes:
            continue
        for nb in neighbors:
            if nb not in boxes:
                continue
            key = (min(bid, nb), max(bid, nb))
            if key in seen:
                continue
            seen.add(key)
            wp = shared_face_center(boxes[bid], boxes[nb])
            if wp is None:
                continue
            ca = np.array(boxes[bid].center)
            cb = np.array(boxes[nb].center)
            ax2.plot(
                [ca[0], wp[0], cb[0]],
                [ca[1], wp[1], cb[1]],
                'k--', linewidth=0.4, alpha=0.3,
            )
            ax2.plot(wp[0], wp[1], 's', color='orange',
                     markersize=2.5, alpha=0.7, zorder=4)

    n_adj = sum(len(v) for v in adjacency.values()) // 2
    ax2.set_xlim(lo_x - 0.1, hi_x + 0.1)
    ax2.set_ylim(lo_y - 0.1, hi_y + 0.1)
    ax2.set_xlabel('q0 (rad)')
    ax2.set_ylabel('q1 (rad)')
    ax2.set_title(
        f'BoxForest — {forest.n_boxes} boxes, {n_adj} adj edges, '
        f'vol={forest.total_volume:.3f}')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    sm = plt.cm.ScalarMappable(
        cmap=cmap_v, norm=plt.Normalize(vmin=0, vmax=max_deg))
    sm.set_array([])
    fig2.colorbar(sm, ax=ax2, shrink=0.6, label='邻接度 (adjacency degree)')
    p2 = str(output_dir / "final_forest.png")
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    saved.append(p2)
    logger.info("  已保存: %s", p2)

    # ── 3. 覆盖率叠加图 ──
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.imshow(collision_map, origin='lower',
               extent=[lo_x, hi_x, lo_y, hi_y],
               cmap='Reds', alpha=0.35, aspect='auto')

    # 按体积着色
    vols = [b.volume for b in boxes.values()]
    max_vol = max(vols) if vols else 1.0
    cmap_vol = cm.plasma

    # 栅格覆盖率
    xs = np.arange(lo_x, hi_x, resolution)
    ys = np.arange(lo_y, hi_y, resolution)
    box_mask = np.zeros_like(collision_map, dtype=bool)

    for bid, box in boxes.items():
        lo0, hi0 = box.joint_intervals[0]
        lo1, hi1 = box.joint_intervals[1]
        w, h = hi0 - lo0, hi1 - lo1
        color = cmap_vol(box.volume / max_vol)
        rect = Rectangle(
            (lo0, lo1), w, h,
            linewidth=0.4, edgecolor=color, facecolor=color, alpha=0.35,
        )
        ax3.add_patch(rect)
        j0 = max(int((lo0 - lo_x) / resolution), 0)
        j1 = min(int(np.ceil((hi0 - lo_x) / resolution)), len(xs))
        i0 = max(int((lo1 - lo_y) / resolution), 0)
        i1 = min(int(np.ceil((hi1 - lo_y) / resolution)), len(ys))
        box_mask[i0:i1, j0:j1] = True

    cspace_area = (hi_x - lo_x) * (hi_y - lo_y)
    coverage = int(np.sum(box_mask)) / collision_map.size * 100
    free_coverage = (int(np.sum(box_mask & (collision_map == 0)))
                     / max(n_free, 1) * 100)

    ax3.set_xlim(lo_x - 0.1, hi_x + 0.1)
    ax3.set_ylim(lo_y - 0.1, hi_y + 0.1)
    ax3.set_xlabel('q0 (rad)')
    ax3.set_ylabel('q1 (rad)')
    ax3.set_title(
        f'BoxForest Overlay — C-space cover={coverage:.1f}%, '
        f'free cover={free_coverage:.1f}%')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2)
    sm3 = plt.cm.ScalarMappable(
        cmap=cmap_vol, norm=plt.Normalize(vmin=0, vmax=max_vol))
    sm3.set_array([])
    fig3.colorbar(sm3, ax=ax3, shrink=0.6, label='Box体积')
    p3 = str(output_dir / "overlay.png")
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    saved.append(p3)
    logger.info("  已保存: %s", p3)

    # ── 4. 工作空间 ──
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    for obs in scene.get_obstacles():
        rect = Rectangle(
            (obs.min_point[0], obs.min_point[1]),
            obs.size[0], obs.size[1],
            linewidth=1.2, edgecolor='red', facecolor='red', alpha=0.4)
        ax4.add_patch(rect)

    all_boxes = sorted(boxes.values(), key=lambda b: b.volume, reverse=True)
    n_show = min(25, len(all_boxes))
    for idx in range(n_show):
        q = all_boxes[idx].center
        positions = robot.get_link_positions(q)
        xs_p = [p[0] for p in positions]
        ys_p = [p[1] for p in positions]
        alpha = 0.2 + 0.6 * (1 - idx / max(n_show - 1, 1))
        color = cm.viridis(idx / max(n_show - 1, 1))
        ax4.plot(xs_p, ys_p, 'o-', color=color, linewidth=1.5,
                 markersize=3, alpha=alpha)

    reach = sum(p['a'] for p in robot.dh_params) or 2.0
    ax4.set_xlim(-reach * 1.3, reach * 1.3)
    ax4.set_ylim(-reach * 1.3, reach * 1.3)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title(f'Workspace: {n_show} arm poses from largest boxes')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    p4 = str(output_dir / "workspace.png")
    fig4.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    saved.append(p4)
    logger.info("  已保存: %s", p4)

    return saved, coverage, free_coverage, free_ratio


# ─────────────────────────────────────────────────────────
#  统计摘要
# ─────────────────────────────────────────────────────────

def write_stats(
    forest: BoxForest,
    n_steps: int, dt: float,
    coverage: float, free_coverage: float, free_ratio: float,
    output_dir: Path,
    args,
    hier_tree: Optional['HierAABBTree'] = None,
    timing_report: Optional[dict] = None,
    expansion_log: Optional[List[dict]] = None,
) -> str:
    boxes_list = sorted(forest.boxes.values(), key=lambda b: b.volume, reverse=True)
    vols = [b.volume for b in boxes_list]
    adjacency = forest.adjacency
    degrees = [len(adjacency.get(bid, set())) for bid in forest.boxes]
    n_adj = sum(len(v) for v in adjacency.values()) // 2
    n_isolated = sum(1 for d in degrees if d == 0)

    lines = [
        "=" * 55,
        "  BoxForest 拓展统计",
        "=" * 55,
        f"  Box 总数         : {forest.n_boxes}",
        f"  总体积           : {forest.total_volume:.4f}",
        f"  邻接边数         : {n_adj}",
        f"  拓展步数         : {n_steps}",
        f"  耗时             : {dt:.2f}s",
        "",
        f"  C-space 覆盖率   : {coverage:.1f}%  (去重 union)",
        f"  自由空间覆盖率   : {free_coverage:.1f}%  (仅 C-free 区域)",
        f"  自由空间比例     : {free_ratio*100:.1f}%",
        "",
        f"  Box 体积 — mean={np.mean(vols):.4f}  "
        f"median={np.median(vols):.4f}  "
        f"max={np.max(vols):.4f}  min={np.min(vols):.6f}",
        f"  邻接度   — mean={np.mean(degrees):.2f}  "
        f"max={max(degrees) if degrees else 0}  "
        f"isolated={n_isolated}",
        "",
        "  前 15 大 Box:",
    ]
    for i, b in enumerate(boxes_list[:15]):
        deg = len(adjacency.get(b.node_id, set()))
        lines.append(
            f"    #{i}: id={b.node_id} vol={b.volume:.4f} "
            f"deg={deg} widths=[{b.widths[0]:.3f}, {b.widths[1]:.3f}]"
        )
    lines.append("")
    lines.append(f"  参数: seed={args.seed} n_obs={args.n_obs} "
                 f"max_boxes={args.max_boxes} max_seeds={args.max_seeds} "
                 f"mode={getattr(args, 'mode', 'legacy')}")
    if getattr(args, 'mode', 'legacy') == 'hier':
        lines.append(f"  max_depth={args.max_depth} boundary_batch={args.boundary_batch}")
    else:
        lines.append(f"  expansion_res={args.expansion_res} max_rounds={args.max_rounds}")

    # HierAABBTree 统计
    if hier_tree is not None:
        ht_stats = hier_tree.get_stats()
        lines.append("")
        lines.append("  HierAABBTree 缓存统计:")
        lines.append(f"    节点数       : {ht_stats['n_nodes']}")
        lines.append(f"    叶节点数     : {ht_stats['n_leaves']}")
        lines.append(f"    最大深度       : {ht_stats['max_depth']}")
        lines.append(f"    平均叶深度     : {ht_stats['avg_depth']:.1f}")
        lines.append(f"    interval FK 调用: {ht_stats['n_fk_calls']}")

    # 分环节计时报告
    if timing_report is not None:
        tr = timing_report
        lines.append("")
        lines.append("-" * 55)
        lines.append("  分环节计时报告:")
        lines.append("-" * 55)
        lines.append(f"    总耗时             : {tr['total']:.2f}s")
        lines.append(f"    find_free_box    : {tr['find_free_box']:.2f}s  ({tr['find_free_box']/tr['total']*100:.0f}%)")
        lines.append(f"    collision_check  : {tr['collision_check']:.2f}s  ({tr['collision_check']/tr['total']*100:.0f}%)")
        lines.append(f"    deoverlap        : {tr['deoverlap']:.2f}s  ({tr['deoverlap']/tr['total']*100:.0f}%)")
        lines.append(f"    sampling         : {tr['sampling']:.2f}s  ({tr['sampling']/tr['total']*100:.0f}%)")
        lines.append(f"    render           : {tr['render']:.2f}s  ({tr['render']/tr['total']*100:.0f}%)")
        lines.append(f"    other            : {tr['other']:.2f}s  ({tr['other']/tr['total']*100:.0f}%)")
        lines.append("")
        lines.append("  采样统计:")
        lines.append(f"    边界采样尝试     : {tr['n_boundary_attempts']}")
        lines.append(f"    远处采样尝试     : {tr['n_farthest_attempts']}")
        lines.append(f"    seed 碰撞拒绝     : {tr['n_seed_collision']}")
        lines.append(f"    seed 已在box内   : {tr['n_seed_inside']}")
        lines.append(f"    find_free_box=None: {tr['n_find_none']}")
        lines.append(f"    find_free_box微小 : {tr['n_find_tiny']}")
        lines.append(f"    停止原因           : {tr['stop_reason']}")

    # 拓展过程详细记录
    if expansion_log:
        lines.append("")
        lines.append("-" * 55)
        lines.append("  拓展过程详细记录:")
        lines.append("-" * 55)
        lines.append(f"  {'step':>5s}  {'id':>5s}  {'vol':>10s}  "
                     f"{'widths':>20s}  {'source':>8s}  "
                     f"{'boxes':>5s}  {'elapsed':>7s}")
        for entry in expansion_log:
            w = entry['widths']
            w_str = '[' + ', '.join(f'{x:.4f}' for x in w) + ']'
            lines.append(
                f"  {entry['step']:5d}  {entry['id']:5d}  "
                f"{entry['vol']:10.6f}  {w_str:>20s}  "
                f"{entry['source']:>8s}  "
                f"{entry['n_boxes']:5d}  {entry['elapsed']:7.2f}s"
            )

    lines.append("=" * 55)

    text = "\n".join(lines)
    print(text)

    p = output_dir / "stats.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


# ─────────────────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BoxForest 增量拓展可视化（无路径规划）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-obs", type=int, default=5,
                        help="障碍物数量 (默认: 5)")
    parser.add_argument("--max-boxes", type=int, default=150,
                        help="最大 box 数 (默认: 150)")
    parser.add_argument("--max-seeds", type=int, default=2000,
                        help="最大采样迭代 (默认: 2000)")
    parser.add_argument("--expansion-res", type=float, default=0.02,
                        help="box 拓展二分精度 (rad)")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="box 拓展最大迭代轮数")
    parser.add_argument("--resolution", type=float, default=0.03,
                        help="碰撞地图扫描精度 (rad)")
    parser.add_argument("--step-interval", type=int, default=3,
                        help="每隔多少步保存一帧 (默认: 3)")
    parser.add_argument("--fps", type=int, default=4,
                        help="动画帧率 (默认: 4)")
    parser.add_argument("--farthest-k", type=int, default=12,
                        help="最远点采样候选数 (默认: 12)")
    parser.add_argument("--mode", type=str, default="hier",
                        choices=["hier", "legacy"],
                        help="拓展模式: hier=层级切分(新), legacy=balanced expansion(旧)")
    parser.add_argument("--max-depth", type=int, default=40,
                        help="HierAABBTree 最大切分深度 (默认: 40)")
    parser.add_argument("--boundary-batch", type=int, default=6,
                        help="DFS 边界采样批量 (默认: 6)")
    parser.add_argument("--min-edge", type=float, default=0.01,
                        help="最小分割边长 (rad, 默认: 0.01)")
    parser.add_argument("--early-stop-window", type=int, default=30,
                        help="早停滑动窗口大小 (默认: 30, 0=禁用)")
    parser.add_argument("--early-stop-min-vol", type=float, default=1e-4,
                        help="早停体积阈值 (默认: 1e-4)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("examples/output") / f"forest_viz_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  BoxForest 增量拓展可视化")
    logger.info("  seed=%d  n_obs=%d  max_boxes=%d  max_seeds=%d  mode=%s",
                args.seed, args.n_obs, args.max_boxes, args.max_seeds, args.mode)
    logger.info("  step_interval=%d  fps=%d",
                args.step_interval, args.fps)
    logger.info("  output: %s", output_dir)
    logger.info("=" * 60)

    # 1. 加载机器人
    robot = load_robot("2dof_planar")
    joint_limits = list(robot.joint_limits)
    logger.info("Robot: %s, %dDOF", robot.name, robot.n_joints)

    # 2. 随机场景
    rng = np.random.default_rng(args.seed)
    scene = random_scene_2d(robot, args.n_obs, rng)
    logger.info("Scene: %d obstacles", scene.n_obstacles)
    for obs in scene.get_obstacles():
        logger.info("  [%s] (%.3f,%.3f)-(%.3f,%.3f) %.3f×%.3f",
                     obs.name,
                     obs.min_point[0], obs.min_point[1],
                     obs.max_point[0], obs.max_point[1],
                     obs.size[0], obs.size[1])
    scene.to_json(str(output_dir / "scene.json"))

    # 3. 扫描碰撞地图
    logger.info("")
    logger.info("▶ 扫描碰撞地图 (resolution=%.3f) ...", args.resolution)
    collision_map = scan_collision_map(
        robot, scene, joint_limits, resolution=args.resolution)

    # 4. 逐步拓展 + 录制帧
    logger.info("")
    hier_tree = None
    timing_report = None
    expansion_log = None
    if args.mode == "hier":
        logger.info("▶ 开始拓展 BoxForest (HierAABBTree 模式) ...")
        (forest, n_steps, dt, frame_paths, hier_tree,
         timing_report, expansion_log) = grow_forest_hier_animated(
            robot=robot,
            scene=scene,
            joint_limits=joint_limits,
            collision_map=collision_map,
            output_dir=output_dir,
            max_boxes=args.max_boxes,
            max_seeds=args.max_seeds,
            boundary_batch=args.boundary_batch,
            step_interval=args.step_interval,
            farthest_k=args.farthest_k,
            rng_seed=args.seed,
            fps=args.fps,
            max_depth=args.max_depth,
            min_edge_length=args.min_edge,
            early_stop_window=args.early_stop_window,
            early_stop_min_vol=args.early_stop_min_vol,
        )
    else:
        logger.info("▶ 开始拓展 BoxForest (legacy balanced expansion 模式) ...")
        forest, n_steps, dt, frame_paths = grow_forest_animated(
            robot=robot,
            scene=scene,
            joint_limits=joint_limits,
            collision_map=collision_map,
            output_dir=output_dir,
            max_boxes=args.max_boxes,
            max_seeds=args.max_seeds,
            expansion_resolution=args.expansion_res,
            max_rounds=args.max_rounds,
            step_interval=args.step_interval,
            farthest_k=args.farthest_k,
            rng_seed=args.seed,
            fps=args.fps,
        )

    # 5. 最终静态图
    logger.info("")
    logger.info("▶ 生成最终静态图 ...")
    saved, coverage, free_coverage, free_ratio = save_final_figures(
        robot=robot,
        scene=scene,
        forest=forest,
        joint_limits=joint_limits,
        collision_map=collision_map,
        output_dir=output_dir,
        resolution=args.resolution,
    )

    # 6. 统计
    logger.info("")
    stats_path = write_stats(
        forest, n_steps, dt,
        coverage, free_coverage, free_ratio,
        output_dir, args,
        hier_tree=hier_tree,
        timing_report=timing_report,
        expansion_log=expansion_log,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("  完成！")
    logger.info("  输出目录: %s", output_dir)
    logger.info("  帧数: %d", len(frame_paths))
    for f in saved:
        logger.info("  %s", Path(f).name)
    logger.info("  stats.txt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
