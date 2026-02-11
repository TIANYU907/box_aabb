#!/usr/bin/env python
"""
examples/animate_box_tree.py - Box Tree 生长过程动画

生成 mp4 动画展示 2DOF 平面机器人在随机场景中的 box tree 生成全过程：
  - 背景：C-space 碰撞地图
  - 红点：随机 seed 碰撞（被拒绝）
  - 黄点：seed 已被现有 box 覆盖
  - 绿★ → box 矩形出现：成功拓展新 box
  - 右侧面板：工作空间中机械臂姿态 + 当前 box 的运动范围
  - 左下角：实时统计信息

输出 mp4 视频到 examples/output/box_tree_anim_<ts>/

用法：
    python -m examples.animate_box_tree
    python -m examples.animate_box_tree --seed 42 --n-obs 4 --max-boxes 200 --fps 15
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from box_aabb.robot import Robot, load_robot
from planner.aabb_cache import AABBCache
from planner.box_expansion import BoxExpander, ExpansionLog
from planner.box_tree import BoxTreeManager
from planner.collision import CollisionChecker
from planner.models import BoxNode
from planner.obstacles import Scene

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("box_tree_anim")


def _sanitize_for_json(obj):
    """递归清理 Python 对象中的非标准 JSON 值 (inf, nan, ndarray)"""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isinf(v) or math.isnan(v):
            return None
        return round(v, 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


# ──────────────────────────────────────────────────────────
#  数据结构
# ──────────────────────────────────────────────────────────

@dataclass
class GrowthEvent:
    """记录 box forest 生长过程中的一个事件"""
    event_type: str       # 'reject_collision', 'reject_covered', 'box_added'
    config: np.ndarray    # seed 配置
    box: Optional[BoxNode] = None
    tree_id: int = -1
    source: str = 'random'  # 'random' / 'boundary'
    expansion_log: Optional[ExpansionLog] = None  # 拓展详情日志


@dataclass
class AnimFrame:
    """一帧动画数据"""
    box: BoxNode                              # 本帧新增的 box
    seed: np.ndarray                          # seed 配置
    tree_id: int
    source: str
    reject_collision: List[np.ndarray] = field(default_factory=list)  # 碰撞被拒 seeds
    reject_covered: List[np.ndarray] = field(default_factory=list)    # 覆盖被拒 seeds
    box_index: int = 0                        # 第几个 box (0-based)
    expansion_progress: float = 1.0           # 拓展进度 0→1（用于动画插值）


# ──────────────────────────────────────────────────────────
#  场景生成
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
#  带事件记录的 box forest 生长
# ──────────────────────────────────────────────────────────

def record_growth(
    robot: Robot,
    scene: Scene,
    joint_limits: List[Tuple[float, float]],
    max_boxes: int,
    max_iters: int,
    expansion_resolution: float,
    max_rounds: int,
    seed_batch: int,
    rng_seed: int,
    expansion_strategy: str = 'balanced',
    balanced_step_fraction: float = 0.5,
    balanced_max_steps: int = 200,
    min_initial_half_width: float = 0.001,
    use_sampling: bool = False,
    sampling_n: int = 80,
    overlap_weight: float = 1.0,
) -> Tuple[List[GrowthEvent], BoxTreeManager]:
    """运行 box forest 生长并记录所有事件"""
    rng = np.random.default_rng(rng_seed)
    aabb_cache = AABBCache.auto_load(robot)
    checker = CollisionChecker(robot, scene, aabb_cache=aabb_cache)
    expander = BoxExpander(
        robot=robot, collision_checker=checker,
        joint_limits=joint_limits,
        expansion_resolution=expansion_resolution,
        max_rounds=max_rounds,
        min_initial_half_width=min_initial_half_width,
        strategy=expansion_strategy,
        balanced_step_fraction=balanced_step_fraction,
        balanced_max_steps=balanced_max_steps,
        use_sampling=use_sampling,
        sampling_n=sampling_n,
        overlap_weight=overlap_weight,
    )
    manager = BoxTreeManager()
    events: List[GrowthEvent] = []

    for iteration in range(max_iters):
        if manager.total_nodes >= max_boxes:
            break

        q = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])

        if checker.check_config_collision(q):
            events.append(GrowthEvent('reject_collision', q.copy()))
            continue

        if manager.find_containing_box(q) is not None:
            events.append(GrowthEvent('reject_covered', q.copy()))
            continue

        nid = manager.allocate_node_id()
        box = expander.expand(q, node_id=nid, rng=rng, enable_log=True,
                              existing_boxes=manager.get_all_boxes())
        exp_log = expander.get_last_log()
        if box is None or box.volume < 1e-6:
            events.append(GrowthEvent('reject_collision', q.copy()))
            continue

        nearest = manager.find_nearest_box(q)
        if nearest is not None and nearest.distance_to_config(q) < 2.0:
            manager.add_box(nearest.tree_id, box, parent_id=nearest.node_id)
        else:
            manager.create_tree(box)
        events.append(GrowthEvent('box_added', q.copy(), box=box,
                                  tree_id=box.tree_id, source='random',
                                  expansion_log=exp_log))

        # 边界重采样
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
                nid2 = manager.allocate_node_id()
                child = expander.expand(qs, node_id=nid2, rng=rng, enable_log=True,
                                        existing_boxes=manager.get_all_boxes())
                child_log = expander.get_last_log()
                if child is None or child.volume < 1e-6:
                    continue
                nearest_in = manager.find_nearest_box_in_tree(box.tree_id, qs)
                if nearest_in is not None:
                    manager.add_box(box.tree_id, child,
                                    parent_id=nearest_in.node_id)
                events.append(GrowthEvent('box_added', qs.copy(), box=child,
                                          tree_id=child.tree_id,
                                          source='boundary',
                                          expansion_log=child_log))

        if (iteration + 1) % 500 == 0:
            logger.info("  growth iter %d/%d: %d boxes",
                        iteration + 1, max_iters, manager.total_nodes)

    # 自动保存 AABB 缓存
    try:
        aabb_cache.auto_save(robot)
    except Exception as e:
        logger.warning("AABB 缓存自动保存失败: %s", e)

    return events, manager


# ──────────────────────────────────────────────────────────
#  事件 → 动画帧序列
# ──────────────────────────────────────────────────────────

def build_frames(
    events: List[GrowthEvent],
    n_expansion_detail: int = 20,
) -> List[AnimFrame]:
    """将事件列表转为动画帧序列

    Args:
        events: 生长事件列表
        n_expansion_detail: 前 N 个 box 显示详细拓展动画 (3步)

    Returns:
        AnimFrame 列表
    """
    frames: List[AnimFrame] = []
    pending_coll: List[np.ndarray] = []
    pending_cov: List[np.ndarray] = []
    box_counter = 0

    for evt in events:
        if evt.event_type == 'reject_collision':
            pending_coll.append(evt.config)
        elif evt.event_type == 'reject_covered':
            pending_cov.append(evt.config)
        elif evt.event_type == 'box_added' and evt.box is not None:
            # 决定拓展动画帧数
            if box_counter < n_expansion_detail:
                # 详细拓展：3 帧 (0.33, 0.67, 1.0)
                for step, progress in enumerate([0.33, 0.67, 1.0]):
                    f = AnimFrame(
                        box=evt.box,
                        seed=evt.config,
                        tree_id=evt.tree_id,
                        source=evt.source,
                        reject_collision=pending_coll.copy() if step == 0 else [],
                        reject_covered=pending_cov.copy() if step == 0 else [],
                        box_index=box_counter,
                        expansion_progress=progress,
                    )
                    frames.append(f)
            else:
                # 普通：1 帧 (立即出现)
                f = AnimFrame(
                    box=evt.box,
                    seed=evt.config,
                    tree_id=evt.tree_id,
                    source=evt.source,
                    reject_collision=pending_coll.copy(),
                    reject_covered=pending_cov.copy(),
                    box_index=box_counter,
                    expansion_progress=1.0,
                )
                frames.append(f)

            pending_coll.clear()
            pending_cov.clear()
            box_counter += 1

    return frames


def interpolate_box(box: BoxNode, seed: np.ndarray, progress: float):
    """插值计算 box 在拓展进度 p ∈ [0,1] 时的区间

    从 seed 中心向外生长到完整 box 区间。
    """
    intervals = []
    for i, (lo, hi) in enumerate(box.joint_intervals):
        s = seed[i]
        i_lo = s - progress * (s - lo)
        i_hi = s + progress * (hi - s)
        intervals.append((i_lo, i_hi))
    return intervals


# ──────────────────────────────────────────────────────────
#  渲染动画
# ──────────────────────────────────────────────────────────

def render_animation(
    frames: List[AnimFrame],
    collision_map: np.ndarray,
    joint_limits: List[Tuple[float, float]],
    robot: Robot,
    scene: Scene,
    output_path: Path,
    total_boxes: int,
    fps: int = 15,
    hold_intro: int = 20,
    hold_outro: int = 40,
):
    """渲染动画到 mp4 文件"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle

    logger.info("渲染动画: %d 帧 + %d intro + %d outro, fps=%d",
                len(frames), hold_intro, hold_outro, fps)

    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]
    reach = sum(p['a'] for p in robot.dh_params)
    if robot.tool_frame:
        reach += robot.tool_frame.get('a', 0)
    reach = reach or 2.0

    # ── 创建画布 ──
    fig, (ax_cs, ax_ws) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(wspace=0.25, left=0.05, right=0.97, top=0.92, bottom=0.08)

    # C-space 背景
    ax_cs.imshow(collision_map, origin='lower',
                 extent=[lo_x, hi_x, lo_y, hi_y],
                 cmap='Reds', alpha=0.35, aspect='auto')
    ax_cs.set_xlim(lo_x - 0.05, hi_x + 0.05)
    ax_cs.set_ylim(lo_y - 0.05, hi_y + 0.05)
    ax_cs.set_xlabel('q0 (rad)', fontsize=9)
    ax_cs.set_ylabel('q1 (rad)', fontsize=9)
    ax_cs.set_aspect('equal')
    ax_cs.grid(True, alpha=0.15)

    # 工作空间背景
    for obs in scene.get_obstacles():
        rect = Rectangle(
            (obs.min_point[0], obs.min_point[1]),
            obs.size[0], obs.size[1],
            linewidth=1.2, edgecolor='red', facecolor='red', alpha=0.35)
        ax_ws.add_patch(rect)
    ax_ws.set_xlim(-reach * 1.3, reach * 1.3)
    ax_ws.set_ylim(-reach * 1.3, reach * 1.3)
    ax_ws.set_xlabel('X (m)', fontsize=9)
    ax_ws.set_ylabel('Y (m)', fontsize=9)
    ax_ws.set_aspect('equal')
    ax_ws.grid(True, alpha=0.15)
    ax_ws.set_title('Workspace', fontsize=10)

    # 动态元素
    tree_cmap = cm.Set2
    reject_coll_sc = ax_cs.scatter([], [], c='red', s=4, alpha=0.5, zorder=5)
    reject_cov_sc = ax_cs.scatter([], [], c='gold', s=4, alpha=0.5, zorder=5)
    seed_sc = ax_cs.scatter([], [], c='lime', s=60, marker='*',
                            zorder=10, edgecolors='darkgreen', linewidths=0.5)

    # 当前 box 高亮框
    highlight_rect = None

    # 当前拓展中的 box (临时)
    expansion_rect = None

    # 机械臂线
    arm_line, = ax_ws.plot([], [], 'o-', color='dodgerblue',
                           lw=2.5, ms=5, zorder=5)
    # box 运动范围的 ghost arms
    ghost_lines = []
    for _ in range(4):
        ln, = ax_ws.plot([], [], 'o-', color='lightblue',
                         lw=1.0, ms=2, alpha=0.4, zorder=3)
        ghost_lines.append(ln)

    # 统计文本
    stats_text = ax_cs.text(
        0.02, 0.02, '', transform=ax_cs.transAxes,
        fontsize=8, verticalalignment='bottom', family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    title_obj = ax_cs.set_title('C-Space: Box Tree Growth', fontsize=10)

    # ── 视频写入器 ──
    import imageio
    try:
        writer = imageio.get_writer(
            str(output_path), fps=fps, codec='libx264',
            quality=8,                        # 1-10, higher = better
            output_params=['-pix_fmt', 'yuv420p'],  # 兼容性
            format='FFMPEG',
        )
    except (ImportError, OSError):
        # ffmpeg 不可用时 fallback 到 GIF
        output_path = output_path.with_suffix('.gif')
        logger.warning("ffmpeg not available, falling back to GIF: %s", output_path)
        writer = imageio.get_writer(str(output_path), fps=fps, format='GIF')

    def capture():
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4) RGBA
        return buf[:, :, :3].copy()                  # → RGB (H, W, 3)

    # ── Intro: 碰撞地图 ──
    title_obj.set_text('C-Space Collision Map')
    stats_text.set_text('Initializing...')
    img = capture()
    for _ in range(hold_intro):
        writer.append_data(img)

    # ── 主循环 ──
    cspace_area = (hi_x - lo_x) * (hi_y - lo_y)
    total_volume = 0.0
    n_boxes_done = 0
    n_reject_coll_total = 0
    n_reject_cov_total = 0
    prev_highlight = None
    prev_expansion = None
    box_count_per_tree: Dict[int, int] = {}

    t_render_start = time.time()

    for fi, frame in enumerate(frames):
        # 清除上一帧的 highlight/expansion
        if prev_highlight is not None:
            prev_highlight.set_linewidth(0.5)
            prev_highlight.set_edgecolor(prev_highlight.get_facecolor())
        if prev_expansion is not None:
            prev_expansion.remove()
            prev_expansion = None

        # 拒绝 seeds 散点
        rc = frame.reject_collision
        n_reject_coll_total += len(rc)
        if rc:
            reject_coll_sc.set_offsets(np.array(rc))
        else:
            reject_coll_sc.set_offsets(np.empty((0, 2)))

        rcv = frame.reject_covered
        n_reject_cov_total += len(rcv)
        if rcv:
            reject_cov_sc.set_offsets(np.array(rcv))
        else:
            reject_cov_sc.set_offsets(np.empty((0, 2)))

        # Seed 标记
        seed_sc.set_offsets([frame.seed])

        # 获取 tree color
        tid = frame.tree_id
        box_count_per_tree.setdefault(tid, 0)
        tc = tree_cmap((tid * 0.13) % 1.0)

        box = frame.box
        progress = frame.expansion_progress

        if progress < 1.0:
            # 拓展动画：临时矩形
            itvs = interpolate_box(box, frame.seed, progress)
            lo0, hi0 = itvs[0]
            lo1, hi1 = itvs[1]
            exp_rect = Rectangle(
                (lo0, lo1), hi0 - lo0, hi1 - lo1,
                linewidth=1.5, edgecolor='lime', facecolor=tc,
                alpha=0.25, linestyle='--', zorder=6)
            ax_cs.add_patch(exp_rect)
            prev_expansion = exp_rect
        else:
            # 完成：添加永久 box patch
            lo0, hi0 = box.joint_intervals[0]
            lo1, hi1 = box.joint_intervals[1]
            rect = Rectangle(
                (lo0, lo1), hi0 - lo0, hi1 - lo1,
                linewidth=0.5, edgecolor=tc, facecolor=tc,
                alpha=0.28, zorder=4)
            ax_cs.add_patch(rect)

            # 高亮当前 box
            hi_rect = Rectangle(
                (lo0, lo1), hi0 - lo0, hi1 - lo1,
                linewidth=2.0, edgecolor='lime', facecolor='none',
                zorder=7)
            ax_cs.add_patch(hi_rect)
            prev_highlight = hi_rect

            total_volume += box.volume
            n_boxes_done += 1
            box_count_per_tree[tid] += 1
            prev_expansion = None

        # 机械臂
        positions = robot.get_link_positions(frame.seed)
        arm_xs = [p[0] for p in positions]
        arm_ys = [p[1] for p in positions]
        arm_line.set_data(arm_xs, arm_ys)

        source_label = 'boundary' if frame.source == 'boundary' else 'random'
        arm_color = 'darkorange' if frame.source == 'boundary' else 'dodgerblue'
        arm_line.set_color(arm_color)

        # Ghost arms: 显示 box 4 角的姿态
        if progress >= 1.0:
            corners = [
                (box.joint_intervals[0][0], box.joint_intervals[1][0]),
                (box.joint_intervals[0][0], box.joint_intervals[1][1]),
                (box.joint_intervals[0][1], box.joint_intervals[1][0]),
                (box.joint_intervals[0][1], box.joint_intervals[1][1]),
            ]
            for gi, (cq0, cq1) in enumerate(corners):
                pos = robot.get_link_positions(np.array([cq0, cq1]))
                ghost_lines[gi].set_data([p[0] for p in pos],
                                         [p[1] for p in pos])
                ghost_lines[gi].set_alpha(0.3)
        else:
            for gl in ghost_lines:
                gl.set_data([], [])

        # 标题
        title_obj.set_text(
            f'C-Space Box Tree Growth  —  Box #{frame.box_index + 1}'
            f'/{total_boxes}  ({source_label})')

        # 统计
        n_trees = len(box_count_per_tree)
        cov = total_volume / cspace_area * 100
        stats_text.set_text(
            f'Boxes:  {n_boxes_done}/{total_boxes}\n'
            f'Trees:  {n_trees}\n'
            f'Volume: {total_volume:.2f}  ({cov:.1f}%)\n'
            f'Reject: coll={n_reject_coll_total}  cov={n_reject_cov_total}')

        # 工作空间标题
        ax_ws.set_title(
            f'Workspace  (tree #{tid}, {source_label})', fontsize=10)

        # 写帧
        writer.append_data(capture())

        if (fi + 1) % 100 == 0:
            elapsed = time.time() - t_render_start
            logger.info("  rendered %d/%d frames (%.1fs)",
                        fi + 1, len(frames), elapsed)

    # ── Outro: 最终状态 ──
    reject_coll_sc.set_offsets(np.empty((0, 2)))
    reject_cov_sc.set_offsets(np.empty((0, 2)))
    seed_sc.set_offsets(np.empty((0, 2)))
    for gl in ghost_lines:
        gl.set_data([], [])
    arm_line.set_data([], [])
    if prev_highlight is not None:
        prev_highlight.remove()
    title_obj.set_text(
        f'Box Tree Complete  —  {n_boxes_done} boxes, '
        f'{len(box_count_per_tree)} trees, '
        f'coverage={total_volume / cspace_area * 100:.1f}%')
    ax_ws.set_title('Workspace (final)', fontsize=10)

    img = capture()
    for _ in range(hold_outro):
        writer.append_data(img)

    writer.close()
    plt.close(fig)
    total_time = time.time() - t_render_start
    total_frames = len(frames) + hold_intro + hold_outro
    logger.info("动画渲染完成: %d 帧, %.1fs, 输出: %s",
                total_frames, total_time, output_path)


# ──────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2DOF Box Tree 生长过程动画")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-obs", type=int, default=4,
                        help="障碍物数量")
    parser.add_argument("--max-boxes", type=int, default=200,
                        help="最大 box 数 (建议 ≤300 控制视频长度)")
    parser.add_argument("--max-iters", type=int, default=3000,
                        help="最大采样迭代")
    parser.add_argument("--expansion-res", type=float, default=0.02,
                        help="box 拓展二分精度 (rad)")
    parser.add_argument("--max-rounds", type=int, default=4,
                        help="box 拓展迭代轮数")
    parser.add_argument("--cmap-resolution", type=float, default=0.05,
                        help="碰撞地图扫描精度 (越小越精细越慢)")
    parser.add_argument("--boundary-batch", type=int, default=6,
                        help="边界再采样 batch size")
    parser.add_argument("--fps", type=int, default=15,
                        help="视频帧率")
    parser.add_argument("--expansion-detail", type=int, default=20,
                        help="前 N 个 box 显示 3 帧拓展动画")
    parser.add_argument("--expansion-strategy", type=str, default='balanced',
                        choices=['balanced', 'greedy'],
                        help="box 拓展策略")
    parser.add_argument("--balanced-step-fraction", type=float, default=0.5,
                        help="balanced 策略比例步长")
    parser.add_argument("--balanced-max-steps", type=int, default=200,
                        help="balanced 策略最大步数")
    parser.add_argument("--min-initial-half-width", type=float, default=0.001,
                        help="box 初始半宽")
    parser.add_argument("--use-sampling", action='store_true', default=False,
                        help="启用 hybrid 碰撞检测（num 方法生成 AABB）")
    parser.add_argument("--sampling-n", type=int, default=80,
                        help="hybrid 碰撞检测采样数")
    parser.add_argument("--overlap-weight", type=float, default=1.0,
                        help="重叠惩罚权重 (0=无惩罚, 1=纯新增体积, >1=激进去重)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("examples/output") / f"box_tree_anim_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  2DOF Box Tree 动画")
    logger.info("  seed=%d  n_obs=%d  max_boxes=%d  fps=%d",
                args.seed, args.n_obs, args.max_boxes, args.fps)
    logger.info("  output: %s", output_dir)
    logger.info("=" * 60)

    # 1. 加载机器人 & 场景
    robot = load_robot("2dof_planar")
    joint_limits = list(robot.joint_limits)
    rng = np.random.default_rng(args.seed)
    scene = random_scene_2d(robot, args.n_obs, rng)
    logger.info("Robot: %s, %dDOF", robot.name, robot.n_joints)
    logger.info("Scene: %d obstacles", scene.n_obstacles)

    # 2. 计算碰撞地图（背景）
    logger.info("扫描 C-space 碰撞地图 (resolution=%.3f) ...",
                args.cmap_resolution)
    checker = CollisionChecker(robot, scene)
    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]
    xs = np.arange(lo_x, hi_x, args.cmap_resolution)
    ys = np.arange(lo_y, hi_y, args.cmap_resolution)
    cmap_grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if checker.check_config_collision(np.array([x, y])):
                cmap_grid[i, j] = 1.0
    free_ratio = float(np.sum(cmap_grid == 0)) / cmap_grid.size
    logger.info("  碰撞地图: %dx%d, 自由空间=%.1f%%",
                len(xs), len(ys), free_ratio * 100)

    # 3. 生长 + 记录事件
    logger.info("生长 box forest (max_boxes=%d) ...", args.max_boxes)
    t0 = time.time()
    events, manager = record_growth(
        robot=robot, scene=scene, joint_limits=joint_limits,
        max_boxes=args.max_boxes, max_iters=args.max_iters,
        expansion_resolution=args.expansion_res,
        max_rounds=args.max_rounds,
        seed_batch=args.boundary_batch, rng_seed=args.seed,
        expansion_strategy=args.expansion_strategy,
        balanced_step_fraction=args.balanced_step_fraction,
        balanced_max_steps=args.balanced_max_steps,
        min_initial_half_width=args.min_initial_half_width,
        use_sampling=args.use_sampling,
        sampling_n=args.sampling_n,
        overlap_weight=args.overlap_weight,
    )
    dt = time.time() - t0
    n_boxes = sum(1 for e in events if e.event_type == 'box_added')
    n_rej = sum(1 for e in events if e.event_type != 'box_added')
    logger.info("  生长完成: %d boxes, %d 拒绝, %.1fs", n_boxes, n_rej, dt)

    # 3.5 写入拓展详情到独立文件
    detail_path = output_dir / "expansion_detail.txt"
    box_idx = 0
    with open(detail_path, 'w', encoding='utf-8') as f:
        f.write("Box Expansion Detail Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Robot: {robot.name}, {robot.n_joints}DOF\n")
        f.write(f"Seed: {args.seed}, Obstacles: {args.n_obs}\n")
        f.write(f"Expansion resolution: {args.expansion_res} rad\n")
        f.write(f"Max rounds: {args.max_rounds}\n")
        f.write(f"Total boxes: {n_boxes}, Rejected: {n_rej}\n")
        f.write("=" * 70 + "\n\n")
        for evt in events:
            if evt.event_type == 'box_added' and evt.expansion_log is not None:
                f.write(f"[source: {evt.source}]  ")
                f.write(f"tree_id={evt.tree_id}\n")
                f.write(evt.expansion_log.to_text(box_index=box_idx))
                f.write("\n")
                box_idx += 1
    logger.info("  拓展详情: %s (%d boxes)", detail_path, box_idx)

    # 3.6 写入 growth_log.json
    growth_log_path = output_dir / "growth_log.json"
    growth_records = []
    for evt in events:
        if evt.event_type != 'box_added' or evt.box is None:
            continue
        b = evt.box
        nearest = manager.find_nearest_box(evt.config)
        ndist = nearest.distance_to_config(evt.config) if nearest else None
        record = {
            "type": evt.event_type,
            "config": evt.config.tolist(),
            "source": evt.source,
            "nearest_box_dist": ndist,
            "box_index": len(growth_records),
            "tree_id": evt.tree_id,
            "volume": b.volume,
            "widths": list(b.widths),
            "intervals": [list(iv) for iv in b.joint_intervals],
        }
        if evt.expansion_log is not None:
            el = evt.expansion_log
            record["expansion"] = {
                "strategy": el.strategy,
                "dim_order": el.dim_order,
                "jacobian_norms": (
                    [el.jacobian_norms[i] for i in range(len(el.jacobian_norms))]
                    if el.jacobian_norms else None
                ),
                "total_rounds": el.total_rounds,
                "total_steps": el.total_steps,
                "final_volume": el.final_volume,
                "early_stop": el.early_stop,
                "early_stop_reason": el.early_stop_reason,
            }
        growth_records.append(record)
    with open(growth_log_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(growth_records), f, indent=2,
                  ensure_ascii=False)
    logger.info("  growth log: %s (%d records)", growth_log_path, len(growth_records))

    # 4. 构建帧序列
    frames = build_frames(events, n_expansion_detail=args.expansion_detail)
    logger.info("  动画帧数: %d (+ intro/outro)", len(frames))

    # 5. 渲染
    video_path = output_dir / "box_tree_growth.mp4"
    render_animation(
        frames=frames,
        collision_map=cmap_grid,
        joint_limits=joint_limits,
        robot=robot, scene=scene,
        output_path=video_path,
        total_boxes=n_boxes,
        fps=args.fps,
    )

    # 6. 保存场景 & 统计
    scene.to_json(str(output_dir / "scene.json"))
    summary = (
        f"Boxes: {n_boxes}, Trees: {manager.n_trees}\n"
        f"Volume: {manager.get_total_volume():.4f}\n"
        f"Free ratio: {free_ratio*100:.1f}%\n"
        f"Events: {len(events)} ({n_rej} rejected)\n"
        f"Frames: {len(frames)}\n"
        f"Video: {video_path}\n"
        f"Growth log: {growth_log_path}\n"
    )
    (output_dir / "stats.txt").write_text(summary, encoding="utf-8")

    logger.info("")
    logger.info("完成！")
    logger.info("  视频: %s", video_path)
    logger.info("  统计: %s", output_dir / "stats.txt")
    logger.info("  播放: 直接双击 mp4 或在浏览器中打开")


if __name__ == "__main__":
    main()
