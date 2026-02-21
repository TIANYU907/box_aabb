"""
examples/viz_2dof_full_pipeline.py — 2DOF 场景完整流程可视化

生成分阶段可视化图:
  1. C-space 碰撞底图
  2. Forest expansion (逐步快照 → GIF)
  3. Coarsen 前后对比
  4. 岛检测 & 桥接
  5. Dijkstra 路径规划 (box序列 + 精炼路径)
  6. 最终总览图

用法:
    cd v3
    python examples/viz_2dof_full_pipeline.py [--seed 42] [--obstacles 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np

# ── v3 path setup ──
_ROOT = Path(__file__).resolve().parents[1]  # v3/
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from forest.models import BoxNode
from forest.connectivity import find_islands, bridge_islands, UnionFind
from forest.safe_box_forest import SafeBoxForest
from forest.hier_aabb_tree import HierAABBTree
from forest.coarsen import coarsen_forest
from planner.sbf_planner import SBFPlanner
from planner.models import SBFConfig, gmean_edge_length
from planner.pipeline import (
    PandaGCSConfig, make_planner_config,
    _build_adjacency_and_islands, find_box_containing,
)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Viz2DConfig:
    seed: int = 42
    robot_name: str = "2dof_planar"
    n_obstacles: int = 8

    q_start: List[float] = field(
        default_factory=lambda: [0.8 * np.pi, 0.2])
    q_goal: List[float] = field(
        default_factory=lambda: [-0.7 * np.pi, -0.4])

    # forest
    max_consecutive_miss: int = 20
    max_boxes: int = 200
    min_box_size: float = 0.02
    goal_bias: float = 0.15
    guided_sample_ratio: float = 0.6
    boundary_expand: bool = True

    # coarsen
    coarsen_max_rounds: int = 20

    # island / bridge
    min_island_size: float = 0.3

    # viz
    snapshot_every: int = 3
    collision_resolution: float = 0.025
    dpi: int = 150
    gif_frame_ms: int = 250

    # scene obstacles
    obs_cx_range: Tuple[float, float] = (-1.8, 1.8)
    obs_cy_range: Tuple[float, float] = (-1.8, 1.8)
    obs_w_range: Tuple[float, float] = (0.3, 0.8)
    obs_h_range: Tuple[float, float] = (0.3, 0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Random scene
# ═══════════════════════════════════════════════════════════════════════════

def build_random_2d_scene(robot, q_start, q_goal, rng, cfg: Viz2DConfig,
                          max_trials=300) -> Scene:
    """生成随机 2D 配置空间障碍物场景，确保起终点无碰撞且直连路径被阻挡。"""
    for _ in range(max_trials):
        scene = Scene()
        for i in range(cfg.n_obstacles):
            cx = float(rng.uniform(*cfg.obs_cx_range))
            cy = float(rng.uniform(*cfg.obs_cy_range))
            w = float(rng.uniform(*cfg.obs_w_range))
            h = float(rng.uniform(*cfg.obs_h_range))
            scene.add_obstacle([cx - w/2, cy - h/2],
                               [cx + w/2, cy + h/2], name=f"obs_{i}")

        checker = CollisionChecker(robot=robot, scene=scene)
        if checker.check_config_collision(q_start):
            continue
        if checker.check_config_collision(q_goal):
            continue
        # 确保直连路径被阻挡(才有规划价值)
        if not checker.check_segment_collision(q_start, q_goal, 0.03):
            continue
        return scene
    raise RuntimeError("无法生成满足条件的随机场景")


# ═══════════════════════════════════════════════════════════════════════════
# C-space collision map
# ═══════════════════════════════════════════════════════════════════════════

def scan_collision_map(robot, scene, joint_limits, resolution=0.03):
    """扫描 2DOF C-space 碰撞区域 → 二值矩阵."""
    checker = CollisionChecker(robot=robot, scene=scene)
    lo_x, hi_x = joint_limits[0]
    lo_y, hi_y = joint_limits[1]
    xs = np.arange(lo_x, hi_x, resolution)
    ys = np.arange(lo_y, hi_y, resolution)
    cmap = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for i, y in enumerate(ys):
        row = np.column_stack([xs, np.full(len(xs), y)])
        cmap[i, :] = checker.check_config_collision_batch(row).astype(np.float32)
    extent = [lo_x, hi_x, lo_y, hi_y]
    return cmap, extent


# ═══════════════════════════════════════════════════════════════════════════
# Forest grow with snapshots
# ═══════════════════════════════════════════════════════════════════════════

def grow_forest_with_snapshots(
    planner: SBFPlanner,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    seed: int,
    max_miss: int,
    max_boxes: int,
    min_box_size: float,
    snapshot_every: int = 3,
):
    """串行 forest 生长, 每 snapshot_every 个 box 拍一次快照."""
    ndim = 2
    rng = np.random.default_rng(seed)
    forest = planner._load_or_create_forest()
    forest.hier_tree = planner.hier_tree
    obs_packed = planner.hier_tree._prepack_obstacles_c(planner.obstacles)

    snapshots = []   # [(n_boxes, boxes_copy, new_box_id)]
    added_since_snap = 0

    def _snap(new_id=-1, force=False):
        nonlocal added_since_snap
        added_since_snap += 1
        if force or added_since_snap >= snapshot_every:
            boxes_copy = {}
            for bid, b in forest.boxes.items():
                boxes_copy[bid] = BoxNode(
                    node_id=b.node_id,
                    joint_intervals=[tuple(iv) for iv in b.joint_intervals],
                    seed_config=b.seed_config.copy() if b.seed_config is not None else None,
                    volume=b.volume)
            snapshots.append((forest.n_boxes, boxes_copy, new_id))
            added_since_snap = 0

    def _try_add(q):
        """尝试在 q 处生长 box, 返回 box id 或 -1."""
        if planner.hier_tree.is_occupied(q):
            return -1
        nid = forest.allocate_id()
        ffb = planner.hier_tree.find_free_box(
            q, planner.obstacles, mark_occupied=True,
            forest_box_id=nid, obs_packed=obs_packed)
        if ffb is None:
            return -1
        vol = 1.0
        for lo, hi in ffb.intervals:
            vol *= max(hi - lo, 0)
        if gmean_edge_length(vol, ndim) < min_box_size:
            return -1
        box = BoxNode(node_id=nid, joint_intervals=ffb.intervals,
                      seed_config=q.copy(), volume=vol)
        if ffb.absorbed_box_ids:
            forest.remove_boxes(ffb.absorbed_box_ids)
        forest.add_box_direct(box)
        return nid

    # seed points
    for qs in [q_start, q_goal]:
        nid = _try_add(qs)
        if nid >= 0:
            _snap(nid)

    # sampling
    intervals = planner.joint_limits
    lows = np.array([lo for lo, _ in intervals])
    highs = np.array([hi for _, hi in intervals])

    consec = 0
    # boundary expand state
    expand_target = None
    expand_fails = 0
    boundary_max_fail = planner.config.boundary_expand_max_failures

    while consec < max_miss:
        if forest.n_boxes >= max_boxes:
            break

        # boundary expand
        if expand_target is not None:
            q = planner._sample_boundary_seed(expand_target, rng)
            if q is None or planner.hier_tree.is_occupied(q):
                expand_fails += 1
                if expand_fails >= boundary_max_fail:
                    expand_target = None
                continue
        else:
            # guided / goal-biased / uniform sampling
            roll = rng.uniform()
            if roll < planner.config.goal_bias:
                noise = rng.normal(0, 0.3, size=ndim)
                q = np.clip(q_goal + noise, lows, highs)
            elif roll < planner.config.goal_bias + 0.6:
                try:
                    q = planner.hier_tree.sample_unoccupied_seed(rng)
                except ValueError:
                    q = None
                if q is None:
                    q = rng.uniform(lows, highs)
            else:
                q = rng.uniform(lows, highs)

            if planner.collision_checker.check_config_collision(q):
                consec += 1
                continue

        nid = _try_add(q)
        if nid >= 0:
            consec = 0
            _snap(nid)
            # trigger boundary expand
            expand_target = forest.boxes[nid]
            expand_fails = 0
        else:
            consec += 1
            if expand_target is not None:
                expand_fails += 1
                if expand_fails >= boundary_max_fail:
                    expand_target = None

    # final snapshot
    if not snapshots or snapshots[-1][0] != forest.n_boxes:
        _snap(-1, force=True)

    return snapshots, forest


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_collision_bg(ax, cmap_data, extent):
    """在 ax 上绘制碰撞底图."""
    ax.imshow(cmap_data, origin="lower", extent=extent,
              cmap="Reds", alpha=0.30, aspect="auto")


def _draw_boxes(ax, boxes, color_by="volume", highlight_ids=None,
                alpha=0.30, edge_alpha=0.8, lw=0.5):
    """绘制 box 集合."""
    highlight_ids = highlight_ids or set()
    cmap = plt.cm.viridis

    if color_by == "volume":
        vols = [b.volume for b in boxes.values()]
        vmin = min(vols) if vols else 0
        vmax = max(vols) if vols else 1
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for bid, box in boxes.items():
        lo_x, hi_x = box.joint_intervals[0]
        lo_y, hi_y = box.joint_intervals[1]

        if bid in highlight_ids:
            ec, fc, a = "#ff6600", "#ff9933", 0.50
            rect_lw = 1.5
        else:
            if color_by == "volume":
                c = cmap(norm(box.volume))
            else:
                c = (0.2, 0.5, 0.8, 1.0)
            ec, fc, a = c, c, alpha
            rect_lw = lw

        rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                          linewidth=rect_lw, edgecolor=ec,
                          facecolor=fc, alpha=a)
        ax.add_patch(rect)


def _draw_start_goal(ax, q_start, q_goal):
    """标记起点和终点."""
    ax.plot(q_start[0], q_start[1], 'o', color='cyan', markersize=10,
            markeredgecolor='black', markeredgewidth=1.2, zorder=20,
            label='Start')
    ax.plot(q_goal[0], q_goal[1], '*', color='yellow', markersize=14,
            markeredgecolor='black', markeredgewidth=1.0, zorder=20,
            label='Goal')


def _draw_islands(ax, boxes, islands):
    """按岛着色绘制 boxes."""
    cmap = plt.cm.tab20
    n_islands = max(len(islands), 1)
    node_island = {}
    for idx, island in enumerate(islands):
        for bid in island:
            node_island[bid] = idx

    for bid, box in boxes.items():
        lo_x, hi_x = box.joint_intervals[0]
        lo_y, hi_y = box.joint_intervals[1]
        isl_idx = node_island.get(bid, 0)
        c = cmap(isl_idx / n_islands)
        rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                          linewidth=0.5, edgecolor=c, facecolor=c, alpha=0.35)
        ax.add_patch(rect)


def _geodesic_points(qa, qb, period, n_pts=200):
    """沿环面最短路径采样 n_pts 个点, 归一化到 [-period/2, period/2)."""
    half = period / 2.0
    diff = ((qb - qa) + half) % period - half
    pts = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        q = qa + t * diff
        q = ((q + half) % period) - half
        pts.append(q)
    return pts


def _draw_wrapped_line(ax, qa, qb, period, n_pts=200, **kwargs):
    """绘制环面最短路径线段, 在周期边界处自动断开."""
    if period is None or period <= 0:
        ax.plot([qa[0], qb[0]], [qa[1], qb[1]], **kwargs)
        return
    pts = _geodesic_points(qa, qb, period, n_pts)
    # 检测相邻点是否跳变 (跳变 = 周期 wrap)
    segments = [[pts[0]]]
    half = period / 2.0
    for i in range(1, len(pts)):
        if np.any(np.abs(pts[i] - pts[i - 1]) > half):
            segments.append([pts[i]])  # 新段
        else:
            segments[-1].append(pts[i])
    for seg in segments:
        if len(seg) < 2:
            continue
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        ax.plot(xs, ys, **kwargs)


def _draw_bridge_edges(ax, bridge_edges, period=None):
    """绘制桥接边 (周期感知)."""
    for edge in bridge_edges:
        _draw_wrapped_line(
            ax, edge.source_config, edge.target_config, period,
            color='lime', linewidth=2.0, alpha=0.9, zorder=5)


def _draw_bridge_boxes(ax, bridge_boxes):
    """高亮桥接 box."""
    for bb in bridge_boxes:
        lo_x, hi_x = bb.joint_intervals[0]
        lo_y, hi_y = bb.joint_intervals[1]
        rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                          linewidth=2.0, edgecolor='lime',
                          facecolor='lime', alpha=0.35, zorder=4)
        ax.add_patch(rect)


def _draw_path(ax, waypoints, color='#00ff00', lw=2.5, label='Path',
               zorder=15, period=None):
    """绘制路径 (周期感知)."""
    # 画节点 markers
    xs = [w[0] for w in waypoints]
    ys = [w[1] for w in waypoints]
    ax.plot(xs, ys, 'o', color=color, markersize=4,
            markeredgecolor='black', markeredgewidth=0.5, zorder=zorder + 1)
    # 画线段 (每段独立, 检测 wrap)
    first_seg = True
    for i in range(len(waypoints) - 1):
        kw = dict(color=color, linewidth=lw, alpha=0.9, zorder=zorder)
        if first_seg:
            kw['label'] = label
            first_seg = False
        _draw_wrapped_line(ax, waypoints[i], waypoints[i + 1], period, **kw)


def _draw_box_sequence(ax, box_seq, boxes, alpha=0.25):
    """高亮 Dijkstra 选出的 box 序列."""
    for bid in box_seq:
        if bid not in boxes:
            continue
        box = boxes[bid]
        lo_x, hi_x = box.joint_intervals[0]
        lo_y, hi_y = box.joint_intervals[1]
        rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                          linewidth=1.5, edgecolor='#00ccff',
                          facecolor='#00ccff', alpha=alpha, zorder=8)
        ax.add_patch(rect)


# ═══════════════════════════════════════════════════════════════════════════
# Geodesic Dijkstra (for periodic C-space)
# ═══════════════════════════════════════════════════════════════════════════

def _dijkstra_box_graph_geodesic(boxes, adj, src, tgt, period):
    """Dijkstra on box graph using geodesic (torus) distance."""
    import heapq
    centers = {}
    for bid, box in boxes.items():
        centers[bid] = np.array(
            [(lo + hi) / 2 for lo, hi in box.joint_intervals])

    half = period / 2.0

    dist_map = {bid: float('inf') for bid in boxes}
    prev_map = {bid: None for bid in boxes}
    dist_map[src] = 0.0
    heap = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist_map[u]:
            continue
        if u == tgt:
            break
        cu = centers[u]
        for v in adj.get(u, set()):
            diff = ((centers[v] - cu) + half) % period - half
            w = float(np.linalg.norm(diff))
            nd = d + w
            if nd < dist_map[v]:
                dist_map[v] = nd
                prev_map[v] = u
                heapq.heappush(heap, (nd, v))

    if dist_map[tgt] == float('inf'):
        return None, float('inf')

    seq = []
    cur = tgt
    while cur is not None:
        seq.append(cur)
        cur = prev_map[cur]
    seq.reverse()
    return seq, dist_map[tgt]


def _setup_ax(ax, extent, title=""):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("q₀ (rad)", fontsize=9)
    ax.set_ylabel("q₁ (rad)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════════
# GIF composition
# ═══════════════════════════════════════════════════════════════════════════

def compose_gif(frames_dir, gif_path, duration_ms=300):
    frame_paths = sorted(Path(frames_dir).glob("step_*.png"))
    if not frame_paths:
        return False
    try:
        from PIL import Image
    except ImportError:
        print("  [WARNING] Pillow not installed, GIF not generated")
        return False

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
              for p in frame_paths]
    durations = [duration_ms] * len(images)
    durations[-1] = 2000  # 最后一帧停留 2s
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False)
    for img in images:
        img.close()
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2DOF 完整流程可视化")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obstacles", type=int, default=8)
    args = parser.parse_args()

    cfg = Viz2DConfig()
    cfg.seed = args.seed
    cfg.n_obstacles = args.obstacles

    rng = np.random.default_rng(cfg.seed)
    robot = load_robot(cfg.robot_name)
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    ndim = 2

    out_dir = _ROOT / "output" / "viz_2dof_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"  2DOF Full Pipeline Visualization")
    print(f"  seed={cfg.seed}, obstacles={cfg.n_obstacles}")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # Phase 0: 场景 & 碰撞底图
    # ══════════════════════════════════════════════════════════════
    print("\n[Phase 0] Building scene & collision map ...")
    scene = build_random_2d_scene(robot, q_start, q_goal, rng, cfg)
    t0 = time.perf_counter()
    cmap_data, extent = scan_collision_map(
        robot, scene, robot.joint_limits, cfg.collision_resolution)
    cmap_ms = (time.perf_counter() - t0) * 1000
    print(f"  Collision map: {cmap_data.shape}, {cmap_ms:.0f}ms")

    # 保存碰撞底图
    fig0, ax0 = plt.subplots(1, 1, figsize=(8, 7))
    _draw_collision_bg(ax0, cmap_data, extent)
    _draw_start_goal(ax0, q_start, q_goal)
    ax0.legend(loc="upper right", fontsize=8)

    # 绘制 C-space 障碍物边界
    obs_list = scene.get_obstacles()
    for obs in obs_list:
        lo_x, lo_y = obs.min_point[0], obs.min_point[1]
        hi_x, hi_y = obs.max_point[0], obs.max_point[1]
        rect = Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                          linewidth=0.8, edgecolor='red',
                          facecolor='none', linestyle='--', alpha=0.5)
        ax0.add_patch(rect)

    _setup_ax(ax0, extent, f"Phase 0: C-space Collision Map ({cfg.n_obstacles} obs)")
    fig0.tight_layout()
    fig0.savefig(out_dir / "phase0_collision_map.png", dpi=cfg.dpi)
    plt.close(fig0)
    print(f"  Saved: phase0_collision_map.png")

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Forest grow (with snapshots)
    # ══════════════════════════════════════════════════════════════
    print("\n[Phase 1] Growing forest ...")
    planner_cfg = PandaGCSConfig()
    planner_cfg.seed = cfg.seed
    planner_cfg.max_boxes = cfg.max_boxes
    planner_cfg.min_box_size = cfg.min_box_size
    planner_cfg.max_consecutive_miss = cfg.max_consecutive_miss
    planner_cfg.goal_bias = cfg.goal_bias
    planner_cfg.guided_sample_ratio = cfg.guided_sample_ratio
    planner_cfg.boundary_expand = cfg.boundary_expand

    sbf_config = make_planner_config(planner_cfg)
    planner = SBFPlanner(robot=robot, scene=scene, config=sbf_config,
                         no_cache=True)  # 2D 不需要大缓存

    t0 = time.perf_counter()
    snapshots, forest_obj = grow_forest_with_snapshots(
        planner, q_start, q_goal,
        seed=cfg.seed, max_miss=cfg.max_consecutive_miss,
        max_boxes=cfg.max_boxes, min_box_size=cfg.min_box_size,
        snapshot_every=cfg.snapshot_every,
    )
    grow_ms = (time.perf_counter() - t0) * 1000
    n_before_coarsen = len(forest_obj.boxes)
    print(f"  Grown: {n_before_coarsen} boxes, {len(snapshots)} snapshots, "
          f"{grow_ms:.0f}ms")

    # 渲染每帧
    print("  Rendering growth frames ...")
    for idx, (n_boxes, boxes_snap, new_id) in enumerate(snapshots):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        _draw_collision_bg(ax, cmap_data, extent)
        _draw_boxes(ax, boxes_snap, highlight_ids={new_id} if new_id >= 0 else set())
        _draw_start_goal(ax, q_start, q_goal)
        _setup_ax(ax, extent,
                  f"Phase 1: Forest Growth — {n_boxes} boxes")
        fig.tight_layout()
        fig.savefig(frames_dir / f"step_{idx:04d}.png", dpi=cfg.dpi)
        plt.close(fig)

    # GIF
    gif_path = out_dir / "phase1_growth.gif"
    gif_ok = compose_gif(frames_dir, gif_path, cfg.gif_frame_ms)
    print(f"  GIF: {'OK' if gif_ok else 'FAILED'} ({len(snapshots)} frames)")

    # 最终 forest 静态图
    final_boxes_pre = snapshots[-1][1] if snapshots else {}
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7))
    _draw_collision_bg(ax1, cmap_data, extent)
    _draw_boxes(ax1, final_boxes_pre, color_by="volume")
    _draw_start_goal(ax1, q_start, q_goal)
    ax1.legend(loc="upper right", fontsize=8)
    _setup_ax(ax1, extent,
              f"Phase 1: Forest Complete — {n_before_coarsen} boxes")
    fig1.tight_layout()
    fig1.savefig(out_dir / "phase1_forest_final.png", dpi=cfg.dpi)
    plt.close(fig1)

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Coarsen
    # ══════════════════════════════════════════════════════════════
    print("\n[Phase 2] Coarsening ...")
    t0 = time.perf_counter()
    coarsen_stats = coarsen_forest(
        tree=planner.hier_tree, forest=forest_obj,
        obstacles=planner.obstacles, safety_margin=0.0,
        max_rounds=cfg.coarsen_max_rounds)
    coarsen_ms = (time.perf_counter() - t0) * 1000
    n_after_coarsen = len(forest_obj.boxes)
    print(f"  Coarsen: {n_before_coarsen} → {n_after_coarsen} boxes "
          f"({coarsen_stats.n_merges} merges, {coarsen_ms:.0f}ms)")

    boxes_after_coarsen = {bid: BoxNode(
        node_id=b.node_id,
        joint_intervals=[tuple(iv) for iv in b.joint_intervals],
        seed_config=b.seed_config.copy() if b.seed_config is not None else None,
        volume=b.volume)
        for bid, b in forest_obj.boxes.items()}

    # Before/After coarsen 对比图
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 7))
    _draw_collision_bg(ax2a, cmap_data, extent)
    _draw_boxes(ax2a, final_boxes_pre, color_by="volume")
    _draw_start_goal(ax2a, q_start, q_goal)
    _setup_ax(ax2a, extent, f"Before Coarsen — {n_before_coarsen} boxes")

    _draw_collision_bg(ax2b, cmap_data, extent)
    _draw_boxes(ax2b, boxes_after_coarsen, color_by="volume")
    _draw_start_goal(ax2b, q_start, q_goal)
    _setup_ax(ax2b, extent, f"After Coarsen — {n_after_coarsen} boxes")

    fig2.suptitle("Phase 2: Coarsen", fontsize=12, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(out_dir / "phase2_coarsen.png", dpi=cfg.dpi)
    plt.close(fig2)
    print(f"  Saved: phase2_coarsen.png")

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Adjacency + Islands + Bridge
    # ══════════════════════════════════════════════════════════════
    print("\n[Phase 3] Adjacency, islands & bridging ...")
    boxes = forest_obj.boxes
    adj, uf, islands = _build_adjacency_and_islands(boxes)
    n_islands_before = len(islands)
    print(f"  Islands: {n_islands_before}, boxes: {len(boxes)}")

    src = find_box_containing(q_start, boxes)
    tgt = find_box_containing(q_goal, boxes)

    # 周期 (所有 revolute joint 假设同周期)
    jl = robot.joint_limits[0]
    period = float(jl[1] - jl[0])  # 2π

    bridge_edges = []
    bridge_boxes_list = []

    if src is not None and tgt is not None and not uf.same(src, tgt):
        print(f"  s-t disconnected, bridging ...")
        t0 = time.perf_counter()
        bridge_result = bridge_islands(
            boxes=boxes,
            collision_checker=planner.collision_checker,
            segment_resolution=0.03,
            max_pairs_per_island_pair=10,
            max_rounds=5,
            period=period,
            hier_tree=planner.hier_tree,
            obstacles=planner.obstacles,
            forest=forest_obj,
            min_box_size=cfg.min_box_size,
            n_bridge_seeds=7,
            min_island_size=cfg.min_island_size,
            precomputed_uf=uf,
            precomputed_islands=islands,
            target_pair=(src, tgt),
        )
        bridge_edges, final_islands, _, bridge_boxes_list, discarded = bridge_result
        bridge_ms = (time.perf_counter() - t0) * 1000
        n_islands_after = len(final_islands)
        print(f"  Bridge: {n_islands_before} → {n_islands_after} islands, "
              f"{len(bridge_edges)} edges, {len(bridge_boxes_list)} boxes "
              f"({bridge_ms:.0f}ms)")

        # 重建 adj
        boxes = forest_obj.boxes
        adj, uf, islands = _build_adjacency_and_islands(boxes)
        # add bridge edges
        for e in bridge_edges:
            s_bid = find_box_containing(e.source_config, boxes)
            t_bid = find_box_containing(e.target_config, boxes)
            if s_bid is not None and t_bid is not None:
                adj.setdefault(s_bid, set()).add(t_bid)
                adj.setdefault(t_bid, set()).add(s_bid)
                uf.union(s_bid, t_bid)
    else:
        print(f"  s-t already connected (or not found)")
        islands = [set(boxes.keys())]

    # Island map
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 7))
    _draw_collision_bg(ax3, cmap_data, extent)
    _draw_islands(ax3, boxes, islands)
    _draw_bridge_boxes(ax3, bridge_boxes_list)
    _draw_bridge_edges(ax3, bridge_edges, period=period)
    _draw_start_goal(ax3, q_start, q_goal)
    ax3.legend(loc="upper right", fontsize=8)
    _setup_ax(ax3, extent,
              f"Phase 3: Islands & Bridge — {len(islands)} islands, "
              f"{len(bridge_edges)} bridges")
    fig3.tight_layout()
    fig3.savefig(out_dir / "phase3_islands_bridge.png", dpi=cfg.dpi)
    plt.close(fig3)
    print(f"  Saved: phase3_islands_bridge.png")

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Path planning (Dijkstra)
    # ══════════════════════════════════════════════════════════════
    print("\n[Phase 4] Path planning (Dijkstra) ...")
    src = find_box_containing(q_start, boxes)
    tgt = find_box_containing(q_goal, boxes)

    path_found = False
    waypoints = []
    box_seq = []
    raw_dist = 0.0
    refined_cost = 0.0

    # 构建 bridge edge 的 box-pair 映射, 用于 Dijkstra 后插入桥接 waypoint
    bridge_edge_map = {}  # (bid_a, bid_b) -> Edge
    for e in bridge_edges:
        s_bid = find_box_containing(e.source_config, boxes)
        t_bid = find_box_containing(e.target_config, boxes)
        if s_bid is not None and t_bid is not None:
            bridge_edge_map[(s_bid, t_bid)] = e
            bridge_edge_map[(t_bid, s_bid)] = e

    if src is not None and tgt is not None:
        # 使用 geodesic 距离的 Dijkstra
        box_seq_result, raw_dist = _dijkstra_box_graph_geodesic(
            boxes, adj, src, tgt, period)
        if box_seq_result is not None:
            box_seq = box_seq_result

            # waypoints: 在桥接处插入桥接端点, 其余用 box center
            raw_waypoints = [q_start.copy()]
            for k in range(len(box_seq) - 1):
                bid_cur = box_seq[k]
                bid_nxt = box_seq[k + 1]
                # 检查是否为桥接边
                be = bridge_edge_map.get((bid_cur, bid_nxt))
                if be is not None:
                    # 确定方向: source_config 属于 bid_cur 还是 bid_nxt
                    s_bid = find_box_containing(be.source_config, boxes)
                    if s_bid == bid_cur:
                        raw_waypoints.append(be.source_config.copy())
                        raw_waypoints.append(be.target_config.copy())
                    else:
                        raw_waypoints.append(be.target_config.copy())
                        raw_waypoints.append(be.source_config.copy())
                elif k + 1 < len(box_seq) - 1:
                    # 普通连接: 用下一个 box 的 center
                    box_nxt = boxes[bid_nxt]
                    c = np.array([(lo + hi) / 2
                                  for lo, hi in box_nxt.joint_intervals])
                    raw_waypoints.append(c)
            raw_waypoints.append(q_goal.copy())

            # 计算 geodesic 路径长度
            half_p = period / 2.0
            refined_cost = 0.0
            for k in range(len(raw_waypoints) - 1):
                diff_geo = ((raw_waypoints[k+1] - raw_waypoints[k])
                            + half_p) % period - half_p
                refined_cost += float(np.linalg.norm(diff_geo))
            waypoints = raw_waypoints
            path_found = True
            print(f"  Path found: {len(box_seq)} boxes, "
                  f"raw_dist={raw_dist:.3f}, geodesic_cost={refined_cost:.3f}, "
                  f"{len(waypoints)} waypoints")
        else:
            print(f"  Dijkstra: no path found (disconnected)")
    else:
        print(f"  Start or goal not in any box "
              f"(src={src}, tgt={tgt})")

    # Path visualization
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 7))
    _draw_collision_bg(ax4, cmap_data, extent)
    _draw_boxes(ax4, boxes, color_by="uniform", alpha=0.15, lw=0.3)
    if box_seq:
        _draw_box_sequence(ax4, box_seq, boxes, alpha=0.30)
    if waypoints:
        _draw_path(ax4, waypoints, color='#00ff00', lw=2.5,
                   label='Refined Path', period=period)
    _draw_start_goal(ax4, q_start, q_goal)
    ax4.legend(loc="upper right", fontsize=8)
    status = f"cost={refined_cost:.3f}" if path_found else "NO PATH"
    _setup_ax(ax4, extent,
              f"Phase 4: Dijkstra Path — {len(box_seq)} boxes, {status}")
    fig4.tight_layout()
    fig4.savefig(out_dir / "phase4_path.png", dpi=cfg.dpi)
    plt.close(fig4)
    print(f"  Saved: phase4_path.png")

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Overview (all-in-one)
    # ══════════════════════════════════════════════════════════════
    print("\n[Phase 5] Generating overview ...")
    fig5, axes = plt.subplots(2, 3, figsize=(22, 14))

    # (0,0) collision map
    ax = axes[0, 0]
    _draw_collision_bg(ax, cmap_data, extent)
    _draw_start_goal(ax, q_start, q_goal)
    _setup_ax(ax, extent, f"① C-space ({cfg.n_obstacles} obs)")

    # (0,1) forest before coarsen
    ax = axes[0, 1]
    _draw_collision_bg(ax, cmap_data, extent)
    _draw_boxes(ax, final_boxes_pre, color_by="volume")
    _draw_start_goal(ax, q_start, q_goal)
    _setup_ax(ax, extent, f"② Forest — {n_before_coarsen} boxes")

    # (0,2) after coarsen
    ax = axes[0, 2]
    _draw_collision_bg(ax, cmap_data, extent)
    _draw_boxes(ax, boxes_after_coarsen, color_by="volume")
    _draw_start_goal(ax, q_start, q_goal)
    _setup_ax(ax, extent, f"③ Coarsen — {n_after_coarsen} boxes")

    # (1,0) islands
    ax = axes[1, 0]
    _draw_collision_bg(ax, cmap_data, extent)
    _draw_islands(ax, boxes, islands)
    _draw_bridge_boxes(ax, bridge_boxes_list)
    _draw_bridge_edges(ax, bridge_edges, period=period)
    _draw_start_goal(ax, q_start, q_goal)
    _setup_ax(ax, extent, f"④ Islands & Bridge — {len(islands)} isl")

    # (1,1) path with boxes
    ax = axes[1, 1]
    _draw_collision_bg(ax, cmap_data, extent)
    _draw_boxes(ax, boxes, color_by="uniform", alpha=0.12, lw=0.2)
    if box_seq:
        _draw_box_sequence(ax, box_seq, boxes, alpha=0.35)
    if waypoints:
        _draw_path(ax, waypoints, color='#00ff00', lw=2.5, period=period)
    _draw_start_goal(ax, q_start, q_goal)
    _setup_ax(ax, extent,
              f"⑤ Dijkstra — {len(box_seq)} boxes, {status}")

    # (1,2) clean path only
    ax = axes[1, 2]
    _draw_collision_bg(ax, cmap_data, extent)
    if waypoints:
        _draw_path(ax, waypoints, color='#00ff00', lw=3.0,
                   label=f'Path (cost={refined_cost:.2f})',
                   period=period)
    _draw_start_goal(ax, q_start, q_goal)
    ax.legend(loc="upper right", fontsize=8)
    _setup_ax(ax, extent, f"⑥ Final Path")

    fig5.suptitle(
        f"SBF 2DOF Pipeline — seed={cfg.seed}, "
        f"{cfg.n_obstacles} obs, {n_after_coarsen} boxes, "
        f"grow={grow_ms:.0f}ms, coarsen={coarsen_ms:.0f}ms",
        fontsize=13, fontweight='bold')
    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    fig5.savefig(out_dir / "phase5_overview.png", dpi=cfg.dpi)
    plt.close(fig5)
    print(f"  Saved: phase5_overview.png")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    summary = {
        "seed": cfg.seed,
        "n_obstacles": cfg.n_obstacles,
        "boxes_before_coarsen": n_before_coarsen,
        "boxes_after_coarsen": n_after_coarsen,
        "coarsen_merges": coarsen_stats.n_merges,
        "n_islands": len(islands),
        "n_bridge_edges": len(bridge_edges),
        "n_bridge_boxes": len(bridge_boxes_list),
        "path_found": path_found,
        "path_cost": float(refined_cost) if path_found else None,
        "path_waypoints": len(waypoints),
        "path_boxes": len(box_seq),
        "grow_ms": grow_ms,
        "coarsen_ms": coarsen_ms,
        "collision_map_ms": cmap_ms,
        "n_snapshots": len(snapshots),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Output: {out_dir}")
    print(f"  Files:")
    for p in sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.gif")):
        print(f"    {p.name}")
    print(f"  Summary: summary.json")
    print(f"  Path: {'FOUND' if path_found else 'NOT FOUND'}"
          f"{f', cost={refined_cost:.3f}' if path_found else ''}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
