import json
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from planner.box_planner import BoxPlanner
from planner.models import PlannerConfig, gmean_edge_length
from forest.models import BoxNode
from forest.connectivity import find_islands, bridge_islands
from common.output import make_output_dir


# ---------------------------------------------------------------------------
# Visualization Config (所有可配置超参数集中在此)
# ---------------------------------------------------------------------------

@dataclass
class VizConfig:
    """2DOF forest expansion 可视化超参数"""

    # 随机种子
    seed: int = 20260219

    # 场景
    n_obstacles: int = 8
    robot_name: str = "2dof_planar"
    q_start: List[float] = field(default_factory=lambda: [0.8 * 3.141592653589793, 0.2])
    q_goal: List[float] = field(default_factory=lambda: [-0.7 * 3.141592653589793, -0.4])

    # 终止条件
    max_consecutive_miss: int = 20   # 连续 N 次采样未成功则停止

    # 采样策略
    goal_bias: float = 0.15          # goal 偏向采样概率
    guided_sample_ratio: float = 0.6 # KD 树引导采样概率
    min_box_size: float = 0.01     # 最小 box 几何均值边长

    # 可视化
    snapshot_every: int = 3          # 每添加 N 个 box 截一帧
    gif_frame_ms: int = 300          # GIF 每帧持续时间 (ms)
    collision_map_resolution: float = 0.03  # 碰撞底图分辨率
    dpi: int = 140                   # 输出图像 DPI

    # 随机场景生成
    obs_cx_range: Tuple[float, float] = (-1.6, 1.6)
    obs_cy_range: Tuple[float, float] = (-1.6, 1.6)
    obs_w_range: Tuple[float, float] = (0.25, 0.65)
    obs_h_range: Tuple[float, float] = (0.25, 0.65)


# ---------------------------------------------------------------------------
# Collision map
# ---------------------------------------------------------------------------

def scan_collision_map(
    robot,
    scene: Scene,
    joint_limits,
    resolution: float = 0.03,
) -> tuple:
    """扫描 2DOF 关节空间碰撞区域。"""
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_forest_snapshot(
    boxes: Dict[int, BoxNode],
    adjacency: Dict[int, Set[int]],
    collision_map,
    extent,
    title: str,
    new_box_id: int = -1,
):
    """在碰撞底图上绘制当前 forest 快照。

    最新添加的 box (`new_box_id`) 用醒目的橙色高亮。
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.imshow(collision_map, origin="lower", extent=extent, cmap="Reds", alpha=0.35, aspect="auto")

    degrees = {bid: len(adjacency.get(bid, set())) for bid in boxes}
    max_deg = max(degrees.values()) if degrees else 1
    cmap_boxes = plt.cm.viridis

    for bid, box in boxes.items():
        lo_x, hi_x = box.joint_intervals[0]
        lo_y, hi_y = box.joint_intervals[1]

        if bid == new_box_id:
            # 新 box 高亮
            ec, fc, alpha = "#ff6600", "#ff9933", 0.45
        else:
            deg = degrees.get(bid, 0)
            c = cmap_boxes(deg / max(max_deg, 1))
            ec, fc, alpha = c, c, 0.28

        rect = Rectangle(
            (lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
            linewidth=0.6, edgecolor=ec, facecolor=fc, alpha=alpha,
        )
        ax.add_patch(rect)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("q0 (rad)")
    ax.set_ylabel("q1 (rad)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    return fig


# ---------------------------------------------------------------------------
# Island 检测与可视化
# ---------------------------------------------------------------------------

def plot_island_map(
    boxes: Dict[int, BoxNode],
    islands: List[Set[int]],
    bridge_edges: list,
    bridge_boxes: list,
    collision_map,
    extent,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    title: str = "Island Map",
):
    """绘制岛检测结果：不同岛不同颜色，bridge box 高亮，segment 用绿线。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.imshow(collision_map, origin="lower", extent=extent,
              cmap="Reds", alpha=0.35, aspect="auto")

    n_islands = len(islands)
    cmap_islands = plt.cm.tab20

    # 建立 node_id -> island_idx 映射
    node_island = {}
    for idx, island in enumerate(islands):
        for bid in island:
            node_island[bid] = idx

    bridge_box_ids = {b.node_id for b in bridge_boxes}

    # 绘制 boxes，按岛着色
    for bid, box in boxes.items():
        lo_x, hi_x = box.joint_intervals[0]
        lo_y, hi_y = box.joint_intervals[1]
        isl_idx = node_island.get(bid, 0)
        c = cmap_islands(isl_idx / max(n_islands, 1))
        if bid in bridge_box_ids:
            # bridge box: 高亮边框
            rect = Rectangle(
                (lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                linewidth=2.0, edgecolor="lime", facecolor=c, alpha=0.50,
                zorder=4,
            )
        else:
            rect = Rectangle(
                (lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                linewidth=0.6, edgecolor=c, facecolor=c, alpha=0.30,
            )
        ax.add_patch(rect)

    # 绘制 bridge edges (segment fallback)
    for edge in bridge_edges:
        ax.plot(
            [edge.source_config[0], edge.target_config[0]],
            [edge.source_config[1], edge.target_config[1]],
            color="lime", linewidth=2.0, alpha=0.9, zorder=5,
        )

    # 标记 start / goal
    ax.plot(q_start[0], q_start[1], 'o', color='cyan', markersize=8,
            markeredgecolor='black', markeredgewidth=1.0, zorder=10, label='start')
    ax.plot(q_goal[0], q_goal[1], '*', color='yellow', markersize=12,
            markeredgecolor='black', markeredgewidth=1.0, zorder=10, label='goal')

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("q0 (rad)")
    ax.set_ylabel("q1 (rad)")
    n_box_br = len(bridge_boxes)
    n_seg_br = len(bridge_edges)
    ax.set_title(
        f"{title}  |  {n_islands} islands, "
        f"{n_box_br} box-bridges, {n_seg_br} seg-bridges",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    return fig


# ---------------------------------------------------------------------------
# GIF composition
# ---------------------------------------------------------------------------

def compose_gif(frames_dir: Path, gif_path: Path, duration_ms: int = 350) -> bool:
    """将 frames 目录中的 PNG 帧合成为 GIF（最后一帧停留更久）。"""
    frame_paths = sorted(frames_dir.glob("step_*.png"))
    if not frame_paths:
        return False

    try:
        from PIL import Image
    except Exception:
        return False

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    if not images:
        return False

    # 最后一帧停留 1.5 秒
    durations = [duration_ms] * len(images)
    durations[-1] = 1500

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    for img in images:
        img.close()

    return True


# ---------------------------------------------------------------------------
# Random scene generation
# ---------------------------------------------------------------------------

def build_random_scene(
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    rng: np.random.Generator,
    cfg: VizConfig,
    max_trials: int = 200,
) -> Scene:
    """生成随机障碍物场景，确保起终点可行且直连路径被阻挡。"""
    for _ in range(max_trials):
        scene = Scene()

        for i in range(cfg.n_obstacles):
            cx = float(rng.uniform(*cfg.obs_cx_range))
            cy = float(rng.uniform(*cfg.obs_cy_range))
            w = float(rng.uniform(*cfg.obs_w_range))
            h = float(rng.uniform(*cfg.obs_h_range))

            lo = [cx - w * 0.5, cy - h * 0.5]
            hi = [cx + w * 0.5, cy + h * 0.5]
            scene.add_obstacle(lo, hi, name=f"obs_{i}")

        checker = CollisionChecker(robot=robot, scene=scene)

        start_collide = checker.check_config_collision(q_start)
        goal_collide = checker.check_config_collision(q_goal)
        direct_collide = checker.check_segment_collision(q_start, q_goal, resolution=0.03)

        if (not start_collide) and (not goal_collide) and direct_collide:
            return scene

    raise RuntimeError("未能生成满足条件的随机场景，请调整随机种子或障碍物参数")


# ---------------------------------------------------------------------------
# Config -> PlannerConfig
# ---------------------------------------------------------------------------

def make_planner_config(cfg: VizConfig) -> PlannerConfig:
    return PlannerConfig(
        max_iterations=999999,       # 不作为终止条件
        max_box_nodes=999999,        # 不作为终止条件
        seed_batch_size=5,
        min_box_size=cfg.min_box_size,
        goal_bias=cfg.goal_bias,
        guided_sample_ratio=cfg.guided_sample_ratio,
        expansion_resolution=0.03,
        max_expansion_rounds=3,
        segment_collision_resolution=0.03,
        connection_max_attempts=50,
        connection_radius=3.0,
        path_shortcut_iters=100,
        verbose=False,
        forest_path=None,
    )


# ---------------------------------------------------------------------------
# Forest expansion with per-box snapshots
# ---------------------------------------------------------------------------

Snapshot = Tuple[int, Dict[int, BoxNode], Dict[int, Set[int]], int]  # (n_boxes, boxes, adj, new_id)


def _try_add_seed(
    planner: BoxPlanner,
    forest,
    q_seed: np.ndarray,
) -> int:
    """尝试将 q_seed 扩展为一个 free box 并加入 forest。

    Returns:
        新添加的 box 的 node_id，若未添加则返回 -1。
    """
    if planner.hier_tree.is_occupied(q_seed):
        return -1

    nid = forest.allocate_id()
    ffb_result = planner.hier_tree.find_free_box(
        q_seed,
        planner.obstacles,
        mark_occupied=True,
        forest_box_id=nid,
    )
    if ffb_result is None:
        return -1

    ivs = ffb_result.intervals
    vol = 1.0
    for lo, hi in ivs:
        vol *= max(hi - lo, 0.0)
    if gmean_edge_length(vol, planner._n_dims) < planner.config.min_box_size:
        return -1
    if ffb_result.absorbed_box_ids:
        forest.remove_boxes(ffb_result.absorbed_box_ids)

    box = BoxNode(
        node_id=nid,
        joint_intervals=ivs,
        seed_config=q_seed.copy(),
        volume=vol,
    )
    forest.add_box_direct(box)
    return nid


def _snapshot(forest, new_id: int) -> Snapshot:
    """深拷贝当前 forest 状态。"""
    boxes_copy = {}
    for bid, b in forest.boxes.items():
        boxes_copy[bid] = BoxNode(
            node_id=b.node_id,
            joint_intervals=[tuple(iv) for iv in b.joint_intervals],
            seed_config=b.seed_config.copy(),
            volume=b.volume,
            parent_id=b.parent_id,
            tree_id=b.tree_id,
        )
    adj_copy = {k: set(v) for k, v in forest.adjacency.items()}
    return (forest.n_boxes, boxes_copy, adj_copy, new_id)


def grow_forest_with_snapshots(
    planner: BoxPlanner,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    seed: int,
    snapshot_every: int = 1,
    max_consecutive_miss: int = 50,
) -> Tuple[list, int, str]:
    """执行单次 forest 拓展，每添加 snapshot_every 个 box 记录一次快照。

    终止条件：连续 max_consecutive_miss 次采样都未能添加新 box 时停止。

    Returns:
        (snapshots, total_attempts, exit_reason)
    """
    rng = np.random.default_rng(seed)
    forest = planner._load_or_create_forest()
    forest.hier_tree = planner.hier_tree

    snapshots: List[Snapshot] = []
    added_since_snap = 0

    def maybe_snap(new_id: int, force: bool = False):
        nonlocal added_since_snap
        added_since_snap += 1
        if force or added_since_snap >= snapshot_every:
            snapshots.append(_snapshot(forest, new_id))
            added_since_snap = 0

    # 1) 种子点 (start / goal)
    for q_seed in [q_start, q_goal]:
        nid = _try_add_seed(planner, forest, q_seed)
        if nid >= 0:
            maybe_snap(nid)

    # 2) 随机扩展（唯一终止条件：连续 miss 达到阈值）
    attempts = 0
    consecutive_miss = 0
    while consecutive_miss < max_consecutive_miss:
        attempts += 1
        q_seed = planner._sample_seed(q_start, q_goal, rng)
        if q_seed is None:
            consecutive_miss += 1
            continue
        nid = _try_add_seed(planner, forest, q_seed)
        if nid >= 0:
            consecutive_miss = 0
            maybe_snap(nid)
        else:
            consecutive_miss += 1

    exit_reason = f"consecutive_miss={max_consecutive_miss}"

    # 确保末尾有快照
    if not snapshots or snapshots[-1][0] != forest.n_boxes:
        snapshots.append(_snapshot(forest, -1))

    return snapshots, attempts, exit_reason, forest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = VizConfig()  # 修改超参数只需改这里
    rng = np.random.default_rng(cfg.seed)

    robot = load_robot(cfg.robot_name)
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)

    # ---------- 场景 ----------
    scene = build_random_scene(
        robot=robot,
        q_start=q_start,
        q_goal=q_goal,
        rng=rng,
        cfg=cfg,
    )

    out_dir = make_output_dir("visualizations", "random_2dof_forest_expansion")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 碰撞底图 ----------
    print("scanning collision map ...")
    collision_map, collision_extent = scan_collision_map(
        robot=robot,
        scene=scene,
        joint_limits=robot.joint_limits,
        resolution=cfg.collision_map_resolution,
    )

    scene_json = out_dir / "scene.json"
    scene.to_json(str(scene_json))

    # ---------- 单次 episode 拓展 ----------
    print(f"expanding forest (max_consecutive_miss={cfg.max_consecutive_miss}) ...")
    planner_cfg = make_planner_config(cfg)
    planner = BoxPlanner(robot=robot, scene=scene, config=planner_cfg)
    snapshots, total_attempts, exit_reason, forest_obj = grow_forest_with_snapshots(
        planner=planner,
        q_start=q_start,
        q_goal=q_goal,
        seed=cfg.seed,
        snapshot_every=cfg.snapshot_every,
        max_consecutive_miss=cfg.max_consecutive_miss,
    )
    print(f"  {len(snapshots)} snapshots, {total_attempts} attempts, exit: {exit_reason}")
    final_boxes = snapshots[-1][0] if snapshots else 0
    hit_rate = final_boxes / max(total_attempts, 1) * 100
    print(f"  final_boxes={final_boxes}, hit_rate={hit_rate:.1f}%")

    # ---------- 岛检测 & 桥接 ----------
    print("detecting islands & bridging ...")
    final_boxes_dict = snapshots[-1][1] if snapshots else {}
    # period 从实际 joint_limits 计算（避免浮点截断导致边界 box 不重叠）
    jl = robot.joint_limits[0]
    period = float(jl[1] - jl[0])
    bridge_edges, final_islands, n_islands_before, bridge_boxes, discarded_islands = bridge_islands(
        boxes=final_boxes_dict,
        collision_checker=planner.collision_checker,
        segment_resolution=planner_cfg.segment_collision_resolution,
        max_pairs_per_island_pair=10,
        max_rounds=5,
        period=period,
        hier_tree=planner.hier_tree,
        obstacles=planner.obstacles,
        forest=forest_obj,
        min_box_size=planner_cfg.min_box_size,
        n_bridge_seeds=7,
        min_island_size=0.5,
    )
    n_islands_after = len(final_islands)
    n_box_bridges = len(bridge_boxes)
    n_seg_bridges = len(bridge_edges)
    n_discarded = len(discarded_islands)
    n_discarded_boxes = sum(len(s) for s in discarded_islands)
    print(f"  islands: {n_islands_before} -> {n_islands_after}, "
          f"box_bridges: {n_box_bridges}, segment_bridges: {n_seg_bridges}, "
          f"discarded: {n_discarded} islands ({n_discarded_boxes} boxes)")

    # ---------- 渲染每帧 ----------
    print("rendering frames ...")
    frame_records = []
    for idx, (n_boxes, boxes, adj, new_id) in enumerate(snapshots):
        title = f"Forest Expansion | boxes={n_boxes}"
        fig = plot_forest_snapshot(
            boxes=boxes,
            adjacency=adj,
            collision_map=collision_map,
            extent=collision_extent,
            title=title,
            new_box_id=new_id,
        )
        frame_path = frames_dir / f"step_{idx:04d}.png"
        if fig is not None:
            fig.savefig(frame_path, dpi=cfg.dpi, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)

        frame_records.append({
            "frame_idx": idx,
            "n_boxes": n_boxes,
            "new_box_id": new_id,
            "frame": str(frame_path),
        })

    # ---------- GIF ----------
    gif_path = out_dir / "expansion.gif"
    gif_ok = compose_gif(frames_dir=frames_dir, gif_path=gif_path, duration_ms=cfg.gif_frame_ms)

    # ---------- 连接后的 forest 图 ----------
    connected_fig = plot_island_map(
        boxes=final_boxes_dict,
        islands=final_islands,
        bridge_edges=bridge_edges,
        bridge_boxes=bridge_boxes,
        collision_map=collision_map,
        extent=collision_extent,
        q_start=q_start,
        q_goal=q_goal,
        title="Island Map (after bridging)",
    )
    connected_png = out_dir / "connected_forest.png"
    if connected_fig is not None:
        import matplotlib.pyplot as plt
        connected_fig.savefig(connected_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(connected_fig)
        print(f"  island map: {connected_png}")

    # ---------- 碰撞底图单独保存 ----------
    collision_map_png = out_dir / "collision_map.png"
    try:
        import matplotlib.pyplot as plt
        fig_c, ax_c = plt.subplots(1, 1, figsize=(8, 6))
        ax_c.imshow(collision_map, origin="lower", extent=collision_extent, cmap="Reds", alpha=0.85, aspect="auto")
        ax_c.set_xlabel("q0 (rad)")
        ax_c.set_ylabel("q1 (rad)")
        ax_c.set_title("C-space collision map")
        ax_c.grid(True, alpha=0.2)
        fig_c.savefig(collision_map_png, dpi=140, bbox_inches="tight")
        plt.close(fig_c)
    except Exception:
        pass

    # ---------- Summary ----------
    summary = {
        "config": {
            "seed": cfg.seed,
            "n_obstacles": cfg.n_obstacles,
            "max_consecutive_miss": cfg.max_consecutive_miss,
            "goal_bias": cfg.goal_bias,
            "guided_sample_ratio": cfg.guided_sample_ratio,
            "min_box_size": cfg.min_box_size,
            "snapshot_every": cfg.snapshot_every,
        },
        "total_attempts": total_attempts,
        "exit_reason": exit_reason,
        "total_snapshots": len(snapshots),
        "final_n_boxes": snapshots[-1][0] if snapshots else 0,
        "n_islands_before": n_islands_before,
        "n_islands_after": n_islands_after,
        "n_box_bridges": n_box_bridges,
        "n_segment_bridges": n_seg_bridges,
        "n_discarded_islands": n_discarded,
        "n_discarded_boxes": n_discarded_boxes,
        "q_start": q_start.tolist(),
        "q_goal": q_goal.tolist(),
        "scene_json": str(scene_json),
        "frames": frame_records,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- README ----------
    md_lines = [
        "# Random Obstacles 2DOF Box Forest Expansion (single episode)",
        "",
        f"- seed: {cfg.seed}",
        f"- n_obstacles: {cfg.n_obstacles}",
        f"- max_consecutive_miss: {cfg.max_consecutive_miss}",
        f"- goal_bias: {cfg.goal_bias}",
        f"- guided_sample_ratio: {cfg.guided_sample_ratio}",
        f"- islands: {n_islands_before} -> {n_islands_after} (discarded {n_discarded})",
        f"- exit_reason: {exit_reason}",
        f"- snapshot_every: {cfg.snapshot_every}",
        f"- total_attempts: {total_attempts}",
        f"- total_snapshots: {len(snapshots)}",
        f"- q_start: {q_start.tolist()}",
        f"- q_goal: {q_goal.tolist()}",
        f"- scene: {scene_json.name}",
        f"- collision_map: {collision_map_png.name}",
        f"- gif: {gif_path.name if gif_ok else 'not generated (Pillow missing or no frames)'}",
        "",
        "## Snapshots",
    ]

    for rec in frame_records:
        md_lines.append(
            f"- frame {rec['frame_idx']:03d}: boxes={rec['n_boxes']}, "
            f"new_box_id={rec['new_box_id']}, file={Path(rec['frame']).name}"
        )

    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\noutput_dir = {out_dir}")
    print(f"frames     = {frames_dir}  ({len(frame_records)} frames)")
    print(f"summary    = {summary_path}")
    if gif_ok:
        print(f"gif        = {gif_path}")
    else:
        print("gif        = NOT_GENERATED")


if __name__ == "__main__":
    main()
