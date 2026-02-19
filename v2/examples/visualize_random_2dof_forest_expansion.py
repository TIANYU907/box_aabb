import json
from pathlib import Path

import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from planner.box_rrt import BoxRRT
from planner.models import PlannerConfig, PlannerResult, gmean_edge_length
from forest.models import BoxNode
from common.output import make_output_dir


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


def plot_forest_with_collision_map(result: PlannerResult, collision_map, extent, title: str):
    """在碰撞底图上叠加 BoxForest。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    forest = result.forest
    if forest is None:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.imshow(collision_map, origin="lower", extent=extent, cmap="Reds", alpha=0.35, aspect="auto")

    boxes = forest.boxes
    adjacency = forest.adjacency
    degrees = {bid: len(adjacency.get(bid, set())) for bid in boxes}
    max_deg = max(degrees.values()) if degrees else 1
    cmap_boxes = plt.cm.viridis

    for bid, box in boxes.items():
        lo_x, hi_x = box.joint_intervals[0]
        lo_y, hi_y = box.joint_intervals[1]
        deg = degrees.get(bid, 0)
        color = cmap_boxes(deg / max(max_deg, 1))
        rect = Rectangle(
            (lo_x, lo_y),
            hi_x - lo_x,
            hi_y - lo_y,
            linewidth=0.6,
            edgecolor=color,
            facecolor=color,
            alpha=0.28,
        )
        ax.add_patch(rect)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("q0 (rad)")
    ax.set_ylabel("q1 (rad)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    return fig


def compose_gif(frames_dir: Path, gif_path: Path, duration_ms: int = 450) -> bool:
    """将 frames 目录中的 PNG 帧合成为 GIF。"""
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

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    for img in images:
        img.close()

    return True


def build_random_scene(
    robot,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    rng: np.random.Generator,
    n_obs: int,
    max_trials: int = 200,
) -> Scene:
    """生成随机障碍物场景，确保起终点可行且直连路径被阻挡。"""
    for _ in range(max_trials):
        scene = Scene()

        for i in range(n_obs):
            cx = float(rng.uniform(-1.6, 1.6))
            cy = float(rng.uniform(-1.6, 1.6))
            w = float(rng.uniform(0.25, 0.65))
            h = float(rng.uniform(0.25, 0.65))

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


def make_config(max_boxes: int) -> PlannerConfig:
    return PlannerConfig(
        max_iterations=max(120, max_boxes * 4),
        max_box_nodes=max_boxes,
        seed_batch_size=5,
        min_box_size=0.001,
        goal_bias=0.15,
        expansion_resolution=0.03,
        max_expansion_rounds=3,
        segment_collision_resolution=0.03,
        connection_max_attempts=50,
        connection_radius=3.0,
        path_shortcut_iters=100,
        verbose=False,
        forest_path=None,
    )


def grow_forest_snapshot(
    planner: BoxRRT,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    seed: int,
) -> tuple:
    """仅执行 BoxForest 拓展（不做图搜索），用于可视化拓展过程。"""
    rng = np.random.default_rng(seed)
    forest = planner._load_or_create_forest()
    forest.hier_tree = planner.hier_tree

    added = 0
    attempts = 0

    for q_seed in [q_start, q_goal]:
        if forest.n_boxes >= planner.config.max_box_nodes:
            break
        if planner.hier_tree.is_occupied(q_seed):
            continue
        nid = forest.allocate_id()
        ffb_result = planner.hier_tree.find_free_box(
            q_seed,
            planner.obstacles,
            mark_occupied=True,
            forest_box_id=nid,
        )
        if ffb_result is None:
            continue

        ivs = ffb_result.intervals
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        if gmean_edge_length(vol, planner._n_dims) < planner.config.min_box_size:
            continue
        if ffb_result.absorbed_box_ids:
            forest.remove_boxes(ffb_result.absorbed_box_ids)

        box = BoxNode(
            node_id=nid,
            joint_intervals=ivs,
            seed_config=q_seed.copy(),
            volume=vol,
        )
        forest.add_box_direct(box)
        added += 1

    while forest.n_boxes < planner.config.max_box_nodes and attempts < planner.config.max_iterations:
        attempts += 1
        q_seed = planner._sample_seed(q_start, q_goal, rng)
        if q_seed is None:
            continue
        if planner.hier_tree.is_occupied(q_seed):
            continue

        nid = forest.allocate_id()
        ffb_result = planner.hier_tree.find_free_box(
            q_seed,
            planner.obstacles,
            mark_occupied=True,
            forest_box_id=nid,
        )
        if ffb_result is None:
            continue

        ivs = ffb_result.intervals
        vol = 1.0
        for lo, hi in ivs:
            vol *= max(hi - lo, 0.0)
        if gmean_edge_length(vol, planner._n_dims) < planner.config.min_box_size:
            continue
        if ffb_result.absorbed_box_ids:
            forest.remove_boxes(ffb_result.absorbed_box_ids)

        box = BoxNode(
            node_id=nid,
            joint_intervals=ivs,
            seed_config=q_seed.copy(),
            volume=vol,
        )
        forest.add_box_direct(box)
        added += 1

    return forest, added, attempts


def main() -> None:
    seed = 20260218
    rng = np.random.default_rng(seed)

    robot = load_robot("2dof_planar")
    q_start = np.array([0.8 * np.pi, 0.2], dtype=np.float64)
    q_goal = np.array([-0.7 * np.pi, -0.4], dtype=np.float64)

    # 可按需改这里
    n_obstacles = 8
    step_start = 10
    step_end = 120
    step_stride = 10  # 更细粒度（相比原先 20）
    steps = list(range(step_start, step_end + 1, step_stride))

    scene = build_random_scene(
        robot=robot,
        q_start=q_start,
        q_goal=q_goal,
        rng=rng,
        n_obs=n_obstacles,
    )

    out_dir = make_output_dir("visualizations", "random_2dof_forest_expansion")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    collision_map, collision_extent = scan_collision_map(
        robot=robot,
        scene=scene,
        joint_limits=robot.joint_limits,
        resolution=0.03,
    )

    # 保存场景数据
    scene_json = out_dir / "scene.json"
    scene.to_json(str(scene_json))

    summary = {
        "seed": seed,
        "n_obstacles": n_obstacles,
        "steps": steps,
        "q_start": q_start.tolist(),
        "q_goal": q_goal.tolist(),
        "scene_json": str(scene_json),
        "results": [],
    }

    for step in steps:
        planner = BoxRRT(robot=robot, scene=scene, config=make_config(step))
        forest, added, attempts = grow_forest_snapshot(
            planner=planner,
            q_start=q_start,
            q_goal=q_goal,
            seed=seed + step,
        )

        result = PlannerResult(
            success=False,
            path=[],
            forest=forest,
            n_boxes_created=forest.n_boxes,
            message=f"forest expansion only: boxes={forest.n_boxes}, added={added}, attempts={attempts}",
        )

        fig = plot_forest_with_collision_map(
            result=result,
            collision_map=collision_map,
            extent=collision_extent,
            title=f"2DOF Random Obstacles + C-space collision | max_box_nodes={step} | boxes={result.n_boxes_created}",
        )

        frame_path = frames_dir / f"step_{step:03d}.png"
        if fig is not None:
            fig.savefig(frame_path, dpi=140, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

        summary["results"].append(
            {
                "max_box_nodes": step,
                "success": bool(result.success),
                "message": result.message,
                "n_boxes_created": int(result.n_boxes_created),
                "n_collision_checks": int(planner.collision_checker.n_collision_checks),
                "attempts": int(attempts),
                "added_boxes": int(added),
                "frame": str(frame_path),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

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

    gif_path = out_dir / "expansion.gif"
    gif_ok = compose_gif(frames_dir=frames_dir, gif_path=gif_path)

    md_lines = [
        "# Random Obstacles 2DOF Box Forest Expansion",
        "",
        f"- seed: {seed}",
        f"- n_obstacles: {n_obstacles}",
        f"- q_start: {q_start.tolist()}",
        f"- q_goal: {q_goal.tolist()}",
        f"- scene: {scene_json.name}",
        f"- collision_map: {collision_map_png.name}",
        f"- gif: {gif_path.name if gif_ok else 'not generated (Pillow missing or no frames)'}",
        "",
        "## Frames",
    ]

    for item in summary["results"]:
        md_lines.append(
            f"- step={item['max_box_nodes']}, boxes={item['n_boxes_created']}, "
            f"added={item['added_boxes']}, attempts={item['attempts']}, frame={Path(item['frame']).name}"
        )

    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"output_dir={out_dir}")
    print(f"frames_dir={frames_dir}")
    print(f"summary={summary_path}")
    if gif_ok:
        print(f"gif={gif_path}")
    else:
        print("gif=NOT_GENERATED")


if __name__ == "__main__":
    main()
