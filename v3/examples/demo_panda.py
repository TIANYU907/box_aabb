"""
examples/demo_panda.py — Panda 7-DOF 路径规划演示

用法:
    python -m v3.examples.demo_panda
"""

import time
from pathlib import Path

import numpy as np

import sys, os
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from aabb.robot import load_robot
from planner.pipeline import (
    PandaGCSConfig,
    build_panda_scene,
    grow_and_prepare,
    run_method_with_bridge,
    _solve_method_dijkstra,
)
from viz.scene_viz import (
    plot_joint_trajectory,
    plot_arm_scene_html,
    plot_arm_poses_html,
    create_animation_html,
    save_plotly_html,
    generate_report,
)
from utils.output import make_output_dir


def main():

    t_total_start = time.perf_counter()

    cfg = PandaGCSConfig()
    if cfg.seed == 0:
        cfg.seed = int(time.time()) % (2**31)
    robot = load_robot("panda")
    ndim = robot.n_joints
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    rng = np.random.default_rng(cfg.seed)

    dist = float(np.linalg.norm(q_goal - q_start))
    print(f"Panda {ndim}-DOF Planner — Dijkstra")
    print(f"  seed        = {cfg.seed}")
    print(f"  q_start     = {np.array2string(q_start, precision=3)}")
    print(f"  q_goal      = {np.array2string(q_goal, precision=3)}")
    print(f"  config dist = {dist:.3f} rad")
    print(f"  obstacles   = {cfg.n_obstacles}")
    print(f"  max_boxes   = {cfg.max_boxes}")

    # ── 1) Scene ──
    t0 = time.perf_counter()
    print("\nBuilding scene ...")
    scene = build_panda_scene(rng, cfg, robot, q_start, q_goal)
    scene_ms = (time.perf_counter() - t0) * 1000
    print(f"  {scene.n_obstacles} obstacles  ({scene_ms:.1f} ms)")
    for obs in scene.get_obstacles():
        mn, mx = obs.min_point, obs.max_point
        sz = mx - mn
        print(f"    {obs.name}: center=({(mn[0]+mx[0])/2:.3f}, "
              f"{(mn[1]+mx[1])/2:.3f}, {(mn[2]+mx[2])/2:.3f})  "
              f"size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})")

    # ── 2) Shared: grow + cache + coarsen ──
    print("\n" + "=" * 60)
    print("  Pipeline: grow + cache + coarsen + bridge + Dijkstra")
    print("=" * 60)
    prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)

    # ── 3) Dijkstra + SOCP refine ──
    print("\n" + "=" * 60)
    print("  Dijkstra + SOCP refine")
    print("=" * 60)
    t0_b = time.perf_counter()
    result_dij = run_method_with_bridge(
        _solve_method_dijkstra, "Dijkstra", prep, cfg, q_start, q_goal, ndim)
    ms_b = (time.perf_counter() - t0_b) * 1000
    if result_dij and result_dij['success']:
        print(f"  Dijkstra: cost={result_dij['cost']:.4f}, "
              f"{len(result_dij['waypoints'])} wp, total {ms_b:.0f}ms")
    else:
        print(f"  Dijkstra: FAILED ({ms_b:.0f}ms)")

    # ── 4) Collect results ──
    all_results = [result_dij]
    best = result_dij if (result_dij and result_dij.get('success')) else None

    # ── 4b) Wait for cache save thread ──
    cache_thread = prep.get('_cache_thread')
    if cache_thread is not None:
        cache_thread.join()
        cr = prep.get('_cache_result', {})
        cache_ms = cr.get('ms', 0.0)
        cache_path = cr.get('path', '?')
        prep['cache_ms'] = cache_ms
        print(f"    [cache] done: {Path(cache_path).name} "
              f"({cache_ms:.0f} ms, parallel)")

    # ── 5) Visualization ──
    out_dir = make_output_dir("visualizations", "gcs_panda")
    print(f"\n{'=' * 60}")
    print(f"  Visualization")
    print(f"{'=' * 60}")
    print(f"  Output: {out_dir}")

    viz_files = []
    if best and best['success']:
        wps = best['waypoints']
        method_name = best['method']
        joint_names = [f"J{i+1}" for i in range(ndim)]

        # (a) Joint trajectory
        t0 = time.perf_counter()
        fig_jt = plot_joint_trajectory(
            wps, q_start, q_goal,
            label=f"Panda ({method_name}) cost={best['cost']:.2f}",
            joint_names=joint_names)
        if fig_jt:
            p = out_dir / "joint_trajectory.html"
            save_plotly_html(fig_jt, p)
            ms = (time.perf_counter() - t0) * 1000
            print(f"    joint_trajectory  {ms:7.0f} ms  -> {p.name}")
            viz_files.append(("joint_trajectory.html", ms))

        # (b) 3D scene
        t0 = time.perf_counter()
        fig_scene = plot_arm_scene_html(
            robot, scene, q_start, q_goal, waypoints=wps,
            title=f"Panda {method_name} — cost={best['cost']:.2f}")
        p = out_dir / "arm_scene.html"
        save_plotly_html(fig_scene, p)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    arm_scene         {ms:7.0f} ms  -> {p.name}")
        viz_files.append(("arm_scene.html", ms))

        # (c) Ghost poses
        t0 = time.perf_counter()
        fig_ghost = plot_arm_poses_html(
            robot, scene, wps, n_ghosts=10,
            title=f"Panda {method_name} — Arm Pose Sequence")
        p = out_dir / "arm_poses.html"
        save_plotly_html(fig_ghost, p)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    arm_poses         {ms:7.0f} ms  -> {p.name}")
        viz_files.append(("arm_poses.html", ms))

        # (d) Animation
        t0 = time.perf_counter()
        fig_anim = create_animation_html(
            robot, scene, wps, n_frames=60,
            title=f"Panda {method_name} Path Animation")
        p = out_dir / "animation.html"
        save_plotly_html(fig_anim, p)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    animation (HTML)  {ms:7.0f} ms  -> {p.name}")
        viz_files.append(("animation.html", ms))

    # ── 6) Report ──
    total_s = time.perf_counter() - t_total_start
    print("\n")
    generate_report(cfg, scene, all_results, q_start, q_goal, ndim,
                    out_dir, total_s, prep, viz_files)


if __name__ == "__main__":
    main()
