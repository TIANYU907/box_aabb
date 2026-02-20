"""
对比 boundary_expand 开/关 在同一场景下的 box 数量与规划效果。

对 2DOF planar 和 Panda 7-DOF 各跑多个 seed，统计：
  - 最终 box 数
  - 规划成功率
  - grow 耗时
  - 路径长度
"""
import time
import numpy as np
from v2._bootstrap import add_v2_paths
add_v2_paths()

from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker
from planner.models import PlannerConfig
from planner.box_planner import BoxPlanner


# ── helpers ──────────────────────────────────────────────────────────────
def make_2dof_scene():
    robot = load_robot("2dof_planar")
    scene = Scene()
    scene.add_obstacle([0.6, -0.2], [0.9, 0.2], "obs")
    q_s = np.array([-1.0, 0.0])
    q_g = np.array([1.0, 0.0])
    return robot, scene, q_s, q_g


def make_panda_scene(seed: int, n_obs: int = 6):
    import math
    rng = np.random.default_rng(seed)
    robot = load_robot("panda")
    q_s = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_g = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])
    for _ in range(200):
        scene = Scene()
        for i in range(n_obs):
            r = rng.uniform(0.2, 0.75)
            th = rng.uniform(-math.pi, math.pi)
            cx, cy = r * np.cos(th), r * np.sin(th)
            cz = rng.uniform(0.15, 0.85)
            h = rng.uniform(0.08, 0.25, size=3)
            scene.add_obstacle(
                [cx - h[0], cy - h[1], cz - h[2]],
                [cx + h[0], cy + h[1], cz + h[2]],
                f"obs_{i}",
            )
        checker = CollisionChecker(robot=robot, scene=scene)
        if not checker.check_config_collision(q_s) and not checker.check_config_collision(q_g):
            return robot, scene, q_s, q_g
    raise RuntimeError("无法生成无碰撞的 Panda 场景")


def run_single(robot, scene, q_s, q_g, seed, expand_on, max_boxes, max_iter):
    cfg = PlannerConfig(
        max_iterations=max_iter,
        max_box_nodes=max_boxes,
        boundary_expand_enabled=expand_on,
        boundary_expand_max_failures=5,
        boundary_expand_epsilon=0.01,
        verbose=False,
    )
    planner = BoxPlanner(robot, scene, config=cfg, no_cache=True)
    t0 = time.perf_counter()
    result = planner.plan(q_s, q_g, seed=seed)
    dt = time.perf_counter() - t0
    return {
        "success": result.success,
        "n_boxes": result.n_boxes_created,
        "path_len": result.path_length,
        "time_s": dt,
    }


# ── main ─────────────────────────────────────────────────────────────────
def main():
    seeds = [42, 123, 256, 314, 999]

    # ===================== 2DOF planar =====================
    print("=" * 70)
    print("  2DOF Planar  (max_boxes=60, max_iter=60)")
    print("=" * 70)
    header = f"{'seed':>6} | {'mode':<10} | {'ok':>3} | {'boxes':>5} | {'path':>8} | {'time':>7}"
    print(header)
    print("-" * len(header))

    robot_2d, scene_2d, qs_2d, qg_2d = make_2dof_scene()
    for s in seeds:
        for expand in [False, True]:
            tag = "ON" if expand else "OFF"
            r = run_single(robot_2d, scene_2d, qs_2d, qg_2d,
                           seed=s, expand_on=expand,
                           max_boxes=60, max_iter=60)
            ok = "Y" if r["success"] else "N"
            pl = f"{r['path_len']:.3f}" if r["success"] else "  -"
            print(f"{s:>6} | {tag:<10} | {ok:>3} | {r['n_boxes']:>5} | {pl:>8} | {r['time_s']:>6.3f}s")

    # ===================== Panda 7-DOF =====================
    print()
    print("=" * 70)
    print("  Panda 7-DOF  (max_boxes=500, max_iter=999999)")
    print("=" * 70)
    header = f"{'seed':>6} | {'mode':<10} | {'ok':>3} | {'boxes':>5} | {'path':>8} | {'time':>7}"
    print(header)
    print("-" * len(header))

    panda_seeds = [42, 123, 256]
    for s in panda_seeds:
        robot_p, scene_p, qs_p, qg_p = make_panda_scene(seed=s, n_obs=6)
        for expand in [False, True]:
            tag = "ON" if expand else "OFF"
            r = run_single(robot_p, scene_p, qs_p, qg_p,
                           seed=s, expand_on=expand,
                           max_boxes=500, max_iter=999999)
            ok = "Y" if r["success"] else "N"
            pl = f"{r['path_len']:.3f}" if r["success"] else "  -"
            print(f"{s:>6} | {tag:<10} | {ok:>3} | {r['n_boxes']:>5} | {pl:>8} | {r['time_s']:>6.3f}s")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
