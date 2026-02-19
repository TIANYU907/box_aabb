#!/usr/bin/env python
"""
benchmarks/forest/bench_panda_multi.py - Panda 7-DOF 多场景对比测试 (v2)

测试不同障碍物数量 × 冷/热缓存 的 box forest 拓展表现。
每个障碍物数量运行两次：第一次冷启动（不使用缓存），第二次使用缓存（热）。

用法：
    python -m v2.benchmarks.forest.bench_panda_multi
    python -m v2.benchmarks.forest.bench_panda_multi --obs-list 5 10 15 20 --max-boxes 200
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from aabb.robot import Robot, load_robot
from common.output import make_output_dir

from v2.benchmarks.forest.bench_panda_forest import random_scene_3d, grow_forest_panda

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
LOG_FMT_SIMPLE = "[%(asctime)s] %(message)s"

logging.basicConfig(level=logging.WARNING, format=LOG_FMT_SIMPLE, datefmt="%H:%M:%S")
logger = logging.getLogger("bench_multi")
console_logger = logging.getLogger("console")


def run_single(
    robot: Robot,
    joint_limits: list,
    n_obs: int,
    rng_seed: int,
    args,
    label: str,
    use_cache: bool = True,
) -> dict:
    """运行单次测试，返回结果字典。"""
    rng = np.random.default_rng(rng_seed)
    scene = random_scene_3d(n_obs, rng)

    t_wall_start = time.time()
    result = grow_forest_panda(
        robot=robot,
        scene=scene,
        joint_limits=joint_limits,
        max_boxes=args.max_boxes,
        max_seeds=args.max_seeds,
        boundary_batch=args.boundary_batch,
        farthest_k=args.farthest_k,
        rng_seed=rng_seed + 1000,
        max_depth=args.max_depth,
        min_edge_length=args.min_edge,
        early_stop_window=args.early_stop_window,
        early_stop_min_size=args.early_stop_min_size,
        use_cache=use_cache,
    )
    result["wall_time"] = time.time() - t_wall_start
    result["label"] = label
    result["n_obs"] = n_obs
    result["rng_seed"] = rng_seed
    result.pop("box_logs", None)
    return result


def generate_comparison_report(all_results: list, args) -> str:
    """生成多场景对比报告。"""
    lines = []
    lines.append("=" * 100)
    lines.append("  Panda 7-DOF Box Forest 多场景对比测试报告 (v2)")
    lines.append("=" * 100)
    lines.append(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  参数: max_boxes={args.max_boxes}  max_seeds={args.max_seeds}  "
                 f"max_depth={args.max_depth}  min_edge={args.min_edge}")
    lines.append(f"        boundary_batch={args.boundary_batch}  farthest_k={args.farthest_k}  "
                 f"early_stop_window={args.early_stop_window}")
    lines.append(f"  障碍物数量: {args.obs_list}")
    lines.append(f"  随机种子: {args.seed}")
    lines.append("")

    lines.append("-" * 100)
    lines.append("  总览表")
    lines.append("-" * 100)
    hdr = (
        f"  {'n_obs':>5s}  {'cache':>6s}  {'boxes':>5s}  {'nsize':>8s}  "
        f"{'adj':>4s}  {'trees':>5s}  {'time':>7s}  {'t_ffb':>7s}  {'t_coll':>7s}  "
        f"{'t_samp':>7s}  {'nodes':>6s}  {'fk':>6s}  {'stop':>25s}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * 106)

    for r in all_results:
        lines.append(
            f"  {r['n_obs']:5d}  {r['label']:>6s}  {r['n_boxes']:5d}  "
            f"{r['nsize_mean']:8.4f}  {r['n_adj']:4d}  {r['n_components']:5d}  "
            f"{r['time_total']:7.2f}  {r['time_find_free_box']:7.2f}  "
            f"{r['time_collision_check']:7.2f}  {r['time_sampling']:7.2f}  "
            f"{r['tree_nodes']:6d}  {r['tree_fk_calls']:6d}  "
            f"{r['stop_reason']:>25s}"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("  冷/热缓存对比（按障碍物数量）")
    lines.append("-" * 100)

    obs_nums = sorted(set(r["n_obs"] for r in all_results))
    for n_obs in obs_nums:
        group = [r for r in all_results if r["n_obs"] == n_obs]
        cold = [r for r in group if r["label"] == "cold"]
        warm = [r for r in group if r["label"] == "warm"]

        lines.append(f"\n  ── n_obs = {n_obs} ──")
        if cold and warm:
            c, w = cold[0], warm[0]
            speedup_total = c["time_total"] / w["time_total"] if w["time_total"] > 0 else float("inf")
            speedup_ffb = (c["time_find_free_box"] / w["time_find_free_box"]
                           if w["time_find_free_box"] > 0 else float("inf"))
            lines.append(f"    冷启动: {c['time_total']:.3f}s  (ffb={c['time_find_free_box']:.3f}s, "
                         f"nodes={c['tree_nodes']}, fk={c['tree_fk_calls']})")
            lines.append(f"    热缓存: {w['time_total']:.3f}s  (ffb={w['time_find_free_box']:.3f}s, "
                         f"nodes={w['tree_nodes']}, fk_new={w['tree_fk_calls']-w['cache_fk_init']})")
            lines.append(f"    总加速比: {speedup_total:.2f}x  |  ffb加速比: {speedup_ffb:.2f}x")
            lines.append(f"    冷 boxes={c['n_boxes']}  nsize_mean={c['nsize_mean']:.4f}  "
                         f"adj={c['n_adj']}  stop={c['stop_reason']}")
            lines.append(f"    热 boxes={w['n_boxes']}  nsize_mean={w['nsize_mean']:.4f}  "
                         f"adj={w['n_adj']}  stop={w['stop_reason']}")
        elif cold:
            c = cold[0]
            lines.append(f"    冷启动: {c['time_total']:.3f}s  boxes={c['n_boxes']}  "
                         f"nsize_mean={c['nsize_mean']:.4f}")

    lines.append("")
    lines.append("-" * 100)
    lines.append("  采样统计（拒绝原因）")
    lines.append("-" * 100)
    hdr2 = (
        f"  {'n_obs':>5s}  {'cache':>6s}  {'iters':>5s}  "
        f"{'boundary':>8s}  {'farthest':>8s}  {'coll_rej':>8s}  "
        f"{'inside':>6s}  {'maxd':>6s}  {'narrow':>6s}  {'tiny':>6s}"
    )
    lines.append(hdr2)
    lines.append("  " + "-" * 78)
    for r in all_results:
        lines.append(
            f"  {r['n_obs']:5d}  {r['label']:>6s}  {r['seed_iters']:5d}  "
            f"{r['n_boundary_attempts']:8d}  {r['n_farthest_attempts']:8d}  "
            f"{r['n_seed_collision']:8d}  {r['n_seed_inside']:6d}  "
            f"{r.get('n_ffb_max_depth', 0):6d}  {r.get('n_ffb_min_edge', 0):6d}  {r['n_find_tiny']:6d}"
        )

    ndim = all_results[0].get('ndim', 7) if all_results else 7
    lines.append("")
    lines.append("-" * 100)
    lines.append(f"  Box 归一化尺寸 vol^(1/{ndim}) (rad)")
    lines.append("-" * 100)
    hdr3 = (
        f"  {'n_obs':>5s}  {'cache':>6s}  {'nsize_mean':>10s}  {'nsize_med':>10s}  "
        f"{'nsize_min':>10s}  {'nsize_max':>10s}  {'deg_mean':>8s}  "
        f"{'isolated':>8s}  {'trees':>5s}  {'largest':>7s}"
    )
    lines.append(hdr3)
    lines.append("  " + "-" * 96)
    for r in all_results:
        lines.append(
            f"  {r['n_obs']:5d}  {r['label']:>6s}  {r['nsize_mean']:10.4f}  "
            f"{r['nsize_median']:10.4f}  {r['nsize_min']:10.4f}  "
            f"{r['nsize_max']:10.4f}  {r['deg_mean']:8.2f}  {r['n_isolated']:8d}  "
            f"{r['n_components']:5d}  {r['largest_component']:7d}"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("  耗时分解 (%)")
    lines.append("-" * 100)
    hdr4 = (
        f"  {'n_obs':>5s}  {'cache':>6s}  {'total':>7s}  "
        f"{'%ffb':>6s}  {'%coll':>6s}  {'%deov':>6s}  {'%samp':>6s}  {'%load':>6s}  {'%save':>6s}  {'%other':>6s}"
    )
    lines.append(hdr4)
    lines.append("  " + "-" * 78)
    for r in all_results:
        dt = r["time_total"]
        if dt > 0:
            p_ffb = r["time_find_free_box"] / dt * 100
            p_coll = r["time_collision_check"] / dt * 100
            p_deov = r["time_deoverlap"] / dt * 100
            p_samp = r["time_sampling"] / dt * 100
            p_load = r["time_load_cache"] / dt * 100
            p_save = r.get("time_save_cache", 0) / dt * 100
            p_other = max(100 - p_ffb - p_coll - p_deov - p_samp - p_load - p_save, 0)
        else:
            p_ffb = p_coll = p_deov = p_samp = p_load = p_save = p_other = 0.0
        lines.append(
            f"  {r['n_obs']:5d}  {r['label']:>6s}  {dt:7.2f}  "
            f"{p_ffb:6.1f}  {p_coll:6.1f}  {p_deov:6.1f}  "
            f"{p_samp:6.1f}  {p_load:6.1f}  {p_save:6.1f}  {p_other:6.1f}"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("  趋势总结（障碍物数量 → 性能影响）")
    lines.append("-" * 100)
    cold_results = [r for r in all_results if r["label"] == "cold"]
    if len(cold_results) >= 2:
        lines.append(f"  {'n_obs':>5s}  {'boxes':>5s}  {'nsize':>8s}  {'time_s':>7s}  "
                     f"{'ffb_s':>7s}  {'fk':>6s}  {'nodes':>6s}  {'coll_rej%':>9s}")
        lines.append("  " + "-" * 60)
        for r in cold_results:
            total_attempts = (r["n_boundary_attempts"] + r["n_farthest_attempts"] *
                              12)
            rej_pct = (r["n_seed_collision"] / max(total_attempts, 1) * 100)
            lines.append(
                f"  {r['n_obs']:5d}  {r['n_boxes']:5d}  {r['nsize_mean']:8.4f}  "
                f"{r['time_total']:7.2f}  {r['time_find_free_box']:7.2f}  "
                f"{r['tree_fk_calls']:6d}  {r['tree_nodes']:6d}  {rej_pct:9.1f}"
            )

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panda 7-DOF 多场景对比测试 (v2)"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (默认: 随机)")
    parser.add_argument("--obs-list", type=int, nargs="+", default=[5, 10, 15, 20],
                        help="障碍物数量列表 (默认: 5 10 15 20)")
    parser.add_argument("--max-boxes", type=int, default=200, help="最大 box 数")
    parser.add_argument("--max-seeds", type=int, default=3000, help="最大采样迭代")
    parser.add_argument("--max-depth", type=int, default=30, help="最大切分深度")
    parser.add_argument("--min-edge", type=float, default=0.05, help="最小分割边长")
    parser.add_argument("--boundary-batch", type=int, default=6, help="DFS 边界采样批量")
    parser.add_argument("--farthest-k", type=int, default=12, help="最远点采样候选数")
    parser.add_argument("--early-stop-window", type=int, default=20, help="早停窗口")
    parser.add_argument("--early-stop-min-size", type=float, default=0.01,
                        help="早停阈值 vol^(1/d) (默认: 0.01 rad)")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time()) % 100000
    master_rng = np.random.default_rng(args.seed)

    output_dir = make_output_dir("benchmarks", "panda_multi")

    log_file = output_dir / "bench_panda_multi.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FMT, datefmt="%H:%M:%S"))

    for log_name in ["bench_multi", "bench_panda", "planner", "forest", "aabb"]:
        log = logging.getLogger(log_name)
        log.setLevel(logging.INFO)
        log.addHandler(file_handler)
        log.propagate = False

    robot = load_robot("panda")
    joint_limits = list(robot.joint_limits)

    console_logger.warning("=" * 80)
    console_logger.warning("  Panda 7-DOF 多场景对比测试 (v2)")
    console_logger.warning(
        f"  master_seed={args.seed}  obs_list={args.obs_list}  max_boxes={args.max_boxes}"
    )
    console_logger.warning(f"  output: {output_dir}")
    console_logger.warning(f"  log: {log_file}")
    console_logger.warning("=" * 80)

    logger.info("=" * 80)
    logger.info("  Panda 7-DOF 多场景对比测试 (v2)")
    logger.info("  master_seed=%d  obs_list=%s  max_boxes=%d",
                args.seed, args.obs_list, args.max_boxes)
    logger.info("  output: %s", output_dir)
    logger.info("=" * 80)

    all_results = []

    for n_obs in args.obs_list:
        scene_seed = int(master_rng.integers(0, 2**31))

        console_logger.warning("")
        console_logger.warning("━" * 80)
        console_logger.warning(f"  n_obs = {n_obs}  |  scene_seed = {scene_seed}")
        console_logger.warning("━" * 80)

        logger.info("")
        logger.info("━" * 80)
        logger.info("  n_obs = %d  |  scene_seed = %d", n_obs, scene_seed)
        logger.info("━" * 80)

        console_logger.warning("  [COLD] 冷启动运行（不使用缓存）...")
        logger.info("  [COLD] 冷启动运行（不使用缓存）...")
        cold_result = run_single(robot, joint_limits, n_obs, scene_seed, args, "cold",
                                 use_cache=False)
        all_results.append(cold_result)

        console_logger.warning(
            f"  [COLD] boxes={cold_result['n_boxes']}  nsize={cold_result['nsize_mean']:.4f}  "
            f"time={cold_result['time_total']:.2f}s  ffb={cold_result['time_find_free_box']:.2f}s"
        )
        logger.info(
            "  [COLD] boxes=%d  nsize=%.4f  time=%.2fs  ffb=%.2fs  "
            "nodes=%d  fk=%d  stop=%s",
            cold_result["n_boxes"], cold_result["nsize_mean"],
            cold_result["time_total"], cold_result["time_find_free_box"],
            cold_result["tree_nodes"], cold_result["tree_fk_calls"],
            cold_result["stop_reason"],
        )

        console_logger.warning("  [WARM] 使用缓存，热启动运行...")
        logger.info("  [WARM] 使用缓存，热启动运行...")
        warm_result = run_single(robot, joint_limits, n_obs, scene_seed, args, "warm",
                                 use_cache=True)
        all_results.append(warm_result)

        console_logger.warning(
            f"  [WARM] boxes={warm_result['n_boxes']}  nsize={warm_result['nsize_mean']:.4f}  "
            f"time={warm_result['time_total']:.2f}s  ffb={warm_result['time_find_free_box']:.2f}s"
        )
        logger.info(
            "  [WARM] boxes=%d  nsize=%.4f  time=%.2fs  ffb=%.2fs  "
            "nodes=%d→%d  fk_init=%d  fk_new=%d  stop=%s",
            warm_result["n_boxes"], warm_result["nsize_mean"],
            warm_result["time_total"], warm_result["time_find_free_box"],
            warm_result["cache_nodes_init"], warm_result["tree_nodes"],
            warm_result["cache_fk_init"],
            warm_result["tree_fk_calls"] - warm_result["cache_fk_init"],
            warm_result["stop_reason"],
        )

        if warm_result["time_total"] > 0:
            sp = cold_result["time_total"] / warm_result["time_total"]
            console_logger.warning(f"  加速比: {sp:.2f}x")
            logger.info("  加速比: %.2fx", sp)

    report = generate_comparison_report(all_results, args)
    report_path = output_dir / "multi_scenario_report.txt"
    report_path.write_text(report, encoding="utf-8")

    console_logger.warning("")
    console_logger.warning("=" * 80)
    console_logger.warning("多场景测试完成汇总")
    console_logger.warning("=" * 80)
    for r in all_results:
        console_logger.warning(
            f"  n_obs={r['n_obs']:2d} [{r['label']:>4s}]: "
            f"boxes={r['n_boxes']:3d}  nsize={r['nsize_mean']:.4f}  "
            f"time={r['time_total']:6.2f}s  stop={r['stop_reason']}"
        )
    console_logger.warning(f"报告: {report_path}")
    console_logger.warning(f"日志: {log_file}")
    console_logger.warning("=" * 80)

    print()
    print(report)
    logger.info("报告已保存: %s", report_path)


if __name__ == "__main__":
    main()
