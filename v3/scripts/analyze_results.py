"""
scripts/analyze_results.py — 分析全部实验结果并输出汇总报告
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
RAW = _ROOT / "experiments" / "output" / "raw"


def _safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return np.mean(vals) if vals else float("nan")


def _safe_median(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return np.median(vals) if vals else float("nan")


def analyze_exp1():
    path = RAW / "exp1_main_comparison.json"
    if not path.exists():
        print("  [SKIP] exp1 not found")
        return
    d = json.load(open(path, encoding="utf-8"))
    groups = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        key = (r["scene"], r["planner"])
        groups[key]["success"].append(bool(r.get("success", False)))
        t = r.get("planning_time", 0)
        if t is None or (isinstance(t, float) and np.isnan(t)):
            t = 0
        groups[key]["time"].append(t)
        cost = r.get("cost")
        if cost is not None and cost != float("inf") and not (isinstance(cost, float) and np.isnan(cost)):
            groups[key]["cost"].append(cost)

    scenes = ["panda_8obs_open", "panda_15obs_moderate", "panda_20obs_dense"]
    planners = ["SBF-Dijkstra", "RRTConnect", "RRT*", "Informed-RRT*", "BiRRT*"]

    for scene in scenes:
        label = scene.replace("panda_", "").replace("_", " ").upper()
        print(f"\n  Scene: {label}")
        hdr = f"  {'Planner':<22s} {'Success':>8s} {'Mean(s)':>10s} {'Med(s)':>10s} {'Cost':>10s}"
        print(hdr)
        print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for planner in planners:
            key = (scene, planner)
            if key not in groups:
                continue
            g = groups[key]
            sr = sum(g["success"]) / len(g["success"]) * 100
            tm = _safe_mean(g["time"])
            tmed = _safe_median(g["time"])
            cs = _safe_mean(g["cost"]) if g["cost"] else float("nan")
            cost_str = f"{cs:>10.2f}" if not np.isnan(cs) else "       N/A"
            print(f"  {planner:<22s} {sr:>7.0f}% {tm:>10.3f} {tmed:>10.3f} {cost_str}")


def analyze_exp2():
    path = RAW / "exp2_forest_reuse.json"
    if not path.exists():
        print("  [SKIP] exp2 not found")
        return
    d = json.load(open(path, encoding="utf-8"))

    planners = ["SBF-Dijkstra", "RRTConnect", "RRT*"]
    k_vals = sorted(set(r.get("K") for r in d["results"] if r.get("K") is not None))

    print(f"\n  {'Planner':<18s} {'K':>5s} {'Cum Time':>10s} {'Avg/Query':>10s} {'Success':>8s}")
    print(f"  {'-'*18} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")
    for planner in planners:
        for K in k_vals:
            matching = [r for r in d["results"]
                        if r["planner"] == planner and r.get("K") == K]
            if not matching:
                continue
            cum = _safe_mean([r["cumulative_time"] for r in matching])
            avg = _safe_mean([r["avg_time_per_query"] for r in matching])
            sr = sum(r["success"] for r in matching) / len(matching) * 100
            print(f"  {planner:<18s} {K:>5d} {cum:>10.3f} {avg:>10.4f} {sr:>7.0f}%")

    # amortized speedup
    print("\n  Amortized speedup at max K:")
    max_K = max(k_vals)
    sbf_avg = [r["avg_time_per_query"] for r in d["results"]
               if r["planner"] == "SBF-Dijkstra" and r.get("K") == max_K]
    rrt_avg = [r["avg_time_per_query"] for r in d["results"]
               if r["planner"] == "RRTConnect" and r.get("K") == max_K]
    if sbf_avg and rrt_avg:
        print(f"    SBF avg/query at K={max_K}: {_safe_mean(sbf_avg):.4f}s")
        print(f"    RRT avg/query at K={max_K}: {_safe_mean(rrt_avg):.4f}s")


def analyze_exp3():
    path = RAW / "exp3_obstacle_change.json"
    if not path.exists():
        print("  [SKIP] exp3 not found")
        return
    d = json.load(open(path, encoding="utf-8"))

    change_types = ["remove", "add", "move"]
    change_counts = sorted(set(r.get("n_changes", 0) for r in d["results"] if r.get("n_changes")))
    planner_order = ["SBF-Incremental", "SBF-FullRebuild", "RRTConnect"]

    for ct in change_types:
        print(f"\n  Change: {ct.upper()}")
        print(f"  {'n':>3s} {'Planner':<20s} {'Success':>8s} {'Time(s)':>10s}")
        print(f"  {'-'*3} {'-'*20} {'-'*8} {'-'*10}")
        for nc in change_counts:
            scene = f"obs_{ct}_{nc}"
            for planner in planner_order:
                matching = [r for r in d["results"]
                            if r["scene"] == scene and r["planner"] == planner]
                if not matching:
                    continue
                sr = sum(r["success"] for r in matching) / len(matching) * 100
                tm_key = "update_time" if "update_time" in matching[0] else "rebuild_time"
                if tm_key not in matching[0]:
                    tm_key = "planning_time"
                tm = _safe_mean([r.get(tm_key, r.get("planning_time", 0)) for r in matching])
                print(f"  {nc:>3d} {planner:<20s} {sr:>7.0f}% {tm:>10.3f}")


def analyze_exp4():
    path = RAW / "exp4_cache_warmstart.json"
    if not path.exists():
        print("  [SKIP] exp4 not found")
        return
    d = json.load(open(path, encoding="utf-8"))

    modes = ["SBF-ColdStart", "SBF-WarmStart", "SBF-CrossSceneCache"]
    print(f"\n  {'Mode':<25s} {'Success':>8s} {'Time(s)':>10s} {'Speedup':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
    for mode in modes:
        matching = [r for r in d["results"] if r["planner"] == mode]
        if not matching:
            continue
        sr = sum(r["success"] for r in matching) / len(matching) * 100
        tm = _safe_mean([r["planning_time"] for r in matching])
        sp_vals = [r.get("speedup") for r in matching if r.get("speedup") is not None]
        sp = _safe_mean(sp_vals) if sp_vals else float("nan")
        sp_str = f"{sp:>7.2f}x" if not np.isnan(sp) else "     N/A"
        print(f"  {mode:<25s} {sr:>7.0f}% {tm:>10.3f} {sp_str}")


def analyze_exp5():
    path = RAW / "exp5_ablation.json"
    if not path.exists():
        print("  [SKIP] exp5 not found")
        return
    d = json.load(open(path, encoding="utf-8"))

    # group by planner (= variant)
    groups = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        key = r["planner"]
        groups[key]["success"].append(bool(r.get("success", False)))
        t = r.get("planning_time", 0) or 0
        groups[key]["time"].append(t)
        cost = r.get("cost")
        if cost is not None and cost != float("inf") and not (isinstance(cost, float) and np.isnan(cost)):
            groups[key]["cost"].append(cost)

    print(f"\n  {'Variant':<30s} {'Success':>8s} {'Time(s)':>10s} {'Cost':>10s}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for variant in sorted(groups.keys()):
        g = groups[variant]
        sr = sum(g["success"]) / len(g["success"]) * 100
        tm = _safe_mean(g["time"])
        cs = _safe_mean(g["cost"]) if g["cost"] else float("nan")
        cost_str = f"{cs:>10.2f}" if not np.isnan(cs) else "       N/A"
        print(f"  {variant:<30s} {sr:>7.0f}% {tm:>10.3f} {cost_str}")


def analyze_exp6():
    path = RAW / "exp6_config_sweep.json"
    if not path.exists():
        print("  [SKIP] exp6 not found")
        return
    d = json.load(open(path, encoding="utf-8"))

    # group by sweep_param → value
    param_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in d["results"]:
        param = r.get("sweep_param", "")
        val = r.get("sweep_value")
        if param and val is not None:
            param_groups[param][val]["success"].append(bool(r.get("success", False)))
            t = r.get("planning_time", 0) or 0
            param_groups[param][val]["time"].append(t)

    for param in sorted(param_groups.keys()):
        print(f"\n  Parameter: {param}")
        print(f"  {'Value':>12s} {'Success':>8s} {'Time(s)':>10s}")
        print(f"  {'-'*12} {'-'*8} {'-'*10}")
        for val in sorted(param_groups[param].keys()):
            g = param_groups[param][val]
            sr = sum(g["success"]) / len(g["success"]) * 100
            tm = _safe_mean(g["time"])
            print(f"  {str(val):>12s} {sr:>7.0f}% {tm:>10.3f}")


def analyze_exp7():
    path = RAW / "exp7_scalability.json"
    if not path.exists():
        print("  [SKIP] exp7 not found")
        return
    d = json.load(open(path, encoding="utf-8"))

    robots = sorted(set(r.get("robot", "") for r in d["results"]))
    planners_in = sorted(set(r["planner"] for r in d["results"]))

    for robot in robots:
        robot_results = [r for r in d["results"] if r.get("robot") == robot]
        n_obs_list = sorted(set(r.get("n_obstacles", 0) for r in robot_results))
        dof = robot_results[0].get("n_dof", "?") if robot_results else "?"
        print(f"\n  Robot: {robot} ({dof}-DOF)")
        header = f"  {'n_obs':>5s}"
        for p in planners_in:
            header += f"  {p+' SR':>14s}  {p+' Time':>12s}"
        print(header)
        print(f"  {'-'*5}" + (f"  {'-'*14}  {'-'*12}" * len(planners_in)))
        for n_obs in n_obs_list:
            line = f"  {n_obs:>5d}"
            for planner in planners_in:
                matching = [r for r in robot_results
                            if r.get("n_obstacles") == n_obs and r["planner"] == planner]
                if matching:
                    sr = sum(r["success"] for r in matching) / len(matching) * 100
                    tm = _safe_mean([r.get("planning_time", 0) for r in matching])
                    line += f"  {sr:>13.0f}%  {tm:>11.3f}s"
                else:
                    line += f"  {'N/A':>14s}  {'N/A':>12s}"
            print(line)


def analyze_exp8():
    path = RAW / "exp8_aabb_tightness.json"
    if not path.exists():
        print("  [SKIP] exp8 not found")
        return
    d = json.load(open(path, encoding="utf-8"))

    sizes = sorted(set(r.get("interval_size") for r in d["results"] if r.get("interval_size") is not None))
    print(f"\n  {'IntSize':>8s} {'Tightness':>10s} {'Vol_IFK':>12s} {'Vol_Num':>12s} {'T_IFK(ms)':>10s} {'T_Num(ms)':>10s}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for size in sizes:
        matching = [r for r in d["results"] if r.get("interval_size") == size]
        tightness = _safe_mean([r["tightness_ratio"] for r in matching])
        vol_ifk = _safe_mean([r["vol_interval_fk"] for r in matching])
        vol_num = _safe_mean([r["vol_numerical"] for r in matching])
        t_ifk = _safe_mean([r["time_interval_fk"] for r in matching]) * 1000
        t_num = _safe_mean([r["time_numerical"] for r in matching]) * 1000
        print(f"  {size:>8.1f} {tightness:>10.4f} {vol_ifk:>12.4f} {vol_num:>12.4f} {t_ifk:>10.2f} {t_num:>10.2f}")


def main():
    print("=" * 80)
    print("  SBF EXPERIMENT RESULTS ANALYSIS")
    print("  " + "=" * 76)

    print("\n" + "=" * 80)
    print("  EXP1: MAIN COMPARISON — SBF vs RRT variants (Panda 7-DOF)")
    print("  " + "-" * 76)
    analyze_exp1()

    print("\n" + "=" * 80)
    print("  EXP2: FOREST REUSE — amortized cost over K queries")
    print("  " + "-" * 76)
    analyze_exp2()

    print("\n" + "=" * 80)
    print("  EXP3: OBSTACLE CHANGE — incremental update vs full rebuild")
    print("  " + "-" * 76)
    analyze_exp3()

    print("\n" + "=" * 80)
    print("  EXP4: CACHE WARMSTART — HCACHE cold vs warm")
    print("  " + "-" * 76)
    analyze_exp4()

    print("\n" + "=" * 80)
    print("  EXP5: ABLATION — component contribution analysis")
    print("  " + "-" * 76)
    analyze_exp5()

    print("\n" + "=" * 80)
    print("  EXP6: CONFIG SWEEP — parameter sensitivity")
    print("  " + "-" * 76)
    analyze_exp6()

    print("\n" + "=" * 80)
    print("  EXP7: SCALABILITY — DOF x obstacles")
    print("  " + "-" * 76)
    analyze_exp7()

    print("\n" + "=" * 80)
    print("  EXP8: AABB TIGHTNESS — interval FK vs numerical sampling")
    print("  " + "-" * 76)
    analyze_exp8()

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
