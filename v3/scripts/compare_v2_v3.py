"""
scripts/compare_v2_v3.py — v2 vs v3 全面性能对比 (subprocess 隔离)

对比维度:
  1. 端到端规划 (6/8 obs): v2 vs v3 含 forest grow + solve
  2. Forest 复用多 query: v3 only
  3. 障碍物增量更新: v2 vs v3 增量 vs 全量
  4. AABB / Interval FK 性能

用法:
    cd v3
    python scripts/compare_v2_v3.py [--quick] [--only 1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]          # v3/
_SRC = _ROOT / "src"
_WORKSPACE = _ROOT.parent
_V2_ROOT = _WORKSPACE / "v2"

for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT_DIR = _ROOT / "experiments" / "output" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# v2 子进程调用
# ═══════════════════════════════════════════════════════════════════

def _run_v2_bench(test_name: str, quick: bool) -> dict:
    """在独立子进程中运行 v2 基准测试."""
    script = _V2_ROOT / "scripts" / "bench_standalone.py"
    cmd = [sys.executable, str(script), "--test", test_name]
    if quick:
        cmd.append("--quick")

    print(f"  [v2 subprocess] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(_V2_ROOT), timeout=600)

    if result.returncode != 0:
        print(f"  [v2 subprocess] FAILED (rc={result.returncode})")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for ln in lines[-10:]:
                print(f"    stderr: {ln}")
        return {}

    stdout = result.stdout
    json_start = stdout.find('{')
    if json_start < 0:
        print(f"  [v2 subprocess] No JSON in stdout")
        return {}

    try:
        return json.loads(stdout[json_start:])
    except json.JSONDecodeError as e:
        print(f"  [v2 subprocess] JSON parse error: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

def _gen_obstacles(n_obs, seed):
    rng = np.random.default_rng(seed)
    obs_data = []
    for _ in range(n_obs):
        cx = float(rng.uniform(-0.6, 0.6))
        cy = float(rng.uniform(-0.6, 0.6))
        cz = float(rng.uniform(0.2, 0.8))
        h = float(rng.uniform(0.06, 0.15))
        obs_data.append(([cx - h, cy - h, cz - h], [cx + h, cy + h, cz + h]))
    return obs_data


# ═══════════════════════════════════════════════════════════════════
# Test 1: End-to-end planning
# ═══════════════════════════════════════════════════════════════════

def test1_e2e(quick=False):
    from aabb.robot import load_robot
    from forest.scene import Scene
    from planner.pipeline import (
        PandaGCSConfig, grow_and_prepare, run_method_with_bridge,
        _solve_method_dijkstra,
    )

    print("\n" + "=" * 80)
    print("  Test 1: End-to-End Planning (v2 vs v3)")
    print("=" * 80)

    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])
    ndim = 7
    obs_counts = [6, 8] if quick else [6, 8, 12]
    n_seeds = 2 if quick else 3
    max_boxes = 200 if quick else 400

    robot = load_robot("panda")
    v3_results = []

    for n_obs in obs_counts:
        for s in range(n_seeds):
            seed = 42 + s
            obs_data = _gen_obstacles(n_obs, seed)

            scene = Scene()
            for i, (mn, mx) in enumerate(obs_data):
                scene.add_obstacle(mn, mx, name=f"obs_{i}")

            cfg = PandaGCSConfig()
            cfg.seed = seed
            cfg.max_boxes = max_boxes
            cfg.n_obstacles = n_obs

            t_total = time.perf_counter()
            t0 = time.perf_counter()
            prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)
            # NOTE: do NOT join cache thread — cache save is an async
            # persistence optimization, not part of planning cost.
            # v2 doesn't save cache after every grow either.
            grow_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            raw = run_method_with_bridge(
                _solve_method_dijkstra, "Dijkstra",
                prep, cfg, q_start, q_goal, ndim)
            solve_ms = (time.perf_counter() - t0) * 1000
            total_ms = (time.perf_counter() - t_total) * 1000

            success = raw is not None and raw.get('success', False)
            cost = raw.get('cost', float('inf')) if raw else float('inf')

            v3_results.append({
                "n_obs": n_obs, "seed": seed,
                "success": success, "cost": cost,
                "n_boxes": len(prep['boxes']),
                "grow_ms": grow_ms, "solve_ms": solve_ms,
                "total_ms": total_ms,
            })
            print(f"  v3 [{n_obs}obs, s={seed}]: "
                  f"{'OK' if success else 'FAIL'} "
                  f"grow={grow_ms:.0f}ms solve={solve_ms:.0f}ms "
                  f"total={total_ms:.0f}ms boxes={len(prep['boxes'])}")

    # v2 via subprocess
    print("\n  Running v2 e2e benchmark in subprocess...")
    v2_data = _run_v2_bench("e2e", quick)
    v2_results = v2_data.get("e2e", [])

    combined = []
    if v2_results:
        print("\n  --- Comparison ---")
        for r3 in v3_results:
            key = (r3['n_obs'], r3['seed'])
            r2 = next((r for r in v2_results
                       if r['n_obs'] == key[0] and r['seed'] == key[1]), None)
            if r2 is None:
                continue
            ratio = r3['total_ms'] / r2['total_ms'] if r2['total_ms'] > 0 else float('inf')
            print(f"  [{key[0]}obs, s={key[1]}] "
                  f"v3={r3['total_ms']:.0f}ms  v2={r2['total_ms']:.0f}ms  "
                  f"ratio={ratio:.2f}x  "
                  f"(v3 boxes={r3['n_boxes']}, v2 boxes={r2['n_boxes']})")
            combined.append({
                "n_obs": key[0], "seed": key[1],
                "v3": r3, "v2": r2, "ratio": ratio,
            })
    else:
        print("  [WARNING] v2 results not available, showing v3 only")
        combined = [{"n_obs": r['n_obs'], "seed": r['seed'],
                     "v3": r, "v2": None, "ratio": None}
                    for r in v3_results]

    return combined


# ═══════════════════════════════════════════════════════════════════
# Test 2: Forest reuse (v3 only)
# ═══════════════════════════════════════════════════════════════════

def test2_forest_reuse(quick=False):
    from aabb.robot import load_robot
    from forest.scene import Scene
    from planner.pipeline import (
        PandaGCSConfig, grow_and_prepare, run_method_with_bridge,
        _solve_method_dijkstra,
    )

    print("\n" + "=" * 80)
    print("  Test 2: Forest Reuse — Multi-Query (v3)")
    print("=" * 80)

    robot = load_robot("panda")
    ndim = 7
    rng = np.random.default_rng(42)
    n_obs = 6
    max_boxes = 300
    n_queries = 5 if quick else 10

    scene = Scene()
    for i in range(n_obs):
        cx = float(rng.uniform(-0.5, 0.5))
        cy = float(rng.uniform(-0.5, 0.5))
        cz = float(rng.uniform(0.2, 0.7))
        h = float(rng.uniform(0.06, 0.12))
        scene.add_obstacle([cx - h, cy - h, cz - h],
                           [cx + h, cy + h, cz + h], name=f"obs_{i}")

    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])

    cfg = PandaGCSConfig()
    cfg.seed = 42
    cfg.max_boxes = max_boxes

    t0 = time.perf_counter()
    prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  Forest built: {len(prep['boxes'])} boxes, {build_ms:.0f}ms")

    query_times = []
    for k in range(n_queries):
        t0 = time.perf_counter()
        raw = run_method_with_bridge(
            _solve_method_dijkstra, "Dijkstra",
            prep, cfg, q_start, q_goal, ndim)
        solve_ms = (time.perf_counter() - t0) * 1000
        success = raw is not None and raw.get('success', False)
        query_times.append(solve_ms)
        if k < 3 or k == n_queries - 1:
            print(f"  Query {k+1}: {'OK' if success else 'FAIL'} "
                  f"solve={solve_ms:.0f}ms")

    avg_query = float(np.mean(query_times))
    print(f"\n  Average query time: {avg_query:.1f}ms")
    print(f"  Build-once amortized ({n_queries} queries): "
          f"{build_ms / n_queries + avg_query:.1f}ms per query")

    return {
        "build_ms": build_ms, "n_queries": n_queries,
        "query_times": query_times, "avg_query_ms": avg_query,
        "amortized_ms": build_ms / n_queries + avg_query,
    }


# ═══════════════════════════════════════════════════════════════════
# Test 3: Incremental obstacle update
# ═══════════════════════════════════════════════════════════════════

def test3_incremental(quick=False):
    import copy
    from aabb.robot import load_robot
    from forest.scene import Scene
    from planner.pipeline import (
        PandaGCSConfig, grow_and_prepare, incremental_obstacle_update,
    )

    print("\n" + "=" * 80)
    print("  Test 3: Incremental Obstacle Update (v2 vs v3)")
    print("=" * 80)

    robot = load_robot("panda")
    ndim = 7
    n_obs = 8
    max_boxes = 300
    regrow_budget = 40

    q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
    q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])

    n_seeds = 2 if quick else 3
    v3_results = []

    for s in range(n_seeds):
        seed = 100 + s
        rng = np.random.default_rng(seed)
        obs_data = []
        for i in range(n_obs):
            cx = float(rng.uniform(-0.6, 0.6))
            cy = float(rng.uniform(-0.6, 0.6))
            cz = float(rng.uniform(0.2, 0.8))
            h = float(rng.uniform(0.06, 0.12))
            obs_data.append(([cx - h, cy - h, cz - h],
                             [cx + h, cy + h, cz + h]))

        scene = Scene()
        for i, (mn, mx) in enumerate(obs_data):
            scene.add_obstacle(mn, mx, name=f"obs_{i}")

        cfg = PandaGCSConfig()
        cfg.seed = seed
        cfg.max_boxes = max_boxes

        t0 = time.perf_counter()
        prep = grow_and_prepare(robot, scene, cfg, q_start, q_goal, ndim)
        build_ms = (time.perf_counter() - t0) * 1000
        print(f"\n  [v3 seed={seed}] build: {len(prep['boxes'])} boxes, "
              f"{build_ms:.0f}ms")

        if len(prep['boxes']) == 0:
            print(f"  [SKIP] No boxes grown for seed {seed}")
            continue

        new_obs = {'min_point': [0.1, 0.1, 0.3],
                   'max_point': [0.25, 0.25, 0.55],
                   'name': 'new_obs_test'}

        incr_scene = Scene()
        for obs in scene.get_obstacles():
            incr_scene.add_obstacle(
                list(obs.min_point), list(obs.max_point), name=obs.name)

        incr_prep = dict(prep)
        incr_prep['boxes'] = dict(prep['boxes'])
        fo = copy.copy(prep['forest_obj'])
        fo.boxes = dict(prep['forest_obj'].boxes)
        fo.adjacency = {k: set(v) for k, v in prep['forest_obj'].adjacency.items()}
        fo._interval_ids = list(prep['forest_obj']._interval_ids)
        fo._interval_id_to_index = dict(prep['forest_obj']._interval_id_to_index)
        if hasattr(prep['forest_obj'], '_intervals_arr') and prep['forest_obj']._intervals_arr is not None:
            fo._intervals_arr = prep['forest_obj']._intervals_arr.copy()
        incr_prep['forest_obj'] = fo

        change_rng = np.random.default_rng(seed + 1000)
        t0 = time.perf_counter()
        stats = incremental_obstacle_update(
            prep=incr_prep, scene=incr_scene,
            added_obstacles=[new_obs],
            removed_obstacle_names=[],
            regrow_budget=regrow_budget, rng=change_rng,
        )
        incr_ms = (time.perf_counter() - t0) * 1000

        full_scene = Scene()
        for i, (mn, mx) in enumerate(obs_data):
            full_scene.add_obstacle(mn, mx, name=f"obs_{i}")
        full_scene.add_obstacle(
            new_obs['min_point'], new_obs['max_point'], name='new_obs_test')

        t0 = time.perf_counter()
        full_prep = grow_and_prepare(robot, full_scene, cfg, q_start, q_goal, ndim)
        full_ms = (time.perf_counter() - t0) * 1000

        # invalidate-only = invalidate + scene update + UF rebuild (no regrow)
        inval_only_ms = (stats['invalidate_ms'] + stats['scene_ms']
                         + stats['uf_rebuild_ms'])
        speedup = full_ms / incr_ms if incr_ms > 0 else float('inf')
        speedup_inval = full_ms / inval_only_ms if inval_only_ms > 0 else float('inf')
        print(f"  [v3] inval-only: {inval_only_ms:.0f}ms "
              f"(inval={stats['invalidate_ms']:.0f}ms, "
              f"scene={stats['scene_ms']:.0f}ms, "
              f"uf={stats['uf_rebuild_ms']:.0f}ms) "
              f"invalidated={stats['n_invalidated']}")
        print(f"  [v3] incr(+regrow): {incr_ms:.0f}ms "
              f"(regrow={stats['regrow_ms']:.0f}ms, "
              f"regrown={stats['n_regrown']})")
        print(f"  [v3] full rebuild: {full_ms:.0f}ms")
        print(f"  [v3] speedup: inval-only={speedup_inval:.1f}x, "
              f"with-regrow={speedup:.1f}x")

        v3_results.append({
            "seed": seed, "build_ms": build_ms,
            "inval_only_ms": inval_only_ms,
            "incr_ms": incr_ms, "full_ms": full_ms,
            "speedup": speedup,
            "speedup_inval_only": speedup_inval,
            "invalidate_ms": stats['invalidate_ms'],
            "scene_ms": stats['scene_ms'],
            "uf_rebuild_ms": stats['uf_rebuild_ms'],
            "regrow_ms": stats['regrow_ms'],
            "n_invalidated": stats['n_invalidated'],
            "n_regrown": stats['n_regrown'],
        })

    # v2 subprocess
    print("\n  Running v2 incremental benchmark in subprocess...")
    v2_data = _run_v2_bench("incr", quick)
    v2_results = v2_data.get("incr", [])

    combined = []
    if v2_results:
        print("\n  --- v2 vs v3 Incremental Comparison (apples-to-apples) ---")
        print("  NOTE: 'inval-only' excludes regrow for fair comparison")
        print("         v2 incremental does NOT include regrow")
        for r3 in v3_results:
            r2 = next((r for r in v2_results if r['seed'] == r3['seed']), None)
            if r2:
                v2_inval = r2.get('invalidate_ms', r2['incr_ms'])
                v3_inval = r3['inval_only_ms']
                print(f"\n  seed={r3['seed']}:")
                print(f"    [invalidate-only] v3={v3_inval:.0f}ms  "
                      f"v2={v2_inval:.0f}ms  "
                      f"ratio={v3_inval / v2_inval:.2f}x" if v2_inval > 0
                      else f"    [invalidate-only] v3={v3_inval:.0f}ms  v2=0ms")
                print(f"    [full incr + regrow] v3={r3['incr_ms']:.0f}ms  "
                      f"v2={r2['incr_ms']:.0f}ms (no regrow)")
                print(f"    [full rebuild]  v3={r3['full_ms']:.0f}ms  "
                      f"v2={r2['full_ms']:.0f}ms")
                print(f"    [speedup vs full] v3-inval={r3['speedup_inval_only']:.1f}x  "
                      f"v3-regrow={r3['speedup']:.1f}x  "
                      f"v2={r2['speedup']:.1f}x")
                combined.append({"v3": r3, "v2": r2})
            else:
                combined.append({"v3": r3, "v2": None})
    else:
        print("  [WARNING] v2 results not available")
        combined = [{"v3": r, "v2": None} for r in v3_results]

    return combined


# ═══════════════════════════════════════════════════════════════════
# Test 4: AABB performance
# ═══════════════════════════════════════════════════════════════════

def test4_aabb(quick=False):
    from aabb.robot import load_robot
    from aabb.interval_fk import compute_interval_aabb

    print("\n" + "=" * 80)
    print("  Test 4: AABB / Interval FK Performance")
    print("=" * 80)

    robot = load_robot("panda")
    n_trials = 100 if quick else 500

    is_cython = False
    try:
        from aabb._interval_fk_cy import compute_fk_full_cy
        is_cython = True
    except ImportError:
        pass
    print(f"\n  Cython available: {is_cython}")

    # Determine zero-length links
    zero_links = set()
    for i, p in enumerate(robot.dh_params):
        if abs(p.get('a', 0)) < 1e-9 and abs(p.get('d', 0)) < 1e-9:
            zero_links.add(i)
    print(f"  Zero-length links: {zero_links}")

    rng = np.random.default_rng(42)
    times_us = []

    for _ in range(n_trials):
        q = rng.uniform(-np.pi, np.pi, size=7)
        dq = rng.uniform(0.1, 1.0, size=7)
        intervals = [(float(q[i] - dq[i]), float(q[i] + dq[i])) for i in range(7)]

        t0 = time.perf_counter()
        compute_interval_aabb(robot, intervals, zero_links)
        elapsed = (time.perf_counter() - t0) * 1e6
        times_us.append(elapsed)

    avg = float(np.mean(times_us))
    p50 = float(np.percentile(times_us, 50))
    p95 = float(np.percentile(times_us, 95))

    print(f"  compute_interval_aabb: "
          f"avg={avg:.1f}μs, p50={p50:.1f}μs, p95={p95:.1f}μs "
          f"({n_trials} trials)")

    return {"cython": is_cython, "n_trials": n_trials,
            "avg_us": avg, "p50_us": p50, "p95_us": p95}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    tests = [1, 2, 3, 4]
    if args.only:
        tests = [int(x) for x in args.only.split(",")]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {"timestamp": ts, "quick": args.quick}

    print("=" * 80)
    print(f"  v2 vs v3 Performance Comparison")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Tests: {tests}")
    print("=" * 80)

    if 1 in tests:
        all_results["test1_e2e"] = test1_e2e(args.quick)
    if 2 in tests:
        all_results["test2_forest_reuse"] = test2_forest_reuse(args.quick)
    if 3 in tests:
        all_results["test3_incremental"] = test3_incremental(args.quick)
    if 4 in tests:
        all_results["test4_aabb"] = test4_aabb(args.quick)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if "test1_e2e" in all_results:
        data = all_results["test1_e2e"]
        ratios = [r['ratio'] for r in data if r.get('ratio') is not None]
        if ratios:
            print(f"\n  Test 1 (E2E): v3/v2 ratio = "
                  f"{np.mean(ratios):.2f}x ± {np.std(ratios):.2f}")

    if "test2_forest_reuse" in all_results:
        r = all_results["test2_forest_reuse"]
        print(f"\n  Test 2 (Reuse): build={r['build_ms']:.0f}ms, "
              f"avg query={r['avg_query_ms']:.1f}ms, "
              f"amortized={r['amortized_ms']:.1f}ms")

    if "test3_incremental" in all_results:
        for item in all_results["test3_incremental"]:
            r3 = item.get('v3', {})
            r2 = item.get('v2')
            s = r3.get('seed', '?')
            print(f"\n  Test 3 (Incr, seed={s}):")
            print(f"    v3: inval-only={r3.get('inval_only_ms',0):.0f}ms, "
                  f"incr+regrow={r3.get('incr_ms',0):.0f}ms, "
                  f"full={r3.get('full_ms',0):.0f}ms, "
                  f"speedup(inval)={r3.get('speedup_inval_only',0):.1f}x, "
                  f"speedup(+regrow)={r3.get('speedup',0):.1f}x")
            if r2:
                print(f"    v2: incr={r2.get('incr_ms',0):.0f}ms "
                      f"(inval={r2.get('invalidate_ms',0):.0f}ms), "
                      f"full={r2.get('full_ms',0):.0f}ms, "
                      f"speedup={r2.get('speedup',0):.1f}x")

    if "test4_aabb" in all_results:
        r = all_results["test4_aabb"]
        print(f"\n  Test 4 (AABB): avg={r['avg_us']:.1f}μs, "
              f"cython={'YES' if r['cython'] else 'NO'}")

    out_path = OUT_DIR / f"v2_v3_comparison_{ts}.json"
    def _ser(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_ser, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
