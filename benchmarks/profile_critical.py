"""Profile critical vs random to find the bottleneck"""
import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
from box_aabb import create_panda_robot, AABBCalculator

robot = create_panda_robot()
calc = AABBCalculator(robot, "Panda")

# Monkey-patch to add timing
original_optimize = calc._optimize_extremes
original_evaluate = calc._evaluate_samples
original_gen_critical = calc._generate_critical_points
original_gen_manifold = calc._generate_manifold_random
original_build = calc._build_link_aabbs
original_prepare = calc._prepare_link
original_expand = calc._expand_reduced

timings = {}

def reset_timings():
    global timings
    timings = {
        'prepare_link': 0, 'gen_critical': 0, 'gen_manifold': 0,
        'expand_reduced': 0, 'evaluate_samples': 0,
        'optimize_extremes': 0, 'build_aabbs': 0,
        'optimize_calls': 0, 'optimize_seeds_total': 0,
        'optimize_scipy_calls': 0, 'evaluate_count': 0,
    }

def timed_optimize(self_ref, link_idx, sorted_rel, mid_q, intervals,
                   seg_extremes, n_sub, extra_seeds=None, n_seeds=3,
                   skip_current_best=False):
    t0 = time.perf_counter()
    # Count seeds that will be used
    seed_count = 0
    if extra_seeds:
        seed_count += min(n_seeds, len(extra_seeds))
    if not skip_current_best:
        seed_count += n_sub  # approx
    timings['optimize_calls'] += 1
    timings['optimize_seeds_total'] += seed_count
    original_optimize(link_idx, sorted_rel, mid_q, intervals,
                      seg_extremes, n_sub, extra_seeds, n_seeds, skip_current_best)
    timings['optimize_extremes'] += time.perf_counter() - t0

def timed_evaluate(self_ref, link_idx, samples, seg_extremes, n_sub):
    t0 = time.perf_counter()
    timings['evaluate_count'] += len(samples)
    original_evaluate(link_idx, samples, seg_extremes, n_sub)
    timings['evaluate_samples'] += time.perf_counter() - t0

# Wrap class methods
import types
calc._optimize_extremes_orig = original_optimize
calc._evaluate_samples_orig = original_evaluate

def patched_optimize(link_idx, sorted_rel, mid_q, intervals,
                     seg_extremes, n_sub, extra_seeds=None, n_seeds=3,
                     skip_current_best=False):
    t0 = time.perf_counter()
    timings['optimize_calls'] += 1
    original_optimize(link_idx, sorted_rel, mid_q, intervals,
                      seg_extremes, n_sub, extra_seeds, n_seeds, skip_current_best)
    timings['optimize_extremes'] += time.perf_counter() - t0

def patched_evaluate(link_idx, samples, seg_extremes, n_sub):
    t0 = time.perf_counter()
    timings['evaluate_count'] += len(samples)
    original_evaluate(link_idx, samples, seg_extremes, n_sub)
    timings['evaluate_samples'] += time.perf_counter() - t0

calc._optimize_extremes = patched_optimize
calc._evaluate_samples = patched_evaluate

# Test with same setup as compare script
random.seed(42)
np.random.seed(42)
intervals = [(-0.5 + random.uniform(-0.2, 0.2), 0.5 + random.uniform(-0.2, 0.2)) for _ in range(7)]
intervals.append((0.0, 0.0))

print("=" * 60)
print("PROFILING CRITICAL SAMPLING")
print("=" * 60)
reset_timings()
t_start = time.perf_counter()

# Time critical point generation separately
t0 = time.perf_counter()
for li in range(1, robot.n_joints + 1):
    prep = calc._prepare_link(li, intervals, True)
    if prep is None:
        continue
    rel, srel, rivl, mq = prep
    pts, cpts = AABBCalculator._generate_critical_points(rivl)
timings['gen_critical'] = time.perf_counter() - t0
print(f"Critical point gen (dry run): {timings['gen_critical']:.4f}s, {len(pts)} pts, {len(cpts)} cpts")

# Now do full critical
reset_timings()
t_start = time.perf_counter()
result_c = calc.compute_envelope(intervals, method='numerical', sampling='critical')
t_critical = time.perf_counter() - t_start

print(f"\nTotal critical time: {t_critical:.4f}s")
print(f"  evaluate_samples:  {timings['evaluate_samples']:.4f}s  ({timings['evaluate_count']} samples)")
print(f"  optimize_extremes: {timings['optimize_extremes']:.4f}s  ({timings['optimize_calls']} calls)")
print(f"  other:             {t_critical - timings['evaluate_samples'] - timings['optimize_extremes']:.4f}s")

# Reset and test random
calc._optimize_extremes = patched_optimize
calc._evaluate_samples = patched_evaluate
print("\n" + "=" * 60)
print("PROFILING RANDOM SAMPLING (n=5000)")
print("=" * 60)
reset_timings()
t_start = time.perf_counter()
result_r = calc.compute_envelope(intervals, method='numerical', sampling='random',
                                  n_random_samples=5000)
t_random = time.perf_counter() - t_start

print(f"\nTotal random time: {t_random:.4f}s")
print(f"  evaluate_samples:  {timings['evaluate_samples']:.4f}s  ({timings['evaluate_count']} samples)")
print(f"  optimize_extremes: {timings['optimize_extremes']:.4f}s  ({timings['optimize_calls']} calls)")
print(f"  other:             {t_random - timings['evaluate_samples'] - timings['optimize_extremes']:.4f}s")

print(f"\n{'='*60}")
print(f"SUMMARY: critical={t_critical:.3f}s  random={t_random:.3f}s  ratio={t_critical/t_random:.2f}x")

# Now profile optimize in detail
print(f"\n{'='*60}")
print("PROFILING OPTIMIZE IN DETAIL")
print("=" * 60)

# Monkey-patch scipy.optimize.minimize to count calls
from scipy.optimize import minimize as scipy_minimize
scipy_call_count = [0]
scipy_total_time = [0.0]
scipy_feval_count = [0]

def counted_minimize(fun, x0, **kwargs):
    scipy_call_count[0] += 1
    t0 = time.perf_counter()
    result = scipy_minimize(fun, x0, **kwargs)
    scipy_total_time[0] += time.perf_counter() - t0
    scipy_feval_count[0] += result.nfev if hasattr(result, 'nfev') else 0
    return result

import scipy.optimize
original_scipy_minimize = scipy.optimize.minimize
scipy.optimize.minimize = counted_minimize

# Re-run critical
reset_timings()
scipy_call_count[0] = 0
scipy_total_time[0] = 0.0
scipy_feval_count[0] = 0

t_start = time.perf_counter()
result_c2 = calc.compute_envelope(intervals, method='numerical', sampling='critical')
t_critical2 = time.perf_counter() - t_start

print(f"\nCritical: scipy.minimize called {scipy_call_count[0]} times")
print(f"  Total scipy time: {scipy_total_time[0]:.4f}s")
print(f"  Total f-evals: {scipy_feval_count[0]}")
print(f"  Avg per call: {scipy_total_time[0]/max(scipy_call_count[0],1)*1000:.2f}ms")
print(f"  optimize wrapper time: {timings['optimize_extremes']:.4f}s")

# Re-run random
reset_timings()
scipy_call_count[0] = 0
scipy_total_time[0] = 0.0
scipy_feval_count[0] = 0

t_start = time.perf_counter()
result_r2 = calc.compute_envelope(intervals, method='numerical', sampling='random',
                                   n_random_samples=5000)
t_random2 = time.perf_counter() - t_start

print(f"\nRandom: scipy.minimize called {scipy_call_count[0]} times")
print(f"  Total scipy time: {scipy_total_time[0]:.4f}s")
print(f"  Total f-evals: {scipy_feval_count[0]}")
print(f"  Avg per call: {scipy_total_time[0]/max(scipy_call_count[0],1)*1000:.2f}ms")
print(f"  optimize wrapper time: {timings['optimize_extremes']:.4f}s")

# Restore
scipy.optimize.minimize = original_scipy_minimize
