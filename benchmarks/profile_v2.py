# Profile critical vs random - ASCII only output
import sys, os, time, random as rng
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
from box_aabb import create_panda_robot, AABBCalculator

robot = create_panda_robot()
calc = AABBCalculator(robot, "Panda")

# Patch to add timing
orig_optimize = AABBCalculator._optimize_extremes
orig_evaluate = AABBCalculator._evaluate_samples

timings = {'eval_time': 0, 'eval_count': 0, 'opt_time': 0, 'opt_calls': 0}

def reset():
    for k in timings:
        timings[k] = 0

_real_opt = calc._optimize_extremes.__func__ if hasattr(calc._optimize_extremes, '__func__') else orig_optimize

def patched_opt(self, *a, **kw):
    t0 = time.perf_counter()
    timings['opt_calls'] += 1
    _real_opt(self, *a, **kw)
    timings['opt_time'] += time.perf_counter() - t0

_real_eval = calc._evaluate_samples.__func__ if hasattr(calc._evaluate_samples, '__func__') else orig_evaluate

def patched_eval(self, link_idx, samples, seg_extremes, n_sub):
    t0 = time.perf_counter()
    timings['eval_count'] += len(samples)
    _real_eval(self, link_idx, samples, seg_extremes, n_sub)
    timings['eval_time'] += time.perf_counter() - t0

AABBCalculator._optimize_extremes = patched_opt
AABBCalculator._evaluate_samples = patched_eval

# Count scipy calls
import scipy.optimize
_real_minimize = scipy.optimize.minimize
scipy_stats = {'calls': 0, 'time': 0.0, 'fevals': 0}

def counted_minimize(fun, x0, **kwargs):
    scipy_stats['calls'] += 1
    t0 = time.perf_counter()
    r = _real_minimize(fun, x0, **kwargs)
    scipy_stats['time'] += time.perf_counter() - t0
    scipy_stats['fevals'] += getattr(r, 'nfev', 0)
    return r

scipy.optimize.minimize = counted_minimize

# Test intervals
rng.seed(42)
np.random.seed(42)
intervals = [(-0.5 + rng.uniform(-0.2, 0.2), 0.5 + rng.uniform(-0.2, 0.2)) for _ in range(7)]
intervals.append((0.0, 0.0))

print("=" * 60)
print("PROFILING CRITICAL")
print("=" * 60)
reset()
for k in scipy_stats: scipy_stats[k] = 0
t0 = time.perf_counter()
rc = calc.compute_envelope(intervals, method='numerical', sampling='critical')
tc = time.perf_counter() - t0
print(f"Total: {tc:.4f}s")
print(f"  evaluate:  {timings['eval_time']:.4f}s  ({timings['eval_count']} samples)")
print(f"  optimize:  {timings['opt_time']:.4f}s  ({timings['opt_calls']} calls)")
print(f"  other:     {tc - timings['eval_time'] - timings['opt_time']:.4f}s")
print(f"  scipy.minimize: {scipy_stats['calls']} calls, {scipy_stats['time']:.4f}s, {scipy_stats['fevals']} fevals")
vc = rc.total_volume()
print(f"  Volume: {vc:.6f}")

print()
print("=" * 60)
print("PROFILING RANDOM (n=5000)")
print("=" * 60)
reset()
for k in scipy_stats: scipy_stats[k] = 0
t0 = time.perf_counter()
rr = calc.compute_envelope(intervals, method='numerical', sampling='random', n_random_samples=5000)
tr = time.perf_counter() - t0
print(f"Total: {tr:.4f}s")
print(f"  evaluate:  {timings['eval_time']:.4f}s  ({timings['eval_count']} samples)")
print(f"  optimize:  {timings['opt_time']:.4f}s  ({timings['opt_calls']} calls)")
print(f"  other:     {tr - timings['eval_time'] - timings['opt_time']:.4f}s")
print(f"  scipy.minimize: {scipy_stats['calls']} calls, {scipy_stats['time']:.4f}s, {scipy_stats['fevals']} fevals")
vr = rr.total_volume()
print(f"  Volume: {vr:.6f}")

print()
print("=" * 60)
ratio = tc / tr if tr > 0 else 0
winner = "Critical FASTER" if tc < tr else "Random FASTER"
print(f"SUMMARY: critical={tc:.3f}s  random={tr:.3f}s  ratio={ratio:.2f}x  [{winner}]")
print(f"Vol critical={vc:.6f}  Vol random={vr:.6f}  diff={vc-vr:.6f}")
gap = "NONE" if vc >= vr - 0.005 else f"Gap={vr-vc:.6f}"
print(f"Quality check: {gap}")

scipy.optimize.minimize = _real_minimize
AABBCalculator._optimize_extremes = orig_optimize
AABBCalculator._evaluate_samples = orig_evaluate
