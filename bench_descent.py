"""Benchmark: find_free_box Cython descent vs Python fallback"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'v2', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'v2'))

from aabb.robot import load_robot
from forest.hier_aabb_tree import HierAABBTree
import numpy as np

robot = load_robot('panda')
jl = robot.joint_limits
ndim = robot.n_joints

# Create tree fresh (no cache)
tree = HierAABBTree(robot, jl)
tree.warmup_fk_cache(6)

# Obstacles from the gcs_planner_panda scene
from forest.scene import Scene
from forest.models import BoxNode
rng = np.random.default_rng(42)
scene = Scene()
for i in range(6):
    center = rng.uniform([-0.6, -0.6, 0.2], [0.6, 0.6, 0.9])
    half = rng.uniform([0.05, 0.05, 0.05], [0.25, 0.25, 0.25])
    scene.add_obstacle(center - half, center + half, f"obs_{i}")
obstacles = scene.get_obstacles()

seed = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
safety_margin = 0.01
obs_packed = tree._prepack_obstacles_c(obstacles, safety_margin)

N = 500

# ── Cython descent ──
tree_cy = HierAABBTree(robot, jl)
tree_cy.warmup_fk_cache(6)
rng_cy = np.random.default_rng(123)

t0 = time.perf_counter()
n_none_cy = 0
for i in range(N):
    s = seed + rng_cy.uniform(-0.5, 0.5, ndim)
    s = np.clip(s, [lo for lo, hi in jl], [hi for lo, hi in jl])
    res = tree_cy.find_free_box(s, obstacles, max_depth=40,
                                 safety_margin=safety_margin, obs_packed=obs_packed)
    if res is None:
        n_none_cy += 1
t1 = time.perf_counter()
ms_cy = (t1 - t0) * 1000
cy_active = getattr(tree_cy, '_use_cy_descent', False)
print(f"Cython descent (active={cy_active}): {N} calls, {n_none_cy} none")
print(f"  Total: {ms_cy:.1f} ms  ({ms_cy/N:.3f} ms/call)")
print(f"  n_nodes={tree_cy.n_nodes}, n_fk={tree_cy.n_fk_calls}")

# ── Python fallback ──
tree_py = HierAABBTree(robot, jl)
tree_py.warmup_fk_cache(6)
tree_py._use_cy_descent = False  # Force Python path
rng_py = np.random.default_rng(123)

t0 = time.perf_counter()
n_none_py = 0
for i in range(N):
    s = seed + rng_py.uniform(-0.5, 0.5, ndim)
    s = np.clip(s, [lo for lo, hi in jl], [hi for lo, hi in jl])
    res = tree_py.find_free_box(s, obstacles, max_depth=40,
                                 safety_margin=safety_margin, obs_packed=obs_packed)
    if res is None:
        n_none_py += 1
t1 = time.perf_counter()
ms_py = (t1 - t0) * 1000
print(f"\nPython descent: {N} calls, {n_none_py} none")
print(f"  Total: {ms_py:.1f} ms  ({ms_py/N:.3f} ms/call)")
print(f"  n_nodes={tree_py.n_nodes}, n_fk={tree_py.n_fk_calls}")

print(f"\nSpeedup: {ms_py/ms_cy:.2f}x")
