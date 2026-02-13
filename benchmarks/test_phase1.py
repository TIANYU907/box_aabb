"""Phase 1 compact AABB test: size reduction + load speedup + correctness"""
import time
import tempfile
from pathlib import Path
import numpy as np

from box_aabb.robot import load_robot
from planner.hier_aabb_tree import HierAABBTree

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bench_panda_forest import random_scene_3d

robot = load_robot("panda")
jl = list(robot.joint_limits)

# 1. load existing v1 cache
t0 = time.time()
tree = HierAABBTree.auto_load(robot, jl)
t_load_v1 = time.time() - t0
stats = tree.get_stats()
print(f"v1 load: {t_load_v1:.3f}s  nodes={stats['n_nodes']}  fk={stats['n_fk_calls']}")

# 2. save as v2 compact
tmp = Path(tempfile.mkdtemp()) / "test_v2.pkl"
t0 = time.time()
tree.save(str(tmp))
t_save_v2 = time.time() - t0
sz_v2 = tmp.stat().st_size
print(f"v2 save: {t_save_v2:.3f}s  size={sz_v2/1024/1024:.2f}MB")

# check old cache size
cache_dir = HierAABBTree._global_cache_dir()
old_fn = cache_dir / HierAABBTree._cache_filename(robot)
if old_fn.exists():
    sz_v1 = old_fn.stat().st_size
    print(f"v1 size: {sz_v1/1024/1024:.2f}MB  ratio: {sz_v2/sz_v1:.2f}x")

# 3. load v2
t0 = time.time()
tree2 = HierAABBTree.load(str(tmp), robot)
t_load_v2 = time.time() - t0
stats2 = tree2.get_stats()
print(f"v2 load: {t_load_v2:.3f}s  nodes={stats2['n_nodes']}  fk={stats2['n_fk_calls']}")
if t_load_v2 > 0:
    print(f"load speedup: {t_load_v1/t_load_v2:.2f}x")

# 4. verify find_free_box still works
rng = np.random.default_rng(42)
scene = random_scene_3d(5, rng)
obs = scene.get_obstacles()

seed = np.array([0.0] * 7)
result = tree2.find_free_box(seed, obs, max_depth=20)
if result:
    print(f"find_free_box OK: widths={[f'{hi-lo:.3f}' for lo, hi in result.intervals]}")
else:
    print("find_free_box: None (may be expected)")

# 5. verify round-trip: save v2 -> load -> save v2 -> load -> same stats
tmp2 = Path(tempfile.mkdtemp()) / "test_v2_rt.pkl"
tree2.save(str(tmp2))
tree3 = HierAABBTree.load(str(tmp2), robot)
stats3 = tree3.get_stats()
stats2_after = tree2.get_stats()
assert stats2_after['n_nodes'] == stats3['n_nodes'], \
    f"node count mismatch: {stats2_after['n_nodes']} vs {stats3['n_nodes']}"
print(f"\nround-trip OK: {stats3['n_nodes']} nodes preserved")
print("Phase 1 PASSED")
