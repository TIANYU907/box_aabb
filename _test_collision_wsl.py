#!/usr/bin/env python3
"""Test collision checker in WSL to verify it works with obstacles."""
import sys, numpy as np
sys.path.insert(0, "/mnt/c/Users/TIAN/Documents/box_aabb/v2/src")
sys.path.insert(0, "/mnt/c/Users/TIAN/Documents/box_aabb/v2")
from aabb.robot import load_robot
from forest.scene import Scene
from forest.collision import CollisionChecker

robot = load_robot("panda")
scene = Scene()
# 10 obstacles from the last run
obs_data = [
  {"min":[-0.383,0.435,0.579],"max":[-0.125,0.635,0.820]},
  {"min":[0.322,-0.198,0.331],"max":[0.616,0.230,0.801]},
  {"min":[-0.140,0.225,0.427],"max":[0.198,0.449,0.857]},
  {"min":[-0.124,0.551,0.646],"max":[0.371,0.831,1.023]},
  {"min":[-0.603,-0.191,0.047],"max":[-0.303,0.267,0.304]},
  {"min":[0.225,-0.544,0.673],"max":[0.685,-0.282,1.095]},
  {"min":[0.081,0.154,0.510],"max":[0.523,0.322,0.745]},
  {"min":[0.441,0.085,0.269],"max":[0.620,0.363,0.723]},
  {"min":[0.094,-0.346,0.732],"max":[0.271,0.231,0.980]},
  {"min":[-0.101,0.484,0.597],"max":[0.132,0.825,1.022]}
]
for od in obs_data:
    scene.add_obstacle(od["min"], od["max"])

checker = CollisionChecker(robot, scene)
q_start = np.array([0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5])
q_goal = np.array([-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8])

print("start collision:", checker.check_config_collision(q_start))
print("goal collision:", checker.check_config_collision(q_goal))

n_collide = 0
collide_indices = []
for i in range(101):
    t = i / 100.0
    q = q_start + t * (q_goal - q_start)
    if checker.check_config_collision(q):
        n_collide += 1
        collide_indices.append(i)

print(f"Straight line: {n_collide}/101 points collide")
if collide_indices:
    print(f"  collide at t = {[i/100.0 for i in collide_indices[:10]]}...")
print(f"scene obstacles: {scene.n_obstacles}")

# Also test: random configs
n_rand_coll = 0
rng = np.random.default_rng(42)
for _ in range(1000):
    q = rng.uniform([-2.8]*7, [2.8]*7)
    if checker.check_config_collision(q):
        n_rand_coll += 1
print(f"Random configs: {n_rand_coll}/1000 collide")
