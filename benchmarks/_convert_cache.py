"""Convert existing v1 cache to v2 native numpy format."""
from box_aabb.robot import load_robot
from planner.hier_aabb_tree import HierAABBTree

robot = load_robot('panda')
jl = list(robot.joint_limits)
tree = HierAABBTree.auto_load(robot, jl)
stats = tree.get_stats()
print(f"Loaded {stats['n_nodes']} nodes")
tree.auto_save()
print("Saved as v2 format")
tree2 = HierAABBTree.auto_load(robot, jl)
s2 = tree2.get_stats()
print(f"Reloaded {s2['n_nodes']} nodes - v2 verified")
