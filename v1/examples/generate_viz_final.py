"""
Generate visualization results for inspection.

Scenarios:
1. 3DOF wall obstacle: start=[0,2,0] -> goal=[0,-2,0], wall at far reach
2. 3DOF two-walls gap: start=[pi/2,0,0] -> goal=[-pi/2,0,0], two walls with narrow gap
3. Panda 7DOF empty scene: shows box decomposition in high dimensions
4-5. Collision heatmaps for both 3DOF scenarios
"""
import os, sys, logging
logging.disable(logging.CRITICAL)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from box_aabb.robot import load_robot
from planner import BoxRRT, Scene, PlannerConfig, evaluate_result
from planner.visualizer import (
    plot_cspace_boxes, plot_workspace_result, plot_cspace_with_collision)

D = os.path.dirname(__file__)
results_log = []

def save(fig, name):
    p = os.path.join(D, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    sz = os.path.getsize(p) // 1024
    results_log.append(f"  -> {name} ({sz} KB)")

def write_log():
    with open(os.path.join(D, 'viz_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_log) + '\n')

# =====================================================
# 1. 3DOF Wall Obstacle
# =====================================================
results_log.append("[1] 3DOF Wall Obstacle")
r3 = load_robot('3dof_planar')
lim3 = r3.joint_limits
s1 = Scene()
s1.add_obstacle([2.05, -0.3], [2.4, 0.3], name="wall")
c1 = PlannerConfig(
    max_iterations=600, max_box_nodes=200, seed_batch_size=5,
    expansion_resolution=0.03, max_expansion_rounds=3, goal_bias=0.15,
    connection_radius=3.0, connection_max_attempts=60, path_shortcut_iters=200,
    segment_collision_resolution=0.03, verbose=False)
p1 = BoxRRT(r3, s1, c1)
res1 = p1.plan(np.array([0., 2., 0.]), np.array([0., -2., 0.]), seed=42)
status1 = f"{'OK' if res1.success else 'FAIL'} pts={len(res1.path)} len={res1.path_length:.3f} t={res1.computation_time:.1f}s"
results_log.append(f"  {status1}")
write_log()
if res1.success:
    save(plot_cspace_boxes(res1, joint_limits=lim3, dim_x=0, dim_y=1,
         title="3DOF Wall: C-space (q0 vs q1)"), "viz_wall_cspace_q0q1.png")
    save(plot_cspace_boxes(res1, joint_limits=lim3, dim_x=0, dim_y=2,
         title="3DOF Wall: C-space (q0 vs q2)"), "viz_wall_cspace_q0q2.png")
    save(plot_workspace_result(r3, s1, res1, n_poses=12),
         "viz_wall_workspace.png")
    m1 = evaluate_result(res1, r3, s1)
    results_log.append(f"  metrics: ratio={m1.length_ratio:.2f}x smooth={m1.smoothness:.3f}")
    write_log()

# =====================================================
# 2. 3DOF Two-Walls Gap
# =====================================================
results_log.append("\n[2] 3DOF Two-Walls Gap")
s2 = Scene()
s2.add_obstacle([2.0, -1.5], [2.2, -0.15], name="wall_bottom")
s2.add_obstacle([2.0, 0.15], [2.2, 1.5], name="wall_top")
c2 = PlannerConfig(
    max_iterations=800, max_box_nodes=300, seed_batch_size=6,
    expansion_resolution=0.02, max_expansion_rounds=4, goal_bias=0.2,
    connection_radius=4.0, connection_max_attempts=80, path_shortcut_iters=200,
    segment_collision_resolution=0.02, verbose=False)
p2 = BoxRRT(r3, s2, c2)
res2 = p2.plan(np.array([1.57, 0., 0.]), np.array([-1.57, 0., 0.]), seed=42)
status2 = f"{'OK' if res2.success else 'FAIL'} pts={len(res2.path) if res2.path else 0} len={res2.path_length:.3f} t={res2.computation_time:.1f}s"
results_log.append(f"  {status2}")
write_log()
if res2.success:
    save(plot_cspace_boxes(res2, joint_limits=lim3, dim_x=0, dim_y=1,
         title="Gap: C-space (q0 vs q1)"), "viz_gap_cspace_q0q1.png")
    save(plot_cspace_boxes(res2, joint_limits=lim3, dim_x=0, dim_y=2,
         title="Gap: C-space (q0 vs q2)"), "viz_gap_cspace_q0q2.png")
    save(plot_workspace_result(r3, s2, res2, n_poses=12),
         "viz_gap_workspace.png")
    m2 = evaluate_result(res2, r3, s2)
    results_log.append(f"  metrics: ratio={m2.length_ratio:.2f}x smooth={m2.smoothness:.3f}")
    write_log()

# =====================================================
# 3. Panda 7DOF (empty scene, free planning)
# =====================================================
results_log.append("\n[3] Panda 7DOF (empty scene)")
rp = load_robot('panda')
limp = rp.joint_limits
sp = Scene()
cp = PlannerConfig(max_iterations=100, max_box_nodes=50, verbose=False)
qps = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785, 0.])
qpg = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.0, -0.3, 0.])
pp = BoxRRT(rp, sp, cp)
resp = pp.plan(qps, qpg, seed=42)
statusp = f"{'OK' if resp.success else 'FAIL'} pts={len(resp.path) if resp.path else 0} len={resp.path_length:.3f} t={resp.computation_time:.1f}s"
results_log.append(f"  {statusp}")
write_log()
if resp.success:
    save(plot_cspace_boxes(resp, joint_limits=limp, dim_x=0, dim_y=1,
         title="Panda (q0 vs q1)"), "viz_panda_cspace_q0q1.png")
    save(plot_cspace_boxes(resp, joint_limits=limp, dim_x=2, dim_y=3,
         title="Panda (q2 vs q3)"), "viz_panda_cspace_q2q3.png")
    save(plot_cspace_boxes(resp, joint_limits=limp, dim_x=4, dim_y=5,
         title="Panda (q4 vs q5)"), "viz_panda_cspace_q4q5.png")
    write_log()

# =====================================================
# 4. Collision heatmap: Wall obstacle
# =====================================================
results_log.append("\n[4] Collision heatmap: Wall (resolution=0.1)")
if res1.success:
    fig4 = plot_cspace_with_collision(r3, s1, joint_limits=lim3, result=res1,
        resolution=0.1, dim_x=0, dim_y=1)
    save(fig4, "viz_wall_collision_map.png")
write_log()

# =====================================================
# 5. Collision heatmap: Two-walls gap
# =====================================================
results_log.append("\n[5] Collision heatmap: Gap (resolution=0.1)")
if res2.success:
    fig5 = plot_cspace_with_collision(r3, s2, joint_limits=lim3, result=res2,
        resolution=0.1, dim_x=0, dim_y=1)
    save(fig5, "viz_gap_collision_map.png")
write_log()

# =====================================================
# Summary
# =====================================================
pngs = sorted(f for f in os.listdir(D) if f.startswith('viz_') and f.endswith('.png'))
results_log.append(f"\nTotal: {len(pngs)} images generated")
for f in pngs:
    sz = os.path.getsize(os.path.join(D, f)) // 1024
    results_log.append(f"  {f} ({sz} KB)")
write_log()
