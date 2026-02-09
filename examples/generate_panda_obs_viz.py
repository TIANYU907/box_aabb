"""
Generate Panda obstacle avoidance visualization.

Panda 7DOF with obstacle: start=[0,0,0,-1.57,0,1.57,0,0] -> goal=[1.57,0,0,-1.57,0,1.57,0,0]
Obstacle: box [0.15,0.10,0.50] -> [0.30,0.30,0.70] in workspace
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
    plot_cspace_boxes, plot_workspace_result)

D = os.path.dirname(__file__)

def save(fig, name):
    p = os.path.join(D, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")

print("=" * 60)
print("Panda 7DOF with obstacle")
print("=" * 60)

rp = load_robot('panda')
sp = Scene()
sp.add_obstacle([0.15, 0.10, 0.50], [0.30, 0.30, 0.70], name="sweep_block")

cfg = PlannerConfig(
    max_iterations=300,
    max_expansion_rounds=2,
    connection_radius=2.0,
    max_box_nodes=60,
    path_shortcut_iters=30,
    seed_batch_size=3,
)

start = np.array([0., 0., 0., -1.57, 0., 1.57, 0., 0.])
goal = np.array([1.57, 0., 0., -1.57, 0., 1.57, 0., 0.])

planner = BoxRRT(rp, sp, cfg)
result = planner.plan(start, goal, seed=42)

status = f"{'OK' if result.success else 'FAIL'}"
status += f" pts={len(result.path) if result.path else 0}"
status += f" len={result.path_length:.3f}"
status += f" t={result.computation_time:.1f}s"
status += f" trees={len(result.box_trees)} boxes={result.n_boxes_created}"
print(f"  Result: {status}")

if result.success:
    limp = rp.joint_limits

    # C-space box projections
    save(plot_cspace_boxes(result, joint_limits=limp, dim_x=0, dim_y=1,
         title="Panda+Obstacle (q0 vs q1)"), "viz_panda_obs_cspace_q0q1.png")
    save(plot_cspace_boxes(result, joint_limits=limp, dim_x=2, dim_y=3,
         title="Panda+Obstacle (q2 vs q3)"), "viz_panda_obs_cspace_q2q3.png")
    save(plot_cspace_boxes(result, joint_limits=limp, dim_x=0, dim_y=3,
         title="Panda+Obstacle (q0 vs q3)"), "viz_panda_obs_cspace_q0q3.png")

    # Workspace visualization with obstacle
    save(plot_workspace_result(rp, sp, result, n_poses=12),
         "viz_panda_obs_workspace.png")

    # Evaluate metrics
    m = evaluate_result(result, rp, sp)
    print(f"  Metrics: ratio={m.length_ratio:.2f}x smooth={m.smoothness:.3f}")
else:
    print("  Planning FAILED - no visualizations generated")

print("\nDone!")
