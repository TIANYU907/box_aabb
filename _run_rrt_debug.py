"""Debug runner for rrt_family_panda."""
import traceback
import sys
import faulthandler
faulthandler.enable()

sys.argv = ['x', '--seed', '42', '--timeout', '15', '--trials', '1']

print("Step 1: importing modules...", flush=True)
try:
    import numpy as np
    from pathlib import Path
    _root = Path(__file__).resolve().parent / "v2"
    sys.path.insert(0, str(_root / "src"))
    sys.path.insert(0, str(_root))

    print("  1a: loading bootstrap...", flush=True)
    from v2._bootstrap import add_v2_paths
    add_v2_paths()

    print("  1b: loading robot...", flush=True)
    from aabb.robot import load_robot
    robot = load_robot("panda")
    print(f"  robot loaded: {robot.n_joints} joints", flush=True)

    print("  1c: loading scene/checker...", flush=True)
    from forest.scene import Scene
    from forest.collision import CollisionChecker
    from common.output import make_output_dir

    print("  1d: loading gcs_planner helpers...", flush=True)
    from v2.examples.gcs_planner_panda import PandaGCSConfig, build_panda_scene

    print("Step 2: building scene...", flush=True)
    cfg = PandaGCSConfig()
    seed = 42
    cfg.seed = seed
    rng = np.random.default_rng(seed)
    q_start = np.array(cfg.q_start, dtype=np.float64)
    q_goal = np.array(cfg.q_goal, dtype=np.float64)
    print(f"  q_start = {q_start}", flush=True)
    print(f"  q_goal  = {q_goal}", flush=True)

    print("  2a: calling build_panda_scene...", flush=True)
    scene = build_panda_scene(rng, cfg, robot, q_start, q_goal)
    print(f"  scene built: {scene.n_obstacles} obstacles", flush=True)

    print("  2b: creating checker...", flush=True)
    checker = CollisionChecker(robot=robot, scene=scene)
    print("  checker created", flush=True)

    print("Step 3: testing collision check...", flush=True)
    ok = checker.check_config(q_start)
    print(f"  q_start collision-free: {ok}", flush=True)

    print("Step 4: testing RRT import...", flush=True)
    from v2.examples.rrt_family_panda import plan_rrt
    print("  plan_rrt imported", flush=True)

    print("Step 5: running short RRT...", flush=True)
    result = plan_rrt(
        q_start=q_start, q_goal=q_goal,
        joint_limits=robot.joint_limits,
        checker=checker,
        timeout=5.0,
        step_size=0.5,
        resolution=0.05,
        goal_bias=0.05,
        goal_tol=0.3,
        seed=42,
    )
    print(f"  RRT result: success={result['success']}, "
          f"time={result['plan_time_s']:.3f}s, nodes={result['n_nodes']}", flush=True)

    print("DONE!", flush=True)

except SystemExit as e:
    print(f"SystemExit code={e.code}", flush=True)
    traceback.print_exc()
except BaseException:
    print("Exception caught:", flush=True)
    traceback.print_exc()
