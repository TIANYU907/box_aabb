import numpy as np

from v2._bootstrap import add_v2_paths

add_v2_paths()

from forest.models import BoxNode, PlannerConfig
from forest.box_forest import BoxForest
from common.output import make_output_dir


def main() -> None:
    forest = BoxForest(
        robot_fingerprint="demo",
        joint_limits=[(-1.0, 1.0), (-1.0, 1.0)],
        config=PlannerConfig(),
    )
    forest.add_box_direct(BoxNode(0, [(-0.8, -0.1), (-0.6, 0.6)], np.array([-0.4, 0.0])))
    forest.add_box_direct(BoxNode(1, [(0.1, 0.8), (-0.6, 0.6)], np.array([0.4, 0.0])))

    out = make_output_dir("reports", "forest_demo")
    save_path = out / "forest.pkl"
    forest.save(str(save_path))
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
