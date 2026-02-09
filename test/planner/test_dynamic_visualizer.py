"""
test/planner/test_dynamic_visualizer.py - 动态可视化测试
"""

import pytest
import numpy as np

from box_aabb.robot import load_robot
from planner.dynamic_visualizer import animate_robot_path, resample_path
from planner.obstacles import Scene


def _has_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:
        return False


@pytest.fixture
def robot_2dof():
    return load_robot('2dof_planar')


@pytest.fixture
def simple_path():
    """简单 2DOF 路径"""
    return [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.3]),
        np.array([1.0, 0.8]),
        np.array([1.5, 1.0]),
    ]


class TestResamplePath:
    """路径重采样测试"""

    def test_resample_basic(self, simple_path):
        resampled = resample_path(simple_path, n_frames=10)
        assert len(resampled) == 10

        # 首尾应接近原始路径
        np.testing.assert_array_almost_equal(resampled[0], simple_path[0])
        np.testing.assert_array_almost_equal(resampled[-1], simple_path[-1])

    def test_resample_single_point(self):
        path = [np.array([1.0, 2.0])]
        resampled = resample_path(path, n_frames=5)
        assert len(resampled) == 1

    def test_resample_two_points(self):
        path = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        resampled = resample_path(path, n_frames=5)
        assert len(resampled) == 5


class TestAnimateRobotPath:
    """动画生成测试"""

    @pytest.mark.skipif(
        not _has_matplotlib(),
        reason="matplotlib 不可用")
    def test_animate_2d(self, robot_2dof, simple_path):
        import matplotlib
        matplotlib.use('Agg')  # 非交互后端

        anim = animate_robot_path(
            robot_2dof, simple_path,
            fps=10, trail_length=5)

        assert anim is not None

    @pytest.mark.skipif(
        not _has_matplotlib(),
        reason="matplotlib 不可用")
    def test_animate_with_scene(self, robot_2dof, simple_path):
        import matplotlib
        matplotlib.use('Agg')

        scene = Scene()
        scene.add_obstacle([0.8, -0.3], [1.2, 0.3], name="block")

        anim = animate_robot_path(
            robot_2dof, simple_path, scene=scene,
            fps=10, ghost_interval=2)

        assert anim is not None

    def test_animate_too_short_path(self, robot_2dof):
        with pytest.raises(ValueError, match="至少需要 2 个路径点"):
            animate_robot_path(robot_2dof, [np.array([0.0, 0.0])])


def _has_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:
        return False
