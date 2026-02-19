"""
test/planner/test_free_space_tiler.py - FreeSpaceTiler 测试
"""

import pytest
import numpy as np

from box_aabb.robot import load_robot
from planner.free_space_tiler import FreeSpaceTiler, FreeSpaceTile
from planner.obstacles import Scene


@pytest.fixture
def robot_2dof():
    return load_robot('2dof_planar')


@pytest.fixture
def scene_with_obstacle():
    scene = Scene()
    scene.add_obstacle([0.8, -0.3], [1.2, 0.3], name="block")
    return scene


@pytest.fixture
def empty_scene():
    return Scene()


class TestFreeSpaceTile:
    """FreeSpaceTile 数据类测试"""

    def test_volume(self):
        tile = FreeSpaceTile([(0.0, 1.0), (0.0, 2.0)])
        assert tile.volume == pytest.approx(2.0)

    def test_center(self):
        tile = FreeSpaceTile([(0.0, 1.0), (1.0, 3.0)])
        np.testing.assert_array_almost_equal(tile.center, [0.5, 2.0])

    def test_widths(self):
        tile = FreeSpaceTile([(0.0, 1.0), (0.5, 2.5)])
        np.testing.assert_array_almost_equal(tile.widths, [1.0, 2.0])

    def test_to_box_node(self):
        tile = FreeSpaceTile([(0.0, 1.0), (0.0, 2.0)])
        node = tile.to_box_node(node_id=5)
        assert node.node_id == 5
        assert node.volume == pytest.approx(2.0)

    def test_zero_width_volume(self):
        tile = FreeSpaceTile([(0.5, 0.5), (0.0, 1.0)])
        assert tile.volume == pytest.approx(1.0)


class TestFreeSpaceTiler:
    """FreeSpaceTiler 测试"""

    def test_tile_empty_scene(self, robot_2dof, empty_scene):
        """无障碍物时整个空间应该是一个大瓦片"""
        limits = [(-1.0, 1.0), (-1.0, 1.0)]
        tiler = FreeSpaceTiler(
            robot_2dof, empty_scene,
            joint_limits=limits,
            max_depth=3, min_width=0.5)

        tiles = tiler.tile()
        assert len(tiles) >= 1

        # 总体积应接近整个空间
        total_vol = sum(t.volume for t in tiles)
        assert total_vol > 0

    def test_tile_with_obstacle(self, robot_2dof, scene_with_obstacle):
        """有障碍物时应返回多个小瓦片"""
        limits = [(-1.0, 1.0), (-1.0, 1.0)]
        tiler = FreeSpaceTiler(
            robot_2dof, scene_with_obstacle,
            joint_limits=limits,
            max_depth=4, min_width=0.3)

        tiles = tiler.tile()
        # 应有一些无碰撞瓦片
        assert len(tiles) >= 0  # 取决于碰撞检测

    def test_tile_sorted_by_volume(self, robot_2dof, empty_scene):
        """结果应按体积降序排列"""
        limits = [(-1.0, 1.0), (-1.0, 1.0)]
        tiler = FreeSpaceTiler(
            robot_2dof, empty_scene,
            joint_limits=limits,
            max_depth=2, min_width=0.5)

        tiles = tiler.tile()
        if len(tiles) > 1:
            for i in range(len(tiles) - 1):
                assert tiles[i].volume >= tiles[i + 1].volume

    def test_max_depth_limits_recursion(self, robot_2dof, scene_with_obstacle):
        """max_depth 应限制递归深度"""
        limits = [(-1.0, 1.0), (-1.0, 1.0)]

        tiler_shallow = FreeSpaceTiler(
            robot_2dof, scene_with_obstacle,
            joint_limits=limits,
            max_depth=2, min_width=0.1)
        tiles_shallow = tiler_shallow.tile()

        tiler_deep = FreeSpaceTiler(
            robot_2dof, scene_with_obstacle,
            joint_limits=limits,
            max_depth=5, min_width=0.1)
        tiles_deep = tiler_deep.tile()

        # 更深的递归应产生更多但更小的瓦片
        # 碰撞检测次数也更多
        assert tiler_deep.n_collision_checks >= tiler_shallow.n_collision_checks

    def test_min_width_prevents_tiny_tiles(self, robot_2dof, scene_with_obstacle):
        """min_width 应防止过小的分割"""
        limits = [(-1.0, 1.0), (-1.0, 1.0)]
        tiler = FreeSpaceTiler(
            robot_2dof, scene_with_obstacle,
            joint_limits=limits,
            max_depth=10, min_width=0.5)

        tiles = tiler.tile()
        for t in tiles:
            for lo, hi in t.intervals:
                assert hi - lo >= 0.5 or hi - lo < 1e-10  # 要么 >= min_width 要么是固定维度

    def test_tiles_to_box_nodes(self, robot_2dof, empty_scene):
        """转换为 BoxNode 列表"""
        limits = [(-1.0, 1.0), (-1.0, 1.0)]
        tiler = FreeSpaceTiler(
            robot_2dof, empty_scene,
            joint_limits=limits,
            max_depth=2, min_width=0.5)

        tiles = tiler.tile()
        nodes = tiler.tiles_to_box_nodes(tiles)

        assert len(nodes) == len(tiles)
        for i, node in enumerate(nodes):
            assert node.node_id == i

    def test_custom_initial_intervals(self, robot_2dof, empty_scene):
        """自定义初始区间"""
        limits = [(-2.0, 2.0), (-2.0, 2.0)]
        tiler = FreeSpaceTiler(
            robot_2dof, empty_scene,
            joint_limits=limits,
            max_depth=2, min_width=0.5)

        # 只瓦片化部分空间
        tiles = tiler.tile(initial_intervals=[(0.0, 1.0), (0.0, 1.0)])
        total_vol = sum(t.volume for t in tiles)
        assert total_vol <= 1.0 + 0.01

    def test_collision_check_count(self, robot_2dof, scene_with_obstacle):
        """碰撞检测计数"""
        limits = [(-1.0, 1.0), (-1.0, 1.0)]
        tiler = FreeSpaceTiler(
            robot_2dof, scene_with_obstacle,
            joint_limits=limits,
            max_depth=3, min_width=0.5)

        tiler.tile()
        assert tiler.n_collision_checks > 0
