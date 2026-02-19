"""test/planner/test_obstacles.py - Scene 与 Obstacle 测试"""
import json
import tempfile
import os
import numpy as np
import pytest

from planner.models import Obstacle
from planner.obstacles import Scene


class TestObstacle:
    """Obstacle 数据模型测试"""

    def test_basic_construction(self):
        obs = Obstacle(min_point=[0, 0, 0], max_point=[1, 1, 1], name="cube")
        assert obs.name == "cube"
        np.testing.assert_array_equal(obs.min_point, [0, 0, 0])
        np.testing.assert_array_equal(obs.max_point, [1, 1, 1])

    def test_auto_array_conversion(self):
        obs = Obstacle(min_point=[1.0, 2.0, 3.0], max_point=[4.0, 5.0, 6.0])
        assert isinstance(obs.min_point, np.ndarray)
        assert obs.min_point.dtype == np.float64

    def test_center(self):
        obs = Obstacle(min_point=[0, 0, 0], max_point=[2, 4, 6])
        np.testing.assert_array_almost_equal(obs.center, [1, 2, 3])

    def test_size(self):
        obs = Obstacle(min_point=[1, 2, 3], max_point=[4, 6, 9])
        np.testing.assert_array_almost_equal(obs.size, [3, 4, 6])

    def test_volume(self):
        obs = Obstacle(min_point=[0, 0, 0], max_point=[2, 3, 4])
        assert obs.volume == pytest.approx(24.0)

    def test_contains_point(self):
        obs = Obstacle(min_point=[0, 0, 0], max_point=[1, 1, 1])
        assert obs.contains_point(np.array([0.5, 0.5, 0.5]))
        assert not obs.contains_point(np.array([2.0, 0.5, 0.5]))

    def test_to_dict(self):
        obs = Obstacle(min_point=[1, 2, 3], max_point=[4, 5, 6], name="test")
        d = obs.to_dict()
        assert d['min'] == [1.0, 2.0, 3.0]
        assert d['max'] == [4.0, 5.0, 6.0]
        assert d['name'] == "test"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            Obstacle(min_point=[0, 0], max_point=[1, 1, 1])


class TestScene:
    """Scene 管理测试"""

    def test_add_obstacle_3d(self):
        scene = Scene()
        obs = scene.add_obstacle([0, 0, 0], [1, 1, 1], name="box")
        assert scene.n_obstacles == 1
        assert obs.name == "box"

    def test_add_obstacle_2d_auto_expand(self):
        """2D 点自动扩展为 3D"""
        scene = Scene()
        obs = scene.add_obstacle([0, 0], [1, 1], name="flat")
        assert obs.min_point.shape == (3,)
        assert obs.max_point.shape == (3,)
        # z 范围应为 -1e3 到 1e3
        assert obs.min_point[2] == pytest.approx(-1e3)
        assert obs.max_point[2] == pytest.approx(1e3)

    def test_auto_naming(self):
        scene = Scene()
        obs0 = scene.add_obstacle([0, 0, 0], [1, 1, 1])
        obs1 = scene.add_obstacle([2, 2, 2], [3, 3, 3])
        assert obs0.name == "obstacle_0"
        assert obs1.name == "obstacle_1"

    def test_remove_obstacle(self):
        scene = Scene()
        scene.add_obstacle([0, 0, 0], [1, 1, 1], name="a")
        scene.add_obstacle([2, 2, 2], [3, 3, 3], name="b")
        assert scene.remove_obstacle("a") is True
        assert scene.n_obstacles == 1
        assert scene.get_obstacle("a") is None

    def test_remove_nonexistent(self):
        scene = Scene()
        assert scene.remove_obstacle("nonexist") is False

    def test_clear(self):
        scene = Scene()
        scene.add_obstacle([0, 0, 0], [1, 1, 1])
        scene.add_obstacle([2, 2, 2], [3, 3, 3])
        scene.clear()
        assert scene.n_obstacles == 0

    def test_get_obstacle_by_name(self):
        scene = Scene()
        scene.add_obstacle([0, 0, 0], [1, 1, 1], name="target")
        obs = scene.get_obstacle("target")
        assert obs is not None
        assert obs.name == "target"

    def test_to_dict_list(self):
        scene = Scene()
        scene.add_obstacle([0, 0, 0], [1, 1, 1], name="a")
        scene.add_obstacle([2, 2, 2], [3, 3, 3], name="b")
        dicts = scene.to_dict_list()
        assert len(dicts) == 2
        assert dicts[0]['name'] == "a"
        assert dicts[1]['name'] == "b"

    def test_json_round_trip(self, tmp_path):
        """JSON 序列化 → 反序列化 round-trip"""
        scene = Scene()
        scene.add_obstacle([0, 0, 0], [1, 1, 1], name="cube1")
        scene.add_obstacle([2, 2, 2], [5, 5, 5], name="cube2")

        filepath = str(tmp_path / "scene.json")
        scene.to_json(filepath)

        loaded = Scene.from_json(filepath)
        assert loaded.n_obstacles == 2
        obs1 = loaded.get_obstacle("cube1")
        assert obs1 is not None
        np.testing.assert_array_almost_equal(obs1.min_point, [0, 0, 0])
        np.testing.assert_array_almost_equal(obs1.max_point, [1, 1, 1])

    def test_from_dict(self):
        data = {
            'obstacles': [
                {'min': [0, 0, 0], 'max': [1, 1, 1], 'name': 'test'},
            ]
        }
        scene = Scene.from_dict(data)
        assert scene.n_obstacles == 1

    def test_from_obstacle_list(self):
        obs_list = [
            {'min': [0, 0, 0], 'max': [1, 1, 1], 'name': 'a'},
            {'min': [3, 3, 3], 'max': [4, 4, 4], 'name': 'b'},
        ]
        scene = Scene.from_obstacle_list(obs_list)
        assert scene.n_obstacles == 2

    def test_repr(self):
        scene = Scene()
        scene.add_obstacle([0, 0, 0], [1, 1, 1])
        assert "n_obstacles=1" in repr(scene)
