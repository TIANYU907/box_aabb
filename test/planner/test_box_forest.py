"""
test/planner/test_box_forest.py - BoxForest 和 BoxForestQuery 测试

注：BoxForest.build() 已在 v5 重构中移除，构建主流程移至 BoxRRT。
    BoxForest 现在是一个纯数据容器（无 build 方法）。
    BoxForestQuery 待重构以适配新 API。
    详见 test_box_forest_v5.py 中的完整测试。
"""

import os
import tempfile
import pytest
import numpy as np

from box_aabb.robot import load_robot
from planner.box_forest import BoxForest
from planner.models import PlannerConfig, BoxNode


@pytest.fixture
def robot_2dof():
    return load_robot('2dof_planar')


@pytest.fixture
def config():
    return PlannerConfig(
        expansion_resolution=0.05,
        max_expansion_rounds=2,
    )


def _make_box(intervals, node_id=0):
    return BoxNode(
        node_id=node_id,
        joint_intervals=intervals,
        seed_config=np.array([(lo + hi) / 2 for lo, hi in intervals]),
    )


class TestBoxForest:
    """BoxForest 基础测试"""

    def test_build_basic(self, robot_2dof, config):
        """创建 BoxForest 并添加 box，验证基本属性"""
        forest = BoxForest(
            robot_fingerprint=robot_2dof.fingerprint(),
            joint_limits=robot_2dof.joint_limits,
            config=config,
        )

        box1 = _make_box([(0.0, 1.0), (0.0, 1.0)], node_id=0)
        box2 = _make_box([(1.0, 2.0), (0.0, 1.0)], node_id=1)
        forest.add_boxes([box1, box2])

        assert forest.n_boxes > 0
        assert forest.total_volume > 0

    def test_build_single_box(self, robot_2dof, config):
        """单个 box 也能正常添加"""
        forest = BoxForest(
            robot_fingerprint=robot_2dof.fingerprint(),
            joint_limits=robot_2dof.joint_limits,
            config=config,
        )

        box = _make_box([(0.0, 1.0), (0.0, 1.0)], node_id=0)
        added = forest.add_boxes([box])

        assert forest.n_boxes > 0
        assert len(added) > 0

    def test_save_and_load(self, robot_2dof, config):
        """序列化和反序列化"""
        forest = BoxForest(
            robot_fingerprint=robot_2dof.fingerprint(),
            joint_limits=robot_2dof.joint_limits,
            config=config,
        )

        box1 = _make_box([(0.0, 1.0), (0.0, 1.0)], node_id=0)
        box2 = _make_box([(1.0, 2.0), (0.0, 1.0)], node_id=1)
        forest.add_boxes([box1, box2])

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            forest.save(filepath)
            loaded = BoxForest.load(filepath, robot_2dof)

            assert loaded.n_boxes == forest.n_boxes
            assert abs(loaded.total_volume - forest.total_volume) < 1e-10
        finally:
            os.unlink(filepath)

    def test_load_wrong_robot(self, robot_2dof, config):
        """加载时机器人指纹不匹配应报错"""
        forest = BoxForest(
            robot_fingerprint=robot_2dof.fingerprint(),
            joint_limits=robot_2dof.joint_limits,
            config=config,
        )

        box = _make_box([(0.0, 1.0), (0.0, 1.0)], node_id=0)
        forest.add_boxes([box])

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            forest.save(filepath)
            robot_3dof = load_robot('3dof_planar')
            with pytest.raises(ValueError, match="指纹不匹配"):
                BoxForest.load(filepath, robot_3dof)
        finally:
            os.unlink(filepath)


@pytest.mark.skip(reason="BoxForestQuery 待重构以适配 v5 BoxForest API")
class TestBoxForestQuery:
    """BoxForestQuery 测试（待 BoxForestQuery 适配新 API 后启用）"""

    def test_query_basic(self):
        pass

    def test_query_direct_connect(self):
        pass

    def test_query_collision_start(self):
        pass

    def test_multiple_queries_same_forest(self):
        pass
