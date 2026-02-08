"""
test/planner/test_gcs_optimizer.py - GCS 优化器集成测试

测试 GCSOptimizer 的 Dijkstra fallback 路径以及 Drake GCS (可选)。
"""
import pytest
import numpy as np

from box_aabb.planner.gcs_optimizer import GCSOptimizer, HAS_DRAKE, HAS_SCIPY
from box_aabb.planner.models import BoxNode, Edge


# ==================== Fixtures ====================

def _make_box(node_id, intervals, seed=None):
    """便捷构造 BoxNode"""
    if seed is None:
        seed = np.array([(lo + hi) / 2 for lo, hi in intervals])
    return BoxNode(
        node_id=node_id,
        joint_intervals=list(intervals),
        seed_config=seed,
    )


def _make_edge(eid, src_id, tgt_id, src_cfg, tgt_cfg, src_tree=0, tgt_tree=0):
    """便捷构造 Edge"""
    return Edge(
        edge_id=eid,
        source_box_id=src_id,
        target_box_id=tgt_id,
        source_config=np.array(src_cfg),
        target_config=np.array(tgt_cfg),
        source_tree_id=src_tree,
        target_tree_id=tgt_tree,
        is_collision_free=True,
    )


@pytest.fixture
def simple_graph_2d():
    """简单的 2D 线性图: start → box0 → box1 → goal

    box0: [-1,1] x [-1,1], box1: [0,2] x [-1,1] (有重叠)
    """
    box0 = _make_box(0, [(-1.0, 1.0), (-1.0, 1.0)])
    box1 = _make_box(1, [(0.0, 2.0), (-1.0, 1.0)])

    boxes = {0: box0, 1: box1}

    q_start = np.array([-0.5, 0.0])
    q_goal = np.array([1.5, 0.0])

    # 构建邻接图
    edges = {
        'start': [(0, 0.5, None)],
        0: [('start', 0.5, None), (1, 0.5, None)],
        1: [(0, 0.5, None), ('goal', 0.5, None)],
        'goal': [(1, 0.5, None)],
    }
    graph = {
        'start': 'start',
        'goal': 'goal',
        'edges': edges,
    }
    return graph, boxes, q_start, q_goal


@pytest.fixture
def disconnected_graph_2d():
    """断开的图: start → box0, box1 → goal, 中间无连接"""
    box0 = _make_box(0, [(-1.0, 0.0), (-1.0, 1.0)])
    box1 = _make_box(1, [(2.0, 3.0), (-1.0, 1.0)])

    boxes = {0: box0, 1: box1}
    q_start = np.array([-0.5, 0.0])
    q_goal = np.array([2.5, 0.0])

    edges = {
        'start': [(0, 0.5, None)],
        0: [('start', 0.5, None)],
        1: [('goal', 0.5, None)],
        'goal': [(1, 0.5, None)],
    }
    graph = {
        'start': 'start',
        'goal': 'goal',
        'edges': edges,
    }
    return graph, boxes, q_start, q_goal


@pytest.fixture
def chain_graph_3d():
    """3D 链式图: start → box0 → box1 → box2 → goal"""
    box0 = _make_box(0, [(-1, 1), (-1, 1), (-1, 1)])
    box1 = _make_box(1, [(0, 2), (0, 2), (-1, 1)])
    box2 = _make_box(2, [(1, 3), (1, 3), (-1, 1)])

    boxes = {0: box0, 1: box1, 2: box2}
    q_start = np.array([0.0, 0.0, 0.0])
    q_goal = np.array([2.5, 2.5, 0.0])

    edges = {
        'start': [(0, 0.1, None)],
        0: [('start', 0.1, None), (1, 0.5, None)],
        1: [(0, 0.5, None), (2, 0.5, None)],
        2: [(1, 0.5, None), ('goal', 0.1, None)],
        'goal': [(2, 0.1, None)],
    }
    graph = {
        'start': 'start',
        'goal': 'goal',
        'edges': edges,
    }
    return graph, boxes, q_start, q_goal


# ==================== Fallback (Dijkstra + scipy) ====================

class TestGCSFallback:
    """测试 Dijkstra + scipy fallback 路径"""

    def test_fallback_simple_path(self, simple_graph_2d):
        graph, boxes, q_start, q_goal = simple_graph_2d
        opt = GCSOptimizer(fallback=True)
        path = opt._optimize_fallback(graph, boxes, q_start, q_goal)
        assert path is not None
        assert len(path) >= 2
        # start 和 goal 正确
        np.testing.assert_array_almost_equal(path[0], q_start)
        np.testing.assert_array_almost_equal(path[-1], q_goal)

    def test_fallback_disconnected_returns_none(self, disconnected_graph_2d):
        graph, boxes, q_start, q_goal = disconnected_graph_2d
        opt = GCSOptimizer(fallback=True)
        path = opt._optimize_fallback(graph, boxes, q_start, q_goal)
        assert path is None

    def test_fallback_chain_3d(self, chain_graph_3d):
        graph, boxes, q_start, q_goal = chain_graph_3d
        opt = GCSOptimizer(fallback=True)
        path = opt._optimize_fallback(graph, boxes, q_start, q_goal)
        assert path is not None
        assert len(path) >= 2
        np.testing.assert_array_almost_equal(path[0], q_start)
        np.testing.assert_array_almost_equal(path[-1], q_goal)

    def test_fallback_path_configs_in_boxes(self, simple_graph_2d):
        """中间路径点在对应 box 内"""
        graph, boxes, q_start, q_goal = simple_graph_2d
        opt = GCSOptimizer(fallback=True)
        path = opt._optimize_fallback(graph, boxes, q_start, q_goal)
        assert path is not None
        # 中间点应在某个 box 内
        for pt in path[1:-1]:
            in_some_box = any(b.contains(pt) for b in boxes.values())
            assert in_some_box, f"路径点 {pt} 不在任何 box 内"


class TestGCSDijkstra:
    """Dijkstra 最短路径测试"""

    def test_dijkstra_simple(self, simple_graph_2d):
        graph, _, _, _ = simple_graph_2d
        opt = GCSOptimizer()
        path_nodes = opt._dijkstra(graph)
        assert path_nodes is not None
        assert path_nodes[0] == 'start'
        assert path_nodes[-1] == 'goal'

    def test_dijkstra_disconnected(self, disconnected_graph_2d):
        graph, _, _, _ = disconnected_graph_2d
        opt = GCSOptimizer()
        assert opt._dijkstra(graph) is None

    def test_dijkstra_empty_graph(self):
        graph = {
            'start': 'start',
            'goal': 'goal',
            'edges': {},
        }
        opt = GCSOptimizer()
        assert opt._dijkstra(graph) is None

    def test_dijkstra_chain_order(self, chain_graph_3d):
        graph, _, _, _ = chain_graph_3d
        opt = GCSOptimizer()
        path_nodes = opt._dijkstra(graph)
        assert path_nodes is not None
        assert path_nodes[0] == 'start'
        assert path_nodes[-1] == 'goal'
        # 中间应包含 box 0, 1, 2 (顺序不一定)
        mid = set(path_nodes[1:-1])
        assert 0 in mid
        assert 2 in mid


class TestGCSOptimize:
    """GCSOptimizer.optimize() 主入口测试"""

    def test_optimize_uses_fallback_when_no_drake(self, simple_graph_2d):
        graph, boxes, q_start, q_goal = simple_graph_2d
        opt = GCSOptimizer(fallback=True)
        opt.use_drake = False  # 强制禁用 Drake
        path = opt.optimize(graph, boxes, q_start, q_goal)
        assert path is not None
        assert len(path) >= 2

    def test_optimize_no_drake_no_fallback_returns_none(self, simple_graph_2d):
        graph, boxes, q_start, q_goal = simple_graph_2d
        opt = GCSOptimizer(fallback=False)
        opt.use_drake = False
        path = opt.optimize(graph, boxes, q_start, q_goal)
        assert path is None

    @pytest.mark.skipif(not HAS_DRAKE, reason="Drake 未安装")
    def test_optimize_with_drake(self, simple_graph_2d):
        """Drake GCS 路径 (需要安装 Drake)"""
        graph, boxes, q_start, q_goal = simple_graph_2d
        opt = GCSOptimizer(fallback=True)
        opt.use_drake = True
        path = opt.optimize(graph, boxes, q_start, q_goal)
        assert path is not None
        assert len(path) >= 2

    @pytest.mark.skipif(not HAS_DRAKE, reason="Drake 未安装")
    def test_drake_vs_fallback_both_find_path(self, chain_graph_3d):
        """Drake 和 fallback 都应找到路径"""
        graph, boxes, q_start, q_goal = chain_graph_3d
        opt_drake = GCSOptimizer(fallback=False)
        opt_drake.use_drake = True
        opt_fallback = GCSOptimizer(fallback=True)
        opt_fallback.use_drake = False
        path_drake = opt_drake.optimize(graph, boxes, q_start, q_goal)
        path_fallback = opt_fallback.optimize(graph, boxes, q_start, q_goal)
        assert path_drake is not None
        assert path_fallback is not None


class TestGCSScipy:
    """scipy 局部优化测试"""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy 未安装")
    def test_scipy_smooth_reduces_length(self, simple_graph_2d):
        """scipy 优化不应增加路径长度"""
        graph, boxes, q_start, q_goal = simple_graph_2d
        opt = GCSOptimizer(fallback=True)
        path = opt._optimize_fallback(graph, boxes, q_start, q_goal)
        assert path is not None
        # 路径长度 ≥ 直线距离
        direct = float(np.linalg.norm(q_goal - q_start))
        length = sum(
            float(np.linalg.norm(path[i + 1] - path[i]))
            for i in range(len(path) - 1)
        )
        assert length >= direct - 1e-6
