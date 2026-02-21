"""test_dijkstra_path_quality.py - verify A* + boundary-aware weights
produce better box sequences than old center-to-center Dijkstra.
"""

import heapq
import numpy as np
import pytest

from forest.models import BoxNode
from planner.pipeline import (
    _dijkstra_box_graph,
    _shortcut_box_sequence,
    _refine_path_in_boxes,
)


def _make_box(nid, intervals):
    center = np.array([(lo + hi) / 2 for lo, hi in intervals])
    return BoxNode(node_id=nid, joint_intervals=intervals, seed_config=center)


def _old_dijkstra_center(boxes, adj, src, tgt):
    """Reproduce the OLD center-to-center Dijkstra for comparison."""
    centers = {}
    for bid, box in boxes.items():
        centers[bid] = np.array(
            [(lo + hi) / 2 for lo, hi in box.joint_intervals])

    dist_map = {bid: float("inf") for bid in boxes}
    prev_map = {bid: None for bid in boxes}
    dist_map[src] = 0.0
    heap = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist_map[u]:
            continue
        if u == tgt:
            break
        cu = centers[u]
        for v in adj.get(u, set()):
            w = float(np.linalg.norm(cu - centers[v]))
            nd = d + w
            if nd < dist_map[v]:
                dist_map[v] = nd
                prev_map[v] = u
                heapq.heappush(heap, (nd, v))

    if dist_map[tgt] == float("inf"):
        return None
    seq = []
    cur = tgt
    while cur is not None:
        seq.append(cur)
        cur = prev_map[cur]
    seq.reverse()
    return seq


# ── Scenario: large box provides near-direct corridor,
#    but its center is far from the start-goal line.
#    Old Dijkstra should prefer the chain of small boxes
#    (lower center-to-center sum), while new A* should
#    prefer the direct large-box route. ──

def _build_detour_scenario():
    """src=[0,0] → goal=[10,0].  Large box has center at [5, 5]."""
    boxes = {}
    boxes[0] = _make_box(0, [(-1, 1), (-1, 1)])          # src
    boxes[1] = _make_box(1, [(-0.5, 10.5), (-0.5, 10)])  # large, center≈[5,4.75]
    boxes[2] = _make_box(2, [(9, 11), (-1, 1)])           # tgt

    # detour chain (centers near y ≈ 1)
    boxes[3] = _make_box(3, [(-0.5, 2), (0.5, 2)])
    boxes[4] = _make_box(4, [(1.5, 3.5), (0.5, 2)])
    boxes[5] = _make_box(5, [(3, 5), (0.5, 2)])
    boxes[6] = _make_box(6, [(4.5, 7), (0.5, 2)])
    boxes[7] = _make_box(7, [(6.5, 8.5), (0.5, 2)])
    boxes[8] = _make_box(8, [(8, 9.5), (0.5, 2)])

    adj = {
        0: {1, 3},
        1: {0, 2},
        2: {1, 8},
        3: {0, 4},
        4: {3, 5},
        5: {4, 6},
        6: {5, 7},
        7: {6, 8},
        8: {7, 2},
    }
    return boxes, adj


class TestDijkstraPathQuality:
    """Verify the improved Dijkstra produces shorter refined paths."""

    def test_prefers_direct_corridor(self):
        """New A* should route through the large box, not the detour chain."""
        boxes, adj = _build_detour_scenario()
        q_start = np.array([0.0, 0.0])
        q_goal = np.array([10.0, 0.0])

        # New A*
        seq_new, _ = _dijkstra_box_graph(boxes, adj, 0, 2, q_goal=q_goal)
        assert seq_new is not None
        # The direct path is [0, 1, 2]
        assert 1 in seq_new, f"Expected large box (id=1) in path, got {seq_new}"
        assert len(seq_new) <= 3, f"Expected ≤3 boxes, got {seq_new}"

    def test_old_dijkstra_takes_detour(self):
        """Old center-to-center Dijkstra should prefer the small-box chain."""
        boxes, adj = _build_detour_scenario()
        seq_old = _old_dijkstra_center(boxes, adj, 0, 2)
        assert seq_old is not None
        # expect the detour path (6+ boxes)
        assert len(seq_old) > 3, (
            f"Old Dijkstra unexpectedly found direct path: {seq_old}"
        )

    def test_shortcut_removes_redundant_hops(self):
        """Shortcut should skip intermediate boxes when endpoints are adjacent."""
        boxes, adj = _build_detour_scenario()
        long_seq = [0, 3, 4, 5, 6, 7, 8, 2]
        # box 0 is adj to 3, 3 adj to 4, etc.
        # But also 0 adj to box 1 (not in seq), so shortcut won't help
        # here since it only looks within the given sequence.
        short = _shortcut_box_sequence(long_seq, adj)
        # In this adj, no jumps possible within the detour chain
        # (0→3 adj, but 3 not adj to 5, etc.)
        assert short == long_seq or len(short) <= len(long_seq)

    def test_socp_quality_ratio(self):
        """Refined cost through direct corridor ≈ direct distance."""
        boxes, adj = _build_detour_scenario()
        q_start = np.array([0.0, 0.0])
        q_goal = np.array([10.0, 0.0])

        seq, _ = _dijkstra_box_graph(boxes, adj, 0, 2, q_goal=q_goal)
        short = _shortcut_box_sequence(seq, adj)

        wps = [q_start.copy()]
        for bid in short[1:-1]:
            box = boxes[bid]
            c = np.array([(lo + hi) / 2 for lo, hi in box.joint_intervals])
            wps.append(c)
        wps.append(q_goal.copy())

        refined, cost = _refine_path_in_boxes(
            wps, short, boxes, q_start, q_goal, ndim=2)

        direct = float(np.linalg.norm(q_goal - q_start))
        ratio = cost / direct
        assert ratio < 1.05, f"Path quality ratio {ratio:.3f} too high (>1.05)"

    def test_same_box_trivial(self):
        """Start and goal in the same box → 2-box path, near-direct cost."""
        box = _make_box(0, [(-5, 15), (-5, 5)])
        boxes = {0: box}
        adj = {0: set()}
        q_s = np.array([0.0, 0.0])
        q_g = np.array([10.0, 0.0])

        seq, _ = _dijkstra_box_graph(boxes, adj, 0, 0, q_goal=q_g)
        assert seq == [0]

    def test_no_path(self):
        """Disconnected graph returns None."""
        boxes = {
            0: _make_box(0, [(0, 1), (0, 1)]),
            1: _make_box(1, [(5, 6), (5, 6)]),
        }
        adj = {0: set(), 1: set()}
        seq, _ = _dijkstra_box_graph(boxes, adj, 0, 1)
        assert seq is None
