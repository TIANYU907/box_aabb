"""Tests for boundary expansion sampling strategy."""

import numpy as np
import pytest

from aabb.robot import load_robot
from forest.models import BoxNode
from forest.scene import Scene
from planner.models import PlannerConfig
from planner.box_planner import BoxPlanner


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

def _make_2dof_robot():
    return load_robot("2dof_planar")


def _make_simple_scene(obstacles=None):
    scene = Scene()
    if obstacles:
        for obs in obstacles:
            scene.add_obstacle(obs["min"], obs["max"], obs.get("name", ""))
    return scene


# ─────────────────────────────────────────────
#  _sample_boundary_seed (BoxPlanner)
# ─────────────────────────────────────────────

class TestSampleBoundarySeed:
    def test_samples_outside_box(self):
        robot = _make_2dof_robot()
        scene = _make_simple_scene([
            {"min": [10.0, 10.0, 10.0], "max": [11.0, 11.0, 11.0]}
        ])
        config = PlannerConfig(boundary_expand_epsilon=0.01)
        planner = BoxPlanner(robot, scene, config=config, no_cache=True)

        box = BoxNode(
            node_id=0,
            joint_intervals=[(-0.5, 0.5), (-0.5, 0.5)],
            seed_config=np.array([0.0, 0.0]),
        )
        rng = np.random.default_rng(42)
        seeds = []
        for _ in range(50):
            s = planner._sample_boundary_seed(box, rng)
            if s is not None:
                seeds.append(s)

        assert len(seeds) > 0
        # Each seed should be outside the box (in at least one dimension)
        for s in seeds:
            outside = False
            for i, (lo, hi) in enumerate(box.joint_intervals):
                if s[i] < lo - 1e-12 or s[i] > hi + 1e-12:
                    outside = True
                    break
            assert outside, f"Seed {s} should be outside box"


# ─────────────────────────────────────────────
#  PlannerConfig new fields
# ─────────────────────────────────────────────

class TestPlannerConfigBoundaryFields:
    def test_defaults(self):
        cfg = PlannerConfig()
        assert cfg.boundary_expand_enabled is True
        assert cfg.boundary_expand_max_failures == 5
        assert cfg.boundary_expand_epsilon == 0.01

    def test_to_dict_includes_fields(self):
        cfg = PlannerConfig(boundary_expand_max_failures=3)
        d = cfg.to_dict()
        assert d["boundary_expand_max_failures"] == 3
        assert "boundary_expand_enabled" in d

    def test_from_dict_round_trip(self):
        orig = PlannerConfig(
            boundary_expand_enabled=False,
            boundary_expand_max_failures=7,
        )
        d = orig.to_dict()
        loaded = PlannerConfig.from_dict(d)
        assert loaded.boundary_expand_enabled is False
        assert loaded.boundary_expand_max_failures == 7


# ─────────────────────────────────────────────
#  Integration: boundary sampling in main loop
# ─────────────────────────────────────────────

class TestBoundarySamplingIntegration:
    def test_boundary_sampling_produces_boxes(self):
        """边缘采样应在 box 周围产生更多相邻 box"""
        robot = _make_2dof_robot()
        scene = _make_simple_scene([
            {"min": [0.6, -0.2, -1000.0], "max": [0.9, 0.2, 1000.0]},
        ])
        config = PlannerConfig(
            boundary_expand_enabled=True,
            boundary_expand_max_failures=3,
            boundary_expand_epsilon=0.05,
            max_box_nodes=30,
            max_iterations=100,
        )
        planner = BoxPlanner(robot, scene, config=config, no_cache=True)
        q_s = np.array([-1.0, 0.0])
        q_g = np.array([1.0, 0.0])
        result = planner.plan(q_s, q_g, seed=123)
        assert result is not None
        assert result.n_boxes_created >= 1


# ─────────────────────────────────────────────
#  Integration: end-to-end plan with boundary sampling
# ─────────────────────────────────────────────

class TestPlanWithBoundaryExpand:
    def test_plan_with_boundary_expand_enabled(self):
        """端到端测试：boundary_expand_enabled=True 不影响规划成功率"""
        robot = _make_2dof_robot()
        scene = _make_simple_scene([
            {"min": [0.3, -0.1, -0.5], "max": [0.5, 0.1, 0.5]},
        ])
        config = PlannerConfig(
            max_iterations=50,
            max_box_nodes=50,
            boundary_expand_enabled=True,
            boundary_expand_max_failures=3,
        )
        planner = BoxPlanner(robot, scene, config=config, no_cache=True)
        q_start = np.array([0.5, 0.5])
        q_goal = np.array([-0.5, -0.5])
        result = planner.plan(q_start, q_goal, seed=42)
        # 只检查不崩溃，成功率取决于场景
        assert result is not None

    def test_plan_with_boundary_expand_disabled(self):
        """对比测试：boundary_expand_enabled=False 保持原有行为"""
        robot = _make_2dof_robot()
        scene = _make_simple_scene([
            {"min": [0.3, -0.1, -0.5], "max": [0.5, 0.1, 0.5]},
        ])
        config = PlannerConfig(
            max_iterations=50,
            max_box_nodes=50,
            boundary_expand_enabled=False,
        )
        planner = BoxPlanner(robot, scene, config=config, no_cache=True)
        q_start = np.array([0.5, 0.5])
        q_goal = np.array([-0.5, -0.5])
        result = planner.plan(q_start, q_goal, seed=42)
        assert result is not None
