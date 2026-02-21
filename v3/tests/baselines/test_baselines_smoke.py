"""
Smoke tests for baselines/ module.

Tests that each adapter can be instantiated, satisfies the BasePlanner ABC,
and can run plan() on a trivial 2-DOF scenario.  Optional dependencies
(pydrake, OMPL) are skipped if not installed.
"""

import numpy as np
import pytest

from baselines.base import BasePlanner, PlanningResult
from baselines.rrt_family import RRTPlanner
from baselines.sbf_adapter import SBFAdapter
from baselines.ompl_adapter import OMPLPlanner
from baselines.iris_gcs import IRISGCSPlanner, _check_drake

from aabb.robot import load_robot
from forest.scene import Scene


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def robot_2dof():
    return load_robot("2dof_planar")


@pytest.fixture
def empty_scene():
    return Scene()


@pytest.fixture
def scene_with_obstacle():
    s = Scene()
    # obstacle far from the typical configs we test
    s.add_obstacle([10.0, 10.0, -0.5], [11.0, 11.0, 0.5], name="far")
    return s


# ═══════════════════════════════════════════════════════════════════════════
# PlanningResult unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanningResult:
    def test_success_result(self):
        path = np.array([[0.0, 0.0], [1.0, 1.0]])
        r = PlanningResult(
            success=True, path=path, cost=1.414,
            planning_time=0.01, first_solution_time=0.005,
            collision_checks=100, nodes_explored=50,
        )
        assert r.success is True
        assert r.n_waypoints == 2
        assert "success" in r.to_dict()

    def test_failure_result(self):
        r = PlanningResult.failure(planning_time=0.5, reason="timeout")
        assert r.success is False
        assert r.path is None
        assert r.cost == float("inf")
        assert r.metadata["reason"] == "timeout"

    def test_to_dict_keys(self):
        r = PlanningResult.failure()
        d = r.to_dict()
        for k in ("success", "cost", "planning_time",
                   "collision_checks", "nodes_explored", "n_waypoints"):
            assert k in d


# ═══════════════════════════════════════════════════════════════════════════
# BasePlanner ABC tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBasePlannerABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BasePlanner()  # type: ignore

    def test_rrt_is_base_planner(self):
        assert issubclass(RRTPlanner, BasePlanner)

    def test_sbf_is_base_planner(self):
        assert issubclass(SBFAdapter, BasePlanner)

    def test_ompl_is_base_planner(self):
        assert issubclass(OMPLPlanner, BasePlanner)

    def test_iris_is_base_planner(self):
        assert issubclass(IRISGCSPlanner, BasePlanner)


# ═══════════════════════════════════════════════════════════════════════════
# RRTPlanner smoke tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRRTPlanner:
    @pytest.mark.parametrize("algo", [
        "RRT", "RRTConnect", "RRT*", "Informed-RRT*", "BiRRT*",
    ])
    def test_instantiate(self, algo):
        p = RRTPlanner(algorithm=algo)
        assert p.name == algo
        assert p.supports_reuse is False

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            RRTPlanner(algorithm="UNKNOWN")

    def test_plan_2dof_rrt_connect(self, robot_2dof, empty_scene):
        """RRTConnect on obstacle-free 2DOF — must succeed."""
        p = RRTPlanner(algorithm="RRTConnect")
        p.setup(robot_2dof, empty_scene, {"seed": 0, "step_size": 0.3})
        result = p.plan(
            start=np.array([-1.0, 0.5]),
            goal=np.array([1.0, -0.5]),
            timeout=10.0,
        )
        assert isinstance(result, PlanningResult)
        assert result.success is True
        assert result.path is not None
        assert result.path.shape[1] == 2
        assert result.planning_time > 0

    def test_plan_2dof_rrt(self, robot_2dof, empty_scene):
        """Basic RRT on obstacle-free 2DOF."""
        p = RRTPlanner(algorithm="RRT")
        p.setup(robot_2dof, empty_scene, {"seed": 0, "step_size": 0.3,
                                           "goal_bias": 0.15,
                                           "goal_tol": 0.5})
        result = p.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([1.0, 1.0]),
            timeout=10.0,
        )
        assert isinstance(result, PlanningResult)
        # RRT with high goal_bias should eventually find it
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════
# SBFAdapter smoke tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSBFAdapter:
    def test_instantiate(self):
        for m in ("dijkstra", "gcs", "visgraph"):
            p = SBFAdapter(method=m)
            assert "SBF" in p.name
            assert p.supports_reuse is True

    def test_plan_2dof_dijkstra(self, robot_2dof, scene_with_obstacle):
        """SBF-Dijkstra on 2DOF — basic plannable scenario."""
        p = SBFAdapter(method="dijkstra")
        p.setup(robot_2dof, scene_with_obstacle,
                {"n_boxes": 40, "seed": 42, "no_cache": True})
        result = p.plan(
            start=np.array([-1.0, 0.5]),
            goal=np.array([1.0, -0.5]),
            timeout=30.0,
        )
        assert isinstance(result, PlanningResult)
        # may or may not succeed with few boxes — just check contract
        assert isinstance(result.success, bool)
        assert result.planning_time > 0

    def test_reset_clears_state(self, robot_2dof, empty_scene):
        p = SBFAdapter(method="dijkstra")
        p.setup(robot_2dof, empty_scene, {"no_cache": True})
        p.plan(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert p._prep is not None
        p.reset()
        assert p._prep is None


# ═══════════════════════════════════════════════════════════════════════════
# OMPLPlanner smoke tests (skipped if OMPL not available)
# ═══════════════════════════════════════════════════════════════════════════

class TestOMPLPlanner:
    def test_instantiate(self):
        p = OMPLPlanner(algorithm="RRTConnect")
        assert "OMPL" in p.name
        assert p.supports_reuse is False

    @pytest.mark.skip(reason="Requires WSL + OMPL; run manually")
    def test_plan_requires_wsl(self, robot_2dof, empty_scene):
        p = OMPLPlanner(algorithm="RRTConnect")
        p.setup(robot_2dof, empty_scene, {"seed": 42})
        result = p.plan(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert isinstance(result, PlanningResult)


# ═══════════════════════════════════════════════════════════════════════════
# IRISGCSPlanner smoke tests (skipped if pydrake not available)
# ═══════════════════════════════════════════════════════════════════════════

class TestIRISGCSPlanner:
    def test_instantiate(self):
        p = IRISGCSPlanner()
        assert p.name == "IRIS-GCS"
        assert p.supports_reuse is True

    def test_plan_without_drake_returns_failure(self, robot_2dof, empty_scene):
        """If Drake is not installed, plan() should gracefully fail."""
        if _check_drake():
            pytest.skip("Drake is available — test N/A")
        p = IRISGCSPlanner()
        p.setup(robot_2dof, empty_scene, {})
        result = p.plan(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert result.success is False
        assert "pydrake" in result.metadata.get("reason", "")
