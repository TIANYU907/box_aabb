"""
test_robot.py — Unit tests for robot.py (Robot class, config system, Panda preset).

Covers:
    - Robot construction & validation
    - name and joint_limits attributes
    - DH transform correctness
    - Forward kinematics (identity at zero, consistency)
    - get_link_positions / get_link_position
    - compute_relevant_joints
    - zero_length_links detection
    - Config system (from_json, from_config, load_robot, list_configs)
    - Panda preset properties
"""

import math
import json
import os
import tempfile
import numpy as np
import pytest

from box_aabb.robot import Robot, create_panda_robot, load_robot, PANDA_JOINT_LIMITS


class TestRobotConstruction:
    """Robot.__init__ and validation."""

    def test_basic_construction(self, simple_2dof_robot):
        assert simple_2dof_robot.n_joints == 2

    def test_panda_joint_count(self, panda_robot):
        assert panda_robot.n_joints == 8

    def test_invalid_joint_type_raises(self):
        dh = [{"alpha": 0, "a": 1, "d": 0, "theta": 0, "type": "linear"}]
        with pytest.raises(ValueError, match="revolute.*prismatic"):
            Robot(dh)

    def test_coupled_pairs_stored(self, panda_robot):
        assert (0, 2) in panda_robot.coupled_pairs
        assert (1, 3) in panda_robot.coupled_pairs

    def test_coupled_triples_stored(self, panda_robot):
        assert (0, 2, 4) in panda_robot.coupled_triples
        assert (1, 3, 5) in panda_robot.coupled_triples


class TestDHTransform:
    """Robot.dh_transform — single joint transform matrix."""

    def test_identity_at_zero(self):
        """DH(0, 0, 0, 0) should be the identity matrix."""
        T = Robot.dh_transform(0, 0, 0, 0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-14)

    def test_pure_translation_d(self):
        """DH(0, 0, d, 0) should translate along Z by d."""
        T = Robot.dh_transform(0, 0, 0.5, 0)
        np.testing.assert_allclose(T[:3, 3], [0, 0, 0.5], atol=1e-14)

    def test_pure_translation_a(self):
        """DH(0, a, 0, 0) should translate along X by a."""
        T = Robot.dh_transform(0, 0.3, 0, 0)
        np.testing.assert_allclose(T[:3, 3], [0.3, 0, 0], atol=1e-14)

    def test_rotation_theta_90(self):
        """DH(0, 0, 0, pi/2) should rotate around Z."""
        T = Robot.dh_transform(0, 0, 0, math.pi / 2)
        expected_rot = np.array([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(T[:3, :3], expected_rot, atol=1e-14)


class TestForwardKinematics:
    """Robot.forward_kinematics and related position queries."""

    def test_fk_identity_at_zero(self, simple_2dof_robot):
        """2-DOF planar: q=[0,0] → end effector at (2, 0, 0)."""
        T = simple_2dof_robot.forward_kinematics([0, 0])
        np.testing.assert_allclose(T[:3, 3], [2, 0, 0], atol=1e-12)

    def test_fk_return_all_count(self, simple_2dof_robot):
        """return_all should return n_joints+1 transforms (base + each joint)."""
        transforms = simple_2dof_robot.forward_kinematics([0, 0],
                                                          return_all=True)
        assert len(transforms) == 3  # base + 2 joints

    def test_fk_wrong_joint_count_raises(self, simple_2dof_robot):
        with pytest.raises(ValueError, match="2"):
            simple_2dof_robot.forward_kinematics([0.0])

    def test_get_link_positions_length(self, panda_robot):
        positions = panda_robot.get_link_positions([0] * 8)
        assert len(positions) == 9  # base + 8 joints

    def test_get_link_position_matches_fk(self, panda_robot):
        """get_link_position(q, i) should match forward_kinematics(q, all)[i]."""
        q = [0.1, -0.2, 0.3, -1.5, 0.1, 1.0, -0.3, 0.0]
        all_pos = panda_robot.get_link_positions(q)
        for i in range(1, 9):
            pos_i = panda_robot.get_link_position(q, i)
            np.testing.assert_allclose(pos_i, all_pos[i], atol=1e-12)

    def test_end_effector_pose(self, panda_robot):
        q = [0] * 8
        pos, rot = panda_robot.end_effector_pose(q)
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        # Rotation should be orthogonal
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)

    def test_planar_fk_half_pi(self, simple_2dof_robot):
        """2-DOF (Modified DH): q=[pi/2, 0] → end at (1, 1, 0)."""
        T = simple_2dof_robot.forward_kinematics([math.pi / 2, 0])
        np.testing.assert_allclose(T[:3, 3], [1, 1, 0], atol=1e-12)

    def test_planar_fk_fold(self, simple_2dof_robot):
        """2-DOF (Modified DH): q=[0, pi] → end at (2, 0, 0).
        In Modified DH, joint 2 rotation applies to future links only;
        link 2's origin stays at (a1+a2, 0, 0) when q1=0."""
        T = simple_2dof_robot.forward_kinematics([0, math.pi])
        np.testing.assert_allclose(T[:3, 3], [2, 0, 0], atol=1e-12)


class TestRelevantJoints:
    """Robot.compute_relevant_joints."""

    def test_link1_no_dependency_planar(self, simple_2dof_robot):
        """Modified DH: Link 1 origin at (a, 0, 0) is independent of q0."""
        rel = simple_2dof_robot.compute_relevant_joints(1)
        # With a=1, d=0, alpha=0: the origin doesn't move with q0
        assert 0 not in rel

    def test_link2_depends_on_q0(self, simple_2dof_robot):
        rel = simple_2dof_robot.compute_relevant_joints(2)
        assert 0 in rel

    def test_panda_higher_links_have_more_deps(self, panda_robot):
        """Higher-index links should depend on at least as many joints."""
        rel3 = panda_robot.compute_relevant_joints(3)
        rel5 = panda_robot.compute_relevant_joints(5)
        assert len(rel5) >= len(rel3)


class TestZeroLengthLinks:
    """Robot.zero_length_links."""

    def test_no_zero_length_for_planar(self, simple_2dof_robot):
        """Both links have a=1, d=0 → no zero-length links."""
        assert len(simple_2dof_robot.zero_length_links) == 0

    def test_panda_has_zero_length_links(self, panda_robot):
        """Panda should have several zero-length links (a≈0, d≈0)."""
        zl = panda_robot.zero_length_links
        assert len(zl) > 0
        # Joints 2, 4, 6, 7 have a=0, d=0 in the Panda DH table
        assert 2 in zl or 4 in zl or 6 in zl  # at least some


class TestPandaPreset:
    """create_panda_robot() and PANDA_JOINT_LIMITS."""

    def test_joint_limits_length(self):
        assert len(PANDA_JOINT_LIMITS) == 8

    def test_joint_limits_ordered(self):
        for lo, hi in PANDA_JOINT_LIMITS:
            assert lo <= hi

    def test_finger_fixed(self):
        lo, hi = PANDA_JOINT_LIMITS[7]
        assert lo == 0 and hi == 0


class TestRobotNameAndLimits:
    """Robot.name and Robot.joint_limits attributes."""

    def test_default_name(self, simple_2dof_robot):
        assert simple_2dof_robot.name == "Robot"

    def test_custom_name(self):
        dh = [{"alpha": 0, "a": 1.0, "d": 0, "theta": 0, "type": "revolute"}]
        r = Robot(dh, name="MyBot")
        assert r.name == "MyBot"

    def test_default_joint_limits_none(self, simple_2dof_robot):
        assert simple_2dof_robot.joint_limits is None

    def test_custom_joint_limits(self):
        dh = [
            {"alpha": 0, "a": 1.0, "d": 0, "theta": 0, "type": "revolute"},
            {"alpha": 0, "a": 1.0, "d": 0, "theta": 0, "type": "revolute"},
        ]
        limits = [(-1.0, 1.0), (-2.0, 2.0)]
        r = Robot(dh, name="LimitBot", joint_limits=limits)
        assert r.joint_limits == [(-1.0, 1.0), (-2.0, 2.0)]

    def test_panda_from_config_has_name(self):
        robot = load_robot('panda')
        assert robot.name == "Panda"

    def test_panda_from_config_has_limits(self):
        robot = load_robot('panda')
        assert robot.joint_limits is not None
        assert len(robot.joint_limits) == 8

    def test_panda_from_config_limits_match_legacy(self):
        robot = load_robot('panda')
        for i, (lo, hi) in enumerate(robot.joint_limits):
            assert abs(lo - PANDA_JOINT_LIMITS[i][0]) < 1e-6
            assert abs(hi - PANDA_JOINT_LIMITS[i][1]) < 1e-6


class TestConfigSystem:
    """Robot.from_json, Robot.from_config, load_robot, list_configs."""

    def test_load_robot_panda(self):
        robot = load_robot('panda')
        assert robot.n_joints == 8
        assert robot.name == "Panda"

    def test_load_robot_case_insensitive(self):
        robot = load_robot('Panda')
        assert robot.name == "Panda"

    def test_load_robot_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="找不到配置"):
            load_robot('nonexistent_robot')

    def test_list_configs_includes_panda(self):
        configs = Robot.list_configs()
        assert 'panda' in configs

    def test_from_config_same_as_load_robot(self):
        r1 = Robot.from_config('panda')
        r2 = load_robot('panda')
        assert r1.name == r2.name
        assert r1.n_joints == r2.n_joints

    def test_from_json_full_format(self, tmp_path):
        """Full config JSON with all fields."""
        cfg = {
            "name": "TestBot",
            "dh_params": [
                {"alpha": 0, "a": 0.5, "d": 0, "theta": 0, "type": "revolute"},
                {"alpha": 0, "a": 0.3, "d": 0, "theta": 0, "type": "revolute"},
            ],
            "joint_limits": [[-1.5, 1.5], [-2.0, 2.0]],
            "coupled_pairs": [[0, 1]],
            "coupled_triples": [],
        }
        p = tmp_path / "test_bot.json"
        p.write_text(json.dumps(cfg), encoding='utf-8')
        robot = Robot.from_json(str(p))
        assert robot.name == "TestBot"
        assert robot.n_joints == 2
        assert robot.joint_limits == [(-1.5, 1.5), (-2.0, 2.0)]
        assert robot.coupled_pairs == [(0, 1)]
        assert robot.coupled_triples == []

    def test_from_json_legacy_list_format(self, tmp_path):
        """Legacy format: bare DH list."""
        dh = [
            {"alpha": 0, "a": 1, "d": 0, "theta": 0, "type": "revolute"},
        ]
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps(dh), encoding='utf-8')
        robot = Robot.from_json(str(p))
        assert robot.n_joints == 1
        assert robot.name == "Robot"
        assert robot.joint_limits is None

    def test_from_json_legacy_dh_key(self, tmp_path):
        """Legacy format: {\"dh\": [...]}."""
        cfg = {"dh": [
            {"alpha": 0, "a": 1, "d": 0, "theta": 0, "type": "revolute"},
        ]}
        p = tmp_path / "legacy_dh.json"
        p.write_text(json.dumps(cfg), encoding='utf-8')
        robot = Robot.from_json(str(p))
        assert robot.n_joints == 1

    def test_create_panda_robot_backward_compat(self):
        """create_panda_robot() still works and returns same result as load_robot."""
        r1 = create_panda_robot()
        r2 = load_robot('panda')
        assert r1.n_joints == r2.n_joints
        assert r1.name == r2.name
        assert len(r1.coupled_pairs) == len(r2.coupled_pairs)
