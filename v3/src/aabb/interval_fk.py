"""
interval_fk.py  –  高速区间正运动学

用纯标量 interval arithmetic + numpy 替代 AffineForm 的 dict 运算，
同时支持增量 FK（只重算 changed_joint 及其之后的变换）。

设计要点
--------
- 变换矩阵 4×4 拆成 (T_lo, T_hi) 两个 (4,4) ndarray
- 利用 DH 矩阵结构：row3=[0,0,0,1]
- 支持 tool_frame（固定变换，无区间）
- 返回值仍为 List[LinkAABBInfo]，与现有接口兼容

Optimization B: AffineForm → 纯 interval: ~3-7× per FK call
Optimization A: 增量 FK（prefix 复用）: 额外 ~43% 平均节省
"""

from __future__ import annotations

import math
from typing import List, Tuple, Set

import numpy as np

from .robot import Robot
from .models import LinkAABBInfo

# ─────────────────────────────────────────────────────
#  区间三角函数（标量版，避免 Python 对象开销）
# ─────────────────────────────────────────────────────

_TWO_PI = 2.0 * math.pi
_HALF_PI = math.pi / 2.0
_THREE_HALF_PI = 3.0 * math.pi / 2.0


def _isin(lo: float, hi: float) -> Tuple[float, float]:
    """sin 的保守区间包络"""
    w = hi - lo
    if w >= _TWO_PI:
        return (-1.0, 1.0)

    s_lo = math.sin(lo)
    s_hi = math.sin(hi)
    r_lo = s_lo if s_lo < s_hi else s_hi
    r_hi = s_hi if s_hi > s_lo else s_lo

    lo_n = lo % _TWO_PI
    hi_n = lo_n + w

    if math.ceil((lo_n - _HALF_PI) / _TWO_PI) <= math.floor(
            (hi_n - _HALF_PI) / _TWO_PI):
        r_hi = 1.0
    if math.ceil((lo_n - _THREE_HALF_PI) / _TWO_PI) <= math.floor(
            (hi_n - _THREE_HALF_PI) / _TWO_PI):
        r_lo = -1.0

    return (r_lo, r_hi)


def _icos(lo: float, hi: float) -> Tuple[float, float]:
    """cos 的保守区间包络"""
    w = hi - lo
    if w >= _TWO_PI:
        return (-1.0, 1.0)

    c_lo = math.cos(lo)
    c_hi = math.cos(hi)
    r_lo = c_lo if c_lo < c_hi else c_hi
    r_hi = c_hi if c_hi > c_lo else c_lo

    lo_n = lo % _TWO_PI
    hi_n = lo_n + w

    if math.ceil(lo_n / _TWO_PI) <= math.floor(hi_n / _TWO_PI):
        r_hi = 1.0
    if math.ceil((lo_n - math.pi) / _TWO_PI) <= math.floor(
            (hi_n - math.pi) / _TWO_PI):
        r_lo = -1.0

    return (r_lo, r_hi)


# ─────────────────────────────────────────────────────
#  DH 区间矩阵构造
# ─────────────────────────────────────────────────────

# row3=[0,0,0,1] 模板，避免每次 np.zeros
_DH_TEMPLATE = np.zeros((4, 4))
_DH_TEMPLATE[3, 3] = 1.0


def _dh_joint_matrix(
    alpha: float, a: float,
    ct_lo: float, ct_hi: float,
    st_lo: float, st_hi: float,
    d_lo: float, d_hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    单关节 DH 区间矩阵 (A_lo, A_hi)，各 (4,4)。
    使用模板 copy 替代 np.zeros，内联 interval*scalar 避免函数调用。
    """
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    A_lo = _DH_TEMPLATE.copy()
    A_hi = _DH_TEMPLATE.copy()

    # row 0
    A_lo[0, 0] = ct_lo;  A_hi[0, 0] = ct_hi
    A_lo[0, 1] = -st_hi; A_hi[0, 1] = -st_lo
    A_lo[0, 3] = a;      A_hi[0, 3] = a

    # row 1: inline interval * scalar
    v1 = st_lo * ca; v2 = st_hi * ca
    if v1 <= v2: A_lo[1, 0] = v1; A_hi[1, 0] = v2
    else:        A_lo[1, 0] = v2; A_hi[1, 0] = v1

    v1 = ct_lo * ca; v2 = ct_hi * ca
    if v1 <= v2: A_lo[1, 1] = v1; A_hi[1, 1] = v2
    else:        A_lo[1, 1] = v2; A_hi[1, 1] = v1

    A_lo[1, 2] = -sa; A_hi[1, 2] = -sa

    v1 = d_lo * (-sa); v2 = d_hi * (-sa)
    if v1 <= v2: A_lo[1, 3] = v1; A_hi[1, 3] = v2
    else:        A_lo[1, 3] = v2; A_hi[1, 3] = v1

    # row 2
    v1 = st_lo * sa; v2 = st_hi * sa
    if v1 <= v2: A_lo[2, 0] = v1; A_hi[2, 0] = v2
    else:        A_lo[2, 0] = v2; A_hi[2, 0] = v1

    v1 = ct_lo * sa; v2 = ct_hi * sa
    if v1 <= v2: A_lo[2, 1] = v1; A_hi[2, 1] = v2
    else:        A_lo[2, 1] = v2; A_hi[2, 1] = v1

    A_lo[2, 2] = ca; A_hi[2, 2] = ca

    v1 = d_lo * ca; v2 = d_hi * ca
    if v1 <= v2: A_lo[2, 3] = v1; A_hi[2, 3] = v2
    else:        A_lo[2, 3] = v2; A_hi[2, 3] = v1

    return A_lo, A_hi


# ─────────────────────────────────────────────────────
#  区间矩阵乘法（利用 DH 结构: row3 = [0,0,0,1]）
# ─────────────────────────────────────────────────────

def _imat_mul_dh(T_lo: np.ndarray, T_hi: np.ndarray,
                 A_lo: np.ndarray, A_hi: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    C = T * A （interval 4×4），利用 A 的 row3=[0,0,0,1] 结构。

    全向量化：消除 Python for-loop（4 行同时计算），
    numpy 函数调用从 ~38 降至 ~14，速度 ~2-3× 提升。
    """
    # A 前 3 行: (3, 4) view
    al = A_lo[:3]
    ah = A_hi[:3]

    # 广播 (4,3,1) × (1,3,4) → (4,3,4)，4 行同时计算
    tl = T_lo[:, :3, np.newaxis]   # (4, 3, 1)
    th = T_hi[:, :3, np.newaxis]   # (4, 3, 1)
    al_e = al[np.newaxis]          # (1, 3, 4)
    ah_e = ah[np.newaxis]          # (1, 3, 4)

    p1 = tl * al_e   # (4, 3, 4)
    p2 = tl * ah_e
    p3 = th * al_e
    p4 = th * ah_e

    mn = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    mx = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))

    R_lo = mn.sum(axis=1)  # (4, 4)
    R_hi = mx.sum(axis=1)

    # k=3 (A[3,:] = [0,0,0,1])  → 只有 j=3 列贡献 T[i,3]*1
    R_lo[:, 3] += T_lo[:, 3]
    R_hi[:, 3] += T_hi[:, 3]

    return R_lo, R_hi


# ─────────────────────────────────────────────────────
#  全量 FK
# ─────────────────────────────────────────────────────

def compute_fk_full(
    robot: Robot,
    intervals: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    全量区间 FK。

    Returns
    -------
    prefix_lo, prefix_hi : (n_transforms, 4, 4)
        cumulative interval transforms.  prefix[0]=I, prefix[k] = A0·…·A(k-1)
    joints_lo, joints_hi : (n_joint_matrices, 4, 4)
        per-joint DH interval matrices (含 tool_frame 若有)
    """
    n_joints = len(robot.dh_params)
    has_tool = robot.tool_frame is not None
    n_tf = n_joints + 1 + (1 if has_tool else 0)
    n_jm = n_joints + (1 if has_tool else 0)

    prefix_lo = np.zeros((n_tf, 4, 4))
    prefix_hi = np.zeros((n_tf, 4, 4))
    joints_lo = np.zeros((n_jm, 4, 4))
    joints_hi = np.zeros((n_jm, 4, 4))

    prefix_lo[0] = np.eye(4)
    prefix_hi[0] = np.eye(4)

    for i, param in enumerate(robot.dh_params):
        lo, hi = intervals[i]
        alpha = param['alpha']
        a_val = param['a']

        if param['type'] == 'revolute':
            theta_lo = lo + param['theta']
            theta_hi = hi + param['theta']
            ct_lo, ct_hi = _icos(theta_lo, theta_hi)
            st_lo, st_hi = _isin(theta_lo, theta_hi)
            d_lo = d_hi = param['d']
        else:
            theta = param['theta']
            ct_lo = ct_hi = math.cos(theta)
            st_lo = st_hi = math.sin(theta)
            d_lo = lo + param['d']
            d_hi = hi + param['d']

        A_lo, A_hi = _dh_joint_matrix(alpha, a_val, ct_lo, ct_hi,
                                       st_lo, st_hi, d_lo, d_hi)
        joints_lo[i] = A_lo
        joints_hi[i] = A_hi

        T_lo, T_hi = _imat_mul_dh(prefix_lo[i], prefix_hi[i], A_lo, A_hi)
        prefix_lo[i + 1] = T_lo
        prefix_hi[i + 1] = T_hi

    if has_tool:
        tf = robot.tool_frame
        ca_t = math.cos(tf['alpha'])
        sa_t = math.sin(tf['alpha'])
        At = np.array([
            [1.0, 0.0, 0.0, tf['a']],
            [0.0, ca_t, -sa_t, -tf['d'] * sa_t],
            [0.0, sa_t, ca_t, tf['d'] * ca_t],
            [0.0, 0.0, 0.0, 1.0],
        ])
        j_idx = n_joints
        joints_lo[j_idx] = At
        joints_hi[j_idx] = At.copy()
        T_lo, T_hi = _imat_mul_dh(prefix_lo[n_joints], prefix_hi[n_joints],
                                    At, At)
        prefix_lo[n_joints + 1] = T_lo
        prefix_hi[n_joints + 1] = T_hi

    return prefix_lo, prefix_hi, joints_lo, joints_hi


# ─────────────────────────────────────────────────────
#  增量 FK
# ─────────────────────────────────────────────────────

def compute_fk_incremental(
    robot: Robot,
    intervals: List[Tuple[float, float]],
    parent_prefix_lo: np.ndarray,
    parent_prefix_hi: np.ndarray,
    parent_joints_lo: np.ndarray,
    parent_joints_hi: np.ndarray,
    changed_joint: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    增量 FK：只重算 changed_joint 及其之后的变换。

    prefix[0..changed_joint] 和 joints[k] (k != changed_joint) 直接复用。
    平均节省 changed_joint / n_joints 的矩阵乘法（Panda ~43%）。
    """
    n_joints = len(robot.dh_params)
    has_tool = robot.tool_frame is not None
    n_tf = n_joints + 1 + (1 if has_tool else 0)
    n_jm = n_joints + (1 if has_tool else 0)

    d = changed_joint

    # 一次性 copy（单次 memcpy 比 empty + 多次 slice copy 更高效）
    prefix_lo = parent_prefix_lo.copy()
    prefix_hi = parent_prefix_hi.copy()
    joints_lo = parent_joints_lo.copy()
    joints_hi = parent_joints_hi.copy()

    # ── 重算 changed_joint ──
    param = robot.dh_params[d]
    lo, hi = intervals[d]
    alpha = param['alpha']
    a_val = param['a']

    if param['type'] == 'revolute':
        theta_lo = lo + param['theta']
        theta_hi = hi + param['theta']
        ct_lo, ct_hi = _icos(theta_lo, theta_hi)
        st_lo, st_hi = _isin(theta_lo, theta_hi)
        d_lo = d_hi = param['d']
    else:
        theta = param['theta']
        ct_lo = ct_hi = math.cos(theta)
        st_lo = st_hi = math.sin(theta)
        d_lo = lo + param['d']
        d_hi = hi + param['d']

    A_lo, A_hi = _dh_joint_matrix(alpha, a_val, ct_lo, ct_hi,
                                   st_lo, st_hi, d_lo, d_hi)
    joints_lo[d] = A_lo
    joints_hi[d] = A_hi

    T_lo, T_hi = _imat_mul_dh(prefix_lo[d], prefix_hi[d], A_lo, A_hi)
    prefix_lo[d + 1] = T_lo
    prefix_hi[d + 1] = T_hi

    # ── 后续关节：joint 矩阵已通过 copy 复用，只需重算 prefix ──
    for k in range(d + 1, n_joints):
        T_lo, T_hi = _imat_mul_dh(prefix_lo[k], prefix_hi[k],
                                    joints_lo[k], joints_hi[k])
        prefix_lo[k + 1] = T_lo
        prefix_hi[k + 1] = T_hi

    # ── tool frame ──
    if has_tool:
        j_idx = n_joints
        T_lo, T_hi = _imat_mul_dh(prefix_lo[n_joints], prefix_hi[n_joints],
                                    joints_lo[j_idx], joints_hi[j_idx])
        prefix_lo[n_joints + 1] = T_lo
        prefix_hi[n_joints + 1] = T_hi

    return prefix_lo, prefix_hi, joints_lo, joints_hi


def _split_fk_pair(
    robot: Robot,
    left_ivs: List[Tuple[float, float]],
    right_ivs: List[Tuple[float, float]],
    parent_prefix_lo: np.ndarray,
    parent_prefix_hi: np.ndarray,
    parent_joints_lo: np.ndarray,
    parent_joints_hi: np.ndarray,
    changed_joint: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """优化 D: 一次 parent copy 同时算左右子增量 FK.

    左右子节点只在 changed_joint 的区间不同, 共享:
    - parent prefix[0..d] (直接 copy 一次)
    - joints[k] for k != d (直接 copy 一次)
    只需分别重算 changed_joint 的 joint 矩阵和后续 prefix 链.
    相比调用两次 compute_fk_incremental, 节省一次完整的 4-array copy.
    """
    n_joints = len(robot.dh_params)
    has_tool = robot.tool_frame is not None
    d = changed_joint

    # ── 共享: copy parent arrays 一次 ──
    shared_jlo = parent_joints_lo.copy()
    shared_jhi = parent_joints_hi.copy()

    # ── LEFT child ──
    l_plo = parent_prefix_lo.copy()
    l_phi = parent_prefix_hi.copy()
    l_jlo = shared_jlo.copy()
    l_jhi = shared_jhi.copy()

    param = robot.dh_params[d]
    alpha = param['alpha']
    a_val = param['a']

    # left joint d
    lo_l, hi_l = left_ivs[d]
    if param['type'] == 'revolute':
        theta_lo = lo_l + param['theta']
        theta_hi = hi_l + param['theta']
        ct_lo, ct_hi = _icos(theta_lo, theta_hi)
        st_lo, st_hi = _isin(theta_lo, theta_hi)
        d_lo = d_hi = param['d']
    else:
        theta = param['theta']
        ct_lo = ct_hi = math.cos(theta)
        st_lo = st_hi = math.sin(theta)
        d_lo = lo_l + param['d']
        d_hi = hi_l + param['d']
    Al_lo, Al_hi = _dh_joint_matrix(alpha, a_val, ct_lo, ct_hi, st_lo, st_hi, d_lo, d_hi)
    l_jlo[d] = Al_lo
    l_jhi[d] = Al_hi

    T_lo, T_hi = _imat_mul_dh(l_plo[d], l_phi[d], Al_lo, Al_hi)
    l_plo[d + 1] = T_lo
    l_phi[d + 1] = T_hi
    for k in range(d + 1, n_joints):
        T_lo, T_hi = _imat_mul_dh(l_plo[k], l_phi[k], l_jlo[k], l_jhi[k])
        l_plo[k + 1] = T_lo
        l_phi[k + 1] = T_hi
    if has_tool:
        j_idx = n_joints
        T_lo, T_hi = _imat_mul_dh(l_plo[n_joints], l_phi[n_joints], l_jlo[j_idx], l_jhi[j_idx])
        l_plo[n_joints + 1] = T_lo
        l_phi[n_joints + 1] = T_hi

    # ── RIGHT child: 共享 shared_jlo/jhi 直接用 (不再 copy) ──
    r_plo = parent_prefix_lo.copy()
    r_phi = parent_prefix_hi.copy()
    r_jlo = shared_jlo  # 直接复用, 下面只改 [d]
    r_jhi = shared_jhi

    lo_r, hi_r = right_ivs[d]
    if param['type'] == 'revolute':
        theta_lo = lo_r + param['theta']
        theta_hi = hi_r + param['theta']
        ct_lo, ct_hi = _icos(theta_lo, theta_hi)
        st_lo, st_hi = _isin(theta_lo, theta_hi)
        d_lo = d_hi = param['d']
    else:
        theta = param['theta']
        ct_lo = ct_hi = math.cos(theta)
        st_lo = st_hi = math.sin(theta)
        d_lo = lo_r + param['d']
        d_hi = hi_r + param['d']
    Ar_lo, Ar_hi = _dh_joint_matrix(alpha, a_val, ct_lo, ct_hi, st_lo, st_hi, d_lo, d_hi)
    r_jlo[d] = Ar_lo
    r_jhi[d] = Ar_hi

    T_lo, T_hi = _imat_mul_dh(r_plo[d], r_phi[d], Ar_lo, Ar_hi)
    r_plo[d + 1] = T_lo
    r_phi[d + 1] = T_hi
    for k in range(d + 1, n_joints):
        T_lo, T_hi = _imat_mul_dh(r_plo[k], r_phi[k], r_jlo[k], r_jhi[k])
        r_plo[k + 1] = T_lo
        r_phi[k + 1] = T_hi
    if has_tool:
        j_idx = n_joints
        T_lo, T_hi = _imat_mul_dh(r_plo[n_joints], r_phi[n_joints], r_jlo[j_idx], r_jhi[j_idx])
        r_plo[n_joints + 1] = T_lo
        r_phi[n_joints + 1] = T_hi

    return l_plo, l_phi, l_jlo, l_jhi, r_plo, r_phi, r_jlo, r_jhi


# ─────────────────────────────────────────────────────
#  prefix transforms → LinkAABBInfo
# ─────────────────────────────────────────────────────

def extract_link_aabbs(
    prefix_lo: np.ndarray,
    prefix_hi: np.ndarray,
    n_links: int,
    zero_length_links: Set[int],
    skip_zero_length: bool = True,
) -> List[LinkAABBInfo]:
    """
    从 prefix transforms 提取 LinkAABBInfo。

    link i (1-based) 的 AABB = union(
        translation(prefix[i-1]),  translation(prefix[i])
    )
    """
    link_aabbs: List[LinkAABBInfo] = []
    for li in range(1, n_links + 1):
        is_zl = skip_zero_length and li in zero_length_links

        s_lo = prefix_lo[li - 1, :3, 3]
        s_hi = prefix_hi[li - 1, :3, 3]
        e_lo = prefix_lo[li, :3, 3]
        e_hi = prefix_hi[li, :3, 3]

        mins = np.minimum(s_lo, e_lo).tolist()
        maxs = np.maximum(s_hi, e_hi).tolist()

        link_aabbs.append(LinkAABBInfo(
            link_index=li,
            link_name=f"Link {li} (Joint {li - 1})",
            min_point=mins,
            max_point=maxs,
            is_zero_length=is_zl,
        ))
    return link_aabbs


# ─────────────────────────────────────────────────────
#  公开 API
# ─────────────────────────────────────────────────────

def compute_interval_aabb(
    robot: Robot,
    intervals: List[Tuple[float, float]],
    zero_length_links: Set[int],
    skip_zero_length: bool = True,
    n_sub: int = 1,
) -> Tuple[List[LinkAABBInfo], int]:
    """
    接口兼容 interval_fk.compute_interval_aabb，但用高速 interval 实现。
    """
    prefix_lo, prefix_hi, _, _ = compute_fk_full(robot, intervals)
    n_links = prefix_lo.shape[0] - 1
    link_aabbs = extract_link_aabbs(
        prefix_lo, prefix_hi, n_links,
        zero_length_links, skip_zero_length)
    return link_aabbs, 0


def compute_interval_aabb_fast(
    robot: Robot,
    intervals: List[Tuple[float, float]],
    zero_length_links: Set[int],
    skip_zero_length: bool = True,
    n_sub: int = 1,
) -> Tuple[List[LinkAABBInfo], int]:
    """快速接口别名，保留旧代码兼容。"""
    return compute_interval_aabb(
        robot=robot,
        intervals=intervals,
        zero_length_links=zero_length_links,
        skip_zero_length=skip_zero_length,
        n_sub=n_sub,
    )


# ─────────────────────────────────────────────────────
#  Cython 加速层：若可用则替换 compute_fk_full / compute_fk_incremental
# ─────────────────────────────────────────────────────

try:
    from ._interval_fk_core import (
        compute_fk_full_cy as _compute_fk_full_cy,
        compute_fk_incremental_cy as _compute_fk_incremental_cy,
    )

    # 保留纯 Python 版本（可供测试 / 回退调用）
    compute_fk_full_py = compute_fk_full
    compute_fk_incremental_py = compute_fk_incremental

    def compute_fk_full(
        robot: Robot,
        intervals: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cython-accelerated full interval FK."""
        return _compute_fk_full_cy(
            robot._dh_alpha, robot._dh_a, robot._dh_d, robot._dh_theta,
            robot._dh_joint_type,
            robot.tool_frame is not None,
            robot._tool_alpha, robot._tool_a, robot._tool_d,
            intervals,
        )

    def compute_fk_incremental(
        robot: Robot,
        intervals: List[Tuple[float, float]],
        parent_prefix_lo: np.ndarray,
        parent_prefix_hi: np.ndarray,
        parent_joints_lo: np.ndarray,
        parent_joints_hi: np.ndarray,
        changed_joint: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cython-accelerated incremental interval FK."""
        return _compute_fk_incremental_cy(
            robot._dh_alpha, robot._dh_a, robot._dh_d, robot._dh_theta,
            robot._dh_joint_type,
            robot.tool_frame is not None,
            robot._tool_alpha, robot._tool_a, robot._tool_d,
            intervals,
            parent_prefix_lo, parent_prefix_hi,
            parent_joints_lo, parent_joints_hi,
            changed_joint,
        )

except ImportError:
    pass
