# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
_interval_fk_core.pyx — Cython 加速的区间正运动学核心

用 C 级别标量循环 + 栈上 4×4 矩阵替代 Python/numpy 区间 FK，
在 find_free_box 热路径上预期获得 5–15× 加速。

函数签名与 interval_fk.py 的 compute_fk_full / compute_fk_incremental
保持一致（由 interval_fk.py 中的 wrapper 透明转发）。
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin, ceil, floor, fmod
from libc.string cimport memcpy, memset

cnp.import_array()

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t   ITYPE_t

# ── 常量 ──
DEF MAX_JOINTS = 32
DEF _TWO_PI        = 6.283185307179586    # 2π
DEF _HALF_PI       = 1.5707963267948966   # π/2
DEF _THREE_HALF_PI = 4.71238898038469     # 3π/2
DEF _PI            = 3.141592653589793    # π


# ═════════════════════════════════════════════
#  区间三角函数 (nogil, inline)
# ═════════════════════════════════════════════

cdef inline void _isin(
    double lo, double hi,
    double* r_lo, double* r_hi,
) noexcept nogil:
    """sin 的保守区间包络  [r_lo, r_hi] ⊇ {sin(x) : x ∈ [lo, hi]}"""
    cdef double w = hi - lo
    cdef double s_lo, s_hi, lo_n, hi_n

    if w >= _TWO_PI:
        r_lo[0] = -1.0
        r_hi[0] = 1.0
        return

    s_lo = sin(lo)
    s_hi = sin(hi)
    if s_lo < s_hi:
        r_lo[0] = s_lo
        r_hi[0] = s_hi
    else:
        r_lo[0] = s_hi
        r_hi[0] = s_lo

    # 归一化 lo 到 [0, 2π)
    lo_n = fmod(lo, _TWO_PI)
    if lo_n < 0.0:
        lo_n += _TWO_PI
    hi_n = lo_n + w

    # π/2 在区间内 → sin=1
    if ceil((lo_n - _HALF_PI) / _TWO_PI) <= floor((hi_n - _HALF_PI) / _TWO_PI):
        r_hi[0] = 1.0
    # 3π/2 在区间内 → sin=-1
    if ceil((lo_n - _THREE_HALF_PI) / _TWO_PI) <= floor((hi_n - _THREE_HALF_PI) / _TWO_PI):
        r_lo[0] = -1.0


cdef inline void _icos(
    double lo, double hi,
    double* r_lo, double* r_hi,
) noexcept nogil:
    """cos 的保守区间包络  [r_lo, r_hi] ⊇ {cos(x) : x ∈ [lo, hi]}"""
    cdef double w = hi - lo
    cdef double c_lo, c_hi, lo_n, hi_n

    if w >= _TWO_PI:
        r_lo[0] = -1.0
        r_hi[0] = 1.0
        return

    c_lo = cos(lo)
    c_hi = cos(hi)
    if c_lo < c_hi:
        r_lo[0] = c_lo
        r_hi[0] = c_hi
    else:
        r_lo[0] = c_hi
        r_hi[0] = c_lo

    lo_n = fmod(lo, _TWO_PI)
    if lo_n < 0.0:
        lo_n += _TWO_PI
    hi_n = lo_n + w

    # 0 在区间内 → cos=1
    if ceil(lo_n / _TWO_PI) <= floor(hi_n / _TWO_PI):
        r_hi[0] = 1.0
    # π 在区间内 → cos=-1
    if ceil((lo_n - _PI) / _TWO_PI) <= floor((hi_n - _PI) / _TWO_PI):
        r_lo[0] = -1.0


# ═════════════════════════════════════════════
#  DH 区间矩阵构造 (flat [16], row-major)
# ═════════════════════════════════════════════

cdef inline void _build_dh_joint(
    double alpha, double a,
    double ct_lo, double ct_hi,
    double st_lo, double st_hi,
    double d_lo, double d_hi,
    double* A_lo, double* A_hi,
) noexcept nogil:
    """构造单关节 DH 区间矩阵 (A_lo, A_hi)，flat [16] row-major。"""
    cdef double ca = cos(alpha)
    cdef double sa = sin(alpha)
    cdef double v1, v2

    memset(A_lo, 0, 16 * sizeof(double))
    memset(A_hi, 0, 16 * sizeof(double))
    A_lo[15] = 1.0;  A_hi[15] = 1.0   # row3 = [0,0,0,1]

    # ── row 0: [ct, -st, 0, a] ──
    A_lo[0] = ct_lo;   A_hi[0] = ct_hi       # [0,0]
    A_lo[1] = -st_hi;  A_hi[1] = -st_lo      # [0,1]
    # [0,2] = 0
    A_lo[3] = a;        A_hi[3] = a           # [0,3]

    # ── row 1: [st*ca, ct*ca, -sa, -d*sa] ──
    v1 = st_lo * ca;  v2 = st_hi * ca
    if v1 <= v2:
        A_lo[4] = v1;  A_hi[4] = v2
    else:
        A_lo[4] = v2;  A_hi[4] = v1

    v1 = ct_lo * ca;  v2 = ct_hi * ca
    if v1 <= v2:
        A_lo[5] = v1;  A_hi[5] = v2
    else:
        A_lo[5] = v2;  A_hi[5] = v1

    A_lo[6] = -sa;  A_hi[6] = -sa

    v1 = d_lo * (-sa);  v2 = d_hi * (-sa)
    if v1 <= v2:
        A_lo[7] = v1;  A_hi[7] = v2
    else:
        A_lo[7] = v2;  A_hi[7] = v1

    # ── row 2: [st*sa, ct*sa, ca, d*ca] ──
    v1 = st_lo * sa;  v2 = st_hi * sa
    if v1 <= v2:
        A_lo[8] = v1;  A_hi[8] = v2
    else:
        A_lo[8] = v2;  A_hi[8] = v1

    v1 = ct_lo * sa;  v2 = ct_hi * sa
    if v1 <= v2:
        A_lo[9] = v1;  A_hi[9] = v2
    else:
        A_lo[9] = v2;  A_hi[9] = v1

    A_lo[10] = ca;  A_hi[10] = ca

    v1 = d_lo * ca;  v2 = d_hi * ca
    if v1 <= v2:
        A_lo[11] = v1;  A_hi[11] = v2
    else:
        A_lo[11] = v2;  A_hi[11] = v1


# ═════════════════════════════════════════════
#  区间 4×4 矩阵乘法 (DH 结构优化)
# ═════════════════════════════════════════════

cdef inline double _min4(double a, double b, double c, double d) noexcept nogil:
    cdef double m = a
    if b < m: m = b
    if c < m: m = c
    if d < m: m = d
    return m

cdef inline double _max4(double a, double b, double c, double d) noexcept nogil:
    cdef double m = a
    if b > m: m = b
    if c > m: m = c
    if d > m: m = d
    return m


cdef inline void _imat_mul_dh(
    const double* T_lo, const double* T_hi,
    const double* A_lo, const double* A_hi,
    double* R_lo, double* R_hi,
) noexcept nogil:
    """区间 4×4 矩阵乘法 C = T × A (利用 row3=[0,0,0,1])

    仅计算 rows 0-2；row 3 硬编码 [0,0,0,1]。
    完全展开内层循环以消除分支。
    """
    cdef int i, j, k
    cdef int ti, tk, tj
    cdef double lo_sum, hi_sum
    cdef double p1, p2, p3, p4

    for i in range(3):
        for j in range(4):
            lo_sum = 0.0
            hi_sum = 0.0
            for k in range(3):
                ti = i * 4 + k
                tk = k * 4 + j
                p1 = T_lo[ti] * A_lo[tk]
                p2 = T_lo[ti] * A_hi[tk]
                p3 = T_hi[ti] * A_lo[tk]
                p4 = T_hi[ti] * A_hi[tk]
                lo_sum += _min4(p1, p2, p3, p4)
                hi_sum += _max4(p1, p2, p3, p4)

            if j == 3:
                # A[3][3]=1  →  += T[i][3]*1
                lo_sum += T_lo[i * 4 + 3]
                hi_sum += T_hi[i * 4 + 3]

            tj = i * 4 + j
            R_lo[tj] = lo_sum
            R_hi[tj] = hi_sum

    # row 3 = [0, 0, 0, 1]
    R_lo[12] = 0.0;  R_lo[13] = 0.0;  R_lo[14] = 0.0;  R_lo[15] = 1.0
    R_hi[12] = 0.0;  R_hi[13] = 0.0;  R_hi[14] = 0.0;  R_hi[15] = 1.0


# ═════════════════════════════════════════════
#  辅助: 从 DH 参数构造关节 i 的区间矩阵
# ═════════════════════════════════════════════

cdef inline void _build_joint_i(
    int i,
    const double* alpha_p,
    const double* a_p,
    const double* d_p,
    const double* theta_p,
    const int*    jtype_p,
    const double* ivs_lo,
    const double* ivs_hi,
    double* A_lo, double* A_hi,
) noexcept nogil:
    """根据第 i 个关节的 DH 参数 + 区间，构建 (A_lo, A_hi)。"""
    cdef double lo = ivs_lo[i]
    cdef double hi = ivs_hi[i]
    cdef double alpha_v = alpha_p[i]
    cdef double a_v     = a_p[i]
    cdef double ct_lo, ct_hi, st_lo, st_hi
    cdef double dl, dh, theta_lo, theta_hi, theta

    if jtype_p[i] == 0:        # revolute
        theta_lo = lo + theta_p[i]
        theta_hi = hi + theta_p[i]
        _icos(theta_lo, theta_hi, &ct_lo, &ct_hi)
        _isin(theta_lo, theta_hi, &st_lo, &st_hi)
        dl = d_p[i]
        dh = d_p[i]
    else:                       # prismatic
        theta = theta_p[i]
        ct_lo = cos(theta);  ct_hi = ct_lo
        st_lo = sin(theta);  st_hi = st_lo
        dl = lo + d_p[i]
        dh = hi + d_p[i]

    _build_dh_joint(alpha_v, a_v, ct_lo, ct_hi, st_lo, st_hi, dl, dh, A_lo, A_hi)


# ═════════════════════════════════════════════
#  全量 FK  (Python ← C)
# ═════════════════════════════════════════════

def compute_fk_full_cy(
    cnp.ndarray[DTYPE_t, ndim=1] alpha_arr,
    cnp.ndarray[DTYPE_t, ndim=1] a_arr,
    cnp.ndarray[DTYPE_t, ndim=1] d_arr,
    cnp.ndarray[DTYPE_t, ndim=1] theta_arr,
    cnp.ndarray[ITYPE_t, ndim=1] joint_type_arr,
    bint has_tool,
    double tool_alpha, double tool_a, double tool_d,
    object intervals,
):
    """
    全量区间 FK (Cython)。

    Parameters
    ----------
    alpha_arr … joint_type_arr : 来自 Robot 预打包数组
    has_tool : 是否有 tool_frame
    tool_alpha, tool_a, tool_d : tool_frame DH 参数
    intervals : List[Tuple[float, float]]

    Returns
    -------
    (prefix_lo, prefix_hi, joints_lo, joints_hi)
        各为 numpy float64 数组，形状与 Python 版一致。
    """
    cdef int n_joints = alpha_arr.shape[0]
    cdef int n_tf = n_joints + 1 + (1 if has_tool else 0)
    cdef int n_jm = n_joints + (1 if has_tool else 0)

    if n_joints > MAX_JOINTS:
        raise ValueError(f"n_joints={n_joints} exceeds MAX_JOINTS={MAX_JOINTS}")

    # ── 输出 numpy 数组 ──
    cdef cnp.ndarray plo_np = np.zeros((n_tf, 4, 4), dtype=np.float64)
    cdef cnp.ndarray phi_np = np.zeros((n_tf, 4, 4), dtype=np.float64)
    cdef cnp.ndarray jlo_np = np.zeros((n_jm, 4, 4), dtype=np.float64)
    cdef cnp.ndarray jhi_np = np.zeros((n_jm, 4, 4), dtype=np.float64)

    # C-contiguous memoryview  →  可取指针
    cdef double[:, :, ::1] plo = plo_np
    cdef double[:, :, ::1] phi = phi_np
    cdef double[:, :, ::1] jlo = jlo_np
    cdef double[:, :, ::1] jhi = jhi_np

    # prefix[0] = I
    plo[0, 0, 0] = 1.0;  plo[0, 1, 1] = 1.0
    plo[0, 2, 2] = 1.0;  plo[0, 3, 3] = 1.0
    phi[0, 0, 0] = 1.0;  phi[0, 1, 1] = 1.0
    phi[0, 2, 2] = 1.0;  phi[0, 3, 3] = 1.0

    # ── 提取区间到 C 数组 ──
    cdef double ivs_lo[MAX_JOINTS]
    cdef double ivs_hi[MAX_JOINTS]
    cdef int i
    for i in range(n_joints):
        ivs_lo[i] = intervals[i][0]
        ivs_hi[i] = intervals[i][1]

    # ── DH 参数指针 ──
    cdef double* alpha_p = <double*> alpha_arr.data
    cdef double* a_p     = <double*> a_arr.data
    cdef double* d_p     = <double*> d_arr.data
    cdef double* theta_p = <double*> theta_arr.data
    cdef int*    jtype_p = <int*>    joint_type_arr.data

    # ── 栈上临时矩阵 ──
    cdef double A_lo[16]
    cdef double A_hi[16]

    # ── 主循环 ──
    for i in range(n_joints):
        _build_joint_i(i, alpha_p, a_p, d_p, theta_p, jtype_p,
                       ivs_lo, ivs_hi, A_lo, A_hi)

        # joints[i] = A
        memcpy(&jlo[i, 0, 0], A_lo, 16 * sizeof(double))
        memcpy(&jhi[i, 0, 0], A_hi, 16 * sizeof(double))

        # prefix[i+1] = prefix[i] × A
        _imat_mul_dh(&plo[i, 0, 0], &phi[i, 0, 0],
                     A_lo, A_hi,
                     &plo[i + 1, 0, 0], &phi[i + 1, 0, 0])

    # ── tool frame（固定变换, 无区间）──
    if has_tool:
        _build_tool_matrix(tool_alpha, tool_a, tool_d, A_lo, A_hi)
        memcpy(&jlo[n_joints, 0, 0], A_lo, 16 * sizeof(double))
        memcpy(&jhi[n_joints, 0, 0], A_hi, 16 * sizeof(double))
        _imat_mul_dh(&plo[n_joints, 0, 0], &phi[n_joints, 0, 0],
                     A_lo, A_hi,
                     &plo[n_joints + 1, 0, 0], &phi[n_joints + 1, 0, 0])

    return plo_np, phi_np, jlo_np, jhi_np


# ═════════════════════════════════════════════
#  增量 FK  (Python ← C)
# ═════════════════════════════════════════════

def compute_fk_incremental_cy(
    cnp.ndarray[DTYPE_t, ndim=1] alpha_arr,
    cnp.ndarray[DTYPE_t, ndim=1] a_arr,
    cnp.ndarray[DTYPE_t, ndim=1] d_arr,
    cnp.ndarray[DTYPE_t, ndim=1] theta_arr,
    cnp.ndarray[ITYPE_t, ndim=1] joint_type_arr,
    bint has_tool,
    double tool_alpha, double tool_a, double tool_d,
    object intervals,
    cnp.ndarray[DTYPE_t, ndim=3] parent_prefix_lo,
    cnp.ndarray[DTYPE_t, ndim=3] parent_prefix_hi,
    cnp.ndarray[DTYPE_t, ndim=3] parent_joints_lo,
    cnp.ndarray[DTYPE_t, ndim=3] parent_joints_hi,
    int changed_joint,
):
    """
    增量区间 FK (Cython)：只重算 changed_joint 及其之后的 prefix 变换。

    prefix[0..changed_joint] 和 joints[k] (k ≠ changed_joint) 直接 memcpy 复用。
    """
    cdef int n_joints = alpha_arr.shape[0]
    cdef int n_tf = n_joints + 1 + (1 if has_tool else 0)
    cdef int n_jm = n_joints + (1 if has_tool else 0)
    cdef int d = changed_joint

    if n_joints > MAX_JOINTS:
        raise ValueError(f"n_joints={n_joints} exceeds MAX_JOINTS={MAX_JOINTS}")

    # ── 一次 copy（单次 memcpy 比 empty + 多次切片更快）──
    cdef cnp.ndarray plo_np = parent_prefix_lo.copy()
    cdef cnp.ndarray phi_np = parent_prefix_hi.copy()
    cdef cnp.ndarray jlo_np = parent_joints_lo.copy()
    cdef cnp.ndarray jhi_np = parent_joints_hi.copy()

    cdef double[:, :, ::1] plo = plo_np
    cdef double[:, :, ::1] phi = phi_np
    cdef double[:, :, ::1] jlo = jlo_np
    cdef double[:, :, ::1] jhi = jhi_np

    # ── 提取 changed joint 的区间 ──
    cdef double ivs_lo[MAX_JOINTS]
    cdef double ivs_hi[MAX_JOINTS]
    cdef int i
    for i in range(n_joints):
        ivs_lo[i] = intervals[i][0]
        ivs_hi[i] = intervals[i][1]

    # ── DH 参数指针 ──
    cdef double* alpha_p = <double*> alpha_arr.data
    cdef double* a_p     = <double*> a_arr.data
    cdef double* d_p     = <double*> d_arr.data
    cdef double* theta_p = <double*> theta_arr.data
    cdef int*    jtype_p = <int*>    joint_type_arr.data

    cdef double A_lo[16]
    cdef double A_hi[16]

    # ── 重算 changed_joint 的 DH 矩阵 ──
    _build_joint_i(d, alpha_p, a_p, d_p, theta_p, jtype_p,
                   ivs_lo, ivs_hi, A_lo, A_hi)
    memcpy(&jlo[d, 0, 0], A_lo, 16 * sizeof(double))
    memcpy(&jhi[d, 0, 0], A_hi, 16 * sizeof(double))

    # prefix[d+1] = prefix[d] × A_new
    _imat_mul_dh(&plo[d, 0, 0], &phi[d, 0, 0],
                 A_lo, A_hi,
                 &plo[d + 1, 0, 0], &phi[d + 1, 0, 0])

    # ── 后续关节: joint 矩阵已通过 copy 复用，只需重算 prefix ──
    cdef int k
    for k in range(d + 1, n_joints):
        _imat_mul_dh(&plo[k, 0, 0], &phi[k, 0, 0],
                     &jlo[k, 0, 0], &jhi[k, 0, 0],
                     &plo[k + 1, 0, 0], &phi[k + 1, 0, 0])

    # ── tool frame ──
    if has_tool:
        _imat_mul_dh(&plo[n_joints, 0, 0], &phi[n_joints, 0, 0],
                     &jlo[n_joints, 0, 0], &jhi[n_joints, 0, 0],
                     &plo[n_joints + 1, 0, 0], &phi[n_joints + 1, 0, 0])

    return plo_np, phi_np, jlo_np, jhi_np


# ═════════════════════════════════════════════
#  tool frame 矩阵（无区间, A_lo == A_hi）
# ═════════════════════════════════════════════

cdef inline void _build_tool_matrix(
    double alpha, double a, double d,
    double* A_lo, double* A_hi,
) noexcept nogil:
    """Tool frame DH 矩阵 (theta=0, 无区间 → A_lo == A_hi)。"""
    cdef double ca = cos(alpha)
    cdef double sa = sin(alpha)

    memset(A_lo, 0, 16 * sizeof(double))

    A_lo[0]  = 1.0               # ct = cos(0) = 1
    # A_lo[1]  = 0.0             # -st = 0
    # A_lo[2]  = 0.0
    A_lo[3]  = a

    # A_lo[4]  = 0.0             # st * ca = 0
    A_lo[5]  = ca                # ct * ca
    A_lo[6]  = -sa
    A_lo[7]  = -d * sa

    # A_lo[8]  = 0.0             # st * sa = 0
    A_lo[9]  = sa                # ct * sa
    A_lo[10] = ca
    A_lo[11] = d * ca

    A_lo[15] = 1.0

    memcpy(A_hi, A_lo, 16 * sizeof(double))
