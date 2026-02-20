# _fk_inline.pxi — 区间 FK 内联函数 (从 aabb/_interval_fk_core.pyx 提取)
#
# 供 _hier_core.pyx 通过 ``include`` 指令包含，
# 使 descent_loop 能在 Cython 层直接调用 FK 计算，
# 消除 Python 调度开销。
#
# 前提: 主 .pyx 已 cimport libc.math (cos, sin, ceil, floor, fmod)
#       以及 libc.string (memcpy, memset)。

# ── 编译期常量 ──
DEF MAX_JOINTS = 32
DEF MAX_TF     = 34     # MAX_JOINTS + 2 (base + optional tool)
DEF MAX_LINKS  = 32
DEF _TWO_PI        = 6.283185307179586
DEF _HALF_PI       = 1.5707963267948966
DEF _THREE_HALF_PI = 4.71238898038469
DEF _PI            = 3.141592653589793


# ═══════════════════════════════════════════
#  区间三角函数
# ═══════════════════════════════════════════

cdef inline void _isin(
    double lo, double hi,
    double* r_lo, double* r_hi,
) noexcept nogil:
    cdef double w = hi - lo
    cdef double s_lo, s_hi, lo_n, hi_n
    if w >= _TWO_PI:
        r_lo[0] = -1.0; r_hi[0] = 1.0; return
    s_lo = sin(lo); s_hi = sin(hi)
    if s_lo < s_hi:
        r_lo[0] = s_lo; r_hi[0] = s_hi
    else:
        r_lo[0] = s_hi; r_hi[0] = s_lo
    lo_n = fmod(lo, _TWO_PI)
    if lo_n < 0.0: lo_n += _TWO_PI
    hi_n = lo_n + w
    if ceil((lo_n - _HALF_PI) / _TWO_PI) <= floor((hi_n - _HALF_PI) / _TWO_PI):
        r_hi[0] = 1.0
    if ceil((lo_n - _THREE_HALF_PI) / _TWO_PI) <= floor((hi_n - _THREE_HALF_PI) / _TWO_PI):
        r_lo[0] = -1.0


cdef inline void _icos(
    double lo, double hi,
    double* r_lo, double* r_hi,
) noexcept nogil:
    cdef double w = hi - lo
    cdef double c_lo, c_hi, lo_n, hi_n
    if w >= _TWO_PI:
        r_lo[0] = -1.0; r_hi[0] = 1.0; return
    c_lo = cos(lo); c_hi = cos(hi)
    if c_lo < c_hi:
        r_lo[0] = c_lo; r_hi[0] = c_hi
    else:
        r_lo[0] = c_hi; r_hi[0] = c_lo
    lo_n = fmod(lo, _TWO_PI)
    if lo_n < 0.0: lo_n += _TWO_PI
    hi_n = lo_n + w
    if ceil(lo_n / _TWO_PI) <= floor(hi_n / _TWO_PI):
        r_hi[0] = 1.0
    if ceil((lo_n - _PI) / _TWO_PI) <= floor((hi_n - _PI) / _TWO_PI):
        r_lo[0] = -1.0


# ═══════════════════════════════════════════
#  DH 区间矩阵 (flat [16], row-major)
# ═══════════════════════════════════════════

cdef inline void _build_dh_joint(
    double alpha, double a,
    double ct_lo, double ct_hi,
    double st_lo, double st_hi,
    double d_lo, double d_hi,
    double* A_lo, double* A_hi,
) noexcept nogil:
    cdef double ca = cos(alpha)
    cdef double sa = sin(alpha)
    cdef double v1, v2

    memset(A_lo, 0, 16 * sizeof(double))
    memset(A_hi, 0, 16 * sizeof(double))
    A_lo[15] = 1.0; A_hi[15] = 1.0

    # row 0: [ct, -st, 0, a]
    A_lo[0] = ct_lo;  A_hi[0] = ct_hi
    A_lo[1] = -st_hi; A_hi[1] = -st_lo
    A_lo[3] = a;      A_hi[3] = a

    # row 1: [st*ca, ct*ca, -sa, -d*sa]
    v1 = st_lo * ca; v2 = st_hi * ca
    if v1 <= v2: A_lo[4] = v1; A_hi[4] = v2
    else:        A_lo[4] = v2; A_hi[4] = v1
    v1 = ct_lo * ca; v2 = ct_hi * ca
    if v1 <= v2: A_lo[5] = v1; A_hi[5] = v2
    else:        A_lo[5] = v2; A_hi[5] = v1
    A_lo[6] = -sa; A_hi[6] = -sa
    v1 = d_lo * (-sa); v2 = d_hi * (-sa)
    if v1 <= v2: A_lo[7] = v1; A_hi[7] = v2
    else:        A_lo[7] = v2; A_hi[7] = v1

    # row 2: [st*sa, ct*sa, ca, d*ca]
    v1 = st_lo * sa; v2 = st_hi * sa
    if v1 <= v2: A_lo[8] = v1; A_hi[8] = v2
    else:        A_lo[8] = v2; A_hi[8] = v1
    v1 = ct_lo * sa; v2 = ct_hi * sa
    if v1 <= v2: A_lo[9] = v1; A_hi[9] = v2
    else:        A_lo[9] = v2; A_hi[9] = v1
    A_lo[10] = ca; A_hi[10] = ca
    v1 = d_lo * ca; v2 = d_hi * ca
    if v1 <= v2: A_lo[11] = v1; A_hi[11] = v2
    else:        A_lo[11] = v2; A_hi[11] = v1


# ═══════════════════════════════════════════
#  区间矩阵乘法 (DH 结构优化)
# ═══════════════════════════════════════════

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
    """区间 4×4 乘法 R = T × A (row3 = [0,0,0,1])"""
    cdef int i, j, k, ti, tk, tj
    cdef double lo_sum, hi_sum, p1, p2, p3, p4

    for i in range(3):
        for j in range(4):
            lo_sum = 0.0; hi_sum = 0.0
            for k in range(3):
                ti = i * 4 + k; tk = k * 4 + j
                p1 = T_lo[ti] * A_lo[tk]
                p2 = T_lo[ti] * A_hi[tk]
                p3 = T_hi[ti] * A_lo[tk]
                p4 = T_hi[ti] * A_hi[tk]
                lo_sum += _min4(p1, p2, p3, p4)
                hi_sum += _max4(p1, p2, p3, p4)
            if j == 3:
                lo_sum += T_lo[i * 4 + 3]
                hi_sum += T_hi[i * 4 + 3]
            tj = i * 4 + j
            R_lo[tj] = lo_sum; R_hi[tj] = hi_sum

    R_lo[12] = 0.0; R_lo[13] = 0.0; R_lo[14] = 0.0; R_lo[15] = 1.0
    R_hi[12] = 0.0; R_hi[13] = 0.0; R_hi[14] = 0.0; R_hi[15] = 1.0


# ═══════════════════════════════════════════
#  关节 i 的区间 DH 矩阵
# ═══════════════════════════════════════════

cdef inline void _build_joint_i(
    int i,
    const double* alpha_p, const double* a_p,
    const double* d_p, const double* theta_p,
    const int* jtype_p,
    const double* ivs_lo, const double* ivs_hi,
    double* A_lo, double* A_hi,
) noexcept nogil:
    cdef double lo = ivs_lo[i], hi = ivs_hi[i]
    cdef double alpha_v = alpha_p[i], a_v = a_p[i]
    cdef double ct_lo, ct_hi, st_lo, st_hi, dl, dh
    cdef double theta_lo, theta_hi, theta

    if jtype_p[i] == 0:  # revolute
        theta_lo = lo + theta_p[i]; theta_hi = hi + theta_p[i]
        _icos(theta_lo, theta_hi, &ct_lo, &ct_hi)
        _isin(theta_lo, theta_hi, &st_lo, &st_hi)
        dl = d_p[i]; dh = d_p[i]
    else:               # prismatic
        theta = theta_p[i]
        ct_lo = cos(theta); ct_hi = ct_lo
        st_lo = sin(theta); st_hi = st_lo
        dl = lo + d_p[i]; dh = hi + d_p[i]

    _build_dh_joint(alpha_v, a_v, ct_lo, ct_hi, st_lo, st_hi, dl, dh,
                    A_lo, A_hi)


# ═══════════════════════════════════════════
#  Tool frame 矩阵 (theta=0)
# ═══════════════════════════════════════════

cdef inline void _build_tool_matrix(
    double alpha, double a, double d,
    double* A_lo, double* A_hi,
) noexcept nogil:
    cdef double ca = cos(alpha), sa = sin(alpha)
    memset(A_lo, 0, 16 * sizeof(double))
    A_lo[0]  = 1.0; A_lo[3]  = a
    A_lo[5]  = ca;  A_lo[6]  = -sa;  A_lo[7]  = -d * sa
    A_lo[9]  = sa;  A_lo[10] = ca;   A_lo[11] = d * ca
    A_lo[15] = 1.0
    memcpy(A_hi, A_lo, 16 * sizeof(double))


# ═══════════════════════════════════════════
#  增量 FK (C 级别, nogil)
# ═══════════════════════════════════════════

cdef inline void _incremental_fk_cc(
    int n_joints, bint has_tool,
    const double* parent_plo, const double* parent_phi,
    const double* parent_jlo, const double* parent_jhi,
    const double* alpha_p, const double* a_p,
    const double* d_p, const double* theta_p,
    const int* jtype_p,
    const double* ivs_lo, const double* ivs_hi,
    int changed_joint,
    double* out_plo, double* out_phi,
    double* out_jlo, double* out_jhi,
) noexcept nogil:
    """增量区间 FK: 复制父态，从 changed_joint 开始重算。"""
    cdef int n_tf = n_joints + 1 + (1 if has_tool else 0)
    cdef int n_jm = n_joints + (1 if has_tool else 0)
    cdef int pf_bytes = n_tf * 16 * sizeof(double)
    cdef int jt_bytes = n_jm * 16 * sizeof(double)
    cdef int d = changed_joint
    cdef double A_lo[16]
    cdef double A_hi[16]
    cdef int k

    memcpy(out_plo, parent_plo, pf_bytes)
    memcpy(out_phi, parent_phi, pf_bytes)
    memcpy(out_jlo, parent_jlo, jt_bytes)
    memcpy(out_jhi, parent_jhi, jt_bytes)

    # 重算 joint d
    _build_joint_i(d, alpha_p, a_p, d_p, theta_p, jtype_p,
                   ivs_lo, ivs_hi, A_lo, A_hi)
    memcpy(out_jlo + d * 16, A_lo, 16 * sizeof(double))
    memcpy(out_jhi + d * 16, A_hi, 16 * sizeof(double))

    # prefix[d+1] = prefix[d] × A
    _imat_mul_dh(out_plo + d * 16, out_phi + d * 16,
                 A_lo, A_hi,
                 out_plo + (d + 1) * 16, out_phi + (d + 1) * 16)

    # 后续关节: prefix 逐步重算
    for k in range(d + 1, n_joints):
        _imat_mul_dh(out_plo + k * 16, out_phi + k * 16,
                     out_jlo + k * 16, out_jhi + k * 16,
                     out_plo + (k + 1) * 16, out_phi + (k + 1) * 16)

    # tool frame
    if has_tool:
        _imat_mul_dh(out_plo + n_joints * 16, out_phi + n_joints * 16,
                     out_jlo + n_joints * 16, out_jhi + n_joints * 16,
                     out_plo + (n_joints + 1) * 16,
                     out_phi + (n_joints + 1) * 16)


# ═══════════════════════════════════════════
#  AABB 提取: prefix → (n_links, 6) float32
# ═══════════════════════════════════════════

cdef inline void _extract_compact_cc(
    const double* plo, const double* phi,
    int n_links, float* out,
) noexcept nogil:
    """从 prefix (n_tf, 4, 4) 提取 (n_links, 6) float32 AABB。

    link i: start = prefix[i][:3,3], end = prefix[i+1][:3,3]
    AABB lo = min(start_lo, end_lo), hi = max(start_hi, end_hi)
    """
    cdef int i
    cdef double s_lo, e_lo, s_hi, e_hi
    cdef int off_s, off_e

    for i in range(n_links):
        off_s = i * 16
        off_e = (i + 1) * 16

        # x (col 3 of row 0 → offset 3)
        s_lo = plo[off_s + 3]; e_lo = plo[off_e + 3]
        s_hi = phi[off_s + 3]; e_hi = phi[off_e + 3]
        out[i * 6]     = <float>(s_lo if s_lo < e_lo else e_lo)
        out[i * 6 + 3] = <float>(s_hi if s_hi > e_hi else e_hi)

        # y (col 3 of row 1 → offset 7)
        s_lo = plo[off_s + 7]; e_lo = plo[off_e + 7]
        s_hi = phi[off_s + 7]; e_hi = phi[off_e + 7]
        out[i * 6 + 1] = <float>(s_lo if s_lo < e_lo else e_lo)
        out[i * 6 + 4] = <float>(s_hi if s_hi > e_hi else e_hi)

        # z (col 3 of row 2 → offset 11)
        s_lo = plo[off_s + 11]; e_lo = plo[off_e + 11]
        s_hi = phi[off_s + 11]; e_hi = phi[off_e + 11]
        out[i * 6 + 2] = <float>(s_lo if s_lo < e_lo else e_lo)
        out[i * 6 + 5] = <float>(s_hi if s_hi > e_hi else e_hi)


# ═══════════════════════════════════════════
#  碰撞检测 (平坦化障碍物数组)
# ═══════════════════════════════════════════

cdef inline bint _link_aabbs_collide_flat(
    const float* aabb, const float* obs_flat, int n_obs,
) noexcept nogil:
    """SAT 碰撞: obs_flat 每 7 个 float = [link_idx, lo0, hi0, lo1, hi1, lo2, hi2]"""
    cdef int i, link_idx, off
    cdef float lo0, hi0, lo1, hi1, lo2, hi2

    for i in range(n_obs):
        link_idx = <int>obs_flat[i * 7]
        lo0 = obs_flat[i * 7 + 1]; hi0 = obs_flat[i * 7 + 2]
        lo1 = obs_flat[i * 7 + 3]; hi1 = obs_flat[i * 7 + 4]
        lo2 = obs_flat[i * 7 + 5]; hi2 = obs_flat[i * 7 + 6]

        off = link_idx * 6
        if aabb[off + 3] < lo0 or aabb[off]     > hi0: continue
        if aabb[off + 4] < lo1 or aabb[off + 1] > hi1: continue
        if aabb[off + 5] < lo2 or aabb[off + 2] > hi2: continue
        return True

    return False


# ═══════════════════════════════════════════
#  AABB union (float32 缓冲区)
# ═══════════════════════════════════════════

cdef inline void _union_aabb_buf_c(
    const float* a, const float* b,
    int n_links, float* out,
) noexcept nogil:
    cdef int i
    for i in range(n_links):
        out[i*6]   = a[i*6]   if a[i*6]   < b[i*6]   else b[i*6]
        out[i*6+1] = a[i*6+1] if a[i*6+1] < b[i*6+1] else b[i*6+1]
        out[i*6+2] = a[i*6+2] if a[i*6+2] < b[i*6+2] else b[i*6+2]
        out[i*6+3] = a[i*6+3] if a[i*6+3] > b[i*6+3] else b[i*6+3]
        out[i*6+4] = a[i*6+4] if a[i*6+4] > b[i*6+4] else b[i*6+4]
        out[i*6+5] = a[i*6+5] if a[i*6+5] > b[i*6+5] else b[i*6+5]


# ═══════════════════════════════════════════
#  AABB refine: intersect(old, union(left, right))
#
#  Both old (direct FK on parent interval) and union(left, right)
#  are valid over-approximations. Their intersection is tighter
#  while remaining a valid over-approximation.
#  For min-dims: take max(old, union)  → shrink lower bound upward
#  For max-dims: take min(old, union)  → shrink upper bound downward
# ═══════════════════════════════════════════

cdef inline void _refine_aabb_buf_c(
    float* dst, const float* union_buf,
    int n_links,
) noexcept nogil:
    """dst = intersect(dst, union_buf): tighten dst in-place.

    dst already holds the old (direct FK) AABB.
    union_buf holds union(left_child, right_child) AABB.
    Result: per-link per-axis min→max, max→min clamping.
    """
    cdef int i
    cdef float v
    for i in range(n_links):
        # min dims: take the larger (tighter) lower bound
        v = union_buf[i*6];   dst[i*6]   = v if v > dst[i*6]   else dst[i*6]
        v = union_buf[i*6+1]; dst[i*6+1] = v if v > dst[i*6+1] else dst[i*6+1]
        v = union_buf[i*6+2]; dst[i*6+2] = v if v > dst[i*6+2] else dst[i*6+2]
        # max dims: take the smaller (tighter) upper bound
        v = union_buf[i*6+3]; dst[i*6+3] = v if v < dst[i*6+3] else dst[i*6+3]
        v = union_buf[i*6+4]; dst[i*6+4] = v if v < dst[i*6+4] else dst[i*6+4]
        v = union_buf[i*6+5]; dst[i*6+5] = v if v < dst[i*6+5] else dst[i*6+5]
