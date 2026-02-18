# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t ITYPE_t


def link_position_core(
    cnp.ndarray[DTYPE_t, ndim=1] q,
    int link_idx,
    cnp.ndarray[DTYPE_t, ndim=1] alpha_arr,
    cnp.ndarray[DTYPE_t, ndim=1] a_arr,
    cnp.ndarray[DTYPE_t, ndim=1] d_arr,
    cnp.ndarray[DTYPE_t, ndim=1] theta_arr,
    cnp.ndarray[ITYPE_t, ndim=1] joint_type_arr,
    bint has_tool,
    double tool_alpha,
    double tool_a,
    double tool_d,
):
    cdef int n = alpha_arr.shape[0]
    cdef int max_link = link_idx
    cdef int i, r, c, k
    cdef double alpha, a, d, theta, ca, sa, ct, st
    cdef double T[4][4]
    cdef double A[4][4]
    cdef double R[4][4]

    if max_link > n:
        max_link = n
    if max_link < 0:
        max_link = 0

    # T = I
    for r in range(4):
        for c in range(4):
            T[r][c] = 1.0 if r == c else 0.0

    for i in range(max_link):
        alpha = alpha_arr[i]
        a = a_arr[i]

        if joint_type_arr[i] == 0:
            d = d_arr[i]
            theta = q[i] + theta_arr[i]
        else:
            d = d_arr[i] + q[i]
            theta = theta_arr[i]

        ca = cos(alpha)
        sa = sin(alpha)
        ct = cos(theta)
        st = sin(theta)

        A[0][0] = ct
        A[0][1] = -st
        A[0][2] = 0.0
        A[0][3] = a

        A[1][0] = st * ca
        A[1][1] = ct * ca
        A[1][2] = -sa
        A[1][3] = -d * sa

        A[2][0] = st * sa
        A[2][1] = ct * sa
        A[2][2] = ca
        A[2][3] = d * ca

        A[3][0] = 0.0
        A[3][1] = 0.0
        A[3][2] = 0.0
        A[3][3] = 1.0

        # T = T @ A
        for r in range(4):
            for c in range(4):
                R[r][c] = 0.0
                for k in range(4):
                    R[r][c] += T[r][k] * A[k][c]

        for r in range(4):
            for c in range(4):
                T[r][c] = R[r][c]

    if link_idx > n and has_tool:
        alpha = tool_alpha
        a = tool_a
        d = tool_d
        theta = 0.0

        ca = cos(alpha)
        sa = sin(alpha)
        ct = cos(theta)
        st = sin(theta)

        A[0][0] = ct
        A[0][1] = -st
        A[0][2] = 0.0
        A[0][3] = a

        A[1][0] = st * ca
        A[1][1] = ct * ca
        A[1][2] = -sa
        A[1][3] = -d * sa

        A[2][0] = st * sa
        A[2][1] = ct * sa
        A[2][2] = ca
        A[2][3] = d * ca

        A[3][0] = 0.0
        A[3][1] = 0.0
        A[3][2] = 0.0
        A[3][3] = 1.0

        for r in range(4):
            for c in range(4):
                R[r][c] = 0.0
                for k in range(4):
                    R[r][c] += T[r][k] * A[k][c]

        for r in range(4):
            for c in range(4):
                T[r][c] = R[r][c]

    return np.array([T[0][3], T[1][3], T[2][3]], dtype=np.float64)
