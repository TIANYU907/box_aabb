# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
planner/_hier_core.pyx - HierAABBTree Cython 热路径

提供 C 级别的节点访问器、SAT 碰撞检测、向上传播、
标记占用等核心方法，通过 typed memoryview + 指针运算
实现极低开销的 mmap-backed 节点操作。

所有方法均在持有 GIL 的情况下调用（find_free_box 循环
包含 FK 计算依赖 Python），但通过消除 Python 对象边界
开销获得 3-10× 加速。
"""

import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy, memset
from libc.math cimport fabsf, ldexp

cnp.import_array()

# ─────────────────────────────────────────────
#  节点字段偏移 (与 _hier_layout.py 保持一致)
# ─────────────────────────────────────────────

DEF _OFF_LEFT     = 0
DEF _OFF_RIGHT    = 4
DEF _OFF_PARENT   = 8
DEF _OFF_DEPTH    = 12
DEF _OFF_SPLIT    = 16
DEF _OFF_HAS_AABB = 24
DEF _OFF_DIRTY    = 25
DEF _OFF_AABB     = 28


# ─────────────────────────────────────────────
#  内联节点访问器
# ─────────────────────────────────────────────

cdef inline char* _node_ptr(char* base, int stride, int idx) noexcept nogil:
    return base + <long long>idx * stride

cdef inline int _get_i32(char* node, int off) noexcept nogil:
    return (<int*>(node + off))[0]

cdef inline void _set_i32(char* node, int off, int val) noexcept nogil:
    (<int*>(node + off))[0] = val

cdef inline double _get_f64(char* node, int off) noexcept nogil:
    return (<double*>(node + off))[0]

cdef inline void _set_f64(char* node, int off, double val) noexcept nogil:
    (<double*>(node + off))[0] = val

cdef inline unsigned char _get_u8(char* node, int off) noexcept nogil:
    return (<unsigned char*>(node + off))[0]

cdef inline void _set_u8(char* node, int off, unsigned char val) noexcept nogil:
    (<unsigned char*>(node + off))[0] = val

cdef inline float* _aabb_ptr(char* node) noexcept nogil:
    return <float*>(node + _OFF_AABB)


# ─────────────────────────────────────────────
#  NodeStore — C 级别节点存储管理
# ─────────────────────────────────────────────

cdef class NodeStore:
    """管理 mmap-backed 或内存中的固定 stride 节点数组

    提供 C 级别的字段访问器与核心算法方法。
    HierAABBTree 持有一个 NodeStore 实例作为底层存储。
    """

    cdef:
        # 节点数据缓冲区 (memoryview, 可由 numpy array 或 memmap 支撑)
        cnp.uint8_t[::1] _buf
        char* _base              # _buf 的 C 指针 (缓存避免重复取)
        int _stride              # 每节点字节步长
        int _n_links             # 连杆数
        int _n_dims              # 关节维度
        int _aabb_floats         # n_links * 6
        int _cap                 # 已分配节点容量
        public int next_idx      # 下一个空闲索引

        # 连杆范围 (用于碰撞的遍历)
        int _range_start         # 第一个非零长连杆的 link index
        int _range_end           # 最后一个连杆 + 1
        int _n_non_zero          # 非零长连杆数
        int[32] _non_zero_links  # 非零长连杆索引 (最多 32)

        # 临时数组 (占用状态, 不持久化)
        cnp.uint8_t[::1]  _occupied
        cnp.int32_t[::1]  _subtree_occ
        cnp.float64_t[::1] _subtree_occ_vol   # 子树占用体积分数 (占 root 体积的比例)
        cnp.int32_t[::1]  _forest_id

    def __init__(self, int n_links, int n_dims, int stride, int cap,
                 object zero_length_links=None):
        self._n_links = n_links
        self._n_dims = n_dims
        self._stride = stride
        self._aabb_floats = n_links * 6
        self._cap = cap
        self.next_idx = 1  # root = 0

        # 分配内存缓冲区
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] arr = np.zeros(
            cap * stride, dtype=np.uint8)
        self._buf = arr
        self._base = <char*>&self._buf[0]

        # 初始化 root 节点
        cdef char* root = self._base
        _set_i32(root, _OFF_LEFT, -1)
        _set_i32(root, _OFF_RIGHT, -1)
        _set_i32(root, _OFF_PARENT, -1)
        _set_i32(root, _OFF_DEPTH, 0)
        _set_f64(root, _OFF_SPLIT, 0.0)
        _set_u8(root, _OFF_HAS_AABB, 0)
        _set_u8(root, _OFF_DIRTY, 0)

        # 临时数组
        self._occupied = np.zeros(cap, dtype=np.uint8)
        self._subtree_occ = np.zeros(cap, dtype=np.int32)
        self._subtree_occ_vol = np.zeros(cap, dtype=np.float64)
        self._forest_id = np.full(cap, -1, dtype=np.int32)

        # 连杆元数据
        self._setup_link_meta(zero_length_links)

    cdef void _setup_link_meta(self, object zero_length_links):
        """设置非零长连杆列表，用于碰撞检测跳过零长连杆"""
        cdef set zl = set(zero_length_links) if zero_length_links else set()
        cdef int nl = self._n_links
        cdef int count = 0
        cdef int i

        # range_start: 跳过 link 0 (base)
        self._range_start = 1
        self._range_end = nl

        for i in range(1, nl):  # skip link 0 (base)
            if i not in zl:
                self._non_zero_links[count] = i
                count += 1
        self._n_non_zero = count

    def attach_buffer(self, cnp.uint8_t[::1] buf, int cap):
        """绑定外部缓冲区 (mmap 数据区), 替换当前缓冲区"""
        self._buf = buf
        self._base = <char*>&self._buf[0]
        self._cap = cap
        # 重新分配临时数组
        self._occupied = np.zeros(cap, dtype=np.uint8)
        self._subtree_occ = np.zeros(cap, dtype=np.int32)
        self._subtree_occ_vol = np.zeros(cap, dtype=np.float64)
        self._forest_id = np.full(cap, -1, dtype=np.int32)

    # ── 容量管理 ──

    def ensure_capacity(self, int needed):
        """确保容量足够，不足时 2× 扩容"""
        if needed <= self._cap:
            return
        cdef int new_cap = max(needed, self._cap * 2)
        cdef int old_size = self._cap * self._stride
        cdef int new_size = new_cap * self._stride

        # 扩容节点数据
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] new_arr = np.zeros(
            new_size, dtype=np.uint8)
        new_arr[:old_size] = np.asarray(self._buf)
        self._buf = new_arr
        self._base = <char*>&self._buf[0]

        # 扩容临时数组
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] new_occ = np.zeros(
            new_cap, dtype=np.uint8)
        new_occ[:self._cap] = np.asarray(self._occupied)
        self._occupied = new_occ

        cdef cnp.ndarray[cnp.int32_t, ndim=1] new_sub = np.zeros(
            new_cap, dtype=np.int32)
        new_sub[:self._cap] = np.asarray(self._subtree_occ)
        self._subtree_occ = new_sub

        cdef cnp.ndarray[cnp.float64_t, ndim=1] new_subv = np.zeros(
            new_cap, dtype=np.float64)
        new_subv[:self._cap] = np.asarray(self._subtree_occ_vol)
        self._subtree_occ_vol = new_subv

        cdef cnp.ndarray[cnp.int32_t, ndim=1] new_fid = np.full(
            new_cap, -1, dtype=np.int32)
        new_fid[:self._cap] = np.asarray(self._forest_id)
        self._forest_id = new_fid

        self._cap = new_cap

    # ── 节点分配 ──

    def alloc_node(self, int parent, int depth):
        """分配新节点，返回索引。调用前须确保容量足够。"""
        cdef int idx = self.next_idx
        self.next_idx += 1
        cdef char* node = _node_ptr(self._base, self._stride, idx)
        _set_i32(node, _OFF_LEFT, -1)
        _set_i32(node, _OFF_RIGHT, -1)
        _set_i32(node, _OFF_PARENT, parent)
        _set_i32(node, _OFF_DEPTH, depth)
        _set_f64(node, _OFF_SPLIT, 0.0)
        _set_u8(node, _OFF_HAS_AABB, 0)
        _set_u8(node, _OFF_DIRTY, 1)
        return idx

    # ── 字段读写 (Python 可调用) ──

    def get_left(self, int idx):
        return _get_i32(_node_ptr(self._base, self._stride, idx), _OFF_LEFT)

    def set_left(self, int idx, int val):
        _set_i32(_node_ptr(self._base, self._stride, idx), _OFF_LEFT, val)
        _set_u8(_node_ptr(self._base, self._stride, idx), _OFF_DIRTY, 1)

    def get_right(self, int idx):
        return _get_i32(_node_ptr(self._base, self._stride, idx), _OFF_RIGHT)

    def set_right(self, int idx, int val):
        _set_i32(_node_ptr(self._base, self._stride, idx), _OFF_RIGHT, val)
        _set_u8(_node_ptr(self._base, self._stride, idx), _OFF_DIRTY, 1)

    def get_parent(self, int idx):
        return _get_i32(_node_ptr(self._base, self._stride, idx), _OFF_PARENT)

    def get_depth(self, int idx):
        return _get_i32(_node_ptr(self._base, self._stride, idx), _OFF_DEPTH)

    def get_split_val(self, int idx):
        return _get_f64(_node_ptr(self._base, self._stride, idx), _OFF_SPLIT)

    def set_split_val(self, int idx, double val):
        _set_f64(_node_ptr(self._base, self._stride, idx), _OFF_SPLIT, val)
        _set_u8(_node_ptr(self._base, self._stride, idx), _OFF_DIRTY, 1)

    def get_has_aabb(self, int idx):
        return _get_u8(_node_ptr(self._base, self._stride, idx), _OFF_HAS_AABB)

    def set_has_aabb(self, int idx, unsigned char val):
        _set_u8(_node_ptr(self._base, self._stride, idx), _OFF_HAS_AABB, val)

    def set_parent(self, int idx, int val):
        _set_i32(_node_ptr(self._base, self._stride, idx), _OFF_PARENT, val)

    def set_depth(self, int idx, int val):
        _set_i32(_node_ptr(self._base, self._stride, idx), _OFF_DEPTH, val)

    def get_dirty(self, int idx):
        return _get_u8(_node_ptr(self._base, self._stride, idx), _OFF_DIRTY)

    def set_dirty(self, int idx, unsigned char val):
        _set_u8(_node_ptr(self._base, self._stride, idx), _OFF_DIRTY, val)

    def get_aabb(self, int idx):
        """返回节点 AABB 的 numpy view (n_links, 6), float32"""
        cdef char* node = _node_ptr(self._base, self._stride, idx)
        cdef float* p = _aabb_ptr(node)
        # 创建 numpy 数组引用底层数据 (零拷贝)
        cdef cnp.npy_intp dims[2]
        dims[0] = self._n_links
        dims[1] = 6
        return cnp.PyArray_SimpleNewFromData(
            2, dims, cnp.NPY_FLOAT32, <void*>p)

    def set_aabb(self, int idx, cnp.ndarray[cnp.float32_t, ndim=2] aabb):
        """设置节点 AABB 并标记 dirty"""
        cdef char* node = _node_ptr(self._base, self._stride, idx)
        cdef float* dst = _aabb_ptr(node)
        cdef float* src = <float*>cnp.PyArray_DATA(aabb)
        cdef int nbytes = self._aabb_floats * 4  # sizeof(float32)
        memcpy(dst, src, nbytes)
        _set_u8(node, _OFF_HAS_AABB, 1)
        _set_u8(node, _OFF_DIRTY, 1)

    # ── 占用管理 ──

    def is_occupied(self, int idx):
        return self._occupied[idx] != 0

    def get_subtree_occ(self, int idx):
        return self._subtree_occ[idx]

    def get_subtree_occ_vol(self, int idx):
        """返回子树占用体积分数 (相对 root 总体积)"""
        return self._subtree_occ_vol[idx]

    def get_forest_id(self, int idx):
        return self._forest_id[idx]

    # ── SAT 碰撞检测 (核心热路径) ──

    cdef bint _link_aabbs_collide_c(
        self, float* aabb, object obs_packed,
    ) noexcept:
        """检测单个节点 AABB 与预打包障碍物列表是否碰撞

        obs_packed: list of (link_idx, lo0, hi0, lo1, hi1, lo2, hi2)
        aabb: float* 指向 (n_links, 6) float32

        返回 True 表示碰撞。使用标量 SAT 逐轴判断 + 早退。
        """
        cdef int n_obs = len(obs_packed)
        cdef int i, link_idx
        cdef float lo0, hi0, lo1, hi1, lo2, hi2
        cdef float a_lo0, a_hi0, a_lo1, a_hi1, a_lo2, a_hi2
        cdef int off
        cdef tuple obs

        for i in range(n_obs):
            obs = <tuple>obs_packed[i]
            link_idx = <int>obs[0]
            lo0 = <float>obs[1]
            hi0 = <float>obs[2]
            lo1 = <float>obs[3]
            hi1 = <float>obs[4]
            lo2 = <float>obs[5]
            hi2 = <float>obs[6]

            # AABB 偏移: link_idx * 6
            off = link_idx * 6
            a_lo0 = aabb[off]
            a_hi0 = aabb[off + 3]
            a_lo1 = aabb[off + 1]
            a_hi1 = aabb[off + 4]
            a_lo2 = aabb[off + 2]
            a_hi2 = aabb[off + 5]

            # SAT: 任一轴分离 → 不碰撞此障碍物
            if a_hi0 < lo0 or a_lo0 > hi0:
                continue
            if a_hi1 < lo1 or a_lo1 > hi1:
                continue
            if a_hi2 < lo2 or a_lo2 > hi2:
                continue
            # 三轴均重叠 → 碰撞
            return True

        return False

    def link_aabbs_collide(self, int idx, object obs_packed):
        """Python 可调用的碰撞检测接口"""
        if obs_packed is None:
            return False
        cdef char* node = _node_ptr(self._base, self._stride, idx)
        cdef float* aabb = _aabb_ptr(node)
        return self._link_aabbs_collide_c(aabb, obs_packed)

    # ── AABB union (向上传播的内核) ──

    cdef void _union_aabb_c(
        self, float* dst, float* a, float* b,
    ) noexcept nogil:
        """dst = union(a, b), 原地写 dst"""
        cdef int total = self._aabb_floats
        cdef int half = total // 2  # n_links * 3
        cdef int i
        # min 维度 [0, 1, 2] per link → 取 min
        # max 维度 [3, 4, 5] per link → 取 max
        # 布局: [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z] per link
        for i in range(self._n_links):
            # min_x, min_y, min_z
            dst[i*6]   = a[i*6]   if a[i*6]   < b[i*6]   else b[i*6]
            dst[i*6+1] = a[i*6+1] if a[i*6+1] < b[i*6+1] else b[i*6+1]
            dst[i*6+2] = a[i*6+2] if a[i*6+2] < b[i*6+2] else b[i*6+2]
            # max_x, max_y, max_z
            dst[i*6+3] = a[i*6+3] if a[i*6+3] > b[i*6+3] else b[i*6+3]
            dst[i*6+4] = a[i*6+4] if a[i*6+4] > b[i*6+4] else b[i*6+4]
            dst[i*6+5] = a[i*6+5] if a[i*6+5] > b[i*6+5] else b[i*6+5]

    # ── 向上传播 ──

    cdef void _propagate_up_c(self, int start_idx) noexcept:
        """从 start_idx 向上传播 AABB union, 直到 root 或无变化"""
        cdef int idx = start_idx
        cdef char* node
        cdef char* left_node
        cdef char* right_node
        cdef int left_idx, right_idx
        cdef float* dst
        cdef float* a
        cdef float* b
        cdef float old_val, new_val
        cdef bint changed
        cdef int i

        while idx >= 0:
            node = _node_ptr(self._base, self._stride, idx)
            left_idx = _get_i32(node, _OFF_LEFT)
            right_idx = _get_i32(node, _OFF_RIGHT)

            if left_idx < 0 or right_idx < 0:
                break

            left_node = _node_ptr(self._base, self._stride, left_idx)
            right_node = _node_ptr(self._base, self._stride, right_idx)

            if _get_u8(left_node, _OFF_HAS_AABB) == 0:
                break
            if _get_u8(right_node, _OFF_HAS_AABB) == 0:
                break

            dst = _aabb_ptr(node)
            a = _aabb_ptr(left_node)
            b = _aabb_ptr(right_node)

            # 检查是否有变化 (early-stop)
            changed = False
            for i in range(self._aabb_floats):
                if i % 6 < 3:
                    # min 维度
                    new_val = a[i] if a[i] < b[i] else b[i]
                else:
                    # max 维度
                    new_val = a[i] if a[i] > b[i] else b[i]
                if dst[i] != new_val:
                    dst[i] = new_val
                    changed = True

            if changed:
                _set_u8(node, _OFF_HAS_AABB, 1)
                _set_u8(node, _OFF_DIRTY, 1)
            else:
                break  # 无变化，不再上传

            idx = _get_i32(node, _OFF_PARENT)

    def propagate_up(self, int start_idx):
        """Python 可调用的向上传播接口"""
        self._propagate_up_c(start_idx)

    # ── 标记/重置占用 ──

    cdef void _mark_occupied_c(self, int idx, int forest_box_id) noexcept:
        """标记节点为已占用，并向上传播 subtree_occ / subtree_occ_vol"""
        self._occupied[idx] = 1
        self._forest_id[idx] = forest_box_id
        self._subtree_occ[idx] += 1

        # 体积分数：depth d 的节点占 root 体积的 1/2^d
        cdef char* node = _node_ptr(self._base, self._stride, idx)
        cdef int depth = _get_i32(node, _OFF_DEPTH)
        cdef double vol = ldexp(1.0, -depth)   # 2^(-depth)
        self._subtree_occ_vol[idx] += vol

        cdef int pidx = _get_i32(node, _OFF_PARENT)
        while pidx >= 0:
            self._subtree_occ[pidx] += 1
            self._subtree_occ_vol[pidx] += vol
            node = _node_ptr(self._base, self._stride, pidx)
            pidx = _get_i32(node, _OFF_PARENT)

    def mark_occupied(self, int idx, int forest_box_id=-1):
        self._mark_occupied_c(idx, forest_box_id)

    cdef void _reset_occupation_c(self, int idx) noexcept:
        """重置节点占用状态"""
        self._occupied[idx] = 0
        self._forest_id[idx] = -1

    def reset_occupation(self, int idx):
        self._reset_occupation_c(idx)

    cdef set _collect_forest_ids_c(self, int idx):
        """收集子树中所有 forest_id"""
        cdef set ids = set()
        cdef list stack = [idx]
        cdef int i, left, right, fid
        cdef char* node

        while stack:
            i = stack.pop()
            fid = self._forest_id[i]
            if fid >= 0:
                ids.add(fid)
            node = _node_ptr(self._base, self._stride, i)
            left = _get_i32(node, _OFF_LEFT)
            right = _get_i32(node, _OFF_RIGHT)
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)
        return ids

    def collect_forest_ids(self, int idx):
        return self._collect_forest_ids_c(idx)

    cdef void _clear_subtree_occupation_c(self, int idx) noexcept:
        """清除子树的占用状态"""
        cdef list stack = [idx]
        cdef int i, left, right, pidx
        cdef char* node
        cdef double vol

        while stack:
            i = stack.pop()
            if self._occupied[i]:
                self._occupied[i] = 0
                self._forest_id[i] = -1
                # 计算该节点的体积分数
                node = _node_ptr(self._base, self._stride, i)
                vol = ldexp(1.0, -_get_i32(node, _OFF_DEPTH))
                # 向上减少 subtree_occ / subtree_occ_vol
                pidx = _get_i32(node, _OFF_PARENT)
                while pidx >= 0:
                    self._subtree_occ[pidx] -= 1
                    self._subtree_occ_vol[pidx] -= vol
                    node = _node_ptr(self._base, self._stride, pidx)
                    pidx = _get_i32(node, _OFF_PARENT)
                self._subtree_occ[i] = 0
                self._subtree_occ_vol[i] = 0.0
            node = _node_ptr(self._base, self._stride, i)
            left = _get_i32(node, _OFF_LEFT)
            right = _get_i32(node, _OFF_RIGHT)
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)

    def clear_subtree_occupation(self, int idx):
        self._clear_subtree_occupation_c(idx)

    def reset_all_occupation(self):
        """重置所有节点的占用状态"""
        cdef int i
        for i in range(self.next_idx):
            self._occupied[i] = 0
            self._subtree_occ[i] = 0
            self._subtree_occ_vol[i] = 0.0
            self._forest_id[i] = -1

    # ── 脏节点迭代 (增量保存) ──

    def iter_dirty(self):
        """生成器：产出所有 dirty 节点索引"""
        cdef int i
        cdef char* node
        for i in range(self.next_idx):
            node = _node_ptr(self._base, self._stride, i)
            if _get_u8(node, _OFF_DIRTY):
                yield i

    def clear_all_dirty(self):
        """清除所有节点的 dirty 标记"""
        cdef int i
        cdef char* node
        for i in range(self.next_idx):
            node = _node_ptr(self._base, self._stride, i)
            _set_u8(node, _OFF_DIRTY, 0)

    # ── 统计 ──

    def get_stats(self):
        """返回树统计信息 dict"""
        cdef int n_leaves = 0
        cdef int max_depth = 0
        cdef int total_depth = 0
        cdef list stack = [0]
        cdef int i, d, left
        cdef char* node

        while stack:
            i = stack.pop()
            node = _node_ptr(self._base, self._stride, i)
            left = _get_i32(node, _OFF_LEFT)
            if left < 0:
                n_leaves += 1
                d = _get_i32(node, _OFF_DEPTH)
                total_depth += d
                if d > max_depth:
                    max_depth = d
            else:
                stack.append(left)
                stack.append(_get_i32(node, _OFF_RIGHT))

        cdef float avg = 0.0
        if n_leaves > 0:
            avg = <float>total_depth / n_leaves

        return {
            'n_nodes': self.next_idx,
            'n_leaves': n_leaves,
            'max_depth': max_depth,
            'avg_depth': avg,
        }

    # ── 序列化/反序列化辅助 ──

    def get_raw_buffer(self):
        """返回底层缓冲区的 numpy view (uint8), 用于文件写入"""
        return np.asarray(self._buf)

    def get_node_bytes(self, int idx):
        """返回单个节点的字节切片 (用于增量写入)"""
        cdef int off = idx * self._stride
        return bytes(self._buf[off:off + self._stride])

    @property
    def capacity(self):
        return self._cap

    @property
    def stride(self):
        return self._stride

    @property
    def n_links(self):
        return self._n_links

    @property
    def n_dims(self):
        return self._n_dims
