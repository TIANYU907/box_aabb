"""
planner/_hier_layout.py - 层级 AABB 缓存树节点布局与文件格式 v2

定义固定 stride 节点记录的字段偏移、文件头格式及辅助函数。
供 _hier_core.pyx (Cython) 与 hier_aabb_tree.py (Python 入口) 共享。

节点记录（AoS, 固定 stride, 单 AABB）:
    offset  type         field       bytes   描述
    ──────  ─────────    ─────       ─────   ──────
    0       int32        left        4       左子索引 (-1 = leaf)
    4       int32        right       4       右子索引 (-1 = leaf)
    8       int32        parent      4       父节点索引 (-1 = root)
    12      int32        depth       4       树深度
    16      float64      split_val   8       切分值
    24      uint8        has_aabb    1       是否已计算 AABB
    25      uint8        dirty       1       脏标记 (需增量写回)
    26      uint8[2]     _reserved   2       保留
    28      float32[NL*6] aabb       NL*24   连杆 AABB (min3 max3 per link)
    28+NL*24 ...         _padding    ...     填充到 stride

stride = ceil((28 + n_links * 24) / 64) * 64  (对齐缓存行)

文件格式 HCACHE02:
    [4096B header][node0][node1]...[node_{n_alloc-1}]

    header 结构 (前 80B 固定):
        0       bytes[8]    magic       'HCACHE02'
        8       uint32      version     2
        12      int64       n_nodes     有效节点数
        20      int64       n_alloc     已分配容量
        28      int32       n_dims      关节维度
        32      int32       n_links     连杆数
        36      int64       n_fk_calls  FK 调用计数
        44      int32       stride      节点步长(字节)
        48      bytes[32]   fp_sha256   机器人指纹 SHA256 前 32 字节
        80      float64[nd*2] joint_limits  关节限制 (最多 16 维 = 256B)
        336     ...         _reserved   填充到 4096B
"""

from __future__ import annotations

import hashlib
import struct
from typing import List, Tuple

import numpy as np

# ─────────────────────────────────────────────
#  节点字段偏移 (bytes)
# ─────────────────────────────────────────────

OFF_LEFT = 0        # int32
OFF_RIGHT = 4       # int32
OFF_PARENT = 8      # int32
OFF_DEPTH = 12      # int32
OFF_SPLIT_VAL = 16  # float64
OFF_HAS_AABB = 24   # uint8
OFF_DIRTY = 25      # uint8
OFF_RESERVED = 26   # uint8[2]
OFF_AABB = 28       # float32[n_links * 6]
TOPO_SIZE = 28      # 拓扑字段总长 (不含 AABB)

# ─────────────────────────────────────────────
#  Stride 计算
# ─────────────────────────────────────────────

def compute_stride(n_links: int) -> int:
    """计算每节点步长（字节），对齐到 64B 缓存行"""
    raw = TOPO_SIZE + n_links * 6 * 4  # float32 = 4B
    return ((raw + 63) // 64) * 64


# ─────────────────────────────────────────────
#  numpy structured dtype
# ─────────────────────────────────────────────

def make_node_dtype(n_links: int) -> np.dtype:
    """创建与文件布局一致的 numpy structured dtype

    用于 np.memmap / np.ndarray 直接映射节点数组。
    """
    stride = compute_stride(n_links)
    aabb_bytes = n_links * 6 * 4
    pad_bytes = stride - TOPO_SIZE - aabb_bytes
    fields = [
        ('left', '<i4'),
        ('right', '<i4'),
        ('parent', '<i4'),
        ('depth', '<i4'),
        ('split_val', '<f8'),
        ('has_aabb', 'u1'),
        ('dirty', 'u1'),
        ('_reserved', 'u1', (2,)),
        ('aabb', '<f4', (n_links, 6)),
    ]
    if pad_bytes > 0:
        fields.append(('_pad', 'u1', (pad_bytes,)))
    dt = np.dtype(fields)
    assert dt.itemsize == stride, \
        f"dtype itemsize {dt.itemsize} != stride {stride}"
    return dt


# ─────────────────────────────────────────────
#  文件头格式
# ─────────────────────────────────────────────

HCACHE_MAGIC = b'HCACHE02'
HCACHE_VERSION = 2
HEADER_SIZE = 4096  # 页对齐
MAX_DIMS = 16       # 最多支持 16 关节维度

# 固定头 80B:
#   magic(8) + version(4) + n_nodes(8) + n_alloc(8)
#   + n_dims(4) + n_links(4) + n_fk_calls(8) + stride(4)
#   + fp_sha256(32)
HEADER_FIXED_FMT = '<8s I q q I I q I 32s'
HEADER_FIXED_SIZE = struct.calcsize(HEADER_FIXED_FMT)  # 80

# joint_limits 区域：offset 80, 最大 256B (16 dims × 2 × float64)
HEADER_JL_OFFSET = HEADER_FIXED_SIZE
HEADER_JL_MAX_SIZE = MAX_DIMS * 2 * 8  # 256B

assert HEADER_FIXED_SIZE == 80


def fingerprint_sha256(fp_str: str) -> bytes:
    """将 robot.fingerprint() 字符串哈希为 32 字节 SHA256"""
    return hashlib.sha256(fp_str.encode('utf-8')).digest()


def write_header(
    f,
    *,
    n_nodes: int,
    n_alloc: int,
    n_dims: int,
    n_links: int,
    n_fk_calls: int,
    stride: int,
    fp_str: str,
    joint_limits: List[Tuple[float, float]],
) -> None:
    """写入 4096B 文件头"""
    fp_hash = fingerprint_sha256(fp_str)
    hdr = struct.pack(
        HEADER_FIXED_FMT,
        HCACHE_MAGIC, HCACHE_VERSION,
        n_nodes, n_alloc, n_dims, n_links,
        n_fk_calls, stride, fp_hash,
    )
    # 初始化 4096B 全零
    buf = bytearray(HEADER_SIZE)
    buf[:len(hdr)] = hdr
    # joint_limits
    jl_arr = np.array(joint_limits, dtype=np.float64)  # (n_dims, 2)
    jl_bytes = jl_arr.tobytes()
    buf[HEADER_JL_OFFSET:HEADER_JL_OFFSET + len(jl_bytes)] = jl_bytes
    f.write(bytes(buf))


def read_header(f) -> dict:
    """读取文件头，返回 dict"""
    buf = f.read(HEADER_SIZE)
    if len(buf) < HEADER_SIZE:
        raise ValueError("文件过短，无法读取 HCACHE02 头")

    (magic, version, n_nodes, n_alloc, n_dims, n_links,
     n_fk_calls, stride, fp_hash) = struct.unpack(
        HEADER_FIXED_FMT, buf[:HEADER_FIXED_SIZE])

    if magic != HCACHE_MAGIC:
        raise ValueError(f"无效的 magic: {magic!r}, 期望 {HCACHE_MAGIC!r}")
    if version != HCACHE_VERSION:
        raise ValueError(f"不支持的版本: {version}, 期望 {HCACHE_VERSION}")

    jl_count = n_dims * 2
    jl_bytes = buf[HEADER_JL_OFFSET:HEADER_JL_OFFSET + jl_count * 8]
    jl_arr = np.frombuffer(jl_bytes, dtype=np.float64).reshape(n_dims, 2)
    joint_limits = [(float(jl_arr[i, 0]), float(jl_arr[i, 1]))
                    for i in range(n_dims)]

    return {
        'n_nodes': n_nodes,
        'n_alloc': n_alloc,
        'n_dims': n_dims,
        'n_links': n_links,
        'n_fk_calls': n_fk_calls,
        'stride': stride,
        'fp_hash': fp_hash,
        'joint_limits': joint_limits,
    }


def update_header_field(mm, field: str, value) -> None:
    """原地更新 mmap 中的头部字段 (无需重写整个头)"""
    offsets = {
        'n_nodes': (12, '<q'),
        'n_alloc': (20, '<q'),
        'n_fk_calls': (36, '<q'),
    }
    if field not in offsets:
        raise ValueError(f"不支持的头部字段: {field}")
    off, fmt = offsets[field]
    struct.pack_into(fmt, mm, off, value)
