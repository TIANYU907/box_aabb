# Forest Algorithm Details (v2)

## Abstract

Forest 层将连续 C-space 近似为"无重叠 box 图"，并为规划层提供可复用的离散拓扑骨架。本文档聚焦实现级细节：`box_forest.py`、`deoverlap.py`、`collision.py`、`hier_aabb_tree.py`、`connectivity.py`、`coarsen.py` 及相关模型。

---

## 1. Formalization and Invariants

记自由空间近似集合为

$$
\mathcal{F}=\{B_i\}_{i=1}^{N},\quad
B_i=\prod_{d=1}^{D}[l_{i,d},u_{i,d}].
$$

Forest 维护两个核心对象：

1. 几何集合：`boxes: id -> BoxNode`
2. 拓扑集合：`adjacency: id -> Set[id]`

关键不变量：

- **(I1) 非重叠优先**：除容差尺度外，任意两 box 不应有正体积重叠
- **(I2) 邻接一致**：若 `j ∈ adjacency[i]`，则 `i ∈ adjacency[j]`
- **(I3) 缓存一致**：interval cache 与 `boxes` 同步更新

---

## 2. BoxForest Core (`box_forest.py`)

### 2.1 Data layout

- `boxes`: 稀疏字典，主数据源
- `adjacency`: 无向图邻接表
- `_intervals_arr (N,D,2)`: 向量化邻接与包含判断缓存
- `_interval_id_to_index`: O(1) id→行索引
- `_kdtree`: 可选中心点近邻索引
- `period`: 可选周期参数（支持周期关节空间）

### 2.2 Update modes

#### Mode A: direct append (`add_box_direct`)

在"调用方保证无重叠"的假设下（HierAABBTree 占用跟踪保证）：

- 仅用缓存判定新 box 邻居
- 增量更新边
- 追加缓存行

该模式是在线扩展阶段主路径，复杂度接近 $O(ND)$ 而非全量 $O(N^2D)$。

#### Mode B: batch without adjacency (`add_box_no_adjacency` / `remove_boxes_no_adjacency`)

专为 coarsen 阶段设计的批量操作模式：

- `add_box_no_adjacency(box)`: 添加 box 到字典和缓存，但跳过邻接图更新
- `remove_boxes_no_adjacency(box_ids)`: 移除 box 并清理缓存，但跳过邻接图更新
- `rebuild_adjacency()`: 在所有批量操作完成后，一次性重建全量邻接图

该模式避免了合并过程中每次增删都重新计算邻接的开销（C1 优化）。

### 2.3 Removal semantics (`remove_boxes`)

- 删除节点并剔除双向边
- 从 interval cache 删除对应行并重映射 index
- 标记 KDTree dirty，延迟重建

### 2.4 Partition merge (`merge_partition_forests`)

接收并行扩展的局部结果列表，按 partition_id 排序后合并到全局 forest。

### 2.5 Boundary dedup (`dedup_boundary_boxes`)

对切分面附近重复 box 按分区优先级做一致去重，保留唯一 owner。

### 2.6 Invariant validation (`validate_invariants`)

`validate_invariants(strict=True)` 执行强校验：

- 无正体积重叠
- 邻接对称性
- 引用有效性

`strict=True` 时任何违规直接抛错。

### 2.7 Persistence

`save(filepath)` / `load(filepath, robot)` 使用 pickle 序列化，含 `robot_fingerprint` 校验。

---

## 3. Adjacency (`deoverlap.py`)

### 3.1 Vectorized adjacency

令重叠宽度矩阵

$$
w_{ij}^{(d)}=\min(u_{i,d},u_{j,d})-\max(l_{i,d},l_{j,d}).
$$

实现中构造三类布尔掩码：

- `separated`: $w<-tol$
- `touching`: $-tol\le w\le tol$
- `overlapping`: $w>tol$

邻接条件由掩码计数组合给出，之后只取上三角避免重复，再镜像到全图。

### 3.2 Chunked upper-triangle strategy

当 N 超过 `chunk_threshold`（默认 300）时，按块 `(chunk_size × chunk_size × D)` 计算上三角，降低峰值内存并减少重复比较。

### 3.3 Incremental adjacency

`compute_adjacency_incremental(new_boxes, existing_boxes)` — 仅计算新 box 与已有 box 之间的邻接关系，复杂度 $O(K \cdot N \cdot D)$。

### 3.4 Shared face utilities

- `shared_face(box_a, box_b, tol)` — 提取两 box 共享面的区间描述
- `shared_face_center(box_a, box_b, tol)` — 共享面中心点，用于 waypoint 生成

---

## 4. Collision Layer (`collision.py`)

### 4.1 Single-config test

`check_config_collision(q)`：

1. FK 得到全部关节点
2. 每段连杆生成 AABB
3. 与障碍 AABB 做分离轴测试

### 4.2 Interval-box test

`check_box_collision(intervals)`：

- 调用 AABB 层区间 FK 获取 link 包络
- 任一 link 包络与障碍相交即返回 True

其语义是"保守可行性否证器"：False 可判定安全，True 表示可能碰撞。

### 4.3 Segment test

`check_segment_collision(q_a, q_b, resolution)` 沿关节直线离散采样，逐点调用 single-config test。

### 4.4 Batch collision test

`check_config_collision_batch(configs)`：

- 批量构造 N 个变换矩阵链
- 对每段连杆向量化求 AABB
- 按障碍批量分离轴测试

该接口是 `BoxPlanner._sample_seed` 与高密度扫描场景的关键加速器。

### 4.5 Spatial index gating

障碍数超过 `spatial_index_threshold`（默认 20）时自动构建 `SpatialIndex`（空间哈希），查询时先筛候选障碍集合，再做精确 overlap 测试。通过 obstacle signature 自动失效重建。

`SpatialIndex` 参数：`cell_size`（默认 0.5）。

---

## 5. Hierarchical Box Generator (`hier_aabb_tree.py`)

### 5.1 Node representation and storage

使用 `NodeStore`（SoA + 固定 stride）替代 Python 对象树：

- 拓扑字段：`left/right/parent/depth/split_val`
- 几何字段：单 AABB `(n_links,6)`
- 状态字段：`occupied/subtree_occ/forest_id`
- `forest_ids_array()` — Cython 一次性返回所有节点 forest_id 的 numpy 数组（C3 优化，用于 coarsen 阶段快速构建 box→树节点映射）

支持 `HCACHE02` 二进制持久化与 `mmap r+` 增量写回。

### 5.2 Active split dims

`active_split_dims` 参数允许指定参与切分的维度列表：

- 默认使用全维轮转：`split_dim = depth % n_dims`
- 指定后改为：`split_dim = active_split_dims[depth % len(active_split_dims)]`
- 为空时回退全维轮转

这使得 Panda 第 7 轴等弱影响维度可被跳过，提高树深利用率。

### 5.3 Top-down free-box search

`find_free_box(seed, obstacles, ...)` 两阶段：

1. **下行**：沿 seed 路径惰性分裂，遇到"当前节点无碰撞且子树无占用"即停
2. **上行**：传播子节点 union 精化，并尝试 promotion 吸收可提升子树

输出 `FindFreeBoxResult(intervals, absorbed_box_ids)`。

支持 `constrained_intervals` 约束参数：下行/上行结果不得越过该约束（并行分区扩展时使用）。

### 5.4 Incremental FK at split

`_split` 先取父 FK 缓存，分别对左右子节点调用 `compute_fk_incremental`（或 `_split_fk_pair`）。相较全量 FK，显著降低切分开销。

### 5.5 Occupancy and absorbed-id semantics

- `mark_occupied(result_idx, forest_box_id)`：把树节点绑定到 forest 节点
- promotion 发生时收集被吸收节点 id，返回上层执行 `forest.remove_boxes(absorbed_ids)`

---

## 6. Connectivity (`connectivity.py`)

### 6.1 UnionFind

支持动态操作：

- `__init__(keys)` / `find(x)` / `union(x, y)` / `components()` / `n_components()` / `same(x, y)`
- `add(key)` — 动态添加新键
- `remove_keys(keys)` — 批量移除键

### 6.2 `find_islands(boxes, period=None)`

识别 box 集合中的所有连通分量。支持周期关节空间（`period` 参数指定各维周期）。

### 6.3 `bridge_islands(boxes, collision_checker, ...)`

连接孤立连通分量的核心函数。

**关键参数**：
- `segment_resolution`：线段碰撞检测分辨率
- `max_pairs_per_island_pair`：每对岛之间最多尝试的桥接对数
- `max_rounds`：最大桥接轮次
- `period`：周期关节空间支持
- `hier_tree` / `obstacles` / `forest`：可选，启用 FFB 桥接扩展
- `min_box_size`：最小 box 尺寸
- `n_bridge_seeds`：每次桥接尝试的种子数
- `min_island_size`：小于此数的岛直接丢弃
- `precomputed_uf` / `precomputed_islands`：预计算的 UnionFind 和岛列表（避免重复计算）
- `target_pair`：定向桥接（仅连接指定的两个分量）

**桥接策略**：

1. 使用 `find_islands` 识别所有连通分量
2. 用 `_try_expand_bridge_box` 在岛间扩展桥接 box
3. 采用 dry-run FFB (`mark_occupied=False`) + overlap 检查 + 正式 FFB (`mark_occupied=True`) 的三步策略
4. 向量化桥接-box 邻接更新（B3 优化，使用 `_adjacent_existing_ids_from_cache`）

**批量线段碰撞**：`_check_segments_batch` 向量化批量线段碰撞检测。

**周期支持**：`_overlap_periodic_1d`、`_wrapped_dist_1d`、`_wrapped_center_dist`、`_nearest_point_wrapped`、`_check_segment_wrapped`。

> **设计注意**：早期尝试的 B5 优化（通过 `clear_subtree_occupation` 回滚）已回滚，因为会清除其他 box 的节点。

---

## 7. Coarsen (`coarsen.py`)

### 7.1 算法概述

通过维度扫描检测可合并的相邻 box 并批量执行合并，减少图节点数量。

**合并条件**：两个 box 在某一维度上相邻接触，且在其余所有维度上边界完全一致（profile key 相同）。

### 7.2 实现结构

```text
coarsen_forest(store, forest, max_rounds=20, tol=1e-10):
    for round in range(max_rounds):
        box_to_nodes = _build_box_to_nodes(store, n_nodes)  # C3: bulk forest_ids_array
        merges = []
        for dim in range(n_dims):
            merges += _sweep_merge_dim(store, forest, dim, ...)  # C2: vectorized grouping
        if no merges: break
        _execute_merge_batch(forest, store, merges, ...)   # C5: batch merge
    forest.rebuild_adjacency()                              # C1: rebuild once at end
```

### 7.3 关键实现细节

1. **`_build_box_to_nodes`**：使用 `store.forest_ids_array()` 返回全部 `(n_nodes,)` int32 数组，再用 `np.flatnonzero` 按 box_id 分组

2. **`_sweep_merge_dim`**：
   - 按 profile key（其余维度边界）分组，使用 `np.unique` 对结构化数组分组
   - 组内按活动维度 lo 值排序，扫描连续接触的运行
   - 使用 `.copy()` 复制数组视图，防止 swap-on-delete 导致的 stale view
   - 两阶段架构：检测阶段只读，执行阶段写入

3. **`_execute_merge_batch`**：
   - 对每个合并运行：移除旧 box (`remove_boxes_no_adjacency`) + 创建新 box (`add_box_no_adjacency`)
   - 更新树节点的 forest_id 指向新 box
   - 验证新 box 无碰撞

### 7.4 优化汇总

| 编号 | 优化 | 描述 |
|---|---|---|
| C1 | 跳过邻接更新 | 合并期间使用 `*_no_adjacency` 操作，最后一次 `rebuild_adjacency` |
| C2 | 向量化分组 | NumPy 结构化数组 `np.unique` 替代 Python 循环 |
| C3 | 批量 forest ID | Cython `NodeStore.forest_ids_array()` 一次性获取映射 |
| C4 | 去除线程池 | 移除 ThreadPoolExecutor（GIL 开销大于收益） |
| C5 | 批量合并 | 同组 k 个 box 一次性合并为 1 个 |

### 7.5 复杂度

- 分组：$O(N \cdot D \cdot \log N)$
- 扫描：$O(N \cdot D)$
- 重建邻接：$O(N^2 D)$（仅执行一次）

### 7.6 性能数据

典型 Panda 7-DOF 场景：500 boxes → 486 boxes，12 merges/round，耗时 ~37ms。

---

## 8. Parallel Collision (`parallel_collision.py`)

### 8.1 ParallelCollisionChecker

多进程批量碰撞检测：

- `batch_check_configs(configs) -> np.ndarray[bool]`
- `batch_check_boxes(boxes) -> List[bool]`
- `batch_check_segments(segments) -> List[bool]`
- `filter_collision_free(configs) -> np.ndarray`

### 8.2 SpatialIndex

空间哈希索引：

- `build(obstacles)` — 构建网格索引
- `query(min_point, max_point) -> List[int]` — 查询候选障碍

---

## 9. Persistence and Cross-Scenario Reuse

Forest 与机器人运动学绑定，不与场景绑定。复用流程：

1. 加载历史 forest/hcache
2. 在新场景运行 `validate_boxes` 或查询时惰性碰撞验证
3. 剔除失效节点，保留可复用拓扑骨架

---

## 10. Partition Merge Pipeline

### 10.1 局部结果输入规范

每个 worker 返回：

```text
{
    partition_id: int,
    boxes: [{joint_intervals, seed_config, volume}, ...]
}
```

主进程按 `partition_id` 排序后合并。

### 10.2 合并阶段关键步骤

1. `merge_partition_forests`：写入全局 `boxes/adjacency`
2. `dedup_boundary_boxes`：按分区优先级移除边界重复盒
3. `validate_invariants(strict=True)`：执行强校验

三步全部通过后，结果才可进入 Planner 构图阶段。

### 10.3 重叠根因与修复策略

并行模式取消全空间起终点预扩展，起终点仅在对应分区 worker 内扩展。合并后直接强校验，将重叠问题从"后验容忍"收敛为"前置规避 + 后验阻断"。

---

## 11. Complexity Profile

| 操作 | 复杂度 |
|---|---|
| 邻接（全量） | $O(N^2D)$ |
| 邻接（增量） | $O(ND)$ |
| 碰撞（单段） | $O(L \cdot M)$ |
| 碰撞（批量） | $O(N \cdot L \cdot M)$ 向量化 |
| 树切分 | $O(depth \cdot C_{FK})$ |
| Coarsen | $O(N \cdot D \cdot \log N) + O(N^2D)$ 一次 |

---

## 12. Failure Modes and Safeguards

1. **数值接触误差**：通过 `tol` 判定 touching/overlap，避免边界抖动
2. **缓存错位风险**：所有增删节点都同步维护 `_interval_id_to_index`
3. **场景漂移失效**：障碍 signature 触发 spatial index 重建
4. **过深树退化**：`max_depth/min_edge_length` 双条件停止分裂
5. **stale numpy view**：coarsen 中使用 `.copy()` 防止 swap-on-delete 导致的视图失效

---

## 13. End-to-End Pseudocode

```text
# find_free_box
Input: seed q, obstacle set O, constrained_intervals (optional)
while node collides or occupied:
    if depth/edge threshold violated: return None
    lazily split node (incremental FK for children)
    if constrained_intervals: clip to constraint
    descend to child containing q
propagate union-aabb upward
promote upward while parent remains collision-free
mark occupied and return resulting intervals (+ absorbed ids)

# coarsen_forest
Input: store, forest, max_rounds
for round in max_rounds:
    build box→nodes mapping via forest_ids_array
    for each dim: sweep and collect mergeable runs
    execute batch merges (remove old, add new, update tree nodes)
rebuild_adjacency once

# bridge_islands
Input: boxes, collision_checker, hier_tree, forest
find connected components via find_islands
for each island pair:
    try _try_expand_bridge_box (dry-run FFB + overlap check + formal FFB)
    if bridge box found: update adjacency
return bridge edges, final islands, bridge boxes, discarded islands
```

---

## 14. Practical Tuning Guidelines

1. 若障碍数量多，优先调低 `spatial_cell_size` 并保持索引阈值开启
2. 若 box 数增长快，优先使用 `add_box_direct` 增量路径
3. 若 hcache 已稳定，使用 `save_incremental` 降低 I/O
4. 对高维机械臂，先限制 `max_box_nodes` 再调碰撞分辨率
5. Coarsen 通常在 1-2 轮后收敛，`max_rounds=20` 足够
6. Bridge 时 `min_island_size` 可用于丢弃过小的噪声岛
