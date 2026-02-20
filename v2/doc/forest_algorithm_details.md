# Forest Algorithm Details (v2)

## Abstract

Forest 层将连续 C-space 近似为“无重叠 box 图”，并为规划层提供可复用的离散拓扑骨架。本文档聚焦实现级细节：`box_forest.py`、`deoverlap.py`（邻接检测）、`collision.py`、`hier_aabb_tree.py` 及相关模型。

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

- (I1) 非重叠优先：除容差尺度外，任意两 box 不应有正体积重叠。
- (I2) 邻接一致：若 `j ∈ adjacency[i]`，则 `i ∈ adjacency[j]`。
- (I3) 缓存一致：interval cache 与 `boxes` 同步更新。

---

## 2. BoxForest Core (`box_forest.py`)

### 2.1 Data layout

- `boxes`: 稀疏字典，主数据源。
- `adjacency`: 无向图邻接表。
- `_intervals_arr (N,D,2)`: 向量化邻接与包含判断缓存。
- `_interval_id_to_index`: O(1) id->行索引。
- `_kdtree`: 可选中心点近邻索引。

### 2.2 Update modes

#### Mode A: direct append (`add_box_direct`)

在“调用方保证无重叠”的假设下（HierAABBTree 占用跟踪保证）：

- 仅用缓存判定新 box 邻居；
- 增量更新边；
- 追加缓存行。

该模式是在线扩展阶段主路径，复杂度接近 $O(ND)$ 而非全量 $O(N^2D)$。

#### Mode B: batch without adjacency (`add_box_no_adjacency` / `remove_boxes_no_adjacency`)

专为 coarsen 阶段设计的批量操作模式：

- `add_box_no_adjacency(box)`: 添加 box 到字典和缓存，但跳过邻接图更新。
- `remove_boxes_no_adjacency(box_ids)`: 移除 box 并清理缓存，但跳过邻接图更新。
- `rebuild_adjacency()`: 在所有批量操作完成后，一次性重建全量邻接图。

该模式避免了合并过程中每次增删都重新计算邻接的开销（C1 优化）。

### 2.3 Removal semantics (`remove_boxes`)

- 删除节点并剔除双向边。
- 从 interval cache 删除对应行并重映射 index。
- 标记 KDTree dirty，延迟重建。

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

### 3.4 Chunked upper-triangle strategy

当 N 超过阈值时，按块 `(Bi × Bj × D)` 计算上三角，降低峰值内存并减少重复比较，适合千级 box 场景。

---

## 4. Collision Layer (`collision.py`)

### 4.1 Single-config test

`check_config_collision(q)`：

1. FK 得到全部关节点。
2. 每段连杆生成 AABB。
3. 与障碍 AABB 做分离轴测试。

### 4.2 Interval-box test

`check_box_collision(intervals)`：

- 调用 AABB 层区间 FK 获取 link 包络。
- 任一 link 包络与障碍相交即返回 True。

其语义是“保守可行性否证器”：False 可判定安全，True 表示可能碰撞。

### 4.3 Segment test

`check_segment_collision(q0,q1,res)` 沿关节直线离散采样，逐点调用 single-config test。分辨率越小，安全性越高但耗时增加。

### 4.4 Batch collision test

`check_config_collision_batch(configs)`：

- 批量构造 N 个变换矩阵链；
- 对每段连杆向量化求 AABB；
- 按障碍批量分离轴测试。

该接口是 `BoxPlanner._sample_seed` 与高密度扫描场景的关键加速器。

### 4.5 Spatial index gating

障碍数超过阈值时构建 `SpatialIndex`，通过 obstacle signature 自动失效重建。查询时先筛候选障碍集合，再做精确 overlap 测试。

---

## 5. Hierarchical Box Generator (`hier_aabb_tree.py`)

### 5.1 Node representation and storage

v6 使用 `NodeStore`（SoA + 固定 stride）替代 Python 对象树：

- 拓扑字段：`left/right/parent/depth/split_val`
- 几何字段：单 AABB `(n_links,6)`
- 状态字段：`occupied/subtree_occ/forest_id`
- 新增方法：`forest_ids_array()` —— 通过 Cython 一次性返回所有节点的 forest_id 的 numpy 数组（C3 优化，用于 coarsen 阶段快速构建 box→树节点映射）

并支持 `HCACHE02` 二进制持久化与 `mmap r+` 增量写回。

### 5.2 Top-down free-box search

`find_free_box(seed, obstacles, ...)` 两阶段：

1. 下行：沿 seed 路径惰性分裂，遇到“当前节点无碰撞且子树无占用”即停。
2. 上行：传播子节点 union 精化，并尝试 promotion 吸收可提升子树。

输出 `FindFreeBoxResult(intervals, absorbed_box_ids)`，供 planner 将旧 box 替换为更大安全盒。

### 5.3 Incremental FK at split

`_split` 先取父 FK 缓存，分别对左右子节点调用 `compute_fk_incremental`。相较全量 FK，显著降低切分开销。

### 5.4 Occupancy and absorbed-id semantics

- `mark_occupied(result_idx, forest_box_id)` 把树节点绑定到 forest 节点。
- promotion 发生时收集被吸收节点 id，返回上层执行 `forest.remove_boxes(absorbed_ids)`。

---

## 6. Persistence and Cross-Scenario Reuse

Forest 与机器人运动学绑定，不与场景绑定。复用流程：

1. 加载历史 forest/hcache。
2. 在新场景运行 `validate_boxes` 或查询时惰性碰撞验证。
3. 剔除失效节点，保留可复用拓扑骨架。

这使“离线积累 + 在线适配”成为可能。

---

## 7. Complexity Profile

主要复杂度来源：

- 邻接：全量约 $O(N^2D)$，增量约 $O(ND)$。
- 碰撞：单段近似 $O(L\cdot M)$（L 连杆数，M 候选障碍数）。
- 树切分：近似 $O(depth \cdot C_{FK})$，增量 FK 可降常数项。

在大规模场景中，瓶颈通常由“碰撞验证 + 邻接更新”共同决定。

---

## 8. Failure Modes and Safeguards

1. **数值接触误差**：通过 `tol` 判定 touching/overlap，避免边界抖动。
2. **缓存错位风险**：所有增删节点都同步维护 `_interval_id_to_index`。
3. **场景漂移失效**：障碍 signature 触发 spatial index 重建。
4. **过深树退化**：`max_depth/min_edge_length` 双条件停止分裂。

---

## 9. End-to-End Pseudocode

```text
Input: seed q, obstacle set O
while node collides or occupied:
	if depth/edge threshold violated: return None
	lazily split node (incremental FK for children)
	descend to child containing q
propagate union-aabb upward
promote upward while parent remains collision-free
mark occupied and return resulting intervals (+ absorbed ids)
```

---

## 10. Practical Tuning Guidelines

1. 若障碍数量多，优先调低 spatial cell size 并保持索引阈值开启。
2. 若 box 数增长快，优先使用 `add_box_direct` 增量路径。
3. 若 hcache 已稳定，使用 `save_incremental` 降低 I/O。
4. 对高维机械臂，先限制 `max_box_nodes` 再调碰撞分辨率，避免图构建爆炸。

---

## 11. Coarsen：维度扫描合并（`coarsen.py`）

### 11.1 算法概述

Coarsen 通过维度扫描检测可合并的相邻 box 并批量执行合并，以减少图节点数量。

**合并条件**：两个 box 在某一维度上相邻接触，且在其余所有维度上边界完全一致（profile key 相同）。

### 11.2 实现结构

```text
coarsen_forest(forest, store, collision_checker, max_rounds):
    for round in range(max_rounds):
        box_to_nodes = _build_box_to_nodes(store, n_nodes)  # C3: bulk forest_ids_array
        merges = []
        for dim in range(n_dims):
            merges += _sweep_merge_dim(forest, dim, ...)     # C2: vectorized grouping
        if no merges: break
        _execute_merge_batch(forest, store, merges, ...)     # C5: batch merge
    forest.rebuild_adjacency()                               # C1: rebuild once at end
```

### 11.3 关键实现细节

1. **`_build_box_to_nodes`**：使用 `store.forest_ids_array()` 返回全部 `(n_nodes,)` 的 int32 数组，再用 `np.flatnonzero` 按 box_id 分组。

2. **`_sweep_merge_dim`**：
   - 按 profile key（其余维度边界）分组，使用 `np.unique` 对结构化数组分组
   - 组内按活动维度 lo 值排序，扫描连续接触的运行
   - 使用 `.copy()` 复制数组视图，防止 swap-on-delete 导致的 stale view
   - 采用两阶段架构：检测阶段只读，执行阶段写入

3. **`_execute_merge_batch`**：
   - 对每个合并运行：移除旧 box (`remove_boxes_no_adjacency`) + 创建新 box (`add_box_no_adjacency`)
   - 更新树节点的 forest_id 指向新 box
   - 验证新 box 无碰撞

### 11.4 复杂度

- 分组：$O(N \cdot D \cdot \log N)$（各维度排序）
- 扫描：$O(N \cdot D)$
- 重建邻接：$O(N^2 D)$（仅执行一次）

### 11.5 性能数据

典型 Panda 7-DOF 场景：500 boxes → 486 boxes，12 merges/round，耗时 ~37ms。

---

## 12. 分区并行合并管线（当前实现细节）

### 11.1 局部结果输入规范

每个 worker 返回：

```text
{
	partition_id: int,
	boxes: [
		{joint_intervals, seed_config, volume},
		...
	]
}
```

主进程按 `partition_id` 排序后合并，保证去重规则的确定性。

### 11.2 合并阶段关键步骤

1. `merge_partition_forests`：写入全局 `boxes/adjacency`。
2. `dedup_boundary_boxes`：按分区优先级移除边界重复盒。
3. `validate_invariants(strict=True)`：执行强校验。

只有三步全部通过，结果才可进入 Planner 构图阶段。

### 11.3 重叠根因与修复策略

历史问题：并行模式下若先做全空间起终点扩展，再合并分区扩展结果，
可能在同一区域重复建盒，导致正体积重叠。

修复后策略：

1. 并行模式取消全空间预扩展；
2. 起终点仅在对应分区 worker 内扩展；
3. 合并后直接强校验，不再以 `strict=False` 容错。

该策略将重叠问题从“后验容忍”收敛为“前置规避 + 后验阻断”。

### 11.4 校验失败的最小排障流程

若 `validate_invariants` 抛错，建议按顺序检查：

1. 输入分区是否互斥（切分区间是否存在穿插）；
2. 局部结果是否重复提交（同 seed 多次返回）；
3. 去重容差是否过小导致"数值同盒未识别"。
