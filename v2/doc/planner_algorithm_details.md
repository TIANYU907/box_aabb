# Planner Algorithm Details (v2)

## Abstract

Planner 层实现一种 box-guided 采样规划器：先构建/复用无碰撞 box 森林，再在 box 图上搜索并优化路径。本文档以论文写法给出流程分解、函数级实现细节、复杂度和失败恢复机制。核心文件为 `box_planner.py`、`connector.py`、`gcs_optimizer.py`、`path_smoother.py`。

---

## 1. Problem Setup

给定起终点 $q_s,q_g\in\mathbb{R}^D$ 与障碍场景 $\mathcal{O}$，求连续路径

$$
\pi:[0,1]\to\mathcal{C}_{free},\quad \pi(0)=q_s,\;\pi(1)=q_g.
$$

v2 不直接在连续空间做全局优化，而是先构建 box 覆盖图 $G=(V,E)$，其中节点为无碰撞区间盒，边表示可行过渡，再在图上求离散骨架并还原为连续轨迹。

---

## 2. Main Planner Pipeline (`BoxPlanner.plan`)

### Stage 0: pre-check and normalization

- 将输入转为 `float64`。
- `check_config_collision(q_s/q_g)`，任一点碰撞立即失败。
- 记录时间戳和计数器，确保失败路径也可追踪。

### Stage 0.5: straight-line early exit

调用 `check_segment_collision(q_s,q_g,resolution)`：若无碰撞，直接返回两点路径。该分支能在“可直连”场景将复杂规划退化为 O(1) 级别。

### Stage 1: forest bootstrap

- 若配置 `forest_path`，尝试 `BoxForest.load`。
- 否则新建空 forest。
- 将 `HierAABBTree` 注入 forest，使后续扩展共享层级缓存。

### Stage 2: lazy validity filtering

`forest.validate_boxes(collision_checker)` 在当前场景筛除失效盒，得到 `valid_boxes` 与 `valid_adjacency`。此阶段实现“跨场景复用 + 在线纠偏”。

### Stage 3: seed expansion loop

1. 先对 `q_s`,`q_g` 尝试 `find_free_box`。
2. 进入迭代采样（受 `max_iterations`、`max_box_nodes` 约束）。
3. 每个 seed 经 `find_free_box(..., mark_occupied=True)` 得到新区间。
4. 过滤几何平均边长小于 `min_box_size` 的盒。
5. 若存在 `absorbed_box_ids`，同步从 forest 删除旧节点。

### Stage 4: graph materialization

- `connector.build_adjacency_edges(valid_boxes, valid_adjacency)` 生成 box-graph 边。
- `connect_endpoints_to_forest(q_s,q_g,valid_boxes)` 把端点接入图。

### Stage 5: graph search + repair

- 先运行 Dijkstra（`gcs_optimizer._dijkstra`）。
- 若失败，运行 `bridge_islands`（`connectivity.py`）：在可达/不可达分量间尝试桥接边后重试搜索。

> **注意**：在 Panda 7-DOF 管线（`panda_planner.py`）中，使用专用的 `bridge_islands` 函数（`connectivity.py`）而非 `_bridge_disconnected`。桥接阶段使用 loose-overlap 邻接图（与 coarsen 后重建的 strict 邻接图不同），以确保连通性。

### Stage 6: geometric path reconstruction

从离散 box 序列恢复几何路径：`gcs_optimizer.optimize_box_sequence` 在共享面上布置/优化 waypoint，得到初始可行折线。

### Stage 7: box-aware post-processing

- `shortcut_in_boxes`: 仅当线段采样点始终落在盒并集中才允许删点。
- `smooth_in_boxes`: 移动平均后投影到对应 box 内（`nearest_point_to`）。

### Stage 8: finalize

封装 `PlannerResult`：成功标志、路径、边、碰撞计数、耗时、消息；可选持久化 forest。

## 2.5 Panda 7-DOF 专用管线（`panda_planner.py`）

Panda 场景采用优化后的端到端管线，在基本 BoxPlanner 之上增加了 coarsen 和 bridge 阶段：

1. **grow**: 调用 `BoxPlanner.expand_forest` / `grow_forest` 扩展 box forest
2. **coarsen**: 调用 `coarsen_forest` 进行 dim-sweep 合并（详见§2.6）
3. **adjacency**: 构建 loose-overlap 邻接图（带重叠容差，区别于 Forest 层的 strict face-touching）
4. **bridge**: 调用 `bridge_islands` 连接孤立连通分量（详见§5.3）
5. **Dijkstra**: 图搜索找到最短路径
6. **waypoint**: 在共享面上布置路点，输出可行轨迹

### 两种邻接条件的区别

| 邻接类型 | 函数 | 语义 | 用途 |
|---|---|---|---|
| strict face-touching | `deoverlap.compute_adjacency` | 仅当两 box 在某维度精确接触且其余维度有重叠 | BoxForest 内部邻接图 |
| loose overlap | `_build_adjacency_and_islands` | 当两 box 在所有维度均有重叠(带容差) | bridge / Dijkstra 连通性 |

> strict 邻接对于 500 个 box 产生 ~5 条边，而 loose 邻接产生 ~157 条边。bridge/Dijkstra 需要后者以确保连通性。

## 2.6 Coarsen 阶段详解（`forest/coarsen.py`）

Coarsen 通过维度扫描合并相邻 box，减少图节点数量。当前实现采用了多项向量化优化：

**主要优化点**：
- **C1 跳过邻接更新**: 合并期间使用 `add_box_no_adjacency` / `remove_boxes_no_adjacency`，最后一次性 `rebuild_adjacency`
- **C2 向量化分组**: 使用 NumPy 结构化数组 `np.unique` 对 box 按 profile key 分组（替代 Python 循环）
- **C3 批量 forest ID**: 通过 Cython `NodeStore.forest_ids_array()` 一次性获取全部 forest ID 映射
- **C4 去除线程池**: 移除 ThreadPoolExecutor（GIL 开销大于收益）
- **C5 批量合并**: 将同一分组内的 k 个 box 一次性合并为 1 个

**两阶段架构**：
1. 检测阶段（只读）：扫描所有维度，收集可合并的 box 运行（run）
2. 执行阶段（写入）：批量执行所有合并，避免读写交叉导致的视图失效

**关键实现细节**：
- `_build_box_to_nodes`: 使用 `store.forest_ids_array()` + `np.flatnonzero` 构建 box→树节点映射
- `_sweep_merge_dim`: 对每个维度按 lo 值排序后扫描连续运行，使用 `.copy()` 防止 stale numpy view
- `_execute_merge_batch`: 批量移除旧 box + 创建新 box，并更新树节点 forest_id

**性能**：coarsen 耗时 ~37ms（500 boxes, 12 merges）

---

## 3. Seed Sampling Micro-Mechanism (`_sample_seed`)

候选生成：

$$
q\sim
\begin{cases}
\mathcal{N}(q_g,\sigma^2I) & \text{with prob. } p_{goal}\\
\mathcal{U}(l,u) & \text{otherwise}
\end{cases}
$$

实现细节：

- 一次生成固定批次（默认 20）候选。
- 优先调用 `check_config_collision_batch` 返回布尔向量。
- 取首个无碰撞候选，失败则本轮跳过。

该设计减少 Python 循环与函数调度开销，是采样阶段的关键吞吐优化。

---

## 4. Connection Layer (`connector.py`)

### 4.1 adjacency-edge construction

从 `adjacency` 直接建边，每对邻居通过 `shared_face_center` 生成过渡点；若共享面退化，回退到交集中心。

### 4.2 endpoint attachment

对 `start/goal`：

1. 若落在某 box 内，直接绑定 box id。
2. 否则连向最近 box 表面点并验碰撞。
3. 若失败，尝试次近候选集合。

### 4.3 legacy tree-to-tree fallback

模块仍保留树间重叠连接与最近对连接逻辑，主要用于兼容旧调用路径与局部实验。

---

## 5. Graph Search and Repair

### 5.1 shortest-path backbone

图节点为 `start/goal + box ids`，边权默认关节空间欧氏长度。Dijkstra 输出的是“节点序列骨架”，非最终连续轨迹。

### 5.2 disconnected-graph bridging (`_bridge_disconnected` / `bridge_islands`)

有两种桥接实现：

**经典路径** (`gcs_optimizer._bridge_disconnected`):

迭代步骤：

1. BFS 求 `start` 可达集合 $R$。
2. 在 $R$ 与 $\bar{R}$ 间按中心距离排序候选 box 对。
3. 取表面最近点连线并做碰撞检查。
4. 成功则加桥接边，重新搜索。

**Panda 管线路径** (`connectivity.bridge_islands`):

专为 7-DOF 场景优化的桥接实现：

1. 使用 `find_islands` 识别所有连通分量
2. 用 `_try_expand_bridge_box` 在孤岛间扩展桥接 box
3. 采用 dry-run FFB (mark_occupied=False) + overlap 检查 + 正式 FFB (mark_occupied=True) 的三步策略
4. 向量化桥接-box 邻接更新（B3 优化，使用 `_adjacent_existing_ids_from_cache`）

> **设计注意**：早期尝试的 B5 优化（通过 `clear_subtree_occupation` 回滚）已回滚，因为 `clear_subtree_occupation` 会清除所有后代节点的占用标记，包括其他 box 的节点。

该机制在邻接图“几何上可连但拓扑未连”时有效提升成功率。

---

## 6. Trajectory Optimization (`gcs_optimizer.py`)

### 6.1 Two-mode architecture

- Drake GCS 可用：构建 `GraphOfConvexSets`，以边代价最小化路径。
- 否则 fallback：Dijkstra + box-sequence 几何优化。

### 6.2 Box-sequence optimization

`optimize_box_sequence` 对每对相邻 box 提取共享面：

- 初值：共享面中心。
- 若启用 scipy：对共享面自由维做 L-BFGS-B，有界优化总路径长度。

目标函数：

$$
J=\sum_{k=1}^{m}\|p_k-p_{k-1}\|_2.
$$

固定维（共享面法向）不参与优化，天然满足面约束。

---

## 7. Path Smoothing (`path_smoother.py`)

### 7.1 Box-aware shortcut

随机选 $(i,j)$，若线段内采样点均可被 `boxes[i:j+1]` 某节点覆盖，则删除中间点。该约束防止“几何直连穿出可行盒集”。

### 7.2 Box-aware moving average

对每个中间点做窗口均值，再投影回对应 box：

$$
	ilde{q}_i=\Pi_{B_i}\Big(\frac{1}{|W_i|}\sum_{j\in W_i}q_j\Big)
$$

其中投影由 `nearest_point_to` 实现（逐维 clip）。

### 7.3 Optional batch collision in generic smoothing

`smooth_moving_average` 若检测器支持 batch API，则批量判定候选点碰撞，减少逐点开销。

---

## 8. PlannerConfig: Parameter-to-Behavior Mapping

- `max_iterations`, `max_box_nodes`: 决定图规模上限与时间预算。
- `goal_bias`: 影响收敛速度与局部偏置。
- `min_box_size`: 控制覆盖粒度（太小会图爆炸，太大影响可行性）。
- `connection_radius`, `connection_max_attempts`: 影响连通成功率。
- `segment_collision_resolution`: 影响安全性/耗时平衡。
- `path_shortcut_iters`: 影响路径压缩充分性。
- `use_gcs`, `gcs_bezier_degree`: 控制优化分支与求解复杂度。

---

## 9. Complexity and Bottlenecks

粗略复杂度：

- 采样扩展：$O(K\cdot C_{sample})$，$K$ 为有效迭代数。
- 连通与建图：与边数 $|E|$ 线性相关。
- Dijkstra：$O((|V|+|E|)\log|V|)$。
- 后处理：$O(I\cdot C_{collision})$，$I$ 为平滑/shortcut 迭代。

实测瓶颈通常落在“碰撞检测调用次数”而非图搜索本身。

---

## 10. Failure Modes and Recovery Paths

1. 端点碰撞：直接失败并给出原因。
2. 端点无法接入 forest：返回连接失败。
3. 图搜索失败：自动桥接修复（`bridge_islands` 或 `_bridge_disconnected`）。
4. 高级优化不可用：自动 fallback。
5. 平滑不可改进：保留当前可行路径，保证稳态输出。

---

## 11. End-to-End Pseudocode

```text
Input: q_start, q_goal, scene O
if start/goal in collision: fail
if segment(start, goal) is free: return direct path

forest <- load_or_create()
valid_boxes <- validate_in_scene(forest, O)
expand boxes via goal-biased seed sampling + hierarchical free-box search

# --- Panda 管线专用阶段 (panda_planner.py) ---
coarsen_forest(forest)              # dim-sweep merge, reduce box count
build loose-overlap adjacency       # different from forest's strict adjacency
bridge_islands(...)                  # connect disconnected components

G <- build graph(valid_boxes)
attach start/goal to G
path_nodes <- dijkstra(G)
if path_nodes is None:
	add bridge edges and retry dijkstra

path <- optimize_box_sequence(path_nodes)
path <- shortcut_in_boxes(path)
path <- smooth_in_boxes(path)
return PlannerResult(path, metrics)
```

---

## 12. Reproducible Evaluation Checklist

建议每次实验记录：

1. `PlannerConfig` 全量参数。
2. 场景障碍数量与分布。
3. 总碰撞检测次数、总 box 数、边数。
4. 规划耗时分解（扩展/建图/搜索/平滑）。
5. 成功率与路径长度统计。

---

## 13. 并行主路径与 strict 校验收敛说明

### 13.1 并行路径的执行顺序

当 `parallel_expand=True` 且 `parallel_workers != 1` 时：

1. 构建 KD 子空间分区；
2. 为每个分区分配 worker（进程池优先）；
3. worker 在 `constrained_intervals` 内扩展并返回局部 box；
4. 主进程合并局部结果并执行跨分区补边；
5. 执行 `validate_invariants(strict=True)`；
6. 通过后进入构图与图搜索。

### 13.2 关键行为变更（重叠修复）

并行模式下已取消“全空间起终点预扩展”。

原因：该预扩展与分区内起终点扩展会形成重复覆盖，产生正体积重叠风险。

当前策略：

1. 串行模式保留全空间起终点预扩展；
2. 并行模式由分区 worker 在子空间约束内处理起终点；
3. 合并后以 strict 校验强制阻断任何残余结构错误。

### 13.3 ProcessPool 失败回退语义

若进程池执行失败（环境/序列化等原因）：

1. 记录 warning；
2. 自动回退进程内分区扩展；
3. 保持后续合并与 strict 校验流程不变。

这样可保证“性能可退化、正确性不退化”。

### 13.4 推荐并行参数初值

对于 2-DoF / 7-DoF 常见场景，可从以下配置起步：

1. `parallel_partition_depth=2`
2. `parallel_workers=2~4`
3. `parallel_partition_dims` 先用主运动维
4. `parallel_cross_partition_connect=True`

再根据成功率和耗时分别调 `connection_radius` 与 `max_iterations`。

这组指标可直接支持论文中的 ablation 和 scaling 分析。
