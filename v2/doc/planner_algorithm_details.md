# Planner Algorithm Details (v2)

## Abstract

Planner 层实现一种 box-guided 采样规划器：先构建/复用无碰撞 box 森林，再在 box 图上搜索并优化路径。本文档以论文写法给出流程分解、函数级实现细节、复杂度和失败恢复机制。核心文件为 `box_planner.py`、`box_query.py`、`connector.py`、`gcs_optimizer.py`、`path_smoother.py`。

---

## 1. Problem Setup

给定起终点 $q_s,q_g\in\mathbb{R}^D$ 与障碍场景 $\mathcal{O}$，求连续路径

$$
\pi:[0,1]\to\mathcal{C}_{free},\quad \pi(0)=q_s,\;\pi(1)=q_g.
$$

v2 不直接在连续空间做全局优化，而是先构建 box 覆盖图 $G=(V,E)$，其中节点为无碰撞区间盒，边表示可行过渡，再在图上求离散骨架并还原为连续轨迹。

---

## 2. BoxPlanner Main Pipeline (`_plan_impl`)

### Stage 0: pre-check and normalization

- 将输入转为 `float64`
- `check_config_collision(q_s/q_g)`，任一点碰撞立即失败
- 记录时间戳和计数器

### Stage 0.5: straight-line early exit

`check_segment_collision(q_s, q_g, resolution)`：若无碰撞，直接返回两点路径。

### Stage 1: forest bootstrap

- 若配置 `forest_path`，尝试 `BoxForest.load`
- 否则新建空 forest
- 将 `HierAABBTree` 注入 forest

### Stage 2: lazy validity filtering

`forest.validate_boxes(collision_checker)` 在当前场景筛除失效盒。

### Stage 2.5: BFS connectivity check

检查起终点是否已在 forest 中同一连通分量内。若已连通，跳过扩展直接进入构图搜索阶段。

### Stage 3: seed expansion loop

**串行模式**：

1. 先对 `q_s`, `q_g` 尝试 `find_free_box`
2. 进入迭代采样（受 `max_iterations`、`max_box_nodes` 约束）
3. 每个 seed 经 `find_free_box(..., mark_occupied=True)` 得到新区间
4. 过滤几何平均边长小于 `min_box_size` 的盒
5. 若存在 `absorbed_box_ids`，同步从 forest 删除旧节点

**并行模式**（`parallel_expand=True`）：

1. `_prepare_partitions()` 构建 KD 子空间分区
2. 每个分区由 `_expand_partition_worker` 或 `_partition_expand_worker`（ProcessPoolExecutor）独立扩展
3. `_merge_connect_partitions` 合并局部结果 + 去重 + 不变量校验
4. `connect_across_partitions` 补充跨分区边

### Stage 4: graph materialization

- `connector.build_adjacency_edges(valid_boxes, valid_adjacency)` 生成 box-graph 边
- `connect_endpoints_to_forest(q_s, q_g, valid_boxes)` 把端点接入图
- `build_forest_graph(adj_edges, endpoint_edges, ...)` 组装完整搜索图

### Stage 5: graph search + repair

- 先运行 Dijkstra（`gcs_optimizer._dijkstra`）
- 若失败，运行 `_bridge_disconnected`：在可达/不可达分量间尝试桥接边后重试搜索（最多 5 轮）

### Stage 6: geometric path reconstruction

`gcs_optimizer.optimize_box_sequence` 在共享面上布置/优化 waypoint，得到初始可行折线。

### Stage 7: box-aware post-processing

- `shortcut_in_boxes`: 仅当线段采样点始终落在盒并集中才允许删点
- `smooth_in_boxes`: 移动平均后投影到对应 box 内（`nearest_point_to`）

### Stage 8: finalize

封装 `PlannerResult`：成功标志、路径、边、碰撞计数、耗时、消息；可选持久化 forest。

---

## 3. Panda 7-DOF Pipeline (`panda_planner.py`)

Panda 场景采用优化后的端到端管线：

1. **grow**: 调用 `grow_forest` / `BoxPlanner` 扩展 box forest（支持并行分区）
2. **coarsen**: 调用 `coarsen_forest` 进行 dim-sweep 合并（详见 Forest §7）
3. **adjacency**: 构建 loose-overlap 邻接图（区别于 Forest 层的 strict face-touching）
4. **bridge**: 调用 `bridge_islands` 连接孤立连通分量
5. **Dijkstra**: 图搜索找到最短路径
6. **waypoint**: 在共享面上布置路点，输出可行轨迹

### 两种邻接条件

| 邻接类型 | 函数 | 语义 | 典型边数 |
|---|---|---|---|
| strict face-touching | `deoverlap.compute_adjacency` | 某维度精确接触且其余维度有重叠 | ~5 |
| loose overlap | `_build_adjacency_and_islands` | 所有维度均有重叠（带容差） | ~157 |

> strict 邻接用于 BoxForest 内部结构维护，loose 邻接用于 bridge/Dijkstra 连通性。

### Panda 管线配置（`PandaGCSConfig`）

| 参数 | 含义 |
|---|---|
| `parallel_grow` | 是否并行生长 |
| `n_partitions_depth` | 分区切分深度 |
| `parallel_workers` | worker 数量 |
| `boundary_expand` | 是否启用边界扩展 |
| `coarsen_max_rounds` | coarsen 最大轮次 |

### 可视化输出

`panda_planner.py` 内置多种可视化函数：

- `plot_arm_scene_html` — 场景与机械臂 3D HTML
- `plot_arm_poses_html` — 路径关键帧 3D HTML
- `create_animation_html` — 执行路径动画 HTML
- `plot_joint_trajectory` — 关节轨迹图

---

## 4. BoxForestQuery (`box_query.py`)

轻量级只读查询规划器，在已有 `BoxForest` 上直接搜索路径：

```python
BoxForestQuery(forest, robot, scene, config).plan(q_start, q_goal, seed)
```

流程：Dijkstra on forest adjacency → shortcut → moving-average smooth。不重建 forest。

适用场景：forest 已事先构建（如离线积累），需要快速在线查询。

---

## 5. Seed Sampling (`_sample_seed`)

三层采样策略：

1. **Goal-biased**（概率 `goal_bias`）：$q\sim\mathcal{N}(q_g,\sigma^2I)$，clip 到关节限制
2. **KD-guided**（概率 `guided_sample_ratio`）：在现有 box 附近偏向采样
3. **Uniform**：$q\sim\mathcal{U}(l,u)$

实现细节：

- 一次生成固定批次候选（`seed_batch_size`，默认 5）
- 调用 `check_config_collision_batch` 返回布尔向量
- 取首个无碰撞候选

支持 `sampling_intervals` 约束（并行分区时限制在子空间内）。

### 边界扩展采样（`_sample_boundary_seed`）

在已有 box 外表面附近以 `boundary_expand_epsilon` 距离采样新 seed，改善局部连通性。连续 `boundary_expand_max_failures` 次失败后停止。

---

## 6. Connection Layer (`connector.py`)

### 6.1 `build_adjacency_edges`

从 `adjacency` 直接建边，每对邻居通过 `shared_face_center` 生成过渡点；若共享面退化，回退到交集中心。

### 6.2 `connect_endpoints_to_forest`

对 start/goal：

1. 若落在某 box 内，直接绑定 box id
2. 否则连向最近 box 表面点并验碰撞
3. 若失败，尝试次近候选集合

### 6.3 `build_forest_graph`

组装完整搜索图（邻接边 + 端点边 + 可选跨分区边）。

### 6.4 `connect_across_partitions`

仅处理共享切分面的相邻分区，边界带候选筛选后做碰撞验证，写入跨区 edges。

---

## 7. Graph Search and Repair

### 7.1 Shortest-path backbone

图节点为 `start/goal + box ids`，边权默认关节空间欧氏长度。Dijkstra 输出"节点序列骨架"。

### 7.2 `_bridge_disconnected`（BoxPlanner 内置）

迭代步骤（最多 5 轮）：

1. BFS 求 start 可达集合 $R$
2. 在 $R$ 与 $\bar{R}$ 间按中心距离排序候选 box 对
3. 取表面最近点连线并做碰撞检查
4. 成功则加桥接边，重新搜索

### 7.3 `bridge_islands`（connectivity.py，Panda 管线用）

专为 7-DOF 场景优化：

1. `find_islands` 识别所有连通分量
2. `_try_expand_bridge_box` 在岛间扩展桥接 box（dry-run FFB + overlap 检查 + 正式 FFB）
3. 向量化桥接-box 邻接更新（B3 优化）
4. 返回 `(bridge_edges, final_islands, n_islands_before, bridge_boxes, discarded_islands)`

---

## 8. Trajectory Optimization (`gcs_optimizer.py`)

### 8.1 Two-mode architecture

- Drake GCS 可用（`HAS_DRAKE`）：构建 `GraphOfConvexSets`，以边代价最小化路径
- 否则 fallback：Dijkstra + box-sequence 几何优化

### 8.2 Box-sequence optimization

`optimize_box_sequence(box_sequence, q_start, q_goal, allow_scipy=False)` 对每对相邻 box 提取共享面：

- 初值：共享面中心
- 若启用 scipy（`HAS_SCIPY`）：对共享面自由维做 L-BFGS-B，有界优化总路径长度

目标函数：

$$
J=\sum_{k=1}^{m}\|p_k-p_{k-1}\|_2.
$$

固定维（共享面法向）不参与优化，天然满足面约束。

---

## 9. Path Smoothing (`path_smoother.py`)

### 9.1 通用操作

- `shortcut(path, max_iters, rng)` — 随机 shortcut
- `resample(path, resolution)` — 路径均匀重采样
- `smooth_moving_average(path, window, iters)` — 移动平均平滑（支持批量碰撞检测）

### 9.2 Box-aware shortcut (`shortcut_in_boxes`)

随机选 $(i,j)$，若线段内采样点均可被 `boxes[i:j+1]` 某节点覆盖，则删除中间点。

### 9.3 Box-aware moving average (`smooth_in_boxes`)

$$
\tilde{q}_i=\Pi_{B_i}\Big(\frac{1}{|W_i|}\sum_{j\in W_i}q_j\Big)
$$

投影由 `nearest_point_to` 实现（逐维 clip）。

---

## 10. PlannerConfig: Parameter Reference

### 扩展与搜索

| 参数 | 默认值 | 含义 |
|---|---|---|
| `max_iterations` | 500 | 最大采样迭代 |
| `max_box_nodes` | 200 | 最大 box 数 |
| `seed_batch_size` | 5 | 每次采样批量大小 |
| `min_box_size` | 0.001 | 最小 box 几何均值边长 |
| `goal_bias` | 0.1 | goal-biased 采样概率 |
| `guided_sample_ratio` | 0.6 | KD-guided 采样比例 |
| `expansion_resolution` | 0.01 | 扩展分辨率 |

### 连接

| 参数 | 默认值 | 含义 |
|---|---|---|
| `connection_radius` | — | 连接半径 |
| `connection_max_attempts` | — | 连接尝试次数 |
| `segment_collision_resolution` | — | 线段碰撞检测分辨率 |

### GCS 优化

| 参数 | 默认值 | 含义 |
|---|---|---|
| `use_gcs` | False | 是否启用 Drake GCS |
| `gcs_bezier_degree` | 3 | Bézier 曲线阶数 |

### Forest 构建与查询

| 参数 | 默认值 | 含义 |
|---|---|---|
| `build_n_seeds` | 200 | forest 构建种子数 |
| `query_expand_budget` | 10 | 查询时扩展预算 |
| `forest_path` | None | forest 持久化路径 |
| `overlap_weight` | 1.0 | 重叠权重 |
| `adjacency_tolerance` | 1e-8 | 邻接容差 |
| `hard_overlap_reject` | True | 严格拒绝重叠 |

### 并行扩展

| 参数 | 默认值 | 含义 |
|---|---|---|
| `parallel_expand` | False | 是否启用并行 |
| `parallel_workers` | 0（自动） | worker 数量 |
| `parallel_batch_size` | 32 | 每批采样数 |
| `parallel_partition_depth` | 2 | KD 切分深度 |
| `parallel_partition_dims` | None | 切分维度列表 |
| `parallel_cross_partition_connect` | True | 跨分区补边 |

### 边界扩展

| 参数 | 默认值 | 含义 |
|---|---|---|
| `boundary_expand_enabled` | True | 是否启用 |
| `boundary_expand_max_failures` | 5 | 连续失败停止阈值 |
| `boundary_expand_epsilon` | 0.01 | 外推距离 |

---

## 11. Metrics and Reporting

### PathMetrics

| 字段 | 含义 |
|---|---|
| `path_length` | 路径总长度 |
| `direct_distance` | 起终点直线距离 |
| `length_ratio` | 路径长度 / 直线距离 |
| `smoothness` | 平滑度（平均角度变化） |
| `max_curvature` | 最大曲率 |
| `min_clearance` / `avg_clearance` | 最小/平均间隙 |
| `n_waypoints` | 路点数 |
| `joint_range_usage` | 关节范围利用率 |
| `box_coverage` | box 覆盖率 |
| `n_boxes` | 使用的 box 数 |

### 评估与对比

- `evaluate_result(result, robot, scene, ...)` → `PathMetrics`
- `compare_results(results_dict)` → `Dict[str, PathMetrics]`
- `format_comparison_table(metrics_dict)` → 格式化表格字符串

### 报告生成

`PlannerReportGenerator.generate(...)` 输出包含以下章节的 Markdown：

- Header / Robot / Scene / Endpoints / Config / Result / Box Trees / Graph / Metrics / Path / Files / Forest

---

## 12. Complexity and Bottlenecks

| 阶段 | 复杂度 |
|---|---|
| 采样扩展 | $O(K \cdot C_{sample})$ |
| 连通与建图 | $O(\|E\|)$ |
| Dijkstra | $O((\|V\|+\|E\|)\log\|V\|)$ |
| 后处理 | $O(I \cdot C_{collision})$ |

实测瓶颈通常落在"碰撞检测调用次数"而非图搜索本身。

---

## 13. Failure Modes and Recovery Paths

1. **端点碰撞**：直接失败并给出原因
2. **端点无法接入 forest**：返回连接失败
3. **图搜索失败**：自动桥接修复（`_bridge_disconnected` 或 `bridge_islands`）
4. **高级优化不可用**：自动 fallback（Drake → scipy → Dijkstra-only）
5. **平滑不可改进**：保留当前可行路径
6. **并行 ProcessPool 失败**：回退到进程内扩展

---

## 14. Parallel Expansion Details

### 14.1 执行顺序

1. 构建 KD 子空间分区（`_prepare_partitions`）
2. 为每个分区分配 worker（`ProcessPoolExecutor` 或进程内 fallback）
3. worker 在 `constrained_intervals` 内扩展并返回局部 box
4. 主进程合并局部结果（`_merge_connect_partitions`）
5. 执行 `validate_invariants(strict=True)`
6. 通过后进入构图与图搜索

### 14.2 关键行为

- 并行模式取消全空间起终点预扩展（避免重叠）
- 起终点由对应分区 worker 在子空间约束内处理
- 合并后 strict 校验强制阻断残余结构错误

### 14.3 ProcessPool 失败回退

进程池失败（环境/序列化等原因）时：

1. 记录 warning
2. 自动回退到进程内分区扩展
3. 后续合并与 strict 校验流程不变

### 14.4 推荐初始参数

| 场景 | `partition_depth` | `workers` | `partition_dims` | `cross_partition_connect` |
|---|---|---|---|---|
| 2-DOF | 2 | 2 | 主运动维 | True |
| 7-DOF | 2 | 2~4 | active dims | True |

---

## 15. End-to-End Pseudocode

```text
Input: q_start, q_goal, scene O

# BoxPlanner.plan
if start/goal in collision: fail
if segment(start, goal) is free: return direct path

forest <- load_or_create()
valid_boxes <- validate_in_scene(forest, O)

if start and goal already connected in forest:
    skip expansion
else:
    if parallel_expand:
        partitions <- _prepare_partitions()
        local_results <- parallel [_expand_partition_worker(p) for p in partitions]
        _merge_connect_partitions(forest, local_results)
    else:
        expand boxes via goal-biased/KD-guided/uniform seed sampling
        + hierarchical free-box search + boundary expansion

G <- build_forest_graph(valid_boxes)
attach start/goal to G
path_nodes <- dijkstra(G)
if path_nodes is None:
    add bridge edges and retry dijkstra (up to 5 rounds)

path <- optimize_box_sequence(path_nodes)
path <- shortcut_in_boxes(path)
path <- smooth_in_boxes(path)
return PlannerResult(path, metrics)

# Panda pipeline (panda_planner.py)
grow_forest(planner, q_start, q_goal)
coarsen_forest(store, forest)
build loose-overlap adjacency
bridge_islands(...)
Dijkstra + waypoint optimization
return path
```

---

## 16. Reproducible Evaluation Checklist

建议每次实验记录：

1. `PlannerConfig` 全量参数
2. 场景障碍数量与分布
3. 总碰撞检测次数、总 box 数、边数
4. 规划耗时分解（扩展/建图/搜索/平滑）
5. 成功率与路径长度统计
