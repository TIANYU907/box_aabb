# Planner 模块说明

## 1. 模块定位

- 实现路径：`v2/src/planner/`
- 核心职责：
  1. 组织 BoxPlanner 端到端规划流程
  2. 将 forest 拓扑转为可搜索图
  3. 进行路径搜索、桥接修复与后处理优化

---

## 2. 子模块划分

| 文件 | 职责 |
|---|---|
| `box_planner.py` | `BoxPlanner`（别名 `BoxRRT`）— 端到端 box-guided 规划主流程，支持串行/并行分区扩展 |
| `box_query.py` | `BoxForestQuery` — 在已有 BoxForest 上做只读查询规划（无 forest 重建） |
| `connector.py` | `TreeConnector` — 邻接边构建、端点接入、跨分区补边、legacy 树间连接 |
| `gcs_optimizer.py` | `GCSOptimizer` — Dijkstra 骨架搜索 + 可选 Drake GCS 优化 + scipy waypoint SOCP |
| `path_smoother.py` | `PathSmoother` — box-aware shortcut / moving-average 平滑、通用 shortcut/resample |
| `models.py` | `PlannerConfig`（~40 字段）、`PlannerResult`、`Edge`、`BoxTree`、`gmean_edge_length` |
| `box_tree.py` | `BoxTreeManager` — 树结构管理（创建/添加/查询） |
| `metrics.py` | `PathMetrics` + `evaluate_result` + `compare_results` + `format_comparison_table` |
| `report.py` | `PlannerReportGenerator` — Markdown 报告 |
| `dynamic_visualizer.py` | `animate_robot_path` — matplotlib 动画（2D/3D） |
| `interactive_viewer.py` | `launch_viewer` — TkAgg 交互式 3D 查看器（键盘控制：Space/←→/+−/R/Q） |
| `configs/` | 预置规划参数（`2dof_planar.json`、`panda.json`） |

---

## 3. 关键公开接口

### 3.1 包公开导出（`__init__.py`）

```python
PlannerConfig, PlannerResult, Edge, BoxTree, gmean_edge_length
BoxPlanner, BoxRRT      # BoxRRT 为向后兼容别名
BoxForestQuery
TreeConnector
PathSmoother
GCSOptimizer
PathMetrics, evaluate_result
PlannerReportGenerator
```

### 3.2 核心入口

| 入口 | 用途 |
|---|---|
| `BoxPlanner(robot, scene, config).plan(q_start, q_goal, seed)` | 完整规划（含 forest 构建） |
| `BoxForestQuery(forest, robot, scene, config).plan(q_start, q_goal, seed)` | 轻量查询（已有 forest） |

---

## 4. BoxPlanner 主流程（`_plan_impl`）

10 步管线：

1. **pre-check**：起终点碰撞预检，输入归一化
2. **straight-line**：直连快速返回（O(1) 级别）
3. **forest bootstrap**：加载/新建 forest + HierAABBTree
4. **validate existing**：`validate_boxes` 在当前场景筛除失效盒
5. **BFS connectivity check**：若起终点已在同一连通分量，跳过扩展
6. **expand**：goal-biased seed 采样 + `find_free_box` 扩展（串行或并行分区）
7. **build graph**：`connector.build_adjacency_edges` + `connect_endpoints_to_forest`
8. **Dijkstra + bridge repair**：图搜索，失败时 `_bridge_disconnected` 补边重试
9. **waypoint optimization**：`optimize_box_sequence` 共享面 waypoint 优化
10. **post-processing**：`shortcut_in_boxes` + `smooth_in_boxes` box-aware 平滑

---

## 5. Panda 7-DOF 端到端管线（`panda_planner.py`）

Panda 场景采用优化后的端到端管线，在基本 BoxPlanner 之上增加 coarsen 和 bridge 阶段：

1. **grow**：调用 `BoxPlanner` 扩展 box forest（支持并行分区生长）
2. **coarsen**：调用 `coarsen_forest` 进行 dim-sweep 合并，减少 box 数量
3. **adjacency**：构建 loose-overlap 邻接图（区别于 Forest 层的 strict face-touching）
4. **bridge**：调用 `bridge_islands` 连接孤立连通分量
5. **Dijkstra**：图搜索找到最短路径
6. **waypoint**：在共享面上布置路点、输出可行轨迹

### 两种邻接条件

| 邻接类型 | 函数 | 语义 | 典型边数（500 boxes） |
|---|---|---|---|
| strict face-touching | `deoverlap.compute_adjacency` | 某维度精确接触且其余维度有重叠 | ~5 条 |
| loose overlap | `_build_adjacency_and_islands` | 所有维度均有重叠（带容差） | ~157 条 |

> bridge/Dijkstra 需要 loose 邻接以确保连通性。

### 典型性能（Panda 7-DOF，500 boxes）

| 阶段 | 耗时 |
|---|---|
| grow | ~293ms |
| coarsen | ~37ms（12 merges, 500→486） |
| adjacency | ~11ms |
| bridge | ~97ms |
| Dijkstra | ~61ms |
| **总计** | **~500ms** |

---

## 6. 并行分区扩展

### 6.1 流程

1. KD 切分得到互斥子空间（`_prepare_partitions`）
2. 每个子空间由一个 worker 扩展（`_expand_partition_worker` / `_partition_expand_worker`）
3. 主进程合并局部结果并去重（`_merge_connect_partitions`）
4. 严格不变量校验 `validate_invariants(strict=True)`
5. 执行跨分区补边（`connect_across_partitions`）

### 6.2 关键约束

- 并行模式下取消全空间起终点预扩展，避免重叠
- 合并后执行 strict 校验
- ProcessPool 失败自动回退到进程内扩展

### 6.3 配置参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `parallel_expand` | `False` | 是否启用并行扩展 |
| `parallel_workers` | `0`（自动） | worker 数量 |
| `parallel_batch_size` | `32` | 每批采样数 |
| `parallel_partition_depth` | `2` | KD 切分深度 |
| `parallel_partition_dims` | `None`（全维） | 切分维度列表 |
| `parallel_cross_partition_connect` | `True` | 是否跨分区补边 |

---

## 7. 边界扩展（Boundary Expansion）

在已有 box 外表面附近采样新 seed，改善局部连通性：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `boundary_expand_enabled` | `True` | 是否启用 |
| `boundary_expand_max_failures` | `5` | 连续失败停止阈值 |
| `boundary_expand_epsilon` | `0.01` | 外推距离 |

---

## 8. 三层采样策略（`_sample_seed`）

1. **Goal-biased**：以 `goal_bias` 概率采样 goal 附近
2. **KD-guided**：以 `guided_sample_ratio` 概率在现有 box 附近区域采样
3. **Uniform**：关节空间均匀随机采样

支持 `sampling_intervals` 约束（并行分区时限制在子空间内）。

---

## 9. 路径后处理

### 9.1 box-aware shortcut（`shortcut_in_boxes`）

随机选点对，若线段采样点全部落在 box 并集中，删除中间点。

### 9.2 box-aware moving average（`smooth_in_boxes`）

窗口均值后投影回对应 box（`nearest_point_to` 逐维 clip）。

---

## 10. 常用配置

### 扩展规模

- `max_iterations`：最大采样迭代
- `max_box_nodes`：最大 box 数
- `min_box_size`：最小 box 几何均值边长

### 搜索与连接

- `connection_radius`：连接半径
- `connection_max_attempts`：连接尝试次数
- `segment_collision_resolution`：线段碰撞检测分辨率

### GCS 优化

- `use_gcs`：是否启用 Drake GCS
- `gcs_bezier_degree`：Bézier 曲线阶数

### Forest 持久化

- `forest_path`：forest 保存/加载路径

---

## 11. 输出与报告

- 规划结果：`PlannerResult`（成功标志、路径、边、碰撞计数、耗时、消息）
- 路径指标：`PathMetrics`（路径长度、平滑度、间隙、覆盖率等）
- 报告生成：`PlannerReportGenerator.generate(...)` → Markdown
- 输出目录：`v2/output/reports/...`、`v2/output/benchmarks/...`

---

## 12. Benchmark

```bash
python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot 2dof_planar --trials 8
python -m v2.benchmarks.planner.bench_gcs_fallback
```

详细说明：[benchmark_rrt_vs_marcucci.md](benchmark_rrt_vs_marcucci.md)

对比对象：

1. BoxPlanner（当前实现）
2. OMPL：RRT / RRTConnect / RRTstar / InformedRRTstar / BITstar
3. Drake GCS（Marcucci 方法映射）

---

## 13. 测试

```bash
python -m pytest v2/tests/planner/ -v
```

测试文件：

- `test_boundary_expand.py` — 边界扩展
- `test_box_query_basic.py` — 只读查询规划
- `test_connector_closest_pairs.py` — 连接器最近对
- `test_gcs_fallback_basic.py` — GCS fallback
- `test_metrics_report_basic.py` — 指标与报告
- `test_path_smoother_basic.py` — 路径平滑
- `test_seed_sampling_batch.py` — 种子批量采样
