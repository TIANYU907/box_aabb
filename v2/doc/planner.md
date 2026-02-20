# Planner 模块说明（详细版）

## 1. 模块定位

- 实现路径：`v2/src/planner`
- 核心职责：
  1. 组织 BoxPlanner 端到端规划流程
  2. 将 forest 拓扑转为可搜索图
  3. 进行路径搜索、桥接修复与后处理优化

---

## 2. 核心组件

1. `box_planner.py`
  - `BoxPlanner.plan` 主流程
  - 串行/并行扩展分支
2. `connector.py`
  - 邻接边构建
  - 端点接入
  - 跨分区补边
3. `gcs_optimizer.py`
  - Dijkstra 与可选 GCS 优化
4. `path_smoother.py`
  - box-aware shortcut 与平滑
5. `models.py`
  - `PlannerConfig`、`PlannerResult`、`Edge`

---

## 3. 主流程摘要

Planner 模块现在有两条主线：

### 3.1 经典规划路径（`BoxPlanner.plan`）

1. 起终点碰撞预检
2. 直连快速返回
3. 加载/新建 forest
4. 扩展 boxes（串行或并行分区）
5. 构图并连接 start/goal
6. 图搜索与桥接修复
7. 轨迹优化与平滑
8. 返回完整结果与指标

### 3.2 Panda 7-DOF 端到端管线（`panda_planner.py`）

用于 Panda 机械臂的专用规划入口，采用优化后的管线：

1. **grow**：调用 `BoxPlanner` 扩展 box forest（支持并行分区生长）
2. **coarsen**：调用 `coarsen_forest` 进行 dim-sweep 合并，减少 box 数量
3. **adjacency**：构建 loose-overlap 邻接图（区别于 Forest 层的 strict face-touching）
4. **bridge**：调用 `bridge_islands` 连接孤立连通分量
5. **Dijkstra**：图搜索找到最短路径
6. **waypoint**：在共享面上布置路点、输出可行轨迹

**关键性能指标（典型 Panda 场景）**：
- grow: ~293ms（500 boxes）
- coarsen: ~37ms（12 merges, 500→486）
- adjacency: ~11ms
- bridge: ~97ms
- Dijkstra: ~61ms
- 总规划时间: ~500ms

---

## 4. 并行模式（当前实现）

### 4.1 流程

1. KD 切分得到互斥子空间
2. 每个子空间由一个 worker 扩展
3. 主进程合并局部结果并去重
4. 严格不变量校验
5. 执行跨分区补边

### 4.2 关键约束

1. 并行模式下不再走全空间起终点预扩展，避免重复占用与重叠。
2. 合并后执行 `validate_invariants(strict=True)`。

---

## 5. 常用配置

### 5.1 扩展规模

- `max_iterations`
- `max_box_nodes`
- `min_box_size`

### 5.2 搜索与连接

- `connection_radius`
- `connection_max_attempts`
- `segment_collision_resolution`

### 5.3 并行分区

- `parallel_expand`
- `parallel_workers`
- `parallel_partition_depth`
- `parallel_partition_dims`
- `parallel_cross_partition_connect`

---

## 6. 输出与报告

- 规划结果：`PlannerResult`
- 边集合：`result.edges`
- 统计信息：路径长度、耗时、碰撞检测次数、box 数
- 报告/基准输出：`v2/output/reports/...` 与 `v2/output/benchmarks/...`

---

## 7. Benchmark

- 基准脚本：`v2/benchmarks/planner/bench_rrt_vs_marcucci.py`
- 详细说明：`v2/doc/benchmark_rrt_vs_marcucci.md`

对比对象：

1. BoxPlanner（当前实现）
2. OMPL：RRT / RRTConnect / RRTstar
3. Drake GCS（Marcucci 方法映射）

基准采用严格依赖：缺少 `ompl` 或 `pydrake` 时直接失败，不做降级。
