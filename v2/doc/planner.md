# Planner 模块说明（详细版）

## 1. 模块定位

- 实现路径：`v2/src/planner`
- 核心职责：
  1. 组织 Box-RRT 端到端规划流程
  2. 将 forest 拓扑转为可搜索图
  3. 进行路径搜索、桥接修复与后处理优化

---

## 2. 核心组件

1. `box_rrt.py`
  - `BoxRRT.plan` 主流程
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

1. 起终点碰撞预检
2. 直连快速返回
3. 加载/新建 forest
4. 扩展 boxes（串行或并行分区）
5. 构图并连接 start/goal
6. 图搜索与桥接修复
7. 轨迹优化与平滑
8. 返回完整结果与指标

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
2. 分区边界归属规则固定（配置可控，默认 `right`）。
3. 合并后执行 `validate_invariants(strict=True)`。

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
- `parallel_boundary_owner`
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

1. Box-RRT（当前实现）
2. OMPL：RRT / RRTConnect / RRTstar
3. Drake GCS（Marcucci 方法映射）

基准采用严格依赖：缺少 `ompl` 或 `pydrake` 时直接失败，不做降级。
