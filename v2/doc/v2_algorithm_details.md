# box-aabb v2 论文目录页

## 使用说明

本文档作为 v2 算法文档的"总目录入口"，按论文章节组织并链接到三个实现分册与术语/符号表。

建议阅读顺序：

1. 先看 [术语/符号表](terminology_notation.md)
2. 再看 AABB → Forest → Planner 三个分册
3. 最后回到本页按"实验复现索引"定位代码

---

## 第 0 章 术语与符号

- [术语与符号表（Notation & Glossary）](terminology_notation.md)

---

## 第 I 篇 AABB 子系统（几何包络层）

- [AABB 模块说明](aabb.md)
- [AABB Algorithm Details](aabb_algorithm_details.md)

推荐重点章节：

- Problem Formulation
- Interval-FK: Implementation-Level Pipeline
- Numerical Envelope Path (Critical / Random)
- Complexity and Numerical Considerations

对应核心源码：

- `v2/src/aabb/robot.py`
- `v2/src/aabb/interval_fk.py`
- `v2/src/aabb/calculator.py`
- `v2/src/aabb/strategies/base.py`
- `v2/src/aabb/strategies/critical.py`
- `v2/src/aabb/strategies/random.py`
- `v2/src/aabb/optimization.py`
- `v2/src/aabb/models.py`
- `v2/src/aabb/report.py`
- `v2/src/aabb/visualizer.py`

---

## 第 II 篇 Forest 子系统（自由空间离散层）

- [Forest 模块说明](forest.md)
- [Forest Algorithm Details](forest_algorithm_details.md)

推荐重点章节：

- Formalization and Invariants
- BoxForest Core (add/remove/merge modes)
- Adjacency (vectorized + chunked)
- Collision Layer (single/batch/interval/segment + spatial index)
- Hierarchical Box Generator (active_split_dims, constrained_intervals)
- Connectivity (UnionFind, find_islands, bridge_islands)
- Coarsen (dimension-sweep merge, C1-C5 optimizations)

对应核心源码：

- `v2/src/forest/box_forest.py`
- `v2/src/forest/deoverlap.py`
- `v2/src/forest/collision.py`
- `v2/src/forest/hier_aabb_tree.py`
- `v2/src/forest/connectivity.py`
- `v2/src/forest/coarsen.py`
- `v2/src/forest/parallel_collision.py`
- `v2/src/forest/scene.py`
- `v2/src/forest/models.py`

---

## 第 III 篇 Planner 子系统（搜索与优化层）

- [Planner 模块说明](planner.md)
- [Planner Algorithm Details](planner_algorithm_details.md)

推荐重点章节：

- BoxPlanner Main Pipeline (10-step _plan_impl)
- Panda 7-DOF Pipeline (grow → coarsen → adjacency → bridge → Dijkstra → waypoint)
- BoxForestQuery (lightweight query-only planner)
- Seed Sampling (3-tier: goal-biased / KD-guided / uniform)
- Connection Layer (adjacency edges, endpoints, cross-partition)
- Graph Search and Repair (Dijkstra + bridge)
- Trajectory Optimization (Drake GCS / scipy / Dijkstra-only)
- Path Smoothing (box-aware shortcut / moving average)
- Parallel Expansion (KD partition + ProcessPool + merge + strict validation)
- Metrics and Reporting (PathMetrics, evaluate_result, PlannerReportGenerator)

对应核心源码：

- `v2/src/planner/box_planner.py`
- `v2/src/planner/box_query.py`
- `v2/src/planner/connector.py`
- `v2/src/planner/gcs_optimizer.py`
- `v2/src/planner/path_smoother.py`
- `v2/src/planner/models.py`
- `v2/src/planner/box_tree.py`
- `v2/src/planner/metrics.py`
- `v2/src/planner/report.py`
- `v2/src/planner/dynamic_visualizer.py`
- `v2/src/planner/interactive_viewer.py`

---

## 附录 A：跨篇章概念映射

- AABB 的保守包络为 Forest 的碰撞否证提供上界
- Forest 的无重叠 box 图为 Planner 提供可搜索拓扑骨架
- Planner 的路径平滑受 box 约束反哺，保持与 Forest 拓扑一致性
- Coarsen 减少 box 数量，改善图搜索效率
- Bridge 连接孤立分量，提高规划成功率
- 并行分区扩展提高 forest 构建吞吐

---

## 附录 B：实验复现索引（代码入口）

### 示例入口

- `v2/examples/planning_demo.py` — 2-DOF 基础规划
- `v2/examples/panda_planner.py` — Panda 7-DOF 端到端管线
- `v2/examples/compare_all_planners.py` — BoxPlanner vs OMPL 统一对比
- `v2/examples/visualize_random_2dof_forest_expansion.py` — 扩展可视化
- `v2/examples/aabb_demo.py` — AABB 包络演示
- `v2/examples/forest_demo.py` — BoxForest 构建演示

### Benchmark 入口

- AABB: `v2/benchmarks/aabb/bench_interval_fk.py`, `bench_fk_batch.py`, `bench_fk_scalar_cython.py`
- Forest: `v2/benchmarks/forest/bench_panda_forest.py`, `bench_panda_multi.py`, `bench_adjacency_vectorized.py`, `bench_nearest_kdtree.py`, `bench_promotion_sweep.py`, `bench_promotion_depth.py`, `bench_box_forest_incremental_add.py`
- Planner: `v2/benchmarks/planner/bench_rrt_vs_marcucci.py`, `bench_gcs_fallback.py`

### 常用命令

```bash
# Forest benchmark
python -m v2.benchmarks.forest.bench_panda_forest
python -m v2.benchmarks.forest.bench_panda_multi

# Planner benchmark
python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot 2dof_planar --trials 8

# 测试
python -m pytest v2/tests/ -v --tb=short
```

### 测试入口

- `v2/tests/aabb/` — AABB 层测试
- `v2/tests/forest/` — Forest 层测试
- `v2/tests/planner/` — Planner 层测试

---

## 附录 C：其他文档

- [Benchmark 说明](benchmark_rrt_vs_marcucci.md) — BoxPlanner vs RRT vs GCS 对比
- [性能分析报告](v2_performance_analysis_report.md) — V1 vs V2 性能对比
- [改进思路](改进思路.md) — 算法改进计划与当前进展

---

## 版本备注

- 本目录页用于导航，不再重复展开算法细节
- 具体实现分析、复杂度讨论与伪代码均维护在三个分册中
- 最后更新：2026-02-21
