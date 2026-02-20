# box-aabb v2 论文目录页

## 使用说明

本文档作为 v2 算法文档的“总目录入口”，按论文章节组织并链接到三个实现分册与术语/符号表。

建议阅读顺序：

1. 先看 [术语/符号表](terminology_notation.md)
2. 再看 AABB → Forest → Planner 三个分册
3. 最后回到本页按“实验复现索引”定位代码

---

## 第 0 章 术语与符号

- [术语与符号表（Notation & Glossary）](terminology_notation.md)

---

## 第 I 篇 AABB 子系统（几何包络层）

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

---

## 第 II 篇 Forest 子系统（自由空间离散层）

- [Forest Algorithm Details](forest_algorithm_details.md)

推荐重点章节：

- Formalization and Invariants
- Deoverlap and Adjacency
- Collision Layer
- Hierarchical Box Generator

对应核心源码：

- `v2/src/forest/box_forest.py`
- `v2/src/forest/deoverlap.py`
- `v2/src/forest/collision.py`
- `v2/src/forest/hier_aabb_tree.py`
- `v2/src/forest/models.py`

---

## 第 III 篇 Planner 子系统（搜索与优化层）

- [Planner Algorithm Details](planner_algorithm_details.md)

推荐重点章节：

- Main Planner Pipeline
- Seed Sampling Micro-Mechanism
- Connection Layer
- Graph Search and Repair
- Trajectory Optimization

对应核心源码：

- `v2/src/planner/box_rrt.py`
- `v2/src/planner/connector.py`
- `v2/src/planner/gcs_optimizer.py`
- `v2/src/planner/path_smoother.py`
- `v2/src/planner/models.py`

---

## 附录 A：跨篇章概念映射

- AABB 的保守包络为 Forest 的碰撞否证提供上界；
- Forest 的无重叠 box 图为 Planner 提供可搜索拓扑骨架；
- Planner 的路径平滑受 box 约束反哺，保持与 Forest 拓扑一致性。

---

## 附录 B：实验复现索引（代码入口）

- 示例入口：
  - `v2/examples/planning_demo.py`
  - `v2/examples/visualize_random_2dof_forest_expansion.py`
- 基准入口：
  - `v2/benchmarks/aabb/*`
  - `v2/benchmarks/forest/*`
  - `v2/benchmarks/planner/*`
- Forest benchmark 常用命令：
  - `python -m v2.benchmarks.forest.bench_panda_forest`
  - `python -m v2.benchmarks.forest.bench_panda_multi`
- 测试入口：
  - `v2/tests/aabb/*`
  - `v2/tests/forest/*`
  - `v2/tests/planner/*`

---

## 版本备注

- 本目录页用于导航，不再重复展开算法细节。
- 具体实现分析、复杂度讨论与伪代码均维护在三个分册中。