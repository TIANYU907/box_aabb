# box-aabb v2 算法文档总目录（详细版）

## 0. 文档定位

本文档是 v2 算法说明的统一入口，面向三类读者：

1. 研发：快速定位函数级实现与配置影响面；
2. 研究：复现实验与写作论文时的符号、流程、复杂度依据；
3. 维护：理解当前并行主路径、强校验不变量与收敛边界。

---

## 1. 建议阅读路径

### 1.1 首次阅读

1. `terminology_notation.md`（术语/符号）
2. `aabb_algorithm_details.md`（几何包络层）
3. `forest_algorithm_details.md`（离散自由空间层）
4. `planner_algorithm_details.md`（规划搜索层）

### 1.2 面向并行实现阅读

1. Forest 分册：KD 切分、合并与不变量
2. Planner 分册：并行扩展、跨分区补边、strict 校验
3. 改进计划：`改进思路.md`（阶段任务与后续清理）

---

## 2. 章节映射（论文结构）

### 第 0 篇：术语与符号

- `terminology_notation.md`

核心内容：

- 统一的符号体系（$D, \mathcal{Q}, \mathcal{F}, G=(V,E)$）
- 保守碰撞语义定义
- 复杂度记号与代码对象映射

### 第 I 篇：AABB（几何包络层）

- `aabb_algorithm_details.md`

核心内容：

- 区间 FK 与数值包络并行存在的设计动机
- relevant joints 与零长度连杆处理
- 分段/边界极值优化与复杂度讨论

核心代码：

- `v2/src/aabb/robot.py`
- `v2/src/aabb/interval_fk.py`
- `v2/src/aabb/calculator.py`
- `v2/src/aabb/strategies/*`
- `v2/src/aabb/optimization.py`

### 第 II 篇：Forest（自由空间离散层）

- `forest_algorithm_details.md`

核心内容：

- 无重叠 box 集合与邻接图不变量
- 邻接向量化 / 层级 AABB 树
- 分区合并去重与 `validate_invariants` 强校验

核心代码：

- `v2/src/forest/box_forest.py`
- `v2/src/forest/deoverlap.py`
- `v2/src/forest/collision.py`
- `v2/src/forest/hier_aabb_tree.py`

### 第 III 篇：Planner（搜索优化层）

- `planner_algorithm_details.md`

核心内容：

- BoxRRT 主流程与 seed 机制
- KD 子空间并行扩展与 ProcessPool 路径
- 跨分区补边、图搜索、桥接修复、路径后处理

核心代码：

- `v2/src/planner/box_rrt.py`
- `v2/src/planner/connector.py`
- `v2/src/planner/gcs_optimizer.py`
- `v2/src/planner/path_smoother.py`
- `v2/src/planner/models.py`

---

## 3. 跨层关系（必须一起看）

1. AABB → Forest：
  - AABB 提供“可证伪”的保守碰撞上界；
  - Forest 使用该能力裁剪扩展与验证盒安全性。
2. Forest → Planner：
  - Forest 提供无重叠拓扑骨架与邻接结构；
  - Planner 仅在该骨架上搜索和优化，避免全空间盲搜。
3. Planner → Forest（反哺）：
  - 扩展阶段产生新 box，推动 forest 增密；
  - 并行模式通过主进程合并保持结构一致性。

---

## 4. 关键实现结论（当前版本）

1. 并行分支已回归严格不变量校验：
  - 合并后执行 `validate_invariants(strict=True)`。
2. 重叠根因已规避：
  - 并行模式不再执行“全空间起终点预扩展”，
    改为由分区 worker 在受约束子空间内处理。
3. 进程池失败可回退：
  - 保留进程内分区执行作为功能兜底。

---

## 5. 复现入口索引

### 5.1 测试

- `v2/tests/aabb/*`
- `v2/tests/forest/*`
- `v2/tests/planner/*`

### 5.2 示例

- `v2/examples/*`

### 5.3 基准

- `v2/benchmarks/aabb/*`
- `v2/benchmarks/forest/*`
- `v2/benchmarks/planner/*`

---

## 6. 与其他文档的边界

- 本页只做目录与结构映射，不展开函数级流程推导。
- 公式推导、伪代码、复杂度、失效模式详见三个分册。
- 实施任务与里程碑详见 `改进思路.md`。