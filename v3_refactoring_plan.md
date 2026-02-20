# SafeBoxForest (SBF) v3 重构计划

> 基于 v2 代码库全面审计，制定 v3 重构路线  
> 目标：**论文级实验框架** — 与 RRT 家族、Drake IRIS+GCS 对比，消融实验，复用优势实验  
> 日期：2026-02-21  
> **重构铁律：算法性能不得降低，所有重构必须通过性能回归测试**

---

## 目录

1. [算法命名](#1-算法命名)
2. [v2 现状总结](#2-v2-现状总结)
3. [v3 目标与原则](#3-v3-目标与原则)
4. [性能保障机制](#4-性能保障机制)
5. [待删除的废弃代码](#5-待删除的废弃代码)
6. [v3 目录结构设计](#6-v3-目录结构设计)
7. [核心模块重构](#7-核心模块重构)
8. [实验框架设计](#8-实验框架设计)
9. [文档精简与整合](#9-文档精简与整合)
10. [实施路线图与优先级](#10-实施路线图与优先级)

---

## 1. 算法命名

### 1.1 命名分析

v2 中混用了多个名称（`Box-AABB`、`Box-RRT`、`BoxPlanner`），缺乏一致性，且 "Box-RRT" 容易与 RRT 混淆。v3 需要一个能体现算法本质特征的名称。

**算法核心特征：**
- **Safe**：区间 FK 保证每个 box **可证明无碰撞**（conservative collision-free guarantee）
- **Box**：在 C-space 中构建**无重叠区间盒(box)**，而非采样点
- **Forest**：多棵树并行生长 → bridge 合并为**森林邻接图(graph)**，支持 Dijkstra / GCS
- 森林可**持久化缓存**并**跨查询复用**
- 障碍物变化时可**局部失效+增量修复**，无需全量重建

**命名理由**："Safe" 在机器人学领域是标准术语（safe set, safe corridor, safety-critical），精准凸显区间 FK 保守安全性这一核心贡献——这是 RRT/PRM 等概率方法不具备的理论保证。缩写 **SBF** 简洁且无冲突。

### 1.2 v3 名称方案

| 层级 | v2 名称 | v3 名称 | 说明 |
|---|---|---|---|
| 论文/算法 | Box-AABB / Box-RRT | **SafeBoxForest (SBF)** | Safe(安全) + Box(盒) + Forest(森林) |
| 框架全称 | BOX-AABB v2 | **SafeBoxForest Planner** | 论文标题用 |
| 核心类 | `BoxPlanner` | `SBFPlanner` | 缩写命名，类比 IRIS/GCS 风格 |
| AABB 层 | 不变 | **Interval AABB** | 区间正运动学 AABB |
| Forest 层 | `BoxForest` | `SafeBoxForest` | 核心数据结构，与算法同名 |
| 查询模式 | `BoxForestQuery` | `SBFQuery` | 复用已有 forest 的轻量查询 |
| 配置类 | `PlannerConfig` | `SBFConfig` | 规划器配置 |
| Python 包名 | `v2` | `sbf` | pip installable |

### 1.3 命名映射表（代码重命名）

```
v2                          →  v3
─────────────────────────────────────────────
BoxPlanner                  →  SBFPlanner
BoxRRT (alias)              →  删除
box_planner.py              →  sbf_planner.py
box_rrt.py                  →  删除
PlannerConfig               →  SBFConfig
PlannerResult               →  PlanningResult (统一接口)
BoxPlanner.plan()           →  SBFPlanner.plan()
BoxForest                   →  SafeBoxForest
box_forest.py               →  safe_box_forest.py
BoxForestQuery              →  SBFQuery
box_aabb_adapter.py         →  sbf_adapter.py
```

---

## 2. v2 现状总结

### 2.1 代码规模

| 层 | 目录 | 文件数 | 代码行数 | 说明 |
|---|---|---|---|---|
| AABB | `v2/src/aabb/` | 12 | ~2,464 | 区间FK、临界策略、L-BFGS-B 优化 |
| Forest | `v2/src/forest/` | 12 | ~4,054 | BoxForest(v3: SafeBoxForest)、HierAABBTree、碰撞检测、连通性 |
| Planner | `v2/src/planner/` | 14 | ~4,431 | BoxPlanner、GCS优化器、路径平滑、可视化 |
| Common | `v2/src/common/` | 2 | ~10 | 仅 `make_output_dir()` |
| **合计** | | **40** | **~10,959** | |

### 2.2 附属代码

| 类型 | 文件数 | 说明 |
|---|---|---|
| Examples | 15 | 有大量重复（ `gcs_planner_panda.py` 是空壳，`panda_planner.py` 混杂管线+可视化+demo） |
| Benchmarks | 10 | v2 内部性能基准，部分论文可用 |
| Tests | 10 | 单元测试，覆盖各层 |
| Docs | 11 | 有重叠和过时内容 |
| Paper | 4 | LaTeX 论文草稿 + 实验结果，保留迁移 |

### 2.3 主要问题

1. **废弃代码未清理**：`BoxRRT` 别名、`create_panda_robot()`、`AffineForm`、`HierAABBNode` 遗留类等
2. **职责不清**：`panda_planner.py`（1789行）混杂管线逻辑、GCS 求解器、可视化图、Plotly 动画
3. **Example 与核心代码边界模糊**：关键管线代码（方法的主入口 `grow_and_prepare`）在 examples/ 而非 src/
4. **文档重叠**：`aabb.md` vs `aabb_algorithm_details.md`，`改进思路.md` 中已完成事项仍保留
5. **实验代码分散**：`compare_all_planners.py`、`bench_rrt_vs_marcucci.py`、`rrt_family_panda.py` 各自独立，无统一接口
6. **Marcucci-GCS 对比仅 2-DOF**：论文需要 7-DOF Panda 对比
7. **碰撞检测计数语义不统一**：batch 调用膨胀 CC 计数 ~17×
8. **算法独特优势未在实验中体现**：AABB 缓存复用、Forest 跨查询复用、障碍物变化增量修复等核心优势缺乏专门实验

---

## 3. v3 目标与原则

### 3.1 核心目标

| # | 目标 | 说明 |
|---|---|---|
| G1 | **算法性能不降低** | 所有重构通过性能回归测试，v3 ≥ v2 在全部基准场景 |
| G2 | **论文实验可复现** | 一键运行全部实验，输出 CSV/LaTeX 表格/图表 |
| G3 | **方法对比公平** | 统一碰撞检测后端、统一计时接口、统一场景定义 |
| G4 | **凸显算法优势** | 专项实验展示 cache 复用、forest 复用、增量更新等独特能力 |
| G5 | **消融实验可配置** | 通过配置开关控制 ±coarsen、±bridge、serial/parallel、box 预算等 |
| G6 | **代码精简** | 删除废弃代码，合并重复模块，目标核心代码行 < 8,000 |
| G7 | **清晰分层** | 三层架构（aabb → forest → planner）+ 独立 experiment 层 |

### 3.2 设计原则

- **算法核心只改名不改逻辑**：v3 保留 v2 已验证的全部算法（区间FK、HierAABBTree、SafeBoxForest、Dijkstra+GCS），仅重构组织结构和清理废弃代码
- **配置驱动**：所有实验参数通过 JSON/dataclass 配置，不在代码中硬编码
- **统一接口**：所有 planner（SBF、RRT 家族、IRIS+GCS）实现相同 `BasePlanner` 接口
- **计时标准化**：使用 `time.perf_counter()` 统一计时，精确区分各阶段耗时
- **结果结构化**：所有实验结果输出为 `PlanningResult` dataclass → JSON → CSV → LaTeX

---

## 4. 性能保障机制

### 4.1 性能基准线（v2 参考值）

在开始重构前，记录 v2 基准性能（固定 seed，隔离子进程）：

| 场景 | 指标 | v2 基准值 | 允许波动 |
|---|---|---|---|
| Panda 7-DOF, 8obs, seed=1771593633 | 端到端耗时 | ~500ms | ≤ ±10% |
| Panda 7-DOF, 8obs, seed=1771593633 | grow 阶段 | ~293ms | ≤ ±10% |
| Panda 7-DOF, 8obs, seed=1771593633 | coarsen 阶段 | ~37ms | ≤ ±15% |
| Panda 7-DOF, 8obs, seed=1771593633 | bridge 阶段 | ~97ms | ≤ ±15% |
| 2-DOF, 双障碍, seed=42-46 | 平均耗时 | ~0.214s | ≤ ±10% |
| AABB Critical, Panda 30 seeds | 平均耗时 | ~0.170s | ≤ ±10% |
| HierAABBTree cache load | 二次运行 FK 次数 | 0 new FK | = 0 |
| Forest incremental query | 查询耗时 | << full rebuild | ratio 不恶化 |

### 4.2 回归测试流程

```
每次重构提交必须执行：

1. pytest tests/ -v                    # 功能正确性
2. python -m benchmarks.perf_regression  # 性能回归（新增）
   → 自动对比 v2 基准 JSON，任何指标超限则 FAIL
3. 手动 spot-check：
   - Panda 7-DOF 端到端规划成功
   - HCACHE 加载/保存正常
   - 增量查询正常
```

### 4.3 性能回归测试脚本设计

```python
# tests/perf_regression.py（新增）
"""
自动性能回归检测。
对比 v3 运行结果与 v2_baseline.json 中记录的基准值。
超过允许波动范围则标红 FAIL。
"""
BASELINE_FILE = "tests/v2_baseline.json"

SCENARIOS = [
    {"name": "panda_7dof_8obs", "seed": 1771593633, 
     "metrics": {"total_ms": (500, 0.10), "grow_ms": (293, 0.10)}},
    {"name": "2dof_dual_obs",   "seed": 42,
     "metrics": {"total_ms": (214, 0.10)}},
]

def run_and_compare():
    for s in SCENARIOS:
        result = run_scenario(s)
        for metric, (baseline, tolerance) in s["metrics"].items():
            actual = result[metric]
            if actual > baseline * (1 + tolerance):
                FAIL(f"{s['name']}.{metric}: {actual:.1f}ms > {baseline*(1+tolerance):.1f}ms")
```

### 4.4 不可修改的核心算法路径

以下代码路径在重构中**只允许改名/移动，不允许修改算法逻辑**：

| 核心路径 | 文件 | 理由 |
|---|---|---|
| 区间 FK 计算 | `interval_fk.py`: `compute_fk_full`, `compute_fk_incremental`, `_split_fk_pair` | 性能核心，~43% 增量加速已验证 |
| HierAABBTree 搜索 | `hier_aabb_tree.py`: `find_free_box`, `_split`, `_refine_aabb`, `_ensure_aabb_at` | KD-tree 分裂+提升逻辑 |
| HCACHE 持久化 | `hier_aabb_tree.py`: `save_binary`, `save_incremental`, `load_binary` | 二进制格式兼容性 |
| SafeBoxForest 邻接 | `safe_box_forest.py`: `add_box_direct`, `_adjacent_existing_ids_from_cache` | 向量化邻接检测 |
| 碰撞检测 | `collision.py`: `check_box_collision`, `check_config_collision_batch` | 保守碰撞语义 |
| BFS 波前扩展 | `sbf_planner.py`: anchor BFS 队列 + 主循环 wavefront 队列 + `_sample_boundary_seed` | 连通性核心，显著提升有效 box 密度 |
| UnionFind + bridge | `connectivity.py`: `find_islands`, `bridge_islands` | 连通性修复优化 A-D |
| coarsen | `coarsen.py`: `coarsen_forest` | 维度扫描贪心合并 |
| Dijkstra + GCS | `gcs_optimizer.py`: `optimize`, `optimize_box_sequence` | 路径优化 |
| Forest 失效/修复 | `safe_box_forest.py`: `invalidate_against_obstacle`, `remove_invalidated` | 增量更新核心 |

---

## 5. 待删除的废弃代码

### 3.1 明确废弃项（v3 中删除）

| # | 文件/符号 | 位置 | 原因 |
|---|---|---|---|
| D1 | `create_panda_robot()` | `aabb/robot.py` | 已标注 deprecated，用 `load_robot('panda')` 替代 |
| D2 | `PANDA_JOINT_LIMITS` 常量 | `aabb/robot.py` | 向后兼容遗留，用 `robot.joint_limits` 替代 |
| D3 | `AffineForm` 类 | `aabb/interval_math.py` | 旧方案，主管线使用纯标量区间（`interval_fk.py`），不再需要 |
| D4 | `smart_sin` / `smart_cos` | `aabb/interval_math.py` | 仅给 `AffineForm` 使用，一并删除 |
| D5 | `HierAABBNode` 数据类 | `forest/hier_aabb_tree.py` | 注释明确："遗留数据类——仅保留供 import 兼容。内部不再使用" |
| D6 | `_NodeView.raw_aabb` / `refined_aabb` | `forest/hier_aabb_tree.py` | 向后兼容别名 → 统一用 `aabb` |
| D7 | `compute_adjacency_reference()` | `forest/deoverlap.py` | 仅测试参考实现，v3 测试可直接用 vectorized 版本 |
| D8 | `forest/visualizer.py` | `forest/visualizer.py` | 3 行空壳文件 |
| D9 | `planner/box_rrt.py` | `planner/box_rrt.py` | 3 行向后兼容 shim（`from .box_planner import *`） |
| D10 | `BoxRRT = BoxPlanner` 别名 | `planner/box_planner.py` 末尾 | 不再需要旧名称 |
| D11 | `result.box_trees` legacy 赋值 | `planner/box_planner.py` | 标注 `# legacy 兼容` |
| D12 | `PlannerConfig.from_dict()` 兼容逻辑 | `planner/models.py` | 处理 `min_box_volume→min_box_size` 重命名 |
| D13 | `connector.py` legacy 图构建段 | `planner/connector.py` | 标注 "图构建 (legacy)" |
| D14 | `gcs_planner_panda.py` | `examples/` | 空壳 shim（10 行），直接 re-export `panda_planner.py` |

### 5.2 可精简项（v3 中合并/缩减）

| # | 文件 | 行动 |
|---|---|---|
| S1 | `aabb/interval_math.py` | 删除 `AffineForm`/`smart_sin`/`smart_cos` 后仅保留 `Interval` 类及 `I_sin`/`I_cos`（约 80 行），考虑合并入 `interval_fk.py` |
| S2 | `aabb/report.py` + `aabb/visualizer.py` | 合并为 `aabb/viz.py`（或移入独立 `viz/` 模块） |
| S3 | `planner/report.py` + `planner/visualizer.py` + `planner/dynamic_visualizer.py` + `planner/interactive_viewer.py` | 合并为 `planner/viz.py`，仅保留论文实验需要的可视化 |
| S4 | `common/output.py` | 10 行代码，内联到需要的地方或合并入 utils |
| S5 | `panda_planner.py`（1789行） | 拆分：管线逻辑 → `src/planner/pipeline.py`，可视化 → `viz/`，demo → `examples/` |
| S6 | `ompl_bridge.py` + `ompl_rrt_panda.py` | 合并为统一的 `baselines/ompl_adapter.py` |

---

## 6. v3 目录结构设计

```
v3/
├── pyproject.toml                   # 项目配置（包名: sbf）
├── setup_cython.py                  # Cython 编译
├── README.md                        # 项目说明
│
├── src/                             # ======== 核心库 ========
│   ├── __init__.py
│   │
│   ├── aabb/                        # Layer 1: Interval AABB 计算
│   │   ├── __init__.py
│   │   ├── robot.py                 # Robot 类 + DH FK（清理废弃符号）
│   │   ├── interval_fk.py           # 纯标量区间 FK（合并 interval_math 精简部分）
│   │   ├── models.py                # BoundaryConfig, LinkAABBInfo, AABBEnvelopeResult
│   │   ├── calculator.py            # AABBCalculator 调度
│   │   ├── optimization.py          # L-BFGS-B 优化
│   │   ├── strategies/              # 采样策略
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── critical.py
│   │   │   └── random.py
│   │   ├── configs/                 # 机器人 JSON 配置
│   │   │   ├── panda.json
│   │   │   ├── 2dof_planar.json
│   │   │   └── 3dof_planar.json
│   │   ├── _fk_scalar_core.pyx      # Cython 加速
│   │   └── _interval_fk_core.pyx
│   │
│   ├── forest/                      # Layer 2: SafeBoxForest
│   │   ├── __init__.py
│   │   ├── safe_box_forest.py       # SafeBoxForest 核心（含 invalidate/remove 增量更新）
│   │   ├── collision.py             # CollisionChecker（保守碰撞语义）
│   │   ├── connectivity.py          # UnionFind, find_islands, bridge_islands
│   │   ├── deoverlap.py             # 邻接计算（删除 reference 实现）
│   │   ├── coarsen.py               # Forest 粗化（维度扫描贪心合并）
│   │   ├── hier_aabb_tree.py        # HierAABBTree（KD-tree + HCACHE 持久化）
│   │   ├── models.py                # Obstacle, BoxNode
│   │   ├── scene.py                 # Scene 管理
│   │   ├── parallel_collision.py    # 并行碰撞 + SpatialIndex
│   │   ├── _hier_layout.py          # HCACHE02 二进制格式
│   │   ├── configs/
│   │   │   └── default.json
│   │   └── _hier_core.pyx           # Cython 加速
│   │
│   ├── planner/                     # Layer 3: SBF Planner
│   │   ├── __init__.py
│   │   ├── sbf_planner.py           # ★ SBFPlanner 主入口（原 box_planner.py，重命名）
│   │   ├── pipeline.py              # ★ 新文件：Panda 全管线（从 panda_planner.py 提取）
│   │   ├── sbf_query.py             # SBFQuery 轻量复用查询
│   │   ├── box_tree.py              # BoxTreeManager
│   │   ├── connector.py             # TreeConnector（删除 legacy 段）
│   │   ├── gcs_optimizer.py         # GCS + Drake/scipy 优化器
│   │   ├── path_smoother.py         # 路径平滑
│   │   ├── models.py                # SBFConfig, PlanningResult, Edge
│   │   ├── metrics.py               # PathMetrics
│   │   └── configs/
│   │       ├── panda.json
│   │       └── 2dof_planar.json
│   │
│   ├── baselines/                   # ★ 新模块：对比方法统一封装
│   │   ├── __init__.py
│   │   ├── base.py                  # BasePlanner ABC（统一接口）
│   │   ├── rrt_family.py            # 纯 Python RRT/RRTConnect/RRT*/InformedRRT*/BiRRT*
│   │   ├── ompl_adapter.py          # OMPL C++ 适配器（合并 ompl_bridge + ompl_rrt_panda）
│   │   ├── iris_gcs.py              # ★ 新文件：Drake IRIS+GCS (Marcucci) 适配器
│   │   └── sbf_adapter.py           # ★ SBF 方法封装为统一 BasePlanner 接口
│   │
│   └── utils/                       # ★ 新模块：公共工具
│       ├── __init__.py
│       ├── timing.py                # 统一计时工具（perf_counter + 阶段记录）
│       ├── output.py                # 输出目录管理
│       └── seed.py                  # 随机种子管理
│
├── experiments/                     # ★ 新目录：论文实验脚本
│   ├── __init__.py
│   ├── configs/                     # 实验配置
│   │   ├── scenes/                  # 场景定义
│   │   │   ├── panda_8obs.json
│   │   │   ├── panda_15obs.json
│   │   │   ├── panda_narrow.json
│   │   │   ├── 2dof_simple.json
│   │   │   └── 2dof_cluttered.json
│   │   ├── planners/                # 方法配置
│   │   │   ├── sbf_default.json
│   │   │   ├── sbf_no_coarsen.json
│   │   │   ├── sbf_no_bridge.json
│   │   │   ├── sbf_serial.json
│   │   │   ├── rrt_connect.json
│   │   │   ├── rrt_star.json
│   │   │   └── iris_gcs.json
│   │   └── experiment_sets/         # 实验组合
│   │       ├── main_comparison.json
│   │       ├── forest_reuse.json
│   │       ├── obstacle_change.json
│   │       ├── cache_warmstart.json
│   │       ├── ablation.json
│   │       ├── config_sweep.json
│   │       └── scalability.json
│   │
│   ├── runner.py                    # ★ 统一实验运行器
│   ├── exp_main_comparison.py       # 实验1: SBF vs RRT家族 vs IRIS+GCS
│   ├── exp_forest_reuse.py          # ★ 实验2: Forest 复用（同场景多查询）
│   ├── exp_obstacle_change.py       # ★ 实验3: 障碍物变化增量更新
│   ├── exp_cache_warmstart.py       # ★ 实验4: AABB 缓存热启动
│   ├── exp_ablation.py              # 实验5: 消融实验（±coarsen, ±bridge, ±parallel）
│   ├── exp_config_sweep.py          # 实验6: 配置扫描
│   ├── exp_scalability.py           # 实验7: 可扩展性
│   ├── exp_aabb_tightness.py        # 实验8: AABB 紧致度
│   │
│   ├── analysis/                    # 结果分析与可视化
│   │   ├── aggregate.py             # 聚合多 seed/trial 结果
│   │   ├── tables.py                # 生成 LaTeX 表格
│   │   ├── plots.py                 # 生成论文图表（matplotlib）
│   │   └── statistical_tests.py     # 统计检验（Wilcoxon, bootstrap CI）
│   │
│   └── output/                      # 实验输出（git-ignored）
│       ├── raw/                     # 原始 JSON 结果
│       ├── tables/                  # LaTeX 表格
│       └── figures/                 # PDF/PNG 图表
│
├── viz/                             # ★ 独立可视化模块
│   ├── __init__.py
│   ├── aabb_viz.py                  # AABB 包络可视化
│   ├── forest_viz.py                # Forest box 可视化
│   ├── planner_viz.py               # 静态路径可视化
│   ├── scene_viz.py                 # 3D 场景 + 机械臂渲染
│   └── animation.py                 # 路径动画（Plotly/matplotlib）
│
├── examples/                        # 精简示例
│   ├── demo_2dof.py                 # 2-DOF 快速演示
│   ├── demo_panda.py                # Panda 7-DOF 演示
│   └── demo_comparison.py           # 简单方法对比演示
│
├── tests/                           # 测试（迁移 + 扩充）
│   ├── conftest.py
│   ├── v2_baseline.json             # ★ v2 性能基准值
│   ├── perf_regression.py           # ★ 性能回归测试
│   ├── test_aabb/
│   │   ├── test_calculator.py
│   │   ├── test_interval_fk.py
│   │   └── test_robot.py
│   ├── test_forest/
│   │   ├── test_safe_box_forest.py
│   │   ├── test_collision.py
│   │   ├── test_hier_aabb_tree.py
│   │   └── test_adjacency.py
│   ├── test_planner/
│   │   ├── test_sbf_planner.py
│   │   ├── test_gcs_optimizer.py
│   │   ├── test_path_smoother.py
│   │   └── test_metrics.py
│   └── test_baselines/
│       ├── test_rrt_family.py
│       └── test_planner_interface.py
│
├── doc/                             # 精简文档
│   ├── algorithm.md                 # ★ 合并：算法总述（三层管线 + 伪代码）
│   ├── notation.md                  # 术语与符号（精简）
│   ├── experiment_guide.md          # ★ 新文档：实验复现指南
│   └── paper/                       # 论文相关
│       ├── algorithm_review.tex
│       ├── sbf_paper.tex
│       └── figures/
│
└── scripts/                         # 工具脚本
    ├── setup_env.ps1                # 环境配置
    ├── build_cython.ps1             # 编译 Cython
    └── run_all_experiments.ps1      # 一键运行全部实验
```

---

## 7. 核心模块重构

### 7.1 AABB 层重构

**目标**：保持算法不变，删除废弃代码，精简模块数量

| 行动 | 详情 |
|---|---|
| 删除 `AffineForm` + `smart_sin/cos` | 从 `interval_math.py` 中移除，仅保留 `Interval`、`I_sin`、`I_cos` |
| 合并 `interval_math.py` → `interval_fk.py` | `Interval` 类和区间三角函数直接放入 `interval_fk.py` 顶部 |
| 清理 `robot.py` | 删除 `create_panda_robot()`、`PANDA_JOINT_LIMITS` |
| 删除 `aabb/report.py` | 报告生成移入 `viz/aabb_viz.py` 或删除（论文实验用 experiment 层统一输出） |
| 保留 `aabb/visualizer.py` | 移入 `viz/aabb_viz.py` |

**预计行数**：~1,800 行（减少约 27%）

### 7.2 Forest 层重构

**目标**：清理遗留类，删除空壳，**保持全部核心算法逻辑和缓存机制不变**

| 行动 | 详情 |
|---|---|
| 删除 `HierAABBNode` 遗留类 | 从 `hier_aabb_tree.py` 中移除 |
| 删除 `_NodeView.raw_aabb/refined_aabb` | 统一使用 `.aabb` |
| 删除 `compute_adjacency_reference()` | 从 `deoverlap.py` 中移除 |
| 删除 `forest/visualizer.py` | 3 行空壳 |
| **完整保留** HCACHE 持久化 | `save_binary`, `save_incremental`, `load_binary`, `auto_load`, `auto_save`, `auto_merge_save` |
| **完整保留** 增量更新 | `invalidate_against_obstacle`, `remove_invalidated`, `unoccupy_boxes` |
| **完整保留** incremental FK | `compute_fk_incremental`, `_split_fk_pair`, `warmup_fk_cache` |
| **完整保留** 并行合并 | `merge_partition_forests`, `dedup_boundary_boxes`, `merge_from` |

**预计行数**：~3,800 行（减少约 6%）

### 7.3 Planner 层重构

**目标**：重命名核心类，拆分巨型文件，提取管线逻辑

| 行动 | 详情 |
|---|---|
| **重命名** `BoxPlanner` → `SBFPlanner` | 文件名 `box_planner.py` → `sbf_planner.py` |
| **重命名** `PlannerConfig` → `SBFConfig` | 在 `models.py` 中修改 |
| 删除 `box_rrt.py` | 3 行 shim |
| 删除 `BoxRRT` 别名 | 不再需要 |
| 删除 `result.box_trees` legacy | 移除 `# legacy 兼容` 赋值 |
| 删除 `connector.py` legacy 段 | 移除 "图构建 (legacy)" 段 |
| 裁剪 `SBFConfig.from_dict()` | 移除 `min_box_volume` 兼容映射 |
| **新增 `pipeline.py`** | 从 `panda_planner.py` 提取核心管线：`grow_and_prepare()`, `run_method_with_bridge()`, `build_adjacency_and_islands()`, GCS SOCP 求解器 |
| 可视化代码 → `viz/` | `visualizer.py`, `dynamic_visualizer.py`, `interactive_viewer.py` 全部移入 `viz/` |
| 保留 `report.py` | 轻量化，仅保留 `PlanningResult → dict` 序列化 |
| **完整保留** BFS 波前扩展 | anchor BFS 队列（从 start/goal 向外辐射） + 主循环 wavefront FIFO 队列 + `_sample_boundary_seed()` |
| **完整保留** 配置参数 | `boundary_expand_enabled`、`boundary_expand_max_failures`、`boundary_expand_epsilon` 迁入 `SBFConfig` |

**预计行数**：~3,200 行（减少约 28%）

### 7.4 新增 Baselines 模块

**目标**：所有对比方法实现统一 `BasePlanner` 接口

```python
# src/baselines/base.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class PlanningResult:
    """所有方法的统一结果格式"""
    success: bool
    path: Optional[np.ndarray]         # (N, DOF) waypoints
    cost: float                        # path length (L2)
    planning_time: float               # 秒 (wall-clock, 总耗时)
    first_solution_time: float         # 首次找到解的时间
    collision_checks: int              # 碰撞检测调用次数（单次语义）
    nodes_explored: int                # 节点/box 数量
    phase_times: dict = field(default_factory=dict)  # 各阶段耗时
    metadata: dict = field(default_factory=dict)     # 方法特定信息

class BasePlanner:
    """所有 planner 的统一接口"""
    
    def setup(self, robot, scene, config: dict):
        """设置机器人、场景、配置（可被多次 plan 复用）"""
        raise NotImplementedError
    
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             timeout: float = 30.0) -> PlanningResult:
        """规划路径"""
        raise NotImplementedError
    
    def reset(self):
        """重置内部状态（用于公平计时对比）"""
        pass
    
    @property
    def name(self) -> str:
        raise NotImplementedError
    
    @property
    def supports_reuse(self) -> bool:
        """是否支持跨查询复用已有数据结构"""
        return False
```

**五个适配器**：

| 适配器 | 封装方法 | 来源 | `supports_reuse` |
|---|---|---|---|
| `sbf_adapter.py` | SBF 全管线 | 新写，调用 `pipeline.py` | **True** |
| `rrt_family.py` | RRT/RRTConnect/RRT*/InformedRRT*/BiRRT* | 从 `rrt_family_panda.py` 重构 | False |
| `ompl_adapter.py` | OMPL C++ 全家族 | 合并 `ompl_bridge.py` + `ompl_rrt_panda.py` | False |
| `iris_gcs.py` | Drake IRIS + GCS (Marcucci) | 从 `bench_rrt_vs_marcucci.py` 扩展至 7-DOF | Partial（IRIS 区域可复用） |

### 7.5 新增 Utils 模块

```python
# src/utils/timing.py
import time
from contextlib import contextmanager

class Timer:
    """阶段计时器，用于精确记录管线各阶段耗时"""
    
    def __init__(self):
        self.records = {}
        self._stack = []
    
    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        yield
        self.records[name] = time.perf_counter() - t0
    
    @property
    def total(self) -> float:
        return sum(self.records.values())
    
    def to_dict(self) -> dict:
        return {**self.records, 'total': self.total}
```

---

## 8. 实验框架设计

### 8.1 算法核心优势分析

在设计实验前，先明确 SafeBoxForest 相比采样类方法和凸分解类方法的**独特优势**：

| 优势 | 机制 | 其他方法不具备 |
|---|---|---|
| **A. AABB 缓存持久化** | HierAABBTree 将区间 FK 结果以 HCACHE02 格式持久化到磁盘；二次运行直接加载，跳过全部 FK 计算；增量保存仅写入 dirty/new 节点 | RRT/PRM 每次从零开始；IRIS 区域不可持久化 |
| **B. 增量 FK 复用** | 树节点分裂时通过 `compute_fk_incremental` 复用父节点 FK 前缀，平均节省 ~43% 矩阵乘法 | 无类比机制 |
| **C. Forest 跨查询复用** | 同一场景的 box 邻接图可被多次 `SBFQuery` 复用；新查询仅做 O(depth) 点定位 + Dijkstra；累积查询越多越快 | RRT 每次建树；PRM roadmap 支持复用但无 box 覆盖保证 |
| **D. 障碍物变化增量更新** | `invalidate_against_obstacle` 仅碰撞检测变更障碍物；`remove_invalidated` + `unoccupy` 释放树节点；`regrow` 利用旧 seed 快速回填 | RRT/RRT* 必须完整重建；IRIS 需要重新扩圆 |
| **E. forest 单调增长** | bridge / expand 产生的新 box 持久化到 forest，后续查询直接受益 | 采样树无法被复用 |
| **F. 并行分区扩展** | KD 分区 → 多进程独立扩展 → 合并 + 去重 + 跨区连接 | RRT 并行化困难 |
| **G. 保守碰撞安全** | 区间 FK AABB 保证无碰撞结论可信（"免碰判定"不会漏报） | 采样碰撞概率性 || **H. BFS 波前扩展** | 从 start/goal 锚点向外做广度优先 boundary 采样，生成紧密相邻的 box 簇；每轮从队列前端 box 的表面外侧采样，失败超限则驱逐，新 box 入队尾；队列清空后回退到随机采样 | RRT 无法保证采样点与已探索区域相邻；IRIS 单个区域扩胀但无波前传播 |
### 8.2 统一实验运行器

```python
# experiments/runner.py
class ExperimentRunner:
    """统一实验运行器"""
    
    def __init__(self, experiment_config: dict):
        self.scenes = experiment_config['scenes']
        self.planners = experiment_config['planners']
        self.seeds = experiment_config['seeds']
        self.n_trials = experiment_config['n_trials']
        self.timeout = experiment_config['timeout']
    
    def run(self) -> ExperimentResults:
        """运行所有 (scene × planner × seed × trial) 组合"""
        results = []
        for scene_cfg in self.scenes:
            scene = load_scene(scene_cfg)
            for planner_cfg in self.planners:
                planner = create_planner(planner_cfg)
                for seed in self.seeds:
                    for trial in range(self.n_trials):
                        result = self._run_single(
                            planner, scene, seed + trial, self.timeout)
                        results.append(result)
        return ExperimentResults(results)
    
    def _run_single(self, planner, scene, seed, timeout):
        """单次运行，隔离子进程以避免缓存影响"""
        ...
```

### 8.3 论文实验列表（8 个实验）

---

#### 实验 1：主对比实验（Main Comparison） — 对应论文 Table 1, Figure 2

**目的**：SafeBoxForest 与主流方法全面对比

| 维度 | 设置 |
|---|---|
| **方法** | SBF, RRTConnect, RRT*, InformedRRT*, BIT*, PRM, IRIS+GCS |
| **场景** | Panda 7-DOF: {8, 15, 20} 障碍物 × {开阔, 狭窄通道} |
| **指标** | 成功率, 首次解时间, 总规划时间(mean/median/P95), 路径长度, 碰撞检测次数 |
| **统计** | 50 seeds × 3 trials = 150 runs per (method × scene) |
| **输出** | Table（方法×指标矩阵）, violin plot（时间分布）|

**关键设计**：
- 所有方法使用相同 `CollisionChecker` 后端
- 碰撞检测统一计单次 `check_config_collision()` 调用数
- SBF 使用**冷启动**（无 cache），与其他方法公平对比

---

#### 实验 2：Forest 跨查询复用（Forest Reuse） — ★ 凸显优势 C/E，对应论文 Table 2, Figure 3

**目的**：展示 box 邻接图可跨查询复用的核心优势

**实验设计**：
```
同一场景（Panda 7-DOF, 8 obstacles），K 个不同起终点对：

方案 A: SBF — 首次完整 build；后续 K-1 次使用 SBFQuery（复用 forest）
方案 B: SBF — 每次全量重建（对照）
方案 C: RRTConnect — 每次从零构建树
方案 D: RRT* — 每次从零构建树
方案 E: IRIS+GCS — 每次完整 IRIS 扩圆 + GCS 求解

K = [1, 5, 10, 20, 50, 100]
```

| 指标 | 说明 |
|---|---|
| **累积总耗时** | 完成 K 个查询的总 wall-clock 时间 |
| **每查询平均耗时** | 总时间 / K |
| **加速比** | 方案 A 相对方案 C 的加速倍数 |
| **首查询 vs 后续查询** | 分别报告首次 build 耗时和后续 query 耗时 |
| **成功率** | 每个查询是否成功 |

**预期结论**：K 越大，SBF 累积优势越显著（amortized O(Dijkstra) vs O(full_build)）

**输出**：
- Figure 3a: 累积耗时 vs K 曲线（多方法对比）
- Figure 3b: 每查询平均耗时 vs K 曲线
- Table 2: K=1/10/50/100 时各方法的平均查询时间

---

#### 实验 3：障碍物变化增量更新（Obstacle Change） — ★ 凸显优势 D，对应论文 Table 3, Figure 4

**目的**：展示障碍物动态变化时 SBF 增量更新能力

**实验设计**：
```
初始场景：Panda 7-DOF, 10 obstacles，完成一次完整规划

场景 A — 障碍物消失：
  移除 1/2/3 个障碍物 → 增量 regrow vs 全量重建

场景 B — 障碍物出现：  
  新增 1/2/3 个障碍物 → invalidate + remove + regrow vs 全量重建

场景 C — 障碍物移动：
  1/2/3 个障碍物位移 → disappear(old) + appear(new) vs 全量重建

对照组：RRTConnect / RRT* / IRIS+GCS 每次全量重建
```

| 指标 | 说明 |
|---|---|
| **更新耗时** | 增量更新总时间（invalidate + remove + regrow + reconnect） |
| **全量重建耗时** | 从零完整 build 时间 |
| **加速比** | 增量 / 全量 |
| **box 存活率** | 更新后保留的 box 占原始 box 总数的比例 |
| **路径质量** | 更新后规划路径长度与全量重建的差异 |

**预期结论**：增量更新耗时远小于全量重建（~5-20× 加速），且路径质量接近

**输出**：
- Table 3: (变化类型 × 变化数量) 增量耗时 vs 全量重建耗时
- Figure 4a: 加速比 vs 变化数量
- Figure 4b: box 存活率 vs 变化数量

---

#### 实验 4：AABB 缓存热启动（Cache Warmstart） — ★ 凸显优势 A/B，对应论文 Table 4, Figure 5

**目的**：展示 HCACHE 持久化和增量 FK 的加速效果

**实验设计**：
```
场景：Panda 7-DOF, 多组实验

A — 冷启动 vs 热启动：
  Run 1: 无 cache，记录 FK 调用数、构建时间、HCACHE 大小
  Run 2: 有 cache（auto_load），记录 FK 调用数、构建时间
  Run 3: 有 cache（auto_load + warmup），记录增量 FK 比例

B — 增量保存效率：
  save_binary (full) vs save_incremental (dirty only)
  在不同 tree size (1K/10K/50K nodes) 下对比 I/O 时间和写入字节

C — 跨场景 cache 复用：  
  场景 1 构建 cache → 加载到场景 2（不同障碍物但同一机器人）→ 记录 FK 节省量
  验证 "cache 仅绑定机器人运动学，不绑定障碍物" 的设计

D — 增量 FK vs 全量 FK 对比：
  记录每次 _split 使用 incremental 还是 full FK
  统计增量 FK 平均节省的矩阵乘法次数
```

| 指标 | 说明 |
|---|---|
| **FK 调用次数** | Run 1（首次）vs Run 2（复用） |
| **构建时间** | 冷启动 vs 热启动 |
| **I/O 时间** | full save vs incremental save |
| **incremental FK 比例** | 使用增量 FK 的 split 占总 split 的百分比 |
| **矩阵乘法节省** | 增量 FK 跳过的 mat-mul 数 / 总 mat-mul 数 |

**预期结论**：
- 热启动跳过 ~95%+ FK 调用，构建时间减少 50-80%
- 增量保存在大 tree 上 I/O 时间减少 10-100×
- 跨场景 cache 复用有效（同一机器人不同障碍物）
- 增量 FK 平均节省 ~43% 矩阵乘法

**输出**：
- Table 4: 冷启动 vs 热启动对比表
- Figure 5a: FK 调用数 bar chart（cold/warm/warm+warmup）
- Figure 5b: incremental save I/O vs tree size

---

#### 实验 5：消融实验（Ablation Study） — 对应论文 Table 5, Figure 6

**目的**：量化各管线组件的贡献

| 变体 | 配置差异 | 对应优势 |
|---|---|---|
| **Full pipeline** | 默认配置（coarsen + bridge + parallel + GCS + shortcut） | 全部 |
| **No coarsen** | `use_coarsen=False` | 验证粗化贡献 |
| **No bridge** | `use_bridge=False` | 验证桥接贡献 |
| **Serial grow** | `parallel_workers=0` | 验证并行扩展贡献 (F) |
| **No GCS refine** | `use_gcs=False`，仅 Dijkstra 中心点 | 验证路径优化贡献 |
| **No shortcut** | `path_shortcut_iters=0` | 验证路径快捷贡献 |
| **No incremental FK** | 强制 `compute_fk_full`（禁用父节点前缀复用） | 验证增量 FK 贡献 (B) |
| **No wavefront** | `boundary_expand_enabled=False`（禁用 BFS 波前，仅用随机/目标导向采样） | 验证波前扩展贡献 (H) |

**指标**：规划时间、成功率、路径长度、box 数量、连通分量数  
**输出**：Table 5（消融表）, Figure 6（阶段耗时分解堆叠条形图）

---

#### 实验 6：配置敏感性（Config Sweep） — 对应论文 Figure 7

| 参数 | 扫描范围 | 影响 |
|---|---|---|
| `max_box_nodes` | [100, 200, 400, 800, 1600] | forest 规模 / 覆盖率 / 构建时间 |
| `min_box_size` | [0.01, 0.005, 0.001, 0.0005] | 最小分辨率 / 窄通道能力 |
| `promotion_depth` | [1, 2, 3, 4, 5] | HierAABBTree 提升深度 / box 大小 |
| `goal_bias` | [0.0, 0.05, 0.1, 0.2, 0.3] | 目标偏向采样强度 |
| `sampling_n` | [40, 80, 120, 200] | 采样碰撞检测粒度 |
| `boundary_expand_enabled` | [True, False] | 波前扩展开关 |
| `boundary_expand_max_failures` | [3, 5, 8, 12] | 波前前沿驱逐阈值 |
| `boundary_expand_epsilon` | [0.005, 0.01, 0.02, 0.05] | 边界采样偏移距离 |

**输出**：Figure 7（8 个子图，每参数一条成功率+耗时双 Y 轴线图）

---

#### 实验 7：可扩展性（Scalability） — 对应论文 Figure 8

| 维度 | 变化 |
|---|---|
| **障碍物数量** | 2, 4, 8, 12, 16, 20 |
| **自由度** | 2-DOF, 3-DOF, 7-DOF (Panda) |
| **场景密度** | 稀疏 / 中等 / 密集 |

**对比方法**：SBF vs RRTConnect vs IRIS+GCS  
**输出**：
- Figure 8a: 时间 vs 障碍物数线图
- Figure 8b: 时间 vs DOF 柱状图

---

#### 实验 8：AABB 紧致度（AABB Tightness） — 对应论文 Table 6

| 对比 | 说明 |
|---|---|
| Critical vs Random(N=1000/5000) | 采样效率对比 |
| Interval vs Numerical vs Hybrid | 不同 FK 方法的紧致性 |
| n_subdivisions 敏感性 | [1, 2, 4, 8] |

**输出**：Table 6（紧致度表 + 耗时对比）

---

### 8.4 实验与论文映射

| 论文编号 | 实验 | 核心论点 |
|---|---|---|
| Table 1, Fig 2 | 实验 1: 主对比 | SBF 在时间/成功率上与 RRT 家族可比，路径质量更优 |
| Table 2, Fig 3 | **实验 2: Forest 复用** | ★ SBF 多查询场景下 amortized cost 远低于重建类方法 |
| Table 3, Fig 4 | **实验 3: 障碍物变化** | ★ SBF 支持增量更新，动态场景下大幅优于全量重建 |
| Table 4, Fig 5 | **实验 4: Cache 热启动** | ★ HCACHE 持久化 + 增量 FK 大幅加速重复运行 |
| Table 5, Fig 6 | 实验 5: 消融 | 量化各组件贡献 |
| Fig 7 | 实验 6: 配置敏感性 | 参数选择指导 |
| Fig 8 | 实验 7: 可扩展性 | 高维/复杂场景表现 |
| Table 6 | 实验 8: AABB 紧致度 | 区间 FK 保守性与效率 |

### 8.5 结果输出格式

```
experiments/output/
├── raw/
│   ├── exp1_main_comparison_20260301.json      # 完整原始数据
│   ├── exp2_forest_reuse_20260301.json
│   ├── exp3_obstacle_change_20260301.json
│   ├── exp4_cache_warmstart_20260301.json
│   ├── exp5_ablation_20260301.json
│   └── ...
├── tables/
│   ├── table1_main_comparison.tex
│   ├── table2_forest_reuse.tex
│   ├── table3_obstacle_change.tex
│   ├── table4_cache_warmstart.tex
│   ├── table5_ablation.tex
│   └── table6_aabb_tightness.tex
└── figures/
    ├── fig2_main_comparison_violin.pdf
    ├── fig3a_forest_reuse_cumulative.pdf
    ├── fig3b_forest_reuse_per_query.pdf
    ├── fig4a_obstacle_change_speedup.pdf
    ├── fig4b_obstacle_change_survival.pdf
    ├── fig5a_cache_fk_calls.pdf
    ├── fig5b_incremental_save_io.pdf
    ├── fig6_ablation_breakdown.pdf
    ├── fig7_config_sensitivity.pdf
    └── fig8_scalability.pdf
```

---

## 9. 文档精简与整合

### 9.1 v2 文档清理

| v2 文档 | 行动 | 说明 |
|---|---|---|
| `aabb.md` | **合并** → `algorithm.md` §2 | 与 `aabb_algorithm_details.md` 内容重叠 |
| `aabb_algorithm_details.md` | **合并** → `algorithm.md` §2 | 保留核心算法描述 |
| `forest.md` | **合并** → `algorithm.md` §3 | 与 `forest_algorithm_details.md` 重叠 |
| `forest_algorithm_details.md` | **合并** → `algorithm.md` §3 | 保留核心算法描述 |
| `planner.md` | **合并** → `algorithm.md` §4 | 与 `planner_algorithm_details.md` 重叠 |
| `planner_algorithm_details.md` | **合并** → `algorithm.md` §4 | 保留核心算法描述 |
| `v2_algorithm_details.md` | **删除** | 仅为上述三个文件的目录页 |
| `terminology_notation.md` | **精简** → `notation.md` | 保留符号表，删除冗余解释 |
| `v2_performance_analysis_report.md` | **存档** | 移入 `doc/archive/`，不再维护 |
| `benchmark_rrt_vs_marcucci.md` | **重写** → `experiment_guide.md` | 整合为实验复现指南 |
| `改进思路.md` | **存档** | 已完成事项归档，未完成提取到 v3 TODO |

### 9.2 v3 文档结构

| 文档 | 内容 |
|---|---|
| `README.md` | 项目概述、安装、快速开始、三层架构图、**算法名称：SafeBoxForest Planner (SBF)** |
| `doc/algorithm.md` | **核心文档**：合并三层算法详述（Interval AABB → SafeBoxForest → SBFPlanner），含伪代码、复杂度分析 |
| `doc/notation.md` | 符号表 + 术语映射（代码对应关系） |
| `doc/experiment_guide.md` | **实验复现指南**：环境配置 → 一键运行 → 结果解读 |
| `doc/paper/` | 论文 LaTeX 及相关素材 |

---

## 10. 实施路线图与优先级

### Phase 0：准备 + 性能基准采集（1 天）

- [ ] 运行 v2 全部测试，确认通过
- [ ] **采集 v2 性能基准**：运行标准场景，输出 `tests/v2_baseline.json`
- [ ] 创建 `v3/` 目录结构
- [ ] 复制 v2 核心源码到 v3（保留 git 可追溯）
- [ ] 验证 v3 基础导入正常
- [ ] 迁移 `pyproject.toml` 和 Cython 编译配置
- [ ] **编写 `tests/perf_regression.py`**（性能回归守卫）

### Phase 1：清理废弃代码 + 重命名（1 天） ⚡ 最高优先

- [ ] 执行 §5.1 全部 14 项删除
- [ ] 执行 §5.2 合并精简
- [ ] 执行 §1.3 重命名（`BoxPlanner` → `SBFPlanner` 等）
- [ ] 运行全部测试 + 性能回归确认无 regression
- [ ] 提交：`v3: remove deprecated code, rename to SafeBoxForest (SBF)`

### Phase 2：结构重组（1.5 天） ⚡ 高优先

- [ ] 拆分 `panda_planner.py` → `pipeline.py` + `viz/scene_viz.py` + `examples/demo_panda.py`
- [ ] 合并可视化代码 → `viz/` 模块
- [ ] 创建 `utils/` 模块（timing, output, seed）
- [ ] 更新所有 import 路径
- [ ] 运行全部测试 + 性能回归

### Phase 3：Baselines 统一接口（2 天） ⚡ 高优先

- [ ] 实现 `BasePlanner` ABC 和 `PlanningResult`（含 `supports_reuse` 属性）
- [ ] 封装 `SBFAdapter`（调用 `pipeline.py`，支持 reuse 模式）
- [ ] 重构 `rrt_family.py`（从 `rrt_family_panda.py` 提取，实现接口）
- [ ] 合并 OMPL 适配器 → `ompl_adapter.py`
- [ ] **扩展 `iris_gcs.py`**：将 `bench_rrt_vs_marcucci.py` 中 Marcucci-GCS 方法从 2-DOF 扩展到 7-DOF
- [ ] 为每个 baseline 写 smoke test

### Phase 4：实验框架（2.5 天） ⚡ 核心

- [ ] 实现 `ExperimentRunner`（含子进程隔离、统一计时）
- [ ] 编写实验 1: 主对比（最基础）
- [ ] **编写实验 2: Forest 复用**（★ 核心优势实验）
- [ ] **编写实验 3: 障碍物变化增量更新**（★ 核心优势实验）
- [ ] **编写实验 4: Cache 热启动**（★ 核心优势实验）
- [ ] 编写实验 5-8: 消融 / 配置 / 可扩展性 / AABB 紧致度
- [ ] 编写场景配置 JSON（标准实验场景集）
- [ ] 实现结果聚合 + LaTeX 表格生成 + 图表生成
- [ ] 端到端验证：从配置 → 运行 → 表格/图表

### Phase 5：文档整合（1 天）

- [ ] 合并三层算法文档 → `algorithm.md`（使用 SafeBoxForest 命名）
- [ ] 精简 `notation.md`
- [ ] 编写 `experiment_guide.md`
- [ ] 更新 `README.md`（SafeBoxForest Planner 介绍）
- [ ] 存档 v2 历史文档

### Phase 6：验证与打磨（1 天）

- [ ] 全量测试 + 性能回归 通过
- [ ] Cython 编译验证
- [ ] 运行一组完整实验（至少实验 1-4）验证输出
- [ ] 代码审查：删除 TODO/FIXME/HACK
- [ ] 验证 HCACHE 格式前后兼容
- [ ] 最终提交

---

### 时间估计

| Phase | 耗时 | 累积 |
|---|---|---|
| Phase 0: 准备 + 基准采集 | 1 天 | 1 天 |
| Phase 1: 清理废弃 + 重命名 | 1 天 | 2 天 |
| Phase 2: 结构重组 | 1.5 天 | 3.5 天 |
| Phase 3: Baselines | 2 天 | 5.5 天 |
| Phase 4: 实验框架 | 2.5 天 | 8 天 |
| Phase 5: 文档 | 1 天 | 9 天 |
| Phase 6: 验证 | 1 天 | 10 天 |
| **总计** | | **~10 个工作日** |

---

### 关键风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| **性能回归** | 重构导致算法变慢 | Phase 0 建立基准，每阶段必须通过 `perf_regression.py` |
| Drake IRIS+GCS 7-DOF 扩展难度 | 可能需要额外调研 IRIS region 生成 | 先使用 SBF 的 forest 作为 GCS 输入（已有先例），再尝试原生 IRIS |
| OMPL WSL 桥不稳定 | 影响 RRT 对比实验 | 优先用纯 Python RRT 家族；OMPL 作为补充 |
| 碰撞检测计数不公平 | 影响论文结论 | 统一使用 `check_config_collision()` 单次语义计数 |
| HCACHE 格式兼容性 | 重命名后旧 cache 不可读 | 保持 HCACHE02 格式不变，仅改 Python API 名称 |
| Cython 编译兼容性 | 跨平台问题 | 保持纯 Python fallback，Cython 仅作加速 |

---

> **下一步**：确认本计划后，从 Phase 0 开始执行。建议每个 Phase 完成后做一次 checkpoint 确认。
