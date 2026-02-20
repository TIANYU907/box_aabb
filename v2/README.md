# box-aabb v2

基于区间 AABB 包络的 box-guided 运动规划框架。

## 架构

三层模块化设计，各层独立可测、向上组合：

```
aabb/     ← 区间 AABB 计算（几何包络层）
forest/   ← 无重叠 box 森林与碰撞核心（离散拓扑层）
planner/  ← 路径规划、搜索与优化（规划层）
common/   ← 共享工具（输出目录等）
```

## 核心模块简介

### aabb（`v2/src/aabb/`）

- **`robot.py`**：DH 模型、FK、批量 FK、Cython 加速（可选）
- **`interval_fk.py`**：区间三角函数包络、区间变换链、全量/增量 interval AABB
- **`calculator.py`**：`AABBCalculator` 统一调度入口
- **`strategies/`**：`CriticalStrategy`（关键点优先）、`RandomStrategy`（随机覆盖）
- **`optimization.py`**：L-BFGS-B 边界极值精化

### forest（`v2/src/forest/`）

- **`box_forest.py`**：`BoxForest` — 无重叠 box 集合 + 邻接图 + 区间缓存 + KDTree 加速
- **`collision.py`**：`CollisionChecker` — 单点 / 区间 / 线段 / 批量碰撞检测，内置 `SpatialIndex` 加速
- **`hier_aabb_tree.py`**：`HierAABBTree` — KD 层级 AABB 树，支持 `active_split_dims` 与增量 FK
- **`deoverlap.py`**：向量化邻接检测 `compute_adjacency` / `compute_adjacency_incremental` + 共享面计算
- **`connectivity.py`**：`UnionFind` + `find_islands` + `bridge_islands`（支持周期关节空间）
- **`coarsen.py`**：`coarsen_forest` — 维度扫描合并、两阶段检测+执行
- **`parallel_collision.py`**：`ParallelCollisionChecker` + `SpatialIndex`
- **`scene.py`**：`Scene` 障碍物管理（JSON 序列化）

### planner（`v2/src/planner/`）

- **`box_planner.py`**：`BoxPlanner`（别名 `BoxRRT`）— 端到端 box-guided 规划主流程
- **`box_query.py`**：`BoxForestQuery` — 轻量级只读查询规划器（已有 forest 直接搜索）
- **`connector.py`**：`TreeConnector` — 邻接边构建、端点接入、跨分区补边
- **`gcs_optimizer.py`**：`GCSOptimizer` — Dijkstra + 可选 Drake GCS + scipy waypoint 优化
- **`path_smoother.py`**：`PathSmoother` — box-aware shortcut / moving-average 平滑
- **`models.py`**：`PlannerConfig`（~40字段）、`PlannerResult`、`Edge`、`BoxTree`
- **`metrics.py`**：`PathMetrics` + `evaluate_result` + `compare_results`
- **`report.py`**：`PlannerReportGenerator` — Markdown 报告生成
- **`dynamic_visualizer.py`**：`animate_robot_path` — matplotlib 动画
- **`interactive_viewer.py`**：`launch_viewer` — TkAgg 交互式 3D 查看器

## 快速开始

### 安装

```bash
pip install -e ./v2
```

### 运行测试

```bash
python -m pytest v2/tests/ -v --tb=short
```

### 最简示例（2-DOF 规划）

```python
import numpy as np
from v2._bootstrap import add_v2_paths; add_v2_paths()
from aabb.robot import load_robot
from forest.scene import Scene
from planner.models import PlannerConfig
from planner.box_planner import BoxPlanner

robot = load_robot("2dof_planar")
scene = Scene()
scene.add_obstacle([0.6, -0.2], [0.9, 0.2], "obs")

planner = BoxPlanner(robot, scene, PlannerConfig(max_iterations=50, max_box_nodes=40))
result = planner.plan(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), seed=42)
print(f"success={result.success}, path_len={result.path_length:.3f}")
```

## 示例脚本

| 脚本 | 说明 |
|---|---|
| `v2/examples/aabb_demo.py` | AABB 包络计算演示 |
| `v2/examples/forest_demo.py` | BoxForest 构建演示 |
| `v2/examples/planning_demo.py` | 2-DOF 基础规划演示 |
| `v2/examples/panda_planner.py` | Panda 7-DOF 端到端管线（grow + coarsen + bridge + Dijkstra + waypoint） |
| `v2/examples/compare_all_planners.py` | BoxPlanner vs OMPL (RRT/RRTConnect/RRT*/InformedRRT*/BIT*) 统一对比 |
| `v2/examples/visualize_random_2dof_forest_expansion.py` | Box 扩展可视化 |
| `v2/examples/ompl_bridge.py` | OMPL 接口桥接 |
| `v2/examples/rrt_family_panda.py` | Panda RRT 系列演示 |
| `v2/examples/gcs_planner_2dof.py` / `gcs_planner_panda.py` | GCS 优化管线 |
| `v2/examples/bench_forest_reuse.py` / `bench_obstacle_change.py` | Forest 复用与场景切换 |

## Benchmark

### Forest 基准

```bash
python -m v2.benchmarks.forest.bench_panda_forest
python -m v2.benchmarks.forest.bench_panda_multi
python -m v2.benchmarks.forest.bench_adjacency_vectorized
python -m v2.benchmarks.forest.bench_nearest_kdtree
python -m v2.benchmarks.forest.bench_promotion_sweep
```

### Planner 基准（BoxPlanner vs RRT vs Marcucci-GCS）

```bash
python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot 2dof_planar --trials 8
python -m v2.benchmarks.planner.bench_gcs_fallback
```

### 一键运行

```powershell
./v2/scripts/run_benchmark_oneclick.ps1 -Mode conda -EnvName box-rrt -Trials 8
```

自动安装依赖 → 运行基准 → 生成汇总：`v2/output/benchmarks/planner_rrt_vs_marcucci_latest_summary.md`

### 环境安装

```powershell
# conda（推荐）
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -RunSmokeTest

# 新建环境
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -CreateEnv -PythonVersion 3.10 -RunSmokeTest

# pip
./v2/scripts/setup_benchmark_env.ps1 -Mode pip -RunSmokeTest
```

### 手动安装

```powershell
# conda
conda install -n box-rrt -c conda-forge numpy scipy matplotlib pytest ompl drake -y
conda run -n box-rrt python -m pip install -e ./v2

# pip
pip install numpy scipy matplotlib pytest
pip install -e ./v2
```

### 可选外部依赖

- `ompl`：RRT / RRTConnect / RRTstar / InformedRRTstar / BITstar 基线（benchmark 对比用）
- `pydrake`：Marcucci-style GCS 优化分支（benchmark 对比用）

缺少时 benchmark 对应方法直接跳过或报错，核心 BoxPlanner 无需这些依赖。

## 文档

| 文档 | 内容 |
|---|---|
| [术语与符号表](doc/terminology_notation.md) | 统一符号与名词定义（建议首先阅读） |
| [v2 论文目录页](doc/v2_algorithm_details.md) | 三篇算法分册的导航入口 |
| [AABB 模块说明](doc/aabb.md) | AABB 层架构与配置建议 |
| [AABB 算法细节](doc/aabb_algorithm_details.md) | Interval-FK、数值采样、L-BFGS-B 优化 |
| [Forest 模块说明](doc/forest.md) | Forest 层架构与关键接口 |
| [Forest 算法细节](doc/forest_algorithm_details.md) | BoxForest、邻接、碰撞、HierAABBTree、Coarsen |
| [Planner 模块说明](doc/planner.md) | Planner 层架构与管线说明 |
| [Planner 算法细节](doc/planner_algorithm_details.md) | BoxPlanner 管线、并行扩展、桥接、Panda 管线 |
| [Benchmark 说明](doc/benchmark_rrt_vs_marcucci.md) | BoxPlanner vs RRT vs GCS 对比方法与复现 |
| [性能分析报告](doc/v2_performance_analysis_report.md) | V1 vs V2 性能对比分析 |
| [改进思路](doc/改进思路.md) | 算法改进计划与当前进展 |

## 测试结构

```
v2/tests/
├── conftest.py                       # 共享 fixtures
├── aabb/                             # AABB 层单元测试
│   ├── test_calculator_no_hybrid.py
│   ├── test_interval_fk_fast_alias.py
│   └── test_robot_fk_batch.py
├── forest/                           # Forest 层单元测试
│   ├── test_adjacency_vectorized.py
│   ├── test_box_forest_basic.py
│   ├── test_collision_basic.py
│   └── test_hier_aabb_tree_split_dims.py
└── planner/                          # Planner 层单元测试
    ├── test_boundary_expand.py
    ├── test_box_query_basic.py
    ├── test_connector_closest_pairs.py
    ├── test_gcs_fallback_basic.py
    ├── test_metrics_report_basic.py
    ├── test_path_smoother_basic.py
    └── test_seed_sampling_batch.py
```

当前状态：全部测试通过。
