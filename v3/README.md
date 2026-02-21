# SafeBoxForest Planner (SBF) — v3

基于区间 AABB 包络的 SafeBoxForest 运动规划框架。

## 架构

三层模块化设计，各层独立可测、向上组合：

```
aabb/       ← 区间 AABB 计算（几何包络层）
forest/     ← SafeBoxForest 无重叠 box 森林（离散拓扑层）
planner/    ← SBFPlanner 路径规划与优化（规划层）
baselines/  ← 统一 baseline 接口（RRT / OMPL / IRIS-GCS）
```

辅助模块：

```
viz/        ← 可视化（AABB / 场景 / 路径动画 / 交互 3D）
experiments/← 8 个论文实验 + runner + reporting
utils/      ← 共享工具
```

## 与 v2 的主要变化

| 变更 | 说明 |
|---|---|
| 统一命名 SBF | BoxPlanner→SBFPlanner, BoxForest→SafeBoxForest, PlannerConfig→SBFConfig 等 |
| 模块精简 | 删除 14 个已过时文件，移除 common/ |
| 管线拆分 | panda_planner.py → pipeline.py + viz/scene_viz.py + examples/demo_panda.py |
| baselines 统一接口 | BasePlanner ABC + PlanningResult, 5 个 adapter |
| 实验框架 | ExperimentRunner + 8 个论文实验脚本 + reporting |
| 文档合并 | 7 篇 → 3 篇（algorithm.md, notation.md, experiment_guide.md） |

## 核心模块

### aabb（`src/aabb/`）

- **`robot.py`** — DH 模型、FK / 批量 FK、Cython 可选加速
- **`interval_fk.py`** — 区间三角函数包络、区间变换链、interval AABB
- **`calculator.py`** — `AABBCalculator` 统一入口
- **`strategies/`** — `CriticalStrategy`（关键点优先）/ `RandomStrategy`（随机覆盖）
- **`optimization.py`** — L-BFGS-B 边界极值精化

### forest（`src/forest/`）

- **`safe_box_forest.py`** — `SafeBoxForest` — 无重叠 box 集合 + 邻接图 + 缓存 + KDTree
- **`collision.py`** — `CollisionChecker` — 单点/区间/线段/批量碰撞检测 + SpatialIndex
- **`hier_aabb_tree.py`** — `HierAABBTree` — KD 层级 AABB 树
- **`deoverlap.py`** — 向量化邻接检测 + 共享面计算
- **`connectivity.py`** — `UnionFind` + `find_islands` + `bridge_islands`
- **`coarsen.py`** — `coarsen_forest` — 维度扫描合并
- **`scene.py`** — `Scene` 障碍物管理

### planner（`src/planner/`）

- **`sbf_planner.py`** — `SBFPlanner` — 端到端 box-guided 规划主流程
- **`sbf_query.py`** — `SBFQuery` — 轻量级只读查询（复用 forest）
- **`connector.py`** — `TreeConnector` — 邻接边构建与跨分区补边
- **`gcs_optimizer.py`** — `GCSOptimizer` — Dijkstra + Drake GCS + scipy 优化
- **`path_smoother.py`** — `PathSmoother` — box-aware shortcut / moving-average 平滑
- **`models.py`** — `SBFConfig`（~40字段）/ `SBFResult` / `Edge` / `BoxTree`
- **`pipeline.py`** — Panda 7-DOF 端到端管线（grow + coarsen + bridge + optimize）

### baselines（`src/baselines/`）

- **`base.py`** — `BasePlanner` ABC + `PlanningResult` dataclass
- **`sbf_adapter.py`** — `SBFAdapter`（Dijkstra / GCS / VisGraph 方法）
- **`rrt_family.py`** — `RRTPlanner`（RRT / RRTConnect / RRT* / InformedRRT* / BiRRT*）
- **`ompl_adapter.py`** — `OMPLPlanner`（WSL 子进程桥接）
- **`iris_gcs.py`** — `IRISGCSPlanner`（Drake IRIS + GCS）

## 快速开始

### 安装

```bash
conda create -n box-rrt python=3.10
conda activate box-rrt
pip install numpy scipy matplotlib pytest
pip install -e ./v3
```

### Cython 加速（可选）

```bash
cd v3
python setup_cython.py build_ext --inplace
```

### 运行测试

```bash
python -m pytest v3/tests/ -v --tb=short
```

### 最简示例（2-DOF）

```python
import numpy as np
from v3._bootstrap import add_v3_paths; add_v3_paths()
from aabb.robot import load_robot
from forest.scene import Scene
from planner.models import SBFConfig
from planner.sbf_planner import SBFPlanner

robot = load_robot("2dof_planar")
scene = Scene()
scene.add_obstacle([0.6, -0.2], [0.9, 0.2], "obs")

planner = SBFPlanner(robot, scene, SBFConfig(max_iterations=50, max_box_nodes=40))
result = planner.plan(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), seed=42)
print(f"success={result.success}, path_len={result.path_length:.3f}")
```

### Panda 7-DOF 管线

```python
from v3._bootstrap import add_v3_paths; add_v3_paths()
from planner.pipeline import PandaGCSConfig, build_panda_scene, grow_and_prepare, run_method_with_bridge

config = PandaGCSConfig()
robot, scene = build_panda_scene(n_obstacles=8)
forest_data = grow_and_prepare(robot, scene, config)
result = run_method_with_bridge(forest_data, robot, scene, config)
```

### Baseline 对比

```python
from baselines import SBFAdapter, RRTPlanner

# SBF
sbf = SBFAdapter(method="dijkstra")
sbf.setup(robot, scene, {"max_boxes": 500})
r1 = sbf.plan(q_start, q_goal, timeout=30.0)

# RRT-Connect
rrt = RRTPlanner(algorithm="rrt_connect")
rrt.setup(robot, scene, {"max_nodes": 5000})
r2 = rrt.plan(q_start, q_goal, timeout=30.0)

print(f"SBF: {r1.planning_time:.3f}s  RRT: {r2.planning_time:.3f}s")
```

## 实验

8 个论文实验覆盖：主对比、forest 复用、增量更新、缓存热启动、消融、配置敏感性、可扩展性、AABB 紧致度。

```bash
# Quick 模式（几分钟）
python -m experiments.exp1_main_comparison --quick

# 完整模式
python -m experiments.exp1_main_comparison
```

详见 [doc/experiment_guide.md](doc/experiment_guide.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [算法说明](doc/algorithm.md) | 三层算法（AABB → Forest → Planner）统一描述 |
| [术语与符号](doc/notation.md) | 统一符号表、缩写表、代码对象映射 |
| [实验指南](doc/experiment_guide.md) | 8 个实验的配置、运行与结果解读 |

## 项目结构

```
v3/
├── src/
│   ├── aabb/           # 区间 AABB 计算
│   ├── forest/         # SafeBoxForest
│   ├── planner/        # SBFPlanner + pipeline
│   ├── baselines/      # 统一 baseline 接口
│   └── utils/          # 共享工具
├── tests/
│   ├── aabb/           # AABB 层测试
│   ├── forest/         # Forest 层测试
│   ├── planner/        # Planner 层测试
│   ├── baselines/      # Baseline 冒烟测试
│   ├── conftest.py     # 共享 fixtures
│   └── perf_regression.py  # 性能回归测试
├── experiments/
│   ├── exp1~exp8       # 8 个论文实验
│   ├── runner.py       # ExperimentRunner
│   ├── reporting.py    # LaTeX / matplotlib 报告
│   └── configs/        # 场景 & planner 配置
├── viz/                # 可视化模块
├── examples/           # 示例脚本
├── doc/                # 文档
├── pyproject.toml
└── setup_cython.py
```

## 测试

```bash
# 全量测试
python -m pytest v3/tests/ -v

# 性能回归
python -m pytest v3/tests/perf_regression.py -v

# Baseline 冒烟测试
python -m pytest v3/tests/baselines/ -v
```

当前状态：53 tests passed, 1 skipped (OMPL WSL)。
