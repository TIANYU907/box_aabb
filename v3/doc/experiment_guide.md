# 实验复现指南（Experiment Reproduction Guide）

> SafeBoxForest (SBF) v3 — 8 个实验的配置、运行与结果解读

---

## 1. 环境配置

### 1.1 基础依赖

```bash
conda create -n box-rrt python=3.10
conda activate box-rrt
pip install numpy scipy matplotlib plotly pytest
```

### 1.2 Cython 加速（可选）

```bash
cd v3
python setup_cython.py build_ext --inplace
```

### 1.3 可选依赖

| 依赖 | 用于 | 安装 |
|---|---|---|
| pydrake | IRIS-GCS baseline, GCS 优化 | `pip install drake` |
| cvxpy | GCS fallback 优化 | `pip install cvxpy` |
| pandas | 结果分析 | `pip install pandas` |
| OMPL | OMPL baseline (WSL) | 需要 WSL + pip install ompl |

### 1.4 确认安装

```bash
cd v3
python -m pytest tests/ -v  # 应全部通过
```

---

## 2. 快速运行全部实验

```bash
cd v3

# Quick 模式（几分钟内完成，少量 seed/trial）
python -m experiments.exp1_main_comparison --quick
python -m experiments.exp2_forest_reuse --quick
python -m experiments.exp3_obstacle_change --quick
python -m experiments.exp4_cache_warmstart --quick
python -m experiments.exp5_ablation --quick
python -m experiments.exp6_config_sweep --quick
python -m experiments.exp7_scalability --quick
python -m experiments.exp8_aabb_tightness --quick

# 完整模式（耗时较长，用于生成论文数据）
python -m experiments.exp1_main_comparison
python -m experiments.exp2_forest_reuse
# ... 等
```

---

## 3. 实验说明

### 实验 1：主对比 (Main Comparison)

**文件**: `experiments/exp1_main_comparison.py`

**对比方法**: SBF-Dijkstra, RRTConnect, RRT*, Informed-RRT*, BiRRT*

**场景**: Panda 7-DOF × {8, 15, 20} 障碍物

**完整运行**: 50 seeds × 3 trials = 150 runs per (method × scene)

**输出**: `experiments/output/raw/exp1_main_comparison.json`

**论文映射**: Table 1, Figure 2

---

### 实验 2：Forest 复用 (Forest Reuse) ★

**文件**: `experiments/exp2_forest_reuse.py`

**核心论点**: SBF 首次 build 后多次 plan() 复用 forest，amortized cost 远低于重建类方法

**设计**: K ∈ {1, 5, 10, 20, 50, 100} 个不同起终点查询

**关键指标**: 累积总耗时、每查询平均耗时、SBF 加速比

**输出**: `experiments/output/raw/exp2_forest_reuse.json`

**论文映射**: Table 2, Figure 3

---

### 实验 3：障碍物增量更新 (Obstacle Change) ★

**文件**: `experiments/exp3_obstacle_change.py`

**核心论点**: SBF 支持增量更新，动态场景优于全量重建

**变化类型**: 移除 / 新增 / 移动 × {1, 2, 3} 个障碍物

**输出**: `experiments/output/raw/exp3_obstacle_change.json`

**论文映射**: Table 3, Figure 4

---

### 实验 4：缓存热启动 (Cache Warmstart) ★

**文件**: `experiments/exp4_cache_warmstart.py`

**核心论点**: HCACHE 持久化跳过 FK 计算，大幅加速二次运行

**设计**: 冷启动 vs 热启动 vs 跨场景 cache 复用

**输出**: `experiments/output/raw/exp4_cache_warmstart.json`

**论文映射**: Table 4, Figure 5

---

### 实验 5：消融 (Ablation Study)

**文件**: `experiments/exp5_ablation.py`

**变体**: Full pipeline / No wavefront / No GCS refine / Small forest / Large forest

**输出**: `experiments/output/raw/exp5_ablation.json`

**论文映射**: Table 5, Figure 6

---

### 实验 6：配置敏感性 (Config Sweep)

**文件**: `experiments/exp6_config_sweep.py`

**扫描参数**: max_boxes, min_box_size, goal_bias, boundary_expand_*

**输出**: `experiments/output/raw/exp6_config_sweep.json`

**论文映射**: Figure 7

---

### 实验 7：可扩展性 (Scalability)

**文件**: `experiments/exp7_scalability.py`

**维度**: 障碍物数 {2,4,8,12,16,20} × DOF {2,3,7}

**输出**: `experiments/output/raw/exp7_scalability.json`

**论文映射**: Figure 8

---

### 实验 8：AABB 紧致度 (AABB Tightness)

**文件**: `experiments/exp8_aabb_tightness.py`

**对比**: 区间 FK AABB 体积 vs 数值采样紧致体积

**输出**: `experiments/output/raw/exp8_aabb_tightness.json`

**论文映射**: Table 6

---

## 4. 结果分析与报告生成

```python
# 生成 LaTeX 表格 + 图表
python -m experiments.reporting experiments/output/raw/exp1_main_comparison.json
```

输出目录结构：
```
experiments/output/
├── raw/           # JSON 原始数据
├── tables/        # LaTeX 表格 (.tex)
└── figures/       # 图表 (.pdf)
```

### 自定义分析

```python
from experiments.runner import ExperimentResults

results = ExperimentResults.load("experiments/output/raw/exp1_main_comparison.json")

# Pandas DataFrame
df = results.summary_df()

# 分组统计
stats = results.grouped_stats(("scene", "planner"), "planning_time")
```

---

## 5. 场景配置

标准场景定义在 `experiments/configs/scenes/standard_scenes.json`，包含：

| 场景 | 说明 |
|---|---|
| `panda_8obs_open` | Panda 7-DOF, 8 障碍物, 开阔 |
| `panda_15obs_moderate` | Panda 7-DOF, 15 障碍物, 中等 |
| `panda_20obs_dense` | Panda 7-DOF, 20 障碍物, 密集 |
| `2dof_simple` | 2-DOF 平面, 无障碍 (单测用) |
| `2dof_with_obstacle` | 2-DOF 平面, 有障碍 |
| `panda_8obs_reuse` | Panda 7-DOF, 10 query pairs (复用实验) |

### 自定义场景

```json
{
  "name": "my_scene",
  "robot": "panda",
  "obstacles": [
    {"min": [0.3, -0.2, 0.1], "max": [0.5, 0.2, 0.5], "name": "box1"}
  ],
  "query_pairs": [
    {"start": [0.5, -1.2, 0.5, -2.5, 0.5, 0.8, 1.5],
     "goal":  [-2.0, 1.2, -1.8, -0.5, -2.0, 3.0, -1.8]}
  ]
}
```

---

## 6. Planner 配置

标准 planner 定义在 `experiments/configs/planners/standard_planners.json`。

通过 `BasePlanner` 统一接口，所有方法使用相同的 `setup() → plan()` 调用链：

```python
from baselines import SBFAdapter, RRTPlanner

planner = SBFAdapter(method="dijkstra")
planner.setup(robot, scene, {"no_cache": True, "max_boxes": 500})
result = planner.plan(q_start, q_goal, timeout=30.0)
print(result.success, result.planning_time, result.cost)
```
