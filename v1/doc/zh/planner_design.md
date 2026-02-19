# Box-RRT 路径规划器 — 设计与实现文档

> **版本**: v4.1.0 &nbsp;|&nbsp; **日期**: 2026-02-10 &nbsp;|&nbsp; **机器人支持**: 2/3DOF 平面 + Panda 7DOF

---

## 目录

1. [概述](#1-概述)
2. [系统架构](#2-系统架构)
3. [核心算法流程](#3-核心算法流程)
4. [模块详解](#4-模块详解)
   - 4.1 [数据模型 (models.py)](#41-数据模型-modelspy)
   - 4.2 [碰撞检测 (collision.py)](#42-碰撞检测-collisionpy)
   - 4.3 [Box 拓展 (hier_aabb_tree.py)](#43-box-拓展-hier_aabb_treepy)
   - 4.4 [Box 树管理 (box_tree.py)](#44-box-树管理-box_treepy)
   - 4.5 [主规划器 (box_rrt.py)](#45-主规划器-box_rrtpy)
   - 4.6 [树间连接 (connector.py)](#46-树间连接-connectorpy)
   - 4.7 [路径后处理 (path_smoother.py)](#47-路径后处理-path_smootherpy)
   - 4.8 [GCS 优化 (gcs_optimizer.py)](#48-gcs-优化-gcs_optimizerpy)
   - 4.9 [障碍物场景 (obstacles.py)](#49-障碍物场景-obstaclespy)
   - 4.10 [评价指标 (metrics.py)](#410-评价指标-metricspy)
   - 4.11 [可视化 (visualizer.py)](#411-可视化-visualizerpy)
   - 4.12 [并行碰撞检测 (parallel_collision.py)](#412-并行碰撞检测-parallel_collisionpy)
5. [v4.0 新增功能](#5-v40-新增功能)
   - 5.1 [可复用 Box 森林 (box_forest.py)](#51-可复用-box-森林-box_forestpy)
   - 5.2 [森林查询规划 (box_query.py)](#52-森林查询规划-box_querypy)
   - 5.3 [动态可视化 (dynamic_visualizer.py)](#53-动态可视化-dynamic_visualizerpy)
   - 5.4 [自由空间瓦片化 (free_space_tiler.py)](#54-自由空间瓦片化-free_space_tilerpy)
   - 5.5 [阈值实验 (threshold_experiment.py)](#55-阈值实验-threshold_experimentpy)
6. [底层数学支撑](#6-底层数学支撑)
   - 6.1 [仿射算术 (interval_math.py)](#61-仿射算术-interval_mathpy)
   - 6.2 [区间正运动学 (interval_fk.py)](#62-区间正运动学-interval_fkpy)
7. [关键设计决策](#7-关键设计决策)
8. [使用示例](#8-使用示例)
9. [测试覆盖](#9-测试覆盖)
10. [性能数据](#10-性能数据)

---

## 1. 概述

### 1.1 背景

在机械臂运动规划中，需要找到一条从起始关节配置 $q_{start}$ 到目标配置 $q_{goal}$ 的无碰撞路径。传统方法如 RRT/PRM 在高维空间中收敛慢，且不具备最优性保证。

本项目实现的 **Box-RRT** 算法，借鉴了 Marcucci et al. 在 *Science Robotics* (2023) 上提出的 **Graph of Convex Sets (GCS)** 框架思想：

- 在关节空间（C-space）中将无碰撞区域分解为一组**凸超矩形（box）**
- 调用 **`box_aabb`** 库提供的 **区间/仿射算术** 和 **区间正运动学** 保守验证 box 的无碰撞性
- 在 box 之间建立连接构成 **凸集图**
- 在图上搜索/优化"经过凸集序列"的最短路径

> **架构关系**: `planner` 是独立于 `box_aabb` 的 Python 包。planner 通过 `from box_aabb.robot import Robot` 和 `from box_aabb.interval_fk import compute_interval_aabb` 调用 box_aabb 的机器人模型和区间 FK 能力。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| 保守碰撞检测 | 基于仿射算术的区间 FK，确保 `check_box_collision=False` 时 box 绝对无碰撞 |
| 采样辅助拓展 | 高自由度（>4DOF）时自动启用 hybrid 模式：区间 FK 判碰撞后用采样复核 |
| 平衡拓展策略 | v4.1 默认 balanced 策略，交替步进避免 box 细长（宽度比中位数 4.1 vs greedy 48.5） |
| 启发式维度排序 | 按 Jacobian 列范数排序，优先拓展对末端影响小的关节 |
| 多树并行生长 | 支持多棵 box tree 独立生长后连接 |
| GCS 优化 | 支持 Drake GCS 凸优化（可选），默认 Dijkstra + scipy fallback |
| 路径后处理 | Shortcut 优化 + 移动平均平滑，保证平滑后仍无碰撞 |
| AABB 缓存 | v4.1 全局接入 BoxRRT/BoxForest/动画脚本，避免重复区间 FK |
| JSON 配置 | PlannerConfig 支持 JSON 序列化，预设 2dof/panda 配置文件 |

---

## 2. 系统架构

```
src/
├── box_aabb/                # 区间 AABB 计算库（被 planner 调用）
│   ├── __init__.py
│   ├── aabb_calculator.py
│   ├── interval_fk.py       # 区间正运动学
│   ├── interval_math.py     # 仿射算术引擎
│   ├── models.py
│   ├── optimization.py
│   ├── report.py
│   ├── robot.py             # Modified DH 机器人模型
│   ├── visualizer.py
│   ├── configs/
│   └── strategies/
└── planner/                 # ★ 独立路径规划包（调用 box_aabb）
    ├── __init__.py
    ├── models.py            # 数据模型 (BoxNode, BoxTree, Edge, ...)
    ├── obstacles.py         # 障碍物场景管理 (Scene)
    ├── collision.py         # 碰撞检测 (调用 box_aabb.interval_fk)
    ├── hier_aabb_tree.py    # ★ 层级 AABB 缓存树（Box 拓展）
    ├── box_tree.py          # Box 树管理
    ├── box_rrt.py           # ★ 主规划器入口 (BoxRRT)
    ├── connector.py         # 树间/始末点连接
    ├── path_smoother.py     # 路径 shortcut + 平滑
    ├── gcs_optimizer.py     # GCS 凸优化 / Dijkstra fallback
    ├── metrics.py           # 路径评价指标
    ├── visualizer.py        # C-space / workspace 可视化
    ├── parallel_collision.py # 并行碰撞检测 + 空间索引
    ├── box_forest.py        # ★ 可复用 BoxForest (v4.0)
    ├── box_query.py         # Forest 查询规划 (v4.0)
    ├── dynamic_visualizer.py # 动态可视化动画 (v4.0)
    └── free_space_tiler.py  # 自由空间瓦片化 (v4.0)
```

**依赖关系图**（planner 调用 box_aabb）:

```
  planner 包                              box_aabb 包
  ─────────                              ──────────
                     BoxRRT (box_rrt.py)
                    /    |    |    \     \
        HierAABBTree  TreeMgr Connector  Smoother  GCSOptimizer
              |         |        |          |           |
       CollisionChecker  BoxTree  CollisionChecker  Dijkstra/Drake
              |
              ├─── box_aabb.interval_fk.compute_interval_aabb()
              └─── box_aabb.robot.Robot
                        |
                   box_aabb.interval_math (AffineForm, smart_sin/cos)
```

---

## 3. 核心算法流程

### 3.1 端到端流程

```
plan(q_start, q_goal)
│
├─ Step 0: 验证始末点无碰撞
│
├─ Step 0.5: 尝试直连（线段碰撞检测）
│   └─ 直连成功 → 返回 [q_start, q_goal]
│
├─ Step 1: 从始末点创建初始 box tree
│   ├─ expand(q_start) → tree_0
│   └─ expand(q_goal)  → tree_1
│
├─ Step 2: 主采样循环 (max_iterations 次)
│   ├─ 采样 seed 点（goal_bias 概率偏向目标）
│   ├─ expand(seed) → new_box
│   ├─ 加入最近树 / 创建新树
│   ├─ 边界再采样并拓展（从叶子 box 边界出发）
│   └─ 定期桥接采样（窄通道策略）
│
├─ Step 3: 连接阶段
│   ├─ connect_within_trees()  → 树内重叠 box 连边
│   ├─ connect_between_trees() → 树间尝试线段连接
│   └─ connect_endpoints()     → 始末点连接到 box graph
│
├─ Step 4: 图搜索
│   ├─ Dijkstra 最短路径 / Drake GCS 凸优化
│   └─ 连通性修复（BFS 检测不可达，桥接断开区域）
│
└─ Step 5: 路径后处理
    ├─ shortcut 优化（随机选两点尝试直连）
    └─ 移动平均平滑（保持碰撞安全）
```

### 3.2 Box 拓展算法

单个 box 的拓展过程（通过 `HierAABBTree.find_free_box(seed, obstacles)`）:

```
1. 从根节点出发，递归搜索包含 seed 的叶节点
2. 对叶节点执行碰撞检测（区间 FK + AABB 重叠检测）
3. 若叶节点无碰撞:
     找到包含 seed、彼此排列组合的最大合并 box
     标记已用叶节点为 occupied，避免重复分配
     返回 joint_intervals
4. 若叶节点有碰撞:
     返回 None（seed 不在自由空间中）
```

**碰撞检测**（`CollisionChecker.check_box_collision`）:
```
interval_result = check_box_collision(intervals)   // 区间 FK
if interval_result == False  →  一定安全
if interval_result == True   →  可能碰撞（区间过估导致误报）
```

---

## 4. 模块详解

### 4.1 数据模型 (models.py)

**文件**: `src/box_aabb/planner/models.py` — 297 行

定义 6 个核心数据类：

| 类名 | 说明 | 关键字段 |
|------|------|---------|
| `Obstacle` | 工作空间 AABB 障碍物 | `min_point`, `max_point`, `name` |
| `BoxNode` | C-space 无碰撞 box 节点 | `joint_intervals`, `seed_config`, `volume`, `tree_id` |
| `BoxTree` | Box 树结构 | `nodes: Dict[int, BoxNode]`, `root_id` |
| `Edge` | 两个 box 之间的连接边 | `source_box_id`, `target_box_id`, `source_config`, `target_config`, `cost` |
| `PlannerConfig` | 规划器参数配置 | 25+ 个可调参数（见下表），支持 JSON 文件加载 |
| `PlannerResult` | 规划结果 | `success`, `path`, `box_trees`, `edges`, `computation_time` |

**BoxNode 关键方法**:
- `contains(config)` — 检查配置是否在 box 内
- `distance_to_config(config)` — LΘ距离
- `overlap_with(other)` — 两个 box 是否有交集
- `nearest_point_to(config)` — box 内最近点
- `_compute_volume()` — 体积计算（跳过固定关节的零宽度维度）

**PlannerConfig 参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_iterations` | 500 | 最大采样迭代次数 |
| `max_box_nodes` | 200 | 最大 box 数量 |
| `seed_batch_size` | 5 | 每次边界重采样数量 |
| `min_box_volume` | 1e-6 | box 体积下限 |
| `goal_bias` | 0.1 | 目标偏向采样概率 |
| `expansion_resolution` | 0.01 | 二分搜索精度 (rad) |
| `max_expansion_rounds` | 3 | greedy 策略 box 拓展最大轮数 |
| `jacobian_delta` | 0.01 | Jacobian 数值差分步长 |
| `min_initial_half_width` | 0.001 | box 初始半宽 (rad) |
| `expansion_strategy` | `'balanced'` | 拓展策略：`'balanced'` 或 `'greedy'` |
| `balanced_step_fraction` | 0.5 | balanced 每步推进剩余空间的比例 |
| `balanced_max_steps` | 200 | balanced 策略最大步数 |
| `use_sampling` | None (自动) | hybrid 碰撞检测：`None`=自动, `True`=启用, `False`=关闭 |
| `sampling_n` | 80 | hybrid 碰撞检测采样数 |
| `segment_collision_resolution` | 0.05 | 线段碰撞检测间隔 |
| `connection_max_attempts` | 50 | 树间连接最大尝试次数 |
| `connection_radius` | 2.0 | 连接搜索半径 (rad) |
| `path_shortcut_iters` | 100 | Shortcut 优化迭代次数 |
| `use_gcs` | False | 是否使用 Drake GCS |
| `gcs_bezier_degree` | 3 | Bézier 曲线阶数 |
| `verbose` | False | 详细日志 |

**JSON 配置化** (v4.1):
```python
# 保存 / 加载
config.to_json("my_config.json")
config = PlannerConfig.from_json("src/planner/configs/panda.json")

# 预设配置: default.json, 2dof_planar.json, panda.json
```

---

### 4.2 碰撞检测 (collision.py)

**文件**: `src/box_aabb/planner/collision.py` — ~300 行

`CollisionChecker` 类封装了 4 种碰撞检测方法：

| 方法 | 输入 | 输出 | 原理 |
|------|------|------|------|
| `check_config_collision(q)` | 单个关节配置 | bool（True=碰撞） | FK → 逐 link AABB vs obstacle AABB |
| `check_box_collision(intervals)` | 关节区间列表 | bool（True=可能碰撞） | 区间 FK → 保守 AABB vs obstacle AABB |
| `check_box_collision_sampling(intervals)` | 关节区间 + 采样数 | bool（True=存在碰撞采样） | 在 box 内随机采样逐点检查 |
| `check_segment_collision(q1, q2)` | 两个配置 | bool | 等间隔采样逐点检查 |

**保守性保证**:
- `check_box_collision` 返回 **False** → 该 box 内 **所有** 配置一定无碰撞
- `check_box_collision` 返回 **True** → 可能碰撞（可能是区间过估导致误报）
- `check_box_collision_sampling` 返回 **False** → 概率性安全（80 个采样点均无碰撞）

**采样碰撞检测策略** (`check_box_collision_sampling`):
1. 先检查 box 中心点
2. 随机选取部分顶点（`min(2n, n_samples/3)` 个，n 为维度数）
3. 剩余采样点在 box 内均匀随机

---

### 4.3 Box 拓展 (hier_aabb_tree.py)

**文件**: `src/planner/hier_aabb_tree.py`

旧的 `BoxExpander`（greedy/balanced 策略）已被 **HierAABBTree** 完全替代。详见设计文档（design_document.md）第 7 节。

核心接口：`HierAABBTree.find_free_box(seed, obstacles, mark_occupied=True)` — 在层级 AABB 缓存树中查找包含 seed 的最大无碰撞 box，并标记为已占用。

---

### 4.4 Box 树管理 (box_tree.py)

**文件**: `src/box_aabb/planner/box_tree.py` — ~260 行

`BoxTreeManager` 类管理多棵 box tree 的生命周期：

| 方法 | 说明 |
|------|------|
| `create_tree(root_box)` | 创建新树 |
| `add_box(tree_id, box, parent_id)` | 向树添加子节点 |
| `find_containing_box(config)` | 查找包含配置的 box |
| `find_nearest_box(config)` | 查找最近的 box |
| `get_boundary_samples(tree_id, n)` | 在叶子 box 边界采样 |

**边界采样策略** (`get_boundary_samples`):
1. 以体积为权重选择叶子 box（大 box 更可能被选中）
2. 在选中 box 的边界面上均匀采样：
   - 随机选一个有效维度和一个方向（lo/hi）
   - 固定该维度到边界值
   - 其余维度在 box 内均匀随机

---

### 4.5 主规划器 (box_rrt.py)

**文件**: `src/box_aabb/planner/box_rrt.py` — ~600 行

`BoxRRT` 类是算法的总控制器，初始化并协调所有子模块。

**构造函数** 自动配置：
- `CollisionChecker` — 碰撞检测器
- `HierAABBTree` — 层级 AABB 缓存树（唯一 Box 拓展后端）
- `BoxTreeManager` — 树管理器
- `TreeConnector` — 树间连接器
- `PathSmoother` — 路径平滑器
- `GCSOptimizer` — GCS 优化器

**关键内部方法**:

| 方法 | 说明 |
|------|------|
| `_create_initial_trees()` | 从始末点拓展初始 box tree |
| `_sample_seed()` | 采样无碰撞 seed（goal_bias 偏向 + 均匀随机） |
| `_add_box_to_tree()` | 将 box 加入最近树或创建新树 |
| `_boundary_expand()` | 在叶子 box 边界上再采样拓展 |
| `_bridge_sampling()` | 桥接采样（在两棵树之间窄通道集中采样） |
| `_bridge_disconnected()` | 连通性修复（BFS 找不可达，逐轮桥接） |
| `_graph_search()` | 在邻接图上搜索路径 |

**采样策略**:
- **Goal bias**: 以概率 `goal_bias`（默认 0.1）在目标附近高斯采样
- **均匀随机**: 以概率 `1 - goal_bias` 在关节限制内均匀采样
- **桥接采样**: 每 10 次迭代触发，在两棵树最近 box 对的中间区域集中采样（窄通道突破）

**连通性修复** (`_bridge_disconnected`):
1. 从 start 做 BFS 找到可达节点集
2. 若 goal 不可达，找可达/不可达集合的最近 box 对
3. 尝试线段连接最近的 box 对
4. 最多 5 轮修复，每轮最多新增 20 条桥接边

---

### 4.6 树间连接 (connector.py)

**文件**: `src/box_aabb/planner/connector.py` — ~400 行

`TreeConnector` 类负责构建 box graph 的边：

**三种连接模式**:

1. **树内连接** (`connect_within_trees`):
   - 遍历同一棵树内的所有 box 对
   - 若两个 box 有交集 → 取交集中心作为连接点
   - 在 box 内的连接自然无碰撞

2. **树间连接** (`connect_between_trees`):
   - 先找跨树有交集的 box 对（直接连接）
   - 再找最近 k 对 box，在表面取最近点，线段碰撞检测验证
   - 每对树最多 3 条连接边

3. **始末点连接** (`connect_endpoints`):
   - 检查始末点是否在某个 box 内 → 直接返回
   - 否则找最近 box、线段验证
   - 连接失败时尝试备选 box

**邻接图构建** (`build_adjacency_graph`):
- 节点: box node_id + 特殊节点 `'start'`, `'goal'`
- 边: 连接关系 + 代价（L2 距离）
- 输出格式: `{nodes, edges: {id: [(neighbor, cost, edge), ...]}, start, goal}`

---

### 4.7 路径后处理 (path_smoother.py)

**文件**: `src/box_aabb/planner/path_smoother.py` — ~180 行

`PathSmoother` 类提供三种后处理操作：

| 方法 | 说明 |
|------|------|
| `shortcut(path, max_iters)` | 随机选两个非相邻点尝试直连，移除中间点 |
| `resample(path, resolution)` | 等间距重采样 |
| `smooth_moving_average(path, window, n_iters)` | 窗口移动平均，每步验证碰撞后回退 |

**Shortcut 算法**:
```
for _ in max_iters:
    随机选 i < j, j > i+1
    if segment(path[i], path[j]) 无碰撞:
        path = path[:i+1] + path[j:]  // 移除中间所有点
```

**移动平均平滑**: 保持首尾不变，中间点用窗口均值替代。平滑后若该点碰撞则回退到原值。

---

### 4.8 GCS 优化 (gcs_optimizer.py)

**文件**: `src/box_aabb/planner/gcs_optimizer.py` — ~360 行

`GCSOptimizer` 类支持两种模式：

**Drake GCS 模式** (需要 `pydrake`):
1. 为每个 box 创建 HPolyhedron vertex
2. 根据邻接关系添加 GCS edge
3. 在 edge 上添加 L2 代价 $\|x_u - x_v\|^2$
4. 添加连续性约束 $x_u = x_v$
5. 调用 `SolveShortestPath()` 求解凸松弛

**Fallback 模式** (默认):
1. **Dijkstra** 在邻接图上找最短路径
2. 将 box 序列转为路径点（每个 box 取距前一点最近的内部点）
3. **scipy L-BFGS-B** 局部优化：最小化路径长度，约束每个中间点在对应 box 内

---

### 4.9 障碍物场景 (obstacles.py)

**文件**: `src/box_aabb/planner/obstacles.py` — ~165 行

`Scene` 类管理工作空间中的 AABB 障碍物集合：

| 方法 | 说明 |
|------|------|
| `add_obstacle(min_pt, max_pt, name)` | 添加障碍物 |
| `remove_obstacle(name)` | 按名称删除 |
| `to_json(filepath)` / `from_json(filepath)` | JSON 序列化 |
| `from_dict(data)` / `from_obstacle_list(obstacles)` | 从字典构造 |

---

### 4.10 评价指标 (metrics.py)

**文件**: `src/box_aabb/planner/metrics.py` — ~340 行

`PathMetrics` 数据类包含：

| 指标 | 说明 |
|------|------|
| `path_length` | 路径总长度（关节空间 L2） |
| `length_ratio` | 路径长度 / 直线距离 |
| `smoothness` | 路径平滑度（角度变化方差） |
| `max_curvature` | 最大曲率 |
| `min_clearance` | 最小障碍物间隔 |
| `avg_clearance` | 平均障碍物间隔 |
| `joint_range_usage` | 各关节范围使用比例 |
| `n_waypoints` | 路径点数量 |
| `computation_time` | 计算时间 |

工具函数：
- `evaluate_result(result, robot, scene)` → `PathMetrics`
- `compare_results(results_dict)` → 多结果对比
- `format_comparison_table(metrics_dict)` → 格式化表格

---

### 4.11 可视化 (visualizer.py)

**文件**: `src/box_aabb/planner/visualizer.py` — ~370 行

提供 3 种可视化函数：

| 函数 | 说明 |
|------|------|
| `plot_cspace_boxes(result, dim_x, dim_y)` | C-space 2D 投影：box 矩形 + 路径曲线 |
| `plot_cspace_with_collision(robot, scene, ...)` | C-space 碰撞热力图 + box / 路径叠加 |
| `plot_workspace_result(robot, scene, result)` | 工作空间 2D/3D 机器人姿态序列 + 障碍物 |

自动检测机器人维度（2D 平面 vs 3D 空间）选择绘图方式。

---

### 4.12 并行碰撞检测 (parallel_collision.py)

**文件**: `src/box_aabb/planner/parallel_collision.py` — ~260 行

`ParallelCollisionChecker` 类使用线程池加速批量碰撞检测：

| 方法 | 说明 |
|------|------|
| `batch_check_configs(configs)` | 批量点碰撞检测 |
| `batch_check_boxes(box_intervals_list)` | 批量 box 碰撞检测 |
| `batch_check_segments(segments)` | 批量线段碰撞检测 |
| `filter_collision_free(configs)` | 筛选无碰撞配置 |

`SpatialIndex` 类实现简单的网格空间索引，以 cell_size 为粒度快速查询附近障碍物。

---

## 5. v4.0 新增功能

### 5.1 可复用 Box 森林 (box_forest.py)

**文件**: `src/planner/box_forest.py`

在场景中构建覆盖 C-free 的 box 森林，可持久化后在多次规划中复用。

**核心 API**:

| 方法 | 说明 |
|------|------|
| `BoxForest.build(robot, scene, config)` | 构建森林（无目标偏向，均匀采样） |
| `forest.save(filepath)` | 持久化到文件 (pickle) |
| `BoxForest.load(filepath, robot, scene)` | 从文件加载（验证机器人指纹） |

**构建策略**:
1. 均匀随机采样 `build_n_seeds` 个 seed 点（无 goal_bias）
2. 跳过已被现有 box 覆盖的 seed
3. 拓展 box，加入最近树或创建新树
4. 每个树做少量边界拓展（3 个采样点/棵）
5. 允许最多 `max_box_nodes × 3` 个 box（森林模式更宽松）

**机器人指纹验证**: 加载时校验 `robot.fingerprint()` (SHA256) 确保机器人一致性。

---

### 5.2 森林查询规划 (box_query.py)

**文件**: `src/planner/box_query.py`

基于已有 BoxForest 的快速查询规划，无需重新拓展 box。

```python
forest = BoxForest.load("forest.pkl", robot, scene)
query = BoxForestQuery(forest)
result = query.plan(q_start, q_goal, seed=42)
```

**查询流程**:
1. 验证始末点无碰撞
2. 尝试直连
3. 在始末点附近做少量补充拓展（`query_expand_budget` 个 box）
4. 复用森林的 TreeConnector 连接始末点
5. 图搜索 + 路径平滑

**优势**: 当场景不变、只换始末点时，无需重建 box 覆盖，查询时间远低于完整规划。

---

### 5.3 动态可视化 (dynamic_visualizer.py)

**文件**: `src/planner/dynamic_visualizer.py`

生成机械臂运动路径的动画，自动检测 2D/3D 机器人。

**核心 API**:

```python
from planner.dynamic_visualizer import animate_robot_path, resample_path

# 重采样使动画平滑
smooth_path = resample_path(result.path, n_frames=100)
# 生成动画
anim = animate_robot_path(robot, smooth_path, scene=scene,
                           fps=20, trail_length=30, ghost_interval=10)
# 保存
anim.save("robot.gif", writer='pillow', fps=20)
```

**功能**:
- 2D 平面和 3D 空间自动检测
- 末端执行器轨迹跟踪（可配置尾巴长度）
- 残影显示（ghost_interval 控制间隔）
- 障碍物叠加显示
- 帧计数器
- 等弧长路径重采样（`resample_path`）

---

### 5.4 自由空间瓦片化 (free_space_tiler.py)

**文件**: `src/planner/free_space_tiler.py`

给定 3D 障碍物，反选出尽可能大的无碰撞关节区间包络。

**核心策略**: 自适应单维度二分分割

```
recursive_tile(intervals, depth):
    if not collision(intervals):
        → 加入结果（整个区间无碰撞）
    elif depth >= max_depth:
        → 放弃（太深了）
    else:
        选最宽维度 dim
        mid = (lo[dim] + hi[dim]) / 2
        recursive_tile([lo, mid], depth+1)  # 子区间 1
        recursive_tile([mid, hi], depth+1)  # 子区间 2
```

**关键特性**:
- 每次只沿一个维度分割（避免 $2^N$ 爆炸）
- 优先分割最宽维度
- `min_width` 防止过小瓦片
- `max_depth` 控制递归深度
- 支持 Hybrid 碰撞检测（区间 FK + 采样复核）
- 结果可直接转为 `BoxNode` 列表

---

### 5.5 阈值实验 (threshold_experiment.py)

**文件**: `benchmarks/threshold_experiment.py`

确定区间方法与数值方法的 AABB 体积交叉阈值。

**实验流程**:
1. 固定基准配置，逐关节扫描区间宽度
2. 对每个宽度分别计算 interval FK 和 numerical sampling 的 AABB
3. 画出 体积-宽度 曲线
4. 找 interval 体积首次超过 numerical 2 倍的宽度作为阈值

**输出**: 每个关节的推荐阈值 + 全局中位阈值 + 对比图表

---

## 6. 底层数学支撑

### 6.1 仿射算术 (interval_math.py)

**文件**: `src/box_aabb/interval_math.py` — ~420 行

实现两层区间算术：

**Interval 类** — 经典区间算术 $[lo, hi]$：
- 四则运算：`+`, `-`, `*`, `/`
- 三角函数：`I_sin(x)`, `I_cos(x)`
- 特点：简单但会导致"dependency problem"（例如 $x - x \neq [0, 0]$）

**AffineForm 类** — 仿射算术 $\hat{x} = x_0 + \sum x_i \varepsilon_i$：
- 中心值 $x_0$ + 噪声符号系数 $\{i: x_i\}$
- 加减法：仿射组合，保留所有噪声符号
- **乘法（改进版）**: 保留线性项，二次余项用单个新符号吸收

$$x \cdot y = x_0 y_0 + x_0 \sum y_i \varepsilon_i + y_0 \sum x_i \varepsilon_i + \delta \cdot \varepsilon_{new}$$

其中 $\delta = \left(\sum |x_i|\right) \cdot \left(\sum |y_i|\right)$ 为二次余项上界。

- **smart_sin / smart_cos（Chebyshev 线性化）**: 对窄区间（< π/2）使用 Taylor 展开

$$\sin(\hat{x}) \approx \sin(x_0) + \cos(x_0) \cdot (\hat{x} - x_0) + \delta$$

$$|\delta| \leq \frac{r^2}{2}, \quad r = \sum |x_i|$$

保留所有噪声符号依赖，避免相关性丢失。宽区间时回退到普通 Interval 计算。

**为什么仿射算术很重要**:

在 8 次链式 DH 矩阵乘法中，普通区间算术的"wrapping effect"（包裹效应）导致指数级过估计。仿射算术通过追踪噪声符号相关性，使得相消项（如 $\sin^2\theta + \cos^2\theta - 1$）能被更紧地估计。

### 6.2 区间正运动学 (interval_fk.py)

**文件**: `src/box_aabb/interval_fk.py` — ~150 行

`compute_interval_aabb(robot, intervals, ...)` 函数实现：

1. 将每个关节区间 $[q_i^{lo}, q_i^{hi}]$ 构造为 `AffineForm`（每个关节独立噪声符号）
2. 计算 `smart_sin(q_i)`, `smart_cos(q_i)` — 保留噪声符号依赖
3. 按 Modified DH 参数构造 4×4 齐次变换矩阵（元素为 AffineForm）
4. 链式矩阵乘法 $T_1 \cdot T_2 \cdots T_n$
5. 从最终矩阵的平移分量中提取各 link 端点的区间包围盒

输出为每个 link 的保守 AABB `LinkAABBInfo`（包含 min_point、max_point）。

---

## 7. 关键设计决策

### 7.1 Hybrid 碰撞检测

**问题**: 对 Panda 7DOF 机器人，纯区间 FK 在 8 次矩阵链乘后严重过估计，导致 box 无法拓展（volume ≈ 0）。

**解决方案**: `BoxRRT` 构造时自动检测 DOF 数：
```python
_use_sampling = robot.n_joints > 4
```
高 DOF 时 `HierAABBTree` 配合 `CollisionChecker` 的区间 FK 碰撞检测，通过层级缓存树的细分来降低过估计影响。这种策略：
- 区间 FK 说安全 → **确定性安全**（不采样）
- 区间 FK 说碰撞 + 采样全通过 → **概率性安全**（覆盖误报）
- 区间 FK 说碰撞 + 采样命中 → **确认碰撞**

### 7.2 固定关节的体积处理

**问题**: Panda 第 8 个关节（手指）限制为 $[0, 0]$（固定关节），宽度=0 使得所有 box volume=0，被 `min_box_volume` 过滤掉。

**解决方案**: `BoxNode._compute_volume()` 跳过零宽度维度，只计算活动关节的体积积。

### 7.3 AffineForm 乘法保相关性

**问题**: 原始实现将两个 AffineForm 转为 Interval 相乘，丢失所有噪声符号相关性。

**解决方案**: 保留线性项的仿射乘法。关键公式：
$$x \cdot y = x_0 y_0 + \sum_i (x_0 y_i + y_0 x_i) \varepsilon_i + \delta \varepsilon_{new}$$

### 7.4 连通性修复

**问题**: 采样不充分时可能导致 start 和 goal 不在同一个连通分量。

**解决方案**: `_bridge_disconnected()` 方法做最多 5 轮 BFS + 最近 box 对桥接。

---

## 8. 使用示例

### 8.1 基本用法

```python
from box_aabb.robot import load_robot
from planner import BoxRRT, Scene, PlannerConfig

# 加载机器人
robot = load_robot('panda')  # 或 '3dof_planar'

# 构建障碍物场景
scene = Scene()
scene.add_obstacle([0.15, 0.10, 0.50], [0.30, 0.30, 0.70], name="block")

# 配置规划器
config = PlannerConfig(
    max_iterations=300,
    max_box_nodes=60,
    path_shortcut_iters=30,
)

# 规划
planner = BoxRRT(robot, scene, config)
result = planner.plan(q_start, q_goal, seed=42)

if result.success:
    print(f"路径: {len(result.path)} 个点, 长度 {result.path_length:.3f}")
```

### 8.2 可视化

```python
from planner.visualizer import plot_cspace_boxes, plot_workspace_result

# C-space box 投影
fig1 = plot_cspace_boxes(result, joint_limits=robot.joint_limits,
                          dim_x=0, dim_y=1, title="q0 vs q1")
fig1.savefig("cspace.png")

# 工作空间 3D 可视化
fig2 = plot_workspace_result(robot, scene, result, n_poses=12)
fig2.savefig("workspace.png")
```

### 8.3 评价指标

```python
from planner import evaluate_result

metrics = evaluate_result(result, robot, scene)
print(metrics.summary())
# PathMetrics:
#   path_length = 2.560
#   length_ratio = 1.63x
#   smoothness = 0.607
#   min_clearance = 0.045
```

### 8.4 可复用 Box 森林 (v4.0)

```python
from planner import BoxForest, BoxForestQuery, PlannerConfig

# 构建森林（一次性）
config = PlannerConfig(build_n_seeds=200, max_box_nodes=100)
forest = BoxForest.build(robot, scene, config, seed=42)
forest.save("forest.pkl")

# 多次查询（换始末点）
forest = BoxForest.load("forest.pkl", robot, scene)
query = BoxForestQuery(forest)
result1 = query.plan(q_start_1, q_goal_1)
result2 = query.plan(q_start_2, q_goal_2)  # 复用同一森林
```

### 8.5 动态可视化 (v4.0)

```python
from planner.dynamic_visualizer import animate_robot_path, resample_path

smooth_path = resample_path(result.path, n_frames=100)
anim = animate_robot_path(
    robot, smooth_path, scene=scene,
    fps=20, trail_length=30, ghost_interval=10)
anim.save("robot_motion.gif", writer='pillow', fps=20)
```

### 8.6 自由空间瓦片化 (v4.0)

```python
from planner import FreeSpaceTiler

tiler = FreeSpaceTiler(robot, scene, min_width=0.2, max_depth=6)
tiles = tiler.tile()
print(f"{len(tiles)} 个无碰撞瓦片")
nodes = tiler.tiles_to_box_nodes(tiles)  # 转为 BoxNode
```

---

## 9. 测试覆盖

项目包含全面的测试套件：

| 测试文件 | 测试数量 | 覆盖范围 |
|----------|---------|----------|
| `test_box_rrt.py` | 18 | 主规划器端到端 |
| `test_box_tree.py` | 19 | box 树管理 + 数据模型 |
| `test_collision.py` | 18 | 所有碰撞检测方法 |
| `test_gcs_optimizer.py` | 13 | GCS/Dijkstra/scipy |
| `test_obstacles.py` | 20 | 场景管理 |
| `test_panda_integration.py` | 22 | Panda 7DOF 集成测试 |
| `test_box_forest.py` | 8 | BoxForest + BoxForestQuery (v4.0) |
| `test_dynamic_visualizer.py` | 6 | 动态可视化 (v4.0) |
| `test_free_space_tiler.py` | 13 | 自由空间瓦片化 (v4.0) |
| `test_interval_math.py` | 26 | 仿射算术 |
| `test_calculator.py` | 22 | AABB 计算器 |
| `test_robot.py` | 41 | Modified DH 机器人模型 |
| 其他 | 21 | models, report 等 |

---

## 10. 性能数据

### 10.1 3DOF 平面机器人

| 场景 | 路径点 | 路径长度 | 树数 | Box 数 | 时间 |
|------|--------|---------|------|--------|------|
| Wall | 5 | 4.246 | 2 | — | 4.8s |
| Two-walls gap | 6 | 4.850 | 2 | — | 7.1s |

### 10.2 Panda 7DOF

| 场景 | 路径点 | 路径长度 | 树数 | Box 数 | 时间 | 长度比 | 平滑度 |
|------|--------|---------|------|--------|------|--------|--------|
| 有障碍物 | 5 | 2.560 | 2 | 60 | 49.6s | 1.63x | 0.607 |
| 无障碍物 | 直连 | — | — | — | <1s | 1.0x | — |

**Box 拓展对比** (Panda start 配置):

| 模式 | 关节 0-3 宽度 | 体积 | 时间 |
|------|--------------|------|------|
| 纯区间 FK（修复前） | 0.002 | 0 | 0.29s |
| 纯区间 FK（修复后） | 0.002 | 0 → 正常 | 0.29s |
| Hybrid（区间+采样） | 1.5 ~ 3.3 | 8335 | 0.71s |

---

## 11. 规划失败率分析

> **实验数据** (2DOF 平面, 5 个随机障碍物, 30 次随机试验):
>
> | 结果类型 | 次数 | 比例 |
> |----------|------|------|
> | 直连成功 (无需 box) | 18 | 60.0% |
> | Box-RRT 规划成功 | 0 | 0.0% |
> | 规划失败 | 12 | 40.0% |
>
> **结论**: 当 start→goal 直连被阻挡时，当前 Box-RRT 在 2DOF 上的成功率为 **0%**。

### 11.1 根本原因：C-space 拓扑与保守碰撞检测

**2DOF C-space 的特殊性**
- 2DOF C-space 是 $\mathbb{T}^2$（二维环面），维度极低。
- 一个工作空间中的小 AABB 障碍物在 C-space 中映射为一条连续弯带（C-obstacle band），宽度取决于连杆长度与障碍物尺寸的比值。
- 由于总共只有 2 个维度，单条 C-obstacle 带就可能将 C-space 切割为不连通的两部分。若 $q_{start}$ 和 $q_{goal}$ 落在带的两侧，则**物理上不存在**从一侧到另一侧的无碰撞路径（除非绕环面的另一条拓扑路径存在）。

**区间算术的过估计 (over-approximation)**
- `check_box_collision()` 使用区间/仿射算术 FK，得到的 link AABB 是**保守过估计**——区间越宽，过估计越严重。
- Box 拓展靠二分搜索逼近碰撞边界，但过估计使得 box 比实际 C-free 更小。
- 在 C-obstacle 带附近，过估计使 box 更早停止拓展，留下一条本可通过但被保守判断为碰撞的间隙。

### 11.2 算法层面的瓶颈

| 瓶颈环节 | 代码位置 | 问题 | 影响 |
|----------|---------|------|------|
| **树间连接仅靠直线** | `connector._try_connect_two_trees()` | 取两棵树最近 box 表面点，做线段碰撞检测；若线段穿过 C-obstacle 带则失败 | 无法绕障连接 |
| **桥接采样集中中点** | `box_rrt._bridge_sampling()` | 在两棵树最近 box 中点附近采样；若中点恰好在碰撞带内，采样全部失败 | 无法在窄通道生长 |
| **连通性修复局限** | `box_rrt._bridge_disconnected()` | 5 轮 × 200 候选，但仍是 surface-to-surface 直线检测 | 与树间连接同样的直线局限 |
| **box 体积过滤** | `min_box_volume = 1e-6` | 在碰撞带边缘，box 极小，可能被过滤 | 丢失关键区域的 box |
| **采样均匀性** | `_sample_seed()` 只用均匀/goal-bias | 不会定向在碰撞带的薄弱处（如窄通道入口）采样 | 采样效率低 |

### 11.3 2DOF vs 高 DOF 的差异

| 维度 | C-obstacle 拓扑 | 绕行路径 | Box-RRT 表现 |
|------|----------------|---------|------------|
| 2DOF | 一条带就能"封死"通路 | 必须沿环面绕行（常不存在） | 极差——非直连即失败 |
| 3DOF | 带是 2D 曲面，3D 空间中容易找到绕行间隙 | 通常存在多条 | 中等 |
| 7DOF (Panda) | 带是低维子流形，7D 空间极其"稀疏" | 几乎总存在 | 较好 |

高 DOF 下碰撞区域在 C-space 中是低维度子流形（co-dimension ≥ 5），几乎不可能封死所有通路。因此 Box-RRT 在 7DOF Panda 上实际表现会好得多。

### 11.4 可改进方向

1. **Bi-directional box growing**: 从两棵树表面同时向对方生长 box，而不是只在中点采样。
2. **Non-linear inter-tree connection**: 树间连接时不用直线段，而用多段折线或 RRT-Connect 在两棵树最近 box 之间搜索路径。
3. **Adaptive sampling near C-obstacle boundary**: 用碰撞检测梯度估计碰撞带法线方向，沿切线方向定向采样。
4. **环面拓扑支持**: 对 2DOF，在 $q_i = \pm\pi$ 处添加 wrap-around 连接，利用环面另一侧的通路。
5. **减小过估计**: 对临界区域用更小的区间宽度阈值（`interval_width_threshold`）或直接切换为逐点采样验证。

### 11.5 结论

当前 Box-RRT 本质上是一个 **"直连优先 + box 覆盖"** 算法。它的优势在于：
- 快速判断直连可行性（<5ms）
- 用 box 提供无碰撞区域的凸表示，为 GCS 优化奠定基础
- 高 DOF 空间中 C-obstacle 稀疏时表现良好

其短板在于：
- 树间连接完全依赖直线段碰撞检测，缺乏绕障能力
- 低 DOF 时 C-obstacle 带容易封死通路，算法无法穿越
- 保守碰撞检测在碰撞带附近过度缩小 box

**推荐**: 对于需要真正穿越碰撞带的场景，应将 Box-RRT 与传统 RRT-Connect 混合使用：先用 Box-RRT 覆盖 C-free 大块区域，再用 RRT-Connect 在 box 覆盖不到的窄通道中搜索连接。

---

## 附录: 文件行数统计

| 模块 | 行数 | 说明 |
|------|------|------|
| `models.py` | 297 | 数据模型 |
| `box_rrt.py` | ~600 | 主规划器 |
| `hier_aabb_tree.py` | ~620 | 层级 AABB 缓存树（Box 拓展） |
| `collision.py` | ~300 | 碰撞检测 |
| `box_tree.py` | ~260 | 树管理 |
| `connector.py` | ~400 | 树间连接 |
| `path_smoother.py` | ~180 | 路径平滑 |
| `gcs_optimizer.py` | ~360 | GCS 优化 |
| `obstacles.py` | ~165 | 场景管理 |
| `metrics.py` | ~340 | 评价指标 |
| `visualizer.py` | ~370 | 可视化 |
| `parallel_collision.py` | ~260 | 并行碰撞 |
| `box_forest.py` | ~230 | 可复用 BoxForest (v4.0) |
| `box_query.py` | ~200 | Forest 查询规划 (v4.0) |
| `dynamic_visualizer.py` | ~310 | 动态可视化 (v4.0) |
| `free_space_tiler.py` | ~210 | 自由空间瓦片化 (v4.0) |
| `interval_math.py` | ~420 | 仿射算术 |
| `interval_fk.py` | ~150 | 区间 FK |
| **合计** | **~5,750** | |
