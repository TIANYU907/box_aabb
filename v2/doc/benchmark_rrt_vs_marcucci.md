# Benchmark 说明：BoxPlanner vs 常用 RRT 库 vs Marcucci-GCS

本文档说明 `v2/benchmarks/planner/bench_rrt_vs_marcucci.py` 的对比目标、方法定义、指标与复现流程。

## 1. 对比目标

统一场景与随机种子下，比较以下规划器：

1. `box_planner`：项目实现的 BoxPlanner（区间 box 扩展 + 图搜索 + 后处理）。
2. `ompl_rrt`：OMPL 的 RRT。
3. `ompl_rrtconnect`：OMPL 的 RRTConnect。
4. `ompl_rrtstar`：OMPL 的 RRTstar。
5. `marcucci_gcs_drake`：Drake `GraphOfConvexSets` 优化（对应 Marcucci et al., Science Robotics 2023 的 GCS 思路映射）。

> 注意：本 benchmark 明确采用“严格依赖模式”，缺少 `ompl` 或 `pydrake` 会直接抛错退出，不做降级。

## 2. 当前方法定义

### 2.1 BoxPlanner

- 按 `PlannerConfig` 扩展 BoxForest。
- 通过邻接图搜索得到 box 序列。
- 共享面 waypoint 优化 + box-aware 路径平滑。

> 在 Panda 7-DOF 场景中，还包含优化后的 coarsen + bridge 管线。

### 2.2 OMPL-RRT 系列

- 关节空间：`RealVectorStateSpace(n_joints)`。
- state validity：复用项目 `CollisionChecker` 作为状态碰撞判定。
- 统一边界：来自机器人 `joint_limits`。
- 分别实例化 `RRT` / `RRTConnect` / `RRTstar`。

### 2.3 Marcucci-GCS（Drake）

- 先用 BoxPlanner 生成可用 forest 与 adjacency。
- 构建 forest graph 后调用 `GCSOptimizer(fallback=False)`。
- 严格禁用 fallback，保证该方法失败时显式暴露。

## 3. 场景与实验设置

当前内置两个 2-DoF 场景（在脚本 `_make_cases()` 中定义）：

- `narrow_passage_2dof`
- `multi_obstacle_2dof`

默认参数：

- `--robot 2dof_planar`
- `--trials 8`
- `--seed 42`
- `--ompl-timeout 1.5`

## 4. 指标定义

每个方法在每个场景上统计：

- `success_rate`：成功次数 / 试验次数
- `time_mean_all`：全部运行平均耗时（含失败）
- `time_p50_all`：全部运行中位耗时
- `time_mean_success`：成功运行平均耗时
- `path_len_mean_success`：成功路径平均长度

并保留每次 trial 的原始记录（成功、耗时、路径长度、消息）。

## 5. 输出文件

运行后输出到：

- `v2/output/benchmarks/planner_rrt_vs_marcucci_<timestamp>/raw_results.json`
- `v2/output/benchmarks/planner_rrt_vs_marcucci_<timestamp>/summary.json`
- `v2/output/benchmarks/planner_rrt_vs_marcucci_<timestamp>/summary.md`

## 6. 一键复现

### 6.0 单命令（安装 + 运行 + 最新汇总）

```powershell
./v2/scripts/run_benchmark_oneclick.ps1 -Mode conda -EnvName box-rrt -Trials 8
```

执行后会自动生成/刷新固定别名文件：

- `v2/output/benchmarks/planner_rrt_vs_marcucci_latest_summary.md`

该文件始终指向“最近一次” benchmark 的 `summary.md` 内容，便于报告系统或人工快速读取。

### 6.1 Conda 模式（推荐）

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -RunSmokeTest
```

若需创建新环境：

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -CreateEnv -PythonVersion 3.10 -RunSmokeTest
```

### 6.2 Pip 模式（当前 Python 环境）

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode pip -RunSmokeTest
```

### 6.3 运行 benchmark

```powershell
python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot 2dof_planar --trials 8
```

### 6.4 跳过安装（仅重新跑 benchmark）

```powershell
./v2/scripts/run_benchmark_oneclick.ps1 -Mode conda -EnvName box-rrt -SkipSetup -Trials 8
```

## 7. 结果解读建议

建议优先关注：

1. `success_rate`：鲁棒性
2. `time_mean_all`：工程可用性（总体时延）
3. `path_len_mean_success`：路径质量

并对比 `raw_results.json` 中失败消息，定位是“求解超时”、“图连接失败”还是“碰撞校验失败”。

## 8. 已知边界

- 当前 benchmark 为 2-DoF 场景，适合做基线稳定性与实现正确性回归。
- 将来扩展 7-DoF（如 Panda）时，建议分离“同质量预算”与“同时延预算”两组实验。
- OMPL 与 Drake 版本差异可能导致时间与成功率有偏移，建议在报告中记录版本信息。

## 9. 并行模式实验补充（BoxPlanner 内部对照）

为评估 v2 新增的 KD 子空间并行扩展路径，建议在同一场景追加两组内部对照：

1. `box_planner_serial`
	- `parallel_expand=False`
2. `box_planner_parallel`
	- `parallel_expand=True`
	- `parallel_workers=2/4`
	- `parallel_partition_depth=2`

### 9.1 推荐附加统计字段

在 `raw_results.json` 每条 trial 中补充：

1. `planner_mode`: serial/parallel
2. `n_boxes`: 最终 box 数
3. `n_edges`: 图边数
4. `n_collision_checks`: 碰撞检测次数
5. `invariant_passed`: 是否通过 strict 校验

### 9.2 结果解释建议

1. 若并行耗时下降但成功率下降：优先调 `connection_radius` 与 `parallel_partition_depth`。
2. 若并行耗时不降：检查 `ProcessPool` 是否回退到进程内执行。
3. 若发生结构异常：应由 strict 校验直接失败，不应被吞掉。

### 9.3 报告呈现模板（建议）

至少给出以下表格：

1. 场景 × 模式 的成功率
2. 场景 × 模式 的平均耗时 / P50
3. 场景 × 模式 的平均路径长度（成功样本）
4. 场景 × 模式 的不变量失败率（应接近 0）
