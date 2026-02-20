# Benchmark 说明：BoxPlanner vs OMPL Family vs GCS

本文档说明项目的两个 benchmark / 对比入口：

1. `v2/benchmarks/planner/bench_rrt_vs_marcucci.py` — 2-DOF 定量 benchmark
2. `v2/examples/compare_all_planners.py` — Panda 7-DOF 全方法统一对比

---

## 1. 对比方法一览

### 1.1 BoxPlanner（Box-RRT 管线）

| 标签 | 说明 |
|------|------|
| `box_planner` / `box_planner_serial` | 标准串行模式 (`parallel_grow=False`) |
| `box_planner_parallel_2` | KD 分区并行，2 分区 / 2 workers |
| `box_planner_parallel_4` | KD 分区并行，4 分区 / 4 workers |
| `box_planner_parallel_8` | KD 分区并行，8 分区 / 4 workers |

完整流程：grow forest → coarsen → adjacency → bridge → Dijkstra → GCS/SOCP refine → smooth

配置由 `BOXRRT_CONFIGS` 列表定义（`compare_all_planners.py`）：

```python
BOXRRT_CONFIGS = [
    {"parallel_grow": False},
    {"parallel_grow": True, "n_partitions_depth": 1, "parallel_workers": 2},
    {"parallel_grow": True, "n_partitions_depth": 2, "parallel_workers": 4},
    {"parallel_grow": True, "n_partitions_depth": 3, "parallel_workers": 4},
]
```

### 1.2 OMPL 系列（C++ via WSL bridge）

| 算法 | OMPL 类名 |
|------|-----------|
| RRT | `RRT` |
| RRTConnect | `RRTConnect` |
| RRT* | `RRTstar` |
| InformedRRT* | `InformedRRTstar` |
| BIT* | `BITstar` |

运行方式：

- 通过 `wsl -e bash -c "python3 ompl_bridge.py"` 子进程调用 WSL 中安装的 OMPL
- bridge 脚本位于 `v2/examples/ompl_bridge.py`
- 使用项目自身的 Robot + CollisionChecker 保证碰撞模型一致
- 以 JSON stdin/stdout 通信

### 1.3 Marcucci-GCS（Drake）

- 先用 BoxPlanner 生成 forest 与 adjacency
- 构建 forest graph 后调用 `GCSOptimizer(fallback=False)`
- 严格禁用 fallback，保证该方法失败时显式暴露

---

## 2. Benchmark 入口

### 2.1 bench_rrt_vs_marcucci.py（2-DOF 定量）

内置场景：

- `narrow_passage_2dof`
- `multi_obstacle_2dof`

默认参数：

- `--robot 2dof_planar`
- `--trials 8`
- `--seed 42`
- `--ompl-timeout 1.5`

方法列表：`box_planner`, `ompl_rrt`, `ompl_rrtconnect`, `ompl_rrtstar`, `marcucci_gcs_drake`

> 严格依赖模式：缺少 `ompl` 或 `pydrake` 会直接抛错退出。

### 2.2 compare_all_planners.py（Panda 7-DOF 全对比）

- 对比全部 BoxPlanner 配置（4 种并行模式）× OMPL 5 种算法
- Panda 7-DOF 场景，使用 `panda_planner.py` 管线
- 输出 HTML 可视化 + 统计汇总

---

## 3. 指标定义

每个方法在每个场景上统计：

| 指标 | 说明 |
|------|------|
| `success_rate` | 成功次数 / 试验次数 |
| `time_mean_all` | 全部运行平均耗时（含失败） |
| `time_p50_all` | 全部运行中位耗时 |
| `time_mean_success` | 成功运行平均耗时 |
| `path_len_mean_success` | 成功路径平均长度 |

并保留每次 trial 的原始记录（成功、耗时、路径长度、消息）。

---

## 4. 输出文件

`bench_rrt_vs_marcucci.py` 运行后输出到：

```
v2/output/benchmarks/planner_rrt_vs_marcucci_<timestamp>/
  raw_results.json
  summary.json
  summary.md
```

`compare_all_planners.py` 输出：

```
v2/output/compare_all/
  各方法 HTML 可视化
  统计汇总报告
```

---

## 5. 一键复现

### 5.0 单命令（安装 + 运行 + 最新汇总）

```powershell
./v2/scripts/run_benchmark_oneclick.ps1 -Mode conda -EnvName box-rrt -Trials 8
```

执行后自动生成/刷新：

```
v2/output/benchmarks/planner_rrt_vs_marcucci_latest_summary.md
```

### 5.1 Conda 模式（推荐）

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -RunSmokeTest
```

创建新环境：

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -CreateEnv -PythonVersion 3.10 -RunSmokeTest
```

### 5.2 Pip 模式

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode pip -RunSmokeTest
```

### 5.3 运行 benchmark

```bash
# 2-DOF benchmark
python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot 2dof_planar --trials 8

# Panda 7-DOF 全方法对比
python -m v2.examples.compare_all_planners
```

### 5.4 跳过安装

```powershell
./v2/scripts/run_benchmark_oneclick.ps1 -Mode conda -EnvName box-rrt -SkipSetup -Trials 8
```

---

## 6. 结果解读建议

优先关注：

1. `success_rate` — 鲁棒性
2. `time_mean_all` — 工程可用性（总体时延）
3. `path_len_mean_success` — 路径质量

并对比 `raw_results.json` 中失败消息，定位是"求解超时"、"图连接失败"还是"碰撞校验失败"。

---

## 7. 已知边界

- 2-DOF benchmark 适合基线稳定性与实现正确性回归
- 7-DOF 对比（Panda）通过 `compare_all_planners.py` 已支持
- OMPL 通过 WSL bridge 调用，网络/进程启动有固定开销
- OMPL 与 Drake 版本差异可能导致时间与成功率有偏移，建议在报告中记录版本信息

---

## 8. BoxPlanner 并行模式实验对照

为评估 KD 子空间并行扩展，在同一场景追加内部对照：

| 对照组 | 配置 |
|--------|------|
| `box_planner_serial` | `parallel_grow=False` |
| `box_planner_parallel_2` | `parallel_grow=True, n_partitions_depth=1, parallel_workers=2` |
| `box_planner_parallel_4` | `parallel_grow=True, n_partitions_depth=2, parallel_workers=4` |
| `box_planner_parallel_8` | `parallel_grow=True, n_partitions_depth=3, parallel_workers=4` |

### 8.1 推荐附加统计字段

在 `raw_results.json` 每条 trial 中补充：

| 字段 | 含义 |
|------|------|
| `planner_mode` | serial / parallel |
| `n_boxes` | 最终 box 数 |
| `n_edges` | 图边数 |
| `n_collision_checks` | 碰撞检测次数 |
| `invariant_passed` | 是否通过 strict 校验 |

### 8.2 结果解释

1. 若并行耗时下降但成功率下降：优先调 `connection_radius` 与 `n_partitions_depth`
2. 若并行耗时不降：检查 `ProcessPool` 是否回退到进程内执行
3. 若发生结构异常：应由 strict 校验直接失败，不应被吞掉

### 8.3 报告呈现模板

至少给出以下表格：

1. 场景 × 模式 的成功率
2. 场景 × 模式 的平均耗时 / P50
3. 场景 × 模式 的平均路径长度（成功样本）
4. 场景 × 模式 的不变量失败率（应接近 0）
