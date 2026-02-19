# box-aabb v2（当前实现总览）

`v2/` 是当前主开发分支实现，采用三层解耦架构：

- `src/aabb`：区间 FK 与连杆 AABB 包络
- `src/forest`：无重叠 box 集合、邻接、层级扩展树与碰撞核心
- `src/planner`：Box-RRT 主流程、连接、图搜索与路径后处理

---

## 1. 当前实现状态（2026-02）

### 1.1 已完成

1. 分层迁移完成：AABB / Forest / Planner 均已在 `v2/src` 下运行。
2. KD 子空间并行扩展主线已接入：
	 - 分区构建：`build_kd_partitions`
	 - 并行扩展：`ProcessPoolExecutor`
	 - 合并补边：`merge_partition_forests` + `connect_across_partitions`
3. 合并后不变量强校验已恢复：`validate_invariants(strict=True)`。
4. 回归测试稳定通过（`v2/tests`）。

### 1.2 当前行为要点

- 并行模式下，起终点扩展由分区 worker 在子空间约束内处理，
	避免“全空间预扩展 + 分区扩展”重复造成正体积重叠。

- 当并行进程池不可用时，自动回退到进程内分区执行（保持功能可用）。

---

## 2. 目录结构（关键路径）

```text
v2/
├─ src/
│  ├─ aabb/
│  ├─ forest/
│  └─ planner/
├─ tests/
├─ benchmarks/
├─ scripts/
└─ doc/
```

> 注意：本目录已统一使用 `v2/src/*` 作为实现入口。

---

## 3. 文档导航（建议阅读顺序）

1. `doc/v2_algorithm_details.md`（总目录）
2. `doc/terminology_notation.md`（术语/符号）
3. `doc/aabb_algorithm_details.md`
4. `doc/forest_algorithm_details.md`
5. `doc/planner_algorithm_details.md`
6. `doc/benchmark_rrt_vs_marcucci.md`

---

## 4. 快速开始

### 4.1 运行测试

```powershell
C:/Users/TIAN/.conda/envs/box-rrt/python.exe -m pytest v2/tests -q
```

### 4.2 运行 Forest 基准

```powershell
python -m v2.benchmarks.forest.bench_panda_forest
python -m v2.benchmarks.forest.bench_panda_multi
```

### 4.3 运行 Planner 对比基准

```powershell
python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot 2dof_planar --trials 8
```

---

## 5. 一键环境与基准

### 5.1 一键安装 + 执行 + 最新汇总别名

```powershell
./v2/scripts/run_benchmark_oneclick.ps1 -Mode conda -EnvName box-rrt -Trials 8
```

输出别名文件：

- `v2/output/benchmarks/planner_rrt_vs_marcucci_latest_summary.md`

### 5.2 仅环境准备（conda）

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -RunSmokeTest
```

若需创建环境：

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -CreateEnv -PythonVersion 3.10 -RunSmokeTest
```

### 5.3 当前环境安装（pip）

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode pip -RunSmokeTest
```

---

## 6. 依赖与约束

- 通用：`numpy`, `scipy`, `pytest`, `matplotlib`
- 基准附加：`ompl`, `pydrake`

对比基准采用严格依赖策略：缺失 `ompl` 或 `pydrake` 时直接失败，不做静默降级。

---

## 7. 关键设计约束

1. 保守碰撞语义优先：宁可误报，不可漏报。
2. Forest 保持“无正体积重叠 + 邻接对称”不变量。
3. 并行扩展遵循“子空间隔离写入，主进程合并”原则。
4. 失败路径必须返回可诊断消息（不吞异常上下文）。

---

## 8. 常见问题（FAQ）

### Q1: 为什么并行模式有时仍可能失败规划？

并行只提升扩展吞吐，不保证特定场景必然可达。若图搜索失败，可提高：

- `max_iterations`
- `parallel_partition_depth`
- `connection_radius`

并适当降低 `min_box_size`。

### Q2: 如何确认不是不变量问题导致失败？

当前合并路径已启用严格校验，若存在重叠/邻接结构问题会直接抛出异常。

---

## 9. 进一步阅读

- 详细算法说明：`doc/*_algorithm_details.md`
- 实施计划与收敛路线：`doc/改进思路.md`
- 基准说明：`doc/benchmark_rrt_vs_marcucci.md`
