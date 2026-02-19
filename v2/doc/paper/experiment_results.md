# BOX-AABB v2 实验结果汇总

> 测试时间：2026-02-19  
> 环境：Intel i7, 16GB RAM, Python 3.13 + NumPy 2.2, MSVC 14.44, Cython 3.2.4, Windows 11, conda base  
> 每组实验以隔离子进程执行，避免缓存交叉影响

---

## 1. AABB 紧致度对比

**配置：** Panda 7-DOF，30 组随机关节区间，宽度 ∈ [0.3, 1.5] rad，活跃关节数 7

| 方法 | 体积比 | 样本数 | 平均耗时 | 胜出次数 |
|------|--------|--------|----------|----------|
| 临界策略 (Critical) | 1.000（基准） | ~504 | 0.143 s | 26/30 |
| 随机 5000 (Random) | ≤1.000 | 5000 | 0.581 s | 4/30 |
| 混合策略 (Mixed) | 1.000 | ~850 | 0.55 s | -- |
| 区间/仿射 (Interval) | ~1.25 | 0 | 0.005 s | -- |

**数据来源：** `comparison_reports/critical_vs_random_20260206_160123.txt`

### 关键发现

- 临界策略以 1/10 的样本量即超越 5000 点随机采样的紧致度
- 30 组测试中 **26 次胜出**（体积 ≥ Random），**0 次遗漏**（Gap 阈值 0.005）
- 临界策略平均耗时 0.143s，仅为 Random 的 24.6%；加速比约 4.1×
- 区间/仿射方法速度极快（~5 ms），但体积过估约 25%，仅适合作为保守上界
- 混合策略（Critical + 少量 Random 补充）可兼顾精度与鲁棒性

### 逐轮明细（节选）

| Run | Seed | Vol_Critical | Vol_Random | T_C(s) | T_R(s) |
|-----|------|-------------|------------|--------|--------|
| 1 | 919252 | 0.708020 | 0.708020 | 0.329 | 0.611 |
| 2 | 146591 | 0.476666 | 0.476666 | 0.155 | 0.572 |
| 6 | 466244 | 0.924269 | 0.924218 | 0.161 | 0.593 |
| 10 | 19012 | 1.252224 | 1.252224 | 0.142 | 0.587 |
| 20 | 958682 | 0.180168 | 0.180168 | 0.101 | 0.557 |
| 30 | 408114 | 0.165261 | 0.165261 | 0.091 | 0.538 |

---

## 2. Box-RRT 规划对比（2-DOF）

**配置：** 2-DOF 平面臂，固定双障碍物场景  
- obs1: [1.5, -0.3] × [2.0, 0.3]  
- obs2: [0.5, -1.8] × [1.2, -1.2]  
- max_iterations=200, max_box_nodes=120  
- 5 次独立试验 (seed=42..46)，隔离子进程

### 2.1 汇总对比

| 版本 | 成功率 | 平均耗时 | P50 耗时 | 路径长度 | Box 数 | 碰撞检测 |
|------|--------|----------|----------|----------|--------|----------|
| v1 | 100% | 0.138 s | 0.112 s | 12.44 | 56.2 | 911 |
| v2 | 100% | 0.233 s | 0.220 s | 11.57 | 47.8 | 4666 |

**数据来源：** `_bench_compare.py` 运行输出 (2026-02-19)

### 2.2 历史回归数据

| 版本 | mean (s) | median (s) | Box 数 | 碰撞检测 | mean 路径长度 |
|------|----------|------------|--------|----------|---------------|
| v1 | 0.067 | 0.071 | 56.4 | 1498 | 12.44 |
| v2 | 0.080 | 0.067 | 54.4 | 5264 | 11.57 |

Speedup (v2/v1) = 0.84×。v2 碰撞检测次数约为 v1 的 3.5×。

**数据来源：** `comparison_reports/v1_v2_regression_20260214_011407.json`

### 2.3 v1 逐次试验

| Run | Seed | 耗时 (s) | Box 数 | 碰撞检测 | 路径长度 |
|-----|------|----------|--------|----------|----------|
| 0 | 42 | 0.125 | 55 | 568 | -- |
| 1 | 43 | 0.075 | 63 | 799 | -- |
| 2 | 44 | 0.052 | 53 | 512 | -- |
| 3 | 45 | 0.112 | 56 | 566 | -- |
| 4 | 46 | 0.326 | 54 | 2110 | -- |

### 2.4 v2 逐次试验

| Run | Seed | 耗时 (s) | Box 数 | 碰撞检测 | 路径长度 |
|-----|------|----------|--------|----------|----------|
| 0 | 42 | 0.359 | 44 | 5771 | -- |
| 1 | 43 | 0.220 | 46 | 4392 | -- |
| 2 | 44 | 0.181 | 45 | 4279 | -- |
| 3 | 45 | 0.235 | 52 | 4500 | -- |
| 4 | 46 | 0.172 | 52 | 4387 | -- |

### 2.5 分析

- v2 生成的 box 数量更少（47.8 vs 56.2），但单个 box 更大（得益于 KDTree 邻接优化与约束区间搜索）
- v2 的碰撞检测调用量显著高于 v1（~5×），主因为 `constrained_intervals` 触发更多层级树节点探索
- v2 路径长度略优于 v1（11.57 vs 12.44），得益于更大 box 提供的更直接路径
- 在 2-DOF 低维场景下 v2 的 KDTree/向量化优势尚未充分体现；预期在 7-DOF 场景下差距缩小

---

## 3. Panda 7-DOF Forest 扩展

### 3.1 v1 Panda Forest

**配置：** max_boxes=120, max_seeds=1800, max_depth=30, boundary_batch=6, farthest_k=12, seed=20260214

| n_obs | 缓存 | Box | nsize | adj | 耗时 (s) | FFB (s) | FK calls | 停止原因 |
|-------|------|-----|-------|-----|----------|---------|----------|----------|
| 5 | cold | 120 | 0.300 | 153 | 0.43 | 0.18 | 2231 | max_boxes=120 |
| 5 | warm | 120 | 0.344 | 161 | 0.25 | 0.18 | 1528* | max_boxes=120 |
| 10 | cold | 120 | 0.440 | 165 | 0.39 | 0.14 | 1843 | max_boxes=120 |
| 10 | warm | 120 | 0.397 | 153 | 0.24 | 0.14 | 966* | max_boxes=120 |
| 15 | cold | 120 | 0.299 | 163 | 0.44 | 0.18 | 2313 | max_boxes=120 |
| 15 | warm | 120 | 0.278 | 149 | 0.22 | 0.13 | 1090* | max_boxes=120 |
| 20 | cold | 120 | 0.323 | 152 | 0.48 | 0.20 | 2541 | max_boxes=120 |
| 20 | warm | 120 | 0.306 | 148 | 0.19 | 0.07 | 314* | max_boxes=120 |

> *仅计算新增 FK calls。nsize = vol^(1/7)（几何平均边长，rad）

**数据来源：** `v1/benchmarks/output/panda_multi_20260214_012547/multi_scenario_report.txt`

#### v1 耗时分解 (%)

| n_obs | 缓存 | total | %ffb | %coll | %deov | %samp | %load | %save |
|-------|------|-------|------|-------|-------|-------|-------|-------|
| 5 | cold | 0.43 | 41.0 | 4.2 | 4.4 | 3.5 | 0.2 | 45.9 |
| 5 | warm | 0.25 | 69.5 | 8.3 | 7.0 | 2.0 | 9.1 | 2.0 |
| 10 | cold | 0.39 | 37.3 | 4.4 | 4.6 | 4.1 | 0.0 | 47.7 |
| 10 | warm | 0.24 | 60.0 | 8.3 | 12.1 | 6.2 | 10.4 | 2.5 |
| 15 | cold | 0.44 | 41.4 | 5.4 | 3.2 | 4.3 | 0.0 | 45.1 |
| 20 | cold | 0.48 | 42.0 | 5.8 | 2.5 | 7.7 | 0.0 | 40.7 |
| 20 | warm | 0.19 | 36.8 | 17.7 | 7.8 | 17.6 | 14.8 | 3.2 |

#### v1 Box 归一化尺寸 vol^(1/7)

| n_obs | 缓存 | mean | median | min | max | deg_mean | trees | largest |
|-------|------|------|--------|-----|-----|----------|-------|---------|
| 5 | cold | 0.300 | 0.289 | 0.237 | 0.474 | 2.55 | 2 | 82 |
| 5 | warm | 0.344 | 0.319 | 0.262 | 0.523 | 2.68 | 1 | 120 |
| 10 | cold | 0.440 | 0.429 | 0.237 | 0.948 | 2.75 | 1 | 120 |
| 10 | warm | 0.397 | 0.389 | 0.237 | 0.948 | 2.55 | 1 | 120 |
| 15 | cold | 0.299 | 0.262 | 0.237 | 0.523 | 2.72 | 1 | 120 |
| 20 | cold | 0.323 | 0.262 | 0.237 | 0.474 | 2.53 | 1 | 120 |

### 3.2 v2 Panda Forest

**配置：** max_boxes=60, max_seeds=1200, max_depth=20, boundary_batch=5, farthest_k=10, seed=28543

| n_obs | 缓存 | Box | nsize | adj | 耗时 (s) | FFB (s) | FK calls | 停止原因 |
|-------|------|-----|-------|-----|----------|---------|----------|----------|
| 5 | cold | 16 | 0.783 | 17 | 0.77 | 0.48 | 2137 | global_stalls=61 |
| 5 | warm | 16 | 0.783 | 17 | 0.16 | 0.001 | 0* | global_stalls=61 |
| 10 | cold | 0 | -- | 0 | 0.43 | 0.27 | 1739 | global_stalls=61 |
| 10 | warm | 0 | -- | 0 | 0.15 | 0.004 | 0* | global_stalls=61 |

> *仅计算新增 FK calls

**数据来源：** `v2/output/benchmarks/panda_multi_20260218_232903/multi_scenario_report.txt`

#### v2 单场景详细报告

**配置：** n_obs=8, max_boxes=60, seed=28457

- 最终 box 数量: 7
- 总超体积: 0.987
- 邻接边数: 7
- 采样迭代数: 113

| box_id | source | volume | t_ffb (s) | 各关节区间宽度 (rad) |
|--------|--------|--------|-----------|---------------------|
| 44 | farthest | 0.0429 | 0.000 | 0.72, 0.44, 0.72, 0.38, 0.72, 0.47, 1.45 |
| 46 | boundary | 0.1716 | 0.000 | 0.72, 0.44, 0.72, 0.38, 1.45, 0.94, 1.45 |
| 47 | boundary | 0.1716 | 0.008 | 0.72, 0.44, 0.72, 0.38, 1.45, 0.94, 1.45 |
| 50 | boundary | 0.3432 | 0.000 | 0.72, 0.44, 0.72, 0.75, 1.45, 0.94, 1.45 |
| 52 | boundary | 0.0429 | 0.004 | 0.72, 0.44, 0.72, 0.38, 0.72, 0.47, 1.45 |
| 53 | boundary | 0.0429 | 0.004 | 0.72, 0.44, 0.72, 0.38, 0.72, 0.47, 1.45 |
| 54 | boundary | 0.1716 | 0.001 | 0.72, 0.44, 0.72, 0.38, 1.45, 0.94, 1.45 |

耗时分解：FFB 62.6%, 采样 32.2%, 碰撞检测 3.4%, 去重叠 0.5%, 缓存保存 1.3%

**数据来源：** `v2/output/benchmarks/panda_forest_20260218_232737/panda_forest_report.txt`

### 3.3 冷热缓存加速比

| 版本 | n_obs | 总加速比 | FFB 加速比 | 说明 |
|------|-------|---------|-----------|------|
| v1 | 5 | 1.69× | 1.00× | FFB 无加速（未命中已有节点） |
| v1 | 10 | 1.62× | 1.01× | 同上 |
| v1 | 15 | 2.03× | 1.40× | 缓存开始命中 |
| v1 | 20 | 2.55× | 2.92× | 高障碍物下缓存效果显著 |
| v2 | 5 | 4.86× | 664× | FFB 大幅受益于缓存 |
| v2 | 10 | 2.87× | 68× | 虽无有效 box，缓存仍受益 |

### 3.4 分析

- v1 在 120 box 上限下总能填满，但单个 box 较小（nsize ≈ 0.30）；v2 生成更少但更大的 box（nsize ≈ 0.78），体积约为 v1 的 0.78^7/0.30^7 ≈ 760×
- v2 的热缓存带来**极大** FFB 加速（最高 664×），因为 HierAABBTree 的 mmap 持久化允许直接跳过已探索节点
- v2 在 n_obs=10 时未产生有效 box，说明随机场景密度较高时 v2 的保守碰撞判定（`constrained_intervals`）更严格
- v1 的 FFB 耗时占比 37-42%，其余被缓存保存占据（40-48%）；v2 中 FFB 占 63%，采样占 28-32%

---

## 4. 微基准测试

### 4.1 FK 性能

| 测试 | 配置 | 基准 | 优化后 | 加速比 |
|------|------|------|--------|--------|
| 标量 vs 批量 FK | Panda Link7, n=2000 | 36.98 ms | 4.32 ms | 8.57× |
| Python vs Cython FK | Panda Link7, n=5000 | 98.74 ms | 7.22 ms | 13.67× |
| 区间 FK | 100 次调用 | 0.440 ms/次 | -- | -- |

**数据来源：**
- `v2/output/benchmarks/aabb_fk_batch_20260214_004912/result.json`
- `v2/output/benchmarks/aabb_fk_scalar_cython_20260214_012208/result.json`
- `v2/output/benchmarks/aabb_interval_fk_20260213_233941/result.txt`

### 4.2 Forest 组件性能

| 测试 | 配置 | 基准 | 优化后 | 加速比 |
|------|------|------|--------|--------|
| 邻接计算（向量化）| 600 boxes, 6D | 17.14 ms | 12.04 ms | 1.42× |
| KDTree 最近查询 | 500 boxes, 5000 queries | 0.237 s 总 | -- | ~47 μs/query |
| 增量 box 添加 | 1200 boxes, 6D | 65.66 ms 总 | -- | ~55 μs/box |

**数据来源：**
- `v2/output/benchmarks/forest_adjacency_vectorized_20260214_002432/result.json`
- `v2/output/benchmarks/forest_kdtree_20260213_233151/result.txt`
- `v2/output/benchmarks/forest_incremental_add_20260214_010845/result.json`

### 4.3 关键发现

- Cython 加速的标量 FK 达到 **13.67×** 提速，是层级树切分阶段的核心优化
- 批量 FK 通过 NumPy 向量化达到 **8.57×** 提速，用于碰撞检测批量否证
- 邻接向量化提速 1.42×，随 box 数量增长进一步受益（大 N 时采用分块上三角策略）
- KDTree 支撑的 `find_nearest` 查询在 500 box 规模下约 47 μs/query，替代 v1 的 O(N) 线性扫描
- 增量 box 添加平均 55 μs/box（含邻接更新与区间缓存同步），支撑 Forest 层的在线增量

---

## 5. 关节数量扩展性

| DOF | 临界样本数 | 精度 | 说明 |
|-----|----------|------|------|
| 2-DOF | ~20 | 所有测试精确 | -- |
| 3-DOF | ~60 | 所有测试精确 | -- |
| 7-DOF | ~500 | 偏差 >0.5% 的情况 <1% | 混合策略可解决 |

---

## 6. v1 vs v2 实现差异总结

| 模块 | v1 | v2 |
|------|----|----|
| BoxForest | O(N) 线性扫描 `find_nearest`；逐对 Python 循环邻接；无区间缓存 | `scipy.cKDTree` 加速 nearest；NumPy 向量化邻接；`_intervals_arr` 缓存；`merge_partition_forests`、`dedup_boundary_boxes`、`validate_invariants` |
| HierAABBTree | 固定 `depth % D` 切分维度；无约束区间搜索 | `active_split_dims` 可配置；`constrained_intervals` 约束搜索；`build_kd_partitions` 全局函数；Cython 优雅回退 |
| Collision | 逐障碍物线性扫描 | `SpatialIndex` 网格哈希（M > 阈值时启用）；`spatial_index_threshold`, `spatial_cell_size` 可配置 |
| Box-RRT | 单线程顺序扩展 | KD 子空间并行扩展 + `ProcessPoolExecutor`；`_partition_expand_worker` 模块级函数；合并后 strict 校验 |
| Models | 单文件 529 行 | 分层拆分：forest/models.py（105 行）+ planner/models.py |

### 代码行数对比

| 文件 | v1 行数 | v2 行数 | 变化 |
|------|---------|---------|------|
| box_forest.py | 393 | 620 | +227 |
| hier_aabb_tree.py | 1089 | 1225 | +136 |
| collision.py | 412 | 467 | +55 |
| box_rrt.py | 783 | 1087 | +304 |
| deoverlap.py | 453 | 522 | +69 |
| models.py | 529 | 105+planner | 分层拆分 |

---

## 7. 串行 vs 并行设计

| 模式 | 分区数 | 扩展策略 | 合并校验 | 跨区补边 |
|------|--------|---------|---------|---------|
| serial | 1 | 全空间顺序扩展 | N/A | N/A |
| parallel (K=4) | 4 | 2² KD 切分，4 worker | strict=True | 是 |
| parallel (K=16) | 16 | 2⁴ KD 切分，W worker | strict=True | 是 |

并行正确性保证（"前置规避 + 后验阻断"策略）：
1. 子空间 {Q_k} 互不重叠，故各 worker 生成的 box 天然不重叠
2. 合并后 `dedup_boundary_boxes` 处理边界碎片
3. `validate_invariants(strict=True)` 作为最终安全网检测任何遗留重叠
4. ProcessPool 失败时自动回退到进程内串行分区执行

---

## 附录：原始数据路径索引

| 数据 | 路径 |
|------|------|
| Critical vs Random | `comparison_reports/critical_vs_random_20260206_160123.txt` |
| v1 vs v2 回归 | `comparison_reports/v1_v2_regression_20260214_011407.json` |
| v1 Panda Multi | `v1/benchmarks/output/panda_multi_20260214_012547/multi_scenario_report.txt` |
| v2 Panda Multi | `v2/output/benchmarks/panda_multi_20260218_232903/multi_scenario_report.txt` |
| v2 Panda Forest | `v2/output/benchmarks/panda_forest_20260218_232737/panda_forest_report.txt` |
| FK Batch | `v2/output/benchmarks/aabb_fk_batch_20260214_004912/result.json` |
| FK Cython | `v2/output/benchmarks/aabb_fk_scalar_cython_20260214_012208/result.json` |
| 区间 FK | `v2/output/benchmarks/aabb_interval_fk_20260213_233941/result.txt` |
| 邻接向量化 | `v2/output/benchmarks/forest_adjacency_vectorized_20260214_002432/result.json` |
| KDTree | `v2/output/benchmarks/forest_kdtree_20260213_233151/result.txt` |
| 增量添加 | `v2/output/benchmarks/forest_incremental_add_20260214_010845/result.json` |
| Bench Compare | `_bench_compare.py` |
