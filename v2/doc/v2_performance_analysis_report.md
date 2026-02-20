# V2 性能分析报告：为什么 V2 的耗时和碰撞检测次数多于 V1？

> 生成日期：2026-02-19  
> 基准场景：2-DOF planar robot, 2 obstacles, max_iterations=200, max_box_nodes=120  
> 测试种子：seed = 42, 43, 44

---

## 1. 核心结论

V2 相比 V1 碰撞检测次数增加 **~7.7×**、耗时增加 **~1.6×** 的根本原因是：

| 根因 | CC 增幅贡献 | 时间增幅贡献 |
|------|------------|------------|
| **批量种子采样计数膨胀** | ~90% (3770 CC) | ~72% (54ms) |
| 非种子阶段差异（bridge repair 等） | ~10% (420 CC) | ~28% (22ms) |
| HierAABBTree 差异 | **0%** | **~0%** |
| validate_boxes 差异 | **负贡献** (V2 更少) | **~0%** |

**一句话总结**：V2 的 `_sample_seed()` 改用 `check_config_collision_batch()` 批量接口，每次调用固定计入 20 次碰撞检测（`max_attempts=20`），即使只需 1 个无碰撞配置即可返回。而 V1 使用逐次早停策略，平均仅 1.2 次/调用。

---

## 2. 实验数据

### 2.1 碰撞检测次数分阶段统计

#### V1

| Seed | 总 CC | `_sample_seed` CC | CC/次 | `validate_boxes` CC | 非种子 CC | Boxes |
|------|-------|-------------------|-------|--------------------|-----------|----- |
| 42   | 568   | 227               | 1.1   | 55                 | 341       | 55   |
| 43   | 799   | 235               | 1.2   | 63                 | 564       | 63   |
| 44   | 512   | 238               | 1.2   | 53                 | 274       | 53   |
| **均值** | **626** | **233**       | **1.2** | **57**           | **393**   | **57** |

#### V2

| Seed | 总 CC | `_sample_seed` CC | CC/次 | `validate_boxes` CC | 非种子 CC | Boxes |
|------|-------|-------------------|-------|--------------------|-----------|----- |
| 42   | 5771  | 4000              | 20.0  | 44                 | 1771      | 44   |
| 43   | 4392  | 4000              | 20.0  | 46                 | 392       | 46   |
| 44   | 4279  | 4000              | 20.0  | 45                 | 279       | 45   |
| **均值** | **4814** | **4000**   | **20.0** | **45**           | **814**   | **45** |

### 2.2 时间分阶段统计

#### V1

| Seed | 总时间 | 种子采样时间 | 占比 | 非种子时间 |
|------|--------|-------------|------|-----------|
| 42   | 0.071s | 0.015s      | 21%  | 0.056s    |
| 43   | 0.136s | 0.019s      | 14%  | 0.116s    |
| 44   | 0.155s | 0.041s      | 27%  | 0.114s    |
| **均值** | **0.121s** | **0.025s** | **21%** | **0.095s** |

#### V2

| Seed | 总时间 | 种子采样时间 | 占比 | 非种子时间 |
|------|--------|-------------|------|-----------|
| 42   | 0.213s | 0.057s      | 27%  | 0.156s    |
| 43   | 0.186s | 0.103s      | 55%  | 0.083s    |
| 44   | 0.189s | 0.077s      | 41%  | 0.112s    |
| **均值** | **0.196s** | **0.079s** | **41%** | **0.117s** |

### 2.3 HierAABBTree 内部统计

| 指标 | V1 均值 | V2 均值 | 差异 |
|------|---------|---------|------|
| 树节点数 | 268 | 262 | -2% |
| FK 调用数 | 268 | 262 | -2% |
| 最大深度 | 12 | 12 | 0% |
| 平均深度 | 8.6 | 8.6 | 0% |

> **结论**：HierAABBTree 的内部行为在 V1 和 V2 之间几乎完全一致，不是性能差异的来源。

---

## 3. 根因详细分析

### 3.1 主因：`_sample_seed()` 的批量计数膨胀（~90% CC 增幅）

#### V1 实现（早停策略）

```python
# v1/src/planner/box_rrt.py L682-L705
def _sample_seed(self, q_start, q_goal, rng):
    max_attempts = 20
    for _ in range(max_attempts):
        if rng.uniform() < self.config.goal_bias:
            q = q_goal + rng.normal(0, 0.3, size=self._n_dims)
            q = np.clip(q, ...)
        else:
            q = rng.uniform(lo, hi)  # 逐个生成
        
        if not self.collision_checker.check_config_collision(q):  # +1 CC
            return q  # ← 找到立即返回！
    return None
```

**关键特征**：逐个采样，逐个检测，找到第一个无碰撞配置立即返回。在 2-DOF 场景中（C-free 占比大），平均 **1.2 次尝试**即可找到无碰撞种子。

#### V2 实现（批量策略）

```python
# v2/src/planner/box_planner.py L982-L1010
def _sample_seed(self, q_start, q_goal, rng, sampling_intervals=None):
    max_attempts = 20
    
    # 一次性生成所有 20 个候选
    candidates = np.empty((max_attempts, self._n_dims), dtype=np.float64)
    ...  # 填充 candidates
    
    # 批量碰撞检测：将 20 个候选全部送入向量化 FK
    collisions = self.collision_checker.check_config_collision_batch(candidates)
    #                                                    ↑ 计数器 += 20 (固定!)
    free_idx = np.flatnonzero(~collisions)
    if free_idx.size > 0:
        return candidates[int(free_idx[0])].copy()
    return None
```

**关键特征**：一次性生成全部 20 个候选并通过 `check_config_collision_batch` 向量化检测。`check_config_collision_batch` 内部执行 `self._n_collision_checks += N`（N=20），**无论实际只需要第 1 个**还是全部 20 个。

#### 数值影响

- `max_iterations = 200`，每次迭代调用一次 `_sample_seed`
- V1：200 × 1.2 CC/call = **240 CC**
- V2：200 × 20.0 CC/call = **4000 CC**
- 差值：**3760 CC**（占 V2 总增幅 4188 CC 的 **90%**）

#### 时间影响

V2 批量 FK 虽然使用了 NumPy 向量化，但对 2-DOF 小规模问题反而带来了额外开销：

- 构造 20×2 候选矩阵的 NumPy 数组分配
- 20 次 DH 链式矩阵乘法的 `np.einsum` 调用
- 全局碰撞检测的向量化分离轴测试

V1 的逐个标量 FK + 早停在小规模问题上更高效：

```
V2 种子采样耗时：0.079s (平均)
V1 种子采样耗时：0.025s (平均)
额外开销：0.054s → 占总耗时差 0.075s 的 72%
```

### 3.2 次因：非种子阶段的偶发差异（~10% CC 增幅）

非种子 CC 包含：验证起终点 (2 CC) + 直连检测 (~66 CC) + validate_boxes + 端点连接 + bridge repair + 路径平滑。

| 指标 | V1 均值 | V2 均值 |
|------|---------|---------|
| 非种子 CC | 393 | 814 |
| validate_boxes CC | 57 | 45 |

V2 非种子 CC 偏高的原因：
1. **V2 seed=42 异常偏高**（1771 vs 平均 335），可能触发了 bridge repair
2. V2 生成的 box 更少（45 vs 57），导致图连通性更差，bridge repair 需要更多线段碰撞检测
3. V2 的 `BoxForest.add_box_direct` 使用向量化邻接检测（NumPy `_adjacent_existing_ids_from_cache`），在小规模下有微量额外开销

### 3.3 非因：HierAABBTree 完全一致

V2 的 `find_free_box` 逻辑与 V1 基本相同——KD 二分下行、AABB 碰撞早停、上行 promotion。V2 增加了 `constrained_intervals` 和 `active_split_dims` 参数，但在单区模式下（`parallel_expand=False`，即基准测试配置）这些参数**未被使用**。

树统计完全一致（节点数 ~265，FK 调用 ~265，深度 12/8.6）证实了这一点。

### 3.4 V2 box 数量更少的原因

V2 平均生成 45 个 box vs V1 的 57 个，但使用相同的 200 次迭代。可能原因：

1. 批量种子采样中部分候选落在已占用区域，返回的 seed 空间分布不同
2. 采样策略变化（批量 goal_bias vs 逐次 goal_bias）导致种子分布差异
3. 更少的有效种子 → 更少的成功 `find_free_box` 调用

---

## 4. V2 架构设计分析

### 4.1 批量采样的设计意图

V2 的 `check_config_collision_batch` 是为**高自由度机器人**设计的优化：

- **7-DOF Panda**：单次 FK 较慢（~20μs Cython），向量化 20 次可充分利用 SIMD
- **大批量场景**：当 C-free 比例很低时（如密集障碍），需要更多采样才能找到无碰撞配置，批量检测的向量化优势更明显
- **并行分区**扩展时，每个分区的采样空间受限，批量检测的收益更大

但在 2-DOF 低维场景中，C-free 比例高（通常 >80%），逐个采样 1-2 次即可命中无碰撞配置，向量化的 setup 开销反而浪费。

### 4.2 CC 计数器语义差异

| 版本 | `_sample_seed` CC 计数语义 |
|------|--------------------------|
| V1   | = 实际执行的标量碰撞检测次数（物理意义明确） |
| V2   | = 每批候选数量（N=20），不反映实际单次检测开销 |

V2 的 `check_config_collision_batch` 内部：
```python
self._n_collision_checks += N  # N 个配置一次性向量化处理
```

这导致 CC 计数**不再等价于计算量**，而是等价于**检查的配置数**。向量化的 20 次检测实际计算量可能仅相当于标量的 3-5 次。

---

## 5. 7-DOF Panda Forest 的全局停滞问题

V2 在 Panda 7-DOF + 10+ 障碍物场景下产生 0 个 box（全局停滞），而 V1 在相同条件下可产生 120 个 box。分析如下：

### 5.1 已知现象

| n_obs | V1 boxes | V1 nsize | V2 boxes | V2 nsize |
|-------|----------|----------|----------|----------|
| 5     | 120      | 0.328    | 60       | 0.779    |
| 10    | 120      | 0.242    | 0        | --       |
| 15    | 120      | 0.257    | 0        | --       |
| 20    | 120      | 0.317    | 0        | --       |

### 5.2 可能原因

1. **`global_stalls` 早停机制**：V2 的 bench_panda_multi 可能设置了全局停滞阈值。当连续多次 `find_free_box` 返回 None 时触发早停。
2. **`constrained_intervals` 在并行模式下收缩搜索空间**：分区扩展时每个分区只在子空间内搜索，10+ 障碍物时子空间内几乎无 C-free 区域。
3. **AABB 过估计在高维下更严重**：7-DOF 的区间 FK AABB 过估计比 2-DOF 严重得多，导致更多 false positive 碰撞，`find_free_box` 下行到 max_depth 仍无法找到无碰撞节点。

---

## 6. 优化建议

### 6.1 短期：修复 CC 计数器（低成本，立即可做）

```python
# v2/src/planner/box_planner.py _sample_seed() 中
# 方案 A：改用早停标量检测（与 V1 一致）
for q in candidates:
    if not self.collision_checker.check_config_collision(q):
        return q.copy()
return None

# 方案 B：保留批量但修正计数器
collisions = self.collision_checker.check_config_collision_batch(candidates)
# 用首个命中为止的数量作为实际 CC
first_free = np.argmax(~collisions)
adjusted_cc = first_free + 1 if not collisions.all() else len(candidates)
```

方案 A 预期效果：CC 从 ~4800 降至 ~600（与 V1 一致），时间从 ~0.20s 降至 ~0.12s。

### 6.2 中期：自适应批量大小

```python
def _sample_seed(self, ...):
    batch_size = min(self._n_dims * 2, 20)  # 低维用小批量
    # 或根据 C-free 比例动态调整
    if self._cfree_hit_rate > 0.5:
        batch_size = 4  # C-free 占比高，小批量即可
    else:
        batch_size = 20  # C-free 占比低，需要大批量
```

### 6.3 长期：分层采样策略

对不同自由度采用不同策略：
- **2-4 DOF**：逐个标量采样 + 早停（V1 策略）
- **5-7+ DOF**：批量向量化采样 + 自适应批量大小

---

## 7. 结论

| 项目 | V1 | V2 | 差异倍数 | 主要原因 |
|------|----|----|---------|---------|
| 碰撞检测次数 | 626 | 4814 | 7.7× | 批量计数膨胀（20 CC/call vs 1.2 CC/call） |
| 总耗时 | 0.121s | 0.196s | 1.6× | 批量 FK 开销 + 向量化 setup 成本 |
| 种子采样耗时 | 0.025s | 0.079s | 3.2× | 向量化阵列分配 + 完整 20 候选 FK |
| Box 数量 | 57 | 45 | 0.79× | 采样分布差异 |
| 树结构 | 265 nodes | 262 nodes | ~1.0× | 完全一致 |

**核心发现**：V2 的碰撞检测次数远高于 V1 并非因为底层算法（HierAABBTree、find_free_box）变差，而是因为 `_sample_seed()` 接口改用批量碰撞检测后**计数语义改变**——每次调用固定计入 20 次 CC，但实际仅需 ~1.2 次。在低维稀疏障碍场景中，这一改变导致 CC 统计数字夸大了约 17 倍（20/1.2），但实际计算开销仅增加约 3.2 倍。

---

## 8. 后续更新（2026-02-22）

本报告撰写后，`_sample_seed()` 已升级为 **三层采样策略**（goal-biased → KD-guided → uniform），并支持 `sampling_intervals` 子空间约束。当前版本在 Panda 7-DOF 端到端管线中表现稳定（500 boxes, ~500ms），7-DOF 全局停滞问题已通过 coarsen + bridge 管线解决。

相关代码变更：
- `_sample_seed` 现支持三种采样源：目标偏置、KD-tree 最近节点引导、均匀随机
- `_sample_boundary_seed` 新增边界扩展采样
- `check_config_collision_batch` 自动启用 `SpatialIndex` gating（`spatial_index_threshold=20`）

本报告中 2-DOF 基准数据仍有效作为历史参照。

---

*报告由自动化分析工具生成，实验数据来自 Windows 11 / Python 3.10 / conda box-rrt 环境*
