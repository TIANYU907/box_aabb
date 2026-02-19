# Box Tree 扩展流程详细说明

> **版本**: v4.1.0 &nbsp;|&nbsp; **更新日期**: 2026-02-10

## 1. 概述

Box Tree 是一种在机器人关节空间（C-space）中构建无碰撞自由空间覆盖的数据结构。其核心思想是：从无碰撞的 **seed 配置**出发，通过启发式二分搜索将其扩展为一个尽可能大的**无碰撞超矩形（box）**，再通过边界采样不断生长出新的 box，最终形成一棵或多棵覆盖 C-free 空间的 **box tree**（box forest）。

**v4.1 更新**：引入 **balanced（平衡交替步进）** 拓展策略，解决旧版 greedy 策略导致的 box 形状极度细长问题（宽度比可达 1000:1+）。同时完善了 AABB 缓存的全局接入、参数 JSON 配置化、以及 hybrid 碰撞检测支持。

本文档基于 `planner/box_expansion.py`、`planner/box_tree.py`、`planner/box_forest.py`、`planner/collision.py` 等模块的实现，详细说明整个扩展流程。

---

## 2. 整体流程总览

```
┌──────────────────────────────────────────────────────┐
│                  Box Forest 生长主循环                  │
│                                                      │
│  for iteration in range(max_iters):                  │
│    1. 均匀随机采样 seed q                              │
│    2. 碰撞检测过滤（reject_collision）                  │
│    3. 覆盖检测过滤（reject_covered）                    │
│    4. Box 扩展（BoxExpander.expand）                   │
│    5. 加入 box tree（新建树 / 挂到最近的树）             │
│    6. 边界重采样 → 递归扩展子 box                       │
│                                                      │
│  直到 total_nodes >= max_boxes 或迭代耗尽              │
└──────────────────────────────────────────────────────┘
```

---

## 3. Seed 采样与过滤

### 3.1 均匀随机采样

在每次迭代中，从关节限制 `[lo_i, hi_i]` 范围内对每个关节维度独立均匀采样，生成一个候选 seed 配置 `q`：

$$q_i \sim \text{Uniform}(lo_i, hi_i), \quad i = 0, 1, \ldots, n-1$$

### 3.2 碰撞过滤（reject_collision）

通过 `CollisionChecker.check_config_collision(q)` 对 seed 进行单点碰撞检测：

1. 调用正运动学（FK）计算各连杆端点位置 $\{p_0, p_1, \ldots, p_n\}$
2. 对每对相邻端点 $(p_{i-1}, p_i)$ 构造该连杆段的 AABB：
   $$\text{link\_min} = \min(p_{i-1}, p_i), \quad \text{link\_max} = \max(p_{i-1}, p_i)$$
3. 对每个连杆 AABB 与每个障碍物 AABB 做分离轴测试（SAT）
4. 若任一连杆与任一障碍物重叠，则该 seed 碰撞，记录为 `reject_collision` 事件并跳过

### 3.3 覆盖过滤（reject_covered）

若 seed 已经落在某个已有 box 内部（即该配置已被覆盖），则无需再以它为中心扩展新 box：

```python
if manager.find_containing_box(q) is not None:
    # 已被覆盖，跳过
```

检测方法：遍历所有已有 box，判断 $q_i \in [lo_i^{(\text{box})}, hi_i^{(\text{box})}]$ 对所有维度是否成立。

---

## 4. Box 扩展算法（核心）

Box 扩展是整个流程的核心，由 `BoxExpander.expand()` 实现。从一个无碰撞 seed 出发，通过三个阶段将其扩展为一个尽可能大的无碰撞超矩形。

### 4.1 初始化

初始 box 设为以 seed 为中心、各维度半宽为 `min_initial_half_width`（默认 0.001 rad）的极小超矩形：

$$\text{intervals}_i = \big[\max(lo_i,\; q_i - \epsilon),\; \min(hi_i,\; q_i + \epsilon)\big]$$

其中 $\epsilon = 0.001$。然后对这个极小 box 做碰撞检测，若因过估计导致碰撞，则退化为点区间 $[q_i, q_i]$。

### 4.2 Jacobian 分析与维度排序

**目的**：确定各关节维度的扩展优先级。

**原理**：对于每个关节 $q_i$，计算 Jacobian 矩阵中对应列向量的范数 $\left\|\frac{\partial p}{\partial q_i}\right\|$，即该关节微小变化对末端执行器位移的影响程度。

**计算方法**（数值差分近似）：

$$\left\|\frac{\partial p}{\partial q_i}\right\| \approx \frac{\|p(q + \delta \mathbf{e}_i) - p(q)\|}{\delta}$$

其中 $\delta = 0.01$（`jacobian_delta` 参数），$\mathbf{e}_i$ 为第 $i$ 个单位向量。

**排序规则**：按范数**从小到大**排序。直觉是：
- **范数小** → 该关节变化对末端位移影响小 → 扩展该维度时碰撞状态变化缓慢 → **可以更大胆地扩展**
- **范数大** → 末端位移敏感 → 更容易碰撞 → **谨慎扩展**

**示例**（2DOF 平面机器人）：
```
dim 0 (q0): ||dP/dq0|| = 1.9632   ← 基座关节，影响大
dim 1 (q1): ||dP/dq1|| = 1.0000   ← 末端关节，影响小

扩展顺序: q1 → q0  （先扩展影响小的维度）
```

### 4.3 多轮迭代扩展（Greedy 策略）

> **注意**：v4.1 默认使用 **balanced** 策略（见 4.4）。以下描述旧版 greedy 策略，可通过 `expansion_strategy='greedy'` 启用。

扩展分为多轮（`max_rounds`，默认 3-4 轮），每轮按照 4.2 确定的优先级顺序逐维度扩展。

**多轮迭代的理由**：第一轮扩展时，先扩展的维度（如 q1）只能在其他维度（如 q0）尚未扩展的狭窄区间内进行搜索。当 q0 也扩展后，q1 可能有更大的可扩展空间。因此需要反复迭代直到体积不再增长。

**Greedy 策略的问题**：逐维度贪心搜索时，先处理的维度（Jacobian 范数小）会一次性扩展到极限，导致区间非常宽。当后续维度计算区间 FK 时，先前维度的宽区间会造成严重的过估计，使后续维度几乎无法扩展。典型表现：q1 宽度 2.85 rad，q0 宽度 0.002 rad，宽度比 1425:1。

**轮次终止条件**：
1. **体积不再增长**：若 $V_{\text{after}} \leq V_{\text{before}} \times 1.001$（增幅 < 0.1%），提前停止
2. **达到最大轮数**：执行完 `max_rounds` 轮

### 4.4 平衡交替步进扩展（Balanced 策略，v4.1 默认）

**核心思想**：不再逐维度一次性搜索到极限，而是维护 $2n$ 个候选方向（每个维度的 $+/-$），每步选择**体积增益最大**的候选执行一个自适应小步，交替推进所有维度的边界。

**算法流程**：

```
初始化 candidate_pool = {(dim_i, +1), (dim_i, -1)} for i = 0..n-1

for step in 0..balanced_max_steps:
    if candidate_pool 为空: break

    for each (dim, direction) in candidate_pool:
        remaining = joint_limit - current_boundary
        step_size = remaining × balanced_step_fraction
        target = current_boundary + step_size × direction

        if check_collision(试探区间) == False:
            new_bound = target          # 整步安全
        else:
            new_bound = binary_search(current_boundary, target)
            # 在 [current, target] 之间二分找安全边界

        volume_gain = V(执行后) - V(当前)

    选择 volume_gain 最大的候选执行
    更新 intervals[best_dim]
    移除 remaining < resolution 的候选
```

**关键设计**：

1. **自适应步长**：每步的步长为 `remaining_room × step_fraction`（默认 50%），保证每个维度都有机会逐步推进，而不是一次性占满
2. **体积增益优先**：每步评估所有活跃候选，选择对总体积增长贡献最大的方向执行，避免无效扩展
3. **二分搜索精炼**：当整步不安全时，在 `[current_bound, target]` 之间做二分搜索，精度为 `expansion_resolution`
4. **候选淘汰**：当某个方向的剩余空间 < `resolution` 时自动移除，避免无效计算
5. **连续零增益停止**：若连续 $2n$ 步体积增益为零，提前终止

**效果对比**（2DOF，30 个随机 seed 统计）：

| 策略 | 宽度比 中位数 | 宽度比 最大 | 宽度比 均值 |
|------|-------------|------------|------------|
| greedy | 48.5 | 3141.6 | 634.6 |
| **balanced** | **4.1** | 1291.4 | 97.3 |

中位宽度比从 48.5 降至 4.1，**改善约 12 倍**。

### 4.4 单维度二分搜索

对于一轮中的每个维度 $d$，分别向**正方向（+）**和**负方向（-）**进行二分搜索以找到碰撞边界。

#### 4.4.1 正方向搜索流程

```
当前安全上界: current_hi (= seed[d] + ε 或上一轮结果)
关节上限:     hi_limit

Step 1: 尝试直接跳到关节极限
  - 构造测试区间: intervals[d] = (current_lo, hi_limit)
  - 若无碰撞 → 直接返回 hi_limit（stop_reason: reached_limit）

Step 2: 二分搜索
  safe = current_hi   (已知安全)
  test = hi_limit     (已知碰撞)

  while |test - safe| > resolution:
      mid = (safe + test) / 2
      构造测试区间: intervals[d] = (current_lo, mid)
      if 碰撞:
          test = mid    (收缩搜索范围)
      else:
          safe = mid    (扩展安全边界)

  返回 safe
```

#### 4.4.2 负方向搜索流程

与正方向对称，搜索的是下界：`intervals[d] = (mid, current_hi)`。

**注意**：负方向搜索使用的是正方向搜索更新后的上界 `new_hi`，因此正方向的扩展结果会影响负方向的搜索空间。

#### 4.4.3 停止条件

每次二分搜索有三种可能的停止条件：

| 停止条件 | 含义 |
|---------|------|
| `reached_limit` | 直接扩展到关节极限都无碰撞，无需进一步搜索 |
| `resolution_converged` | safe 与 test 之间的差距已小于 `expansion_resolution`（如 0.02 rad），精度已够 |
| `max_iterations` | 达到 50 次二分迭代上限（理论上极少触发） |

### 4.5 碰撞检测方法

每次二分搜索中的碰撞检测调用 `_check_collision(test_intervals)`，这是整个扩展算法的关键子程序。

#### 4.5.1 区间 FK（保守方法）

对于给定的关节区间 $\{[lo_i, hi_i]\}$，使用**区间算术/仿射算术**进行正运动学计算：

1. 将每个关节区间表示为仿射形式（AffineForm）
2. 通过仿射算术传播 DH 参数链中的 sin/cos 运算
3. 得到每个连杆位置的保守包围盒（AABB）
4. 与障碍物 AABB 做重叠检测

**保守性保证**：
- 若区间 FK 判断为**无碰撞** → **一定无碰撞**（box 内所有配置安全）
- 若区间 FK 判断为**碰撞** → **可能是误报**（过估计导致的假阳性）

这种保守性确保了最终生成的 box 确实是完全无碰撞的。

#### 4.5.2 Hybrid 模式（高 DOF 机器人）

对于高自由度机器人（> 4 DOF），区间算术的过估计可能非常严重。此时启用 hybrid 模式：

1. 先用区间 FK 检测
2. 若区间 FK 判定碰撞，再用**采样法**进行复核：
   - 在 box 内随机采样 $N$ 个配置（默认 80 个）
   - 对每个采样点做精确单点碰撞检测
   - 若所有采样点都无碰撞，则覆盖为安全（概率性）

---

## 5. Box Tree 管理

### 5.1 树的创建与挂载

当一个新 box 被成功扩展后，需要将其加入 box forest：

```python
nearest = manager.find_nearest_box(q)
if nearest is not None and nearest.distance_to_config(q) < connection_radius:
    # 挂到最近 box 的树上
    manager.add_box(nearest.tree_id, box, parent_id=nearest.node_id)
else:
    # 距离太远，新建一棵树
    manager.create_tree(box)
```

- **`find_nearest_box(q)`**：遍历所有树的所有 box，找到距离 seed 最近的 box
- **距离度量**：配置到 box 的最小 L2 距离
  $$d(q, \text{box}) = \sqrt{\sum_i \max(0,\, lo_i - q_i,\, q_i - hi_i)^2}$$
- **连接判定**：若最近距离 < `connection_radius`（默认 2.0），则挂为该 box 的子节点

### 5.2 边界重采样

每成功创建一个新 box 后，在其所属树的叶子节点边界上进行重采样，以生长出更多的相邻 box：

```python
samples = manager.get_boundary_samples(tree_id, n_samples=seed_batch)
```

**采样策略**：
1. **选择 box**：从该树的所有叶子节点中按体积加权随机选取
   $$P(\text{box}) \propto V(\text{box})$$
   体积大的 box 获得更多的采样机会
2. **选择边界面**：在选中的 box 上随机选取一个维度 $d$ 和一个方向（lo 或 hi），将 $q_d$ 固定在该边界值
3. **内部填充**：其余维度在 box 内均匀随机采样

**边界采样的意义**：
- 边界处是"自由空间的边缘"，向外可能还有未被覆盖的自由空间
- 从边界出发扩展的新 box 天然与父 box 相邻，有利于连通性
- 对比纯随机采样，边界采样更高效地填充 C-free 的"缝隙"

---

## 6. 完整的单个 Box 扩展示例

### 6.1 Greedy 策略示例

以旧版 `expansion_detail.txt` 中的 Box #0 为例，展示 greedy 策略的典型表现：

```
=== Box #0  (node_id=0) ===
  Seed: [1.7213, -0.3840]
```

**Step 1: Jacobian 分析**
```
  dim 0 (q0): ||dP/dq0|| = 1.9632
  dim 1 (q1): ||dP/dq1|| = 1.0000
  扩展顺序: q1 → q0
  理由: q1 的 Jacobian 范数更小 (1.0000 < 1.9632)，
        q1 变化对末端位移影响更小，优先扩展
```

**Step 2: Round 1 扩展**

| 维度 | 方向 | 起始 | 结果 | 扩展量 | 步数 | 停止原因 |
|------|------|------|------|--------|------|----------|
| q1 | + | -0.3830 | 0.0851 | 0.4681 | 8 | 精度收敛 |
| q1 | - | -0.3850 | -2.7647 | 2.3797 | 8 | 精度收敛 |
| q0 | + | 1.7223 | 1.7223 | 0.0000 | 7 | 精度收敛 |
| q0 | - | 1.7203 | 1.7203 | 0.0000 | 8 | 精度收敛 |

- q1 方向扩展了很大的范围（约 2.85 rad），说明在该区域末端关节的变化不会导致碰撞
- q0 方向几乎未能扩展（0.0000），因为 q1 区间过宽导致区间 FK 严重过估计
- **宽度比 q1/q0 = 2.8498/0.0020 = 1425:1** — 这就是 greedy 策略的主要缺陷

**最终结果**：
```
  Final q0: [1.7203, 1.7223]  width=0.0020
  Final q1: [-2.7647, 0.0851]  width=2.8498
  Final volume: 0.005700
```

### 6.2 Balanced 策略示例

同样的 seed 使用 balanced 策略时，不会出现某维度独占所有区间的情况。每步只推进 `remaining × 0.5` 的距离，所有维度交替扩展：

```
Step  0: dim=1, dir=+, step=1.5708, actual=1.5708, vol_gain=0.003
Step  1: dim=1, dir=-, step=1.5708, actual=1.5708, vol_gain=0.006
Step  2: dim=0, dir=+, step=1.5708, actual=0.4200, vol_gain=0.004
Step  3: dim=0, dir=-, step=1.5708, actual=0.3800, vol_gain=0.003
Step  4: dim=1, dir=+, step=0.7854, actual=0.7854, vol_gain=0.002
...
```

每步都评估所有方向的体积增益，选最优方向执行。最终各维度宽度更加均衡，典型宽度比降至 4:1 ~ 10:1。

---

## 7. 参数说明

所有参数可通过 `PlannerConfig` 设置，支持 JSON 文件加载（`PlannerConfig.from_json("config.json")`）。

### 7.1 Box 拓展参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `expansion_resolution` | 0.01~0.02 rad | 二分搜索精度，决定 box 边界的定位精度。越小越精确但越慢 |
| `max_expansion_rounds` | 3~4 | greedy 策略每个 box 的最大迭代轮数 |
| `jacobian_delta` | 0.01 | 数值差分计算 Jacobian 范数的步长 |
| `min_initial_half_width` | 0.001 | 初始 box 的半宽（rad） |
| `expansion_strategy` | `'balanced'` | 拓展策略：`'balanced'`（交替步进）或 `'greedy'`（逐维贪心） |
| `balanced_step_fraction` | 0.5 | balanced 策略每步推进剩余空间的比例 (0, 1] |
| `balanced_max_steps` | 200 | balanced 策略最大步数 |
| `use_sampling` | None (自动) | 碰撞检测模式：`None`=自动（>4DOF 启用），`True`=强制启用，`False`=纯区间 FK |
| `sampling_n` | 80 | hybrid 碰撞检测的采样点数 |

### 7.2 Forest 构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_box_nodes` | 200 | box forest 中的最大 box 数量 |
| `build_n_seeds` | 200 | BoxForest 构建的采样种子数 |
| `connection_radius` | 2.0 | 新 box 挂载到已有树的最大距离阈值 |
| `seed_batch_size` | 5 | 每个新 box 触发的边界重采样数量 |
| `min_box_volume` | 1e-6 | box 体积下限（太小的 box 丢弃） |

### 7.3 JSON 配置文件

预设配置文件位于 `src/planner/configs/`：

| 文件 | 适用场景 |
|------|---------|
| `default.json` | 通用默认参数 |
| `2dof_planar.json` | 2DOF 平面机械臂 |
| `panda.json` | Panda 7DOF 机械臂 |

使用方式：
```python
config = PlannerConfig.from_json("src/planner/configs/panda.json")
planner = BoxRRT(robot, scene, config)
```

---

## 8. 算法特性总结

### 8.1 保守性

整个算法的安全性建立在**区间算术的保守性**之上：

- 区间 FK 计算得到的连杆 AABB 是对真实扫掠体积的**外包围**
- 若区间 FK 判定无碰撞，则 box 内所有配置一定无碰撞
- 代价是可能"丢失"一些实际安全但被误判的空间（过估计）

**Hybrid 碰撞检测**（`use_sampling=True`）可以部分缓解过估计：当区间 FK 判定碰撞时，用采样方法复核，若所有采样点均安全则覆盖判定。这在实践中能显著增大 box 面积，代价是安全性从理论保证降为概率保证。

### 8.2 Balanced 策略 vs Greedy 策略

| 特性 | Greedy | Balanced |
|------|--------|---------|
| 扩展方式 | 逐维度一次性搜到极限 | 交替小步推进，选最优方向 |
| 维度宽度比 | 极不均衡（中位 48:1） | **基本均衡（中位 4:1）** |
| 过估计影响 | 先扩展维度的宽区间严重干扰后续维度 | 各维度同步推进，过估计均匀分布 |
| 单 box 体积 | 可能较大（但形状畸形） | 可能较小（但形状规则） |
| C-free 覆盖 | box 细长，覆盖效率低 | **box 紧凑，拼接效率高** |
| 计算成本 | 较低（每维度一次二分） | 较高（每步评估所有候选） |
| 默认 | v4.0 默认 | **v4.1 默认** |

### 8.3 启发式维度排序的有效性

Jacobian 范数排序的启发式在两种策略中均有使用：
- **Greedy**：决定扩展顺序，范数小的优先
- **Balanced**：仍计算维度排序（用于日志记录和初始候选排列），但实际扩展由体积增益驱动

### 8.4 边界采样 vs 随机采样

| 特性 | 随机采样 | 边界采样 |
|------|---------|----------|
| seed 来源 | C-space 均匀随机 | 已有 box 边界面 |
| 覆盖冗余 | 可能落入已有 box（被 reject_covered） | 天然在 box 外部的邻域 |
| 连通性 | 可能形成孤立的新树 | 容易与父树相邻/重叠 |
| 空间利用率 | 低（大量 reject） | 高（定向填充空隙） |

实际运行中，边界采样产生的 box（`source: boundary`）与随机采样的 box（`source: random`）交替出现，共同构成覆盖良好的 box forest。

---

## 9. 流程图

```
                    ┌─────────────┐
                    │  随机采样 q   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 碰撞检测 FK  │──碰撞──→ reject_collision
                    └──────┬──────┘
                           │ 无碰撞
                    ┌──────▼──────┐
                    │ 已被覆盖？   │──是──→ reject_covered
                    └──────┬──────┘
                           │ 否
              ┌────────────▼────────────┐
              │   BoxExpander.expand()   │
              │                         │
              │  1. 初始化极小区间       │
              │  2. Jacobian 维度排序    │
              │  3. 策略分派 ────────┐   │
              │     │                │   │
              │     ▼                ▼   │
              │  balanced          greedy │
              │  ┌──────────┐  ┌───────┐ │
              │  │交替步进    │  │逐维度  │ │
              │  │选最优方向  │  │二分搜索│ │
              │  │二分精炼    │  │多轮迭代│ │
              │  └──────────┘  └───────┘ │
              │                         │
              │  4. 返回 BoxNode        │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │ 挂到树 / 新建树│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  边界重采样   │────→ 递归扩展子 box
                    └─────────────┘
```
