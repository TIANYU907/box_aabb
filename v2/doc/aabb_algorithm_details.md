# AABB Algorithm Details (v2)

## Abstract

本文件描述 v2 中 AABB 子系统的"实现级算法细节"，目标是解释：给定关节区间，如何在可控计算代价下得到**保守且可复用**的连杆包络。实现覆盖 `robot.py`、`interval_fk.py`、`calculator.py`、`strategies/*`、`optimization.py` 与 `models.py`。

---

## 1. Problem Formulation

设机器人关节向量为 $q \in \mathbb{R}^n$，区间盒为

$$
\mathcal{Q}=\prod_{i=1}^{n}[l_i,u_i].
$$

对每条连杆段 $\ell$，需要计算工作空间保守包络

$$
\mathcal{B}_\ell(\mathcal{Q}) \supseteq \{x\mid x\in \mathrm{Link}_\ell(q),\;q\in\mathcal{Q}\}
$$

并输出其轴对齐边界盒（AABB）

$$
AABB_\ell=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]\times[z_{\min},z_{\max}].
$$

该包络用于上层碰撞判定：若 `AABB ∩ Obstacle = ∅`，可直接证明对应区间无碰撞；若相交，仅表示"可能碰撞"。

---

## 2. System Decomposition

### 2.1 `robot.py`: kinematics kernel

- 使用 Modified DH 定义串联链。
- 提供单点 FK (`forward_kinematics` / `get_link_position`) 与批量 FK (`get_link_positions_batch`)。
- 维护 `zero_length_links`、`joint_limits`、`tool_frame` 与 `fingerprint()`。
- 可选加载 Cython 标量核 `_fk_scalar_core`，自动降级到 Python 计算路径。
- 预置模型加载：`load_robot("2dof_planar"/"3dof_planar"/"panda")`，从 `configs/*.json` 读取参数。

### 2.2 `interval_fk.py`: interval propagation kernel

- 区间三角函数包络：`_isin`, `_icos`。
- 区间 DH 矩阵构造：`_dh_joint_matrix`。
- 区间矩阵乘法：`_imat_mul_dh`（利用 DH 结构减少乘法开销）。
- 全量/增量接口：`compute_fk_full`, `compute_fk_incremental`, `compute_interval_aabb`。
- 分裂辅助：`_split_fk_pair` — 用于 HierAABBTree 节点切分时的左右子 FK 对生成。
- 可选 Cython 加速核 `_interval_fk_core`。

### 2.3 `calculator.py` + `strategies/*`

- `AABBCalculator.compute_envelope` 作为调度器。
- 数值法：`CriticalStrategy` 与 `RandomStrategy`。
- 区间法：直接调用 `compute_interval_aabb`。
- 极值精化：`optimization.py::optimize_extremes` (L-BFGS-B)。

### 2.4 `models.py`

- `BoundaryConfig`: 记录边界极值对应关节配置。
- `LinkAABBInfo`: 连杆（或子段）AABB。
- `AABBEnvelopeResult`: 一次计算的完整实验记录（样本量、耗时、分段数）。

---

## 3. Interval-FK: Implementation-Level Pipeline

### 3.1 Interval trigonometric enclosure

对每个区间角变量 $[\theta_l,\theta_u]$，`_isin/_icos` 通过极值点穿越判定给出保守界。若宽度 $\ge 2\pi$，直接退化为 $[-1,1]$。该步骤避免离散采样漏检，是"保守性"的根源。

### 3.2 Joint interval matrix assembly

`_dh_joint_matrix` 将每个关节转成区间齐次变换 $(A_{lo},A_{hi})$。对符号不确定项使用"4 角点 min/max"原则，避免引入仿射对象分配。

### 3.3 Interval chain multiplication

`_imat_mul_dh` 执行 $T\leftarrow T\otimes A$。实现层面：

- 采用广播向量化计算四种乘积组合并归并 min/max。
- 利用 DH 第四行为常量 `[0,0,0,1]`，对最后一列增量更新。
- 输出仍为 `(T_lo,T_hi)` 双矩阵。

### 3.4 Link AABB extraction

每条连杆段由前后两端点表示：

$$
p^{start}_\ell=T_{\ell-1}[0:3,3],\quad p^{end}_\ell=T_{\ell}[0:3,3].
$$

保守盒为分量 min/max：

$$
\mathbf{b}_{\ell}^{min}=\min\big(p^{start}_{lo},p^{end}_{lo}\big),\quad
\mathbf{b}_{\ell}^{max}=\max\big(p^{start}_{hi},p^{end}_{hi}\big).
$$

---

## 4. Incremental FK and Cache Reuse

`compute_fk_incremental` 输入父节点缓存 `(prefix_lo, prefix_hi, joints_lo, joints_hi)` 及 `changed_joint=d`，只重算后缀 $[d,n)$。该机制在树切分与局部更新中将平均复杂度从全量 $O(n)$ 降到近似 $O(n-d)$。

工程要点：

- 先 copy 父缓存（连续内存复制），再覆盖后缀。
- 关节不变前缀严格复用，保证数值一致性。
- tool frame 作为固定末端变换在后缀阶段统一处理。

`_split_fk_pair`：同时生成左右子节点 FK 缓存对，供 HierAABBTree 切分时一步到位。

---

## 5. Numerical Envelope Path (Critical / Random)

### 5.1 Shared execution skeleton (`strategies/base.py`)

对每个连杆执行：

1. `_prepare_link`：筛除零长度或无关节影响连杆。
2. `_process_link`：策略子类执行采样。
3. `_evaluate_samples`：两次 FK（连杆起点/终点）+ 线性插值覆盖子段端点。
4. `_build_link_aabbs`：构造 `LinkAABBInfo` 与边界配置。

这里"分段端点线性插值"是实现关键：`n_sub` 增大时不按段重复 FK，显著节省计算。

### 5.2 Critical strategy (`critical.py`)

三阶段组合：

- 阶段 A：关键点枚举（边界组合、$k\pi/2$、耦合关节约束）。
- 阶段 B：约束流形随机采样（覆盖离散枚举难到达区域）。
- 阶段 C：L-BFGS-B 局部优化（以当前最优和精选种子精化极值）。

### 5.3 Random strategy (`random.py`)

- 大量随机采样 + 可选"避开关键点邻域"。
- 强制加入边界组合防止丢失区间端点极值。
- 最后统一 L-BFGS-B 精化。

---

## 6. Local Optimization (`optimization.py`)

`optimize_extremes` 对每个边界方向（`x_min/x_max/...`）执行有界 L-BFGS-B。

实现细节：

- 优化变量仅为 relevant joints，其他关节固定在中点。
- 种子预筛选采用 "top-1 exploit + farthest-1 explore"。
- 优化目标函数调用 `robot.get_link_position` 的单维投影。
- 收敛后通过 `_update_segs_for_point` 同步更新所有子段极值。

---

## 7. Data Semantics and Reproducibility

### 7.1 Boundary provenance

`BoundaryConfig` 除极值外还存储：

- `relevant_joints`
- `boundary_joints`（命中区间边界）
- `angle_constraints`（耦合约束命中描述）
- `is_aabb_vertex`（是否同属多个面极值）

### 7.2 Reproducible envelope artifacts

`AABBEnvelopeResult` 记录时间戳、样本数、方法名与分段参数；`ReportGenerator` 可直接输出 Markdown 报告，保证实验可复现。

---

## 8. Complexity and Numerical Considerations

### 8.1 Complexity summary

- 区间 FK：每层矩阵运算主导，约 $O(n)$ 关节链长度。
- 数值采样：约 $O(N_{samples}\cdot n_{links})$。
- 局部优化：$O(N_{seeds}\cdot N_{iter}\cdot C_{FK})$。

### 8.2 Numerical safeguards

- 区间端点比较与边界判定中使用小容差 (`1e-6` / `1e-10`)。
- 宽区间触发三角函数退化包络，防止错误收紧。
- 对零长度连杆显式跳过或退化为点 AABB。

### 8.3 Conservative bias

系统明确偏向"宁可误报，不可漏报"：这是为上层可行性保证服务，而非追求最紧几何包络。

---

## 9. End-to-End Pseudocode

```text
Input: joint_intervals Q, method M, strategy S
if M == interval:
    run interval_fk(Q) -> link_aabbs
else:
    for each link l:
        prepare relevant joints
        sample by S (critical/random)
        evaluate samples using two-end FK + segment interpolation
        run L-BFGS-B refinement on boundary objectives
        build LinkAABBInfo with boundary provenance
aggregate to AABBEnvelopeResult
```

---

## 10. Practical Recommendations

1. 需要安全证明优先：选 `method=interval`。
2. 需要更紧包络：选 `numerical+critical` 并提高 `n_subdivisions`。
3. 需要吞吐量：启用批量 FK (`get_link_positions_batch`) / Cython (`_fk_scalar_core`)，控制优化种子数与迭代上限。
4. 论文实验报告建议同时记录：方法、采样规模、分段数、总耗时与最终体积。

---

## 11. 接口收敛与配置透传

### 11.1 上层配置透传原则

AABB 层的参数应由上层统一配置并透传，避免在中间层硬编码：

1. `n_subdivisions`：分段粒度，影响包络紧致度与计算代价
2. `skip_zero_length`：零长度连杆跳过策略
3. `method` 与策略配置：`interval / numerical`

### 11.2 relevant joints 的一致性要求

当 `relevant_joints` 在策略层可用时，需保证：

1. 与 `Robot` 模型的关节依赖一致
2. 不影响保守语义（可跳过无影响维，不可跳过有效维）
3. 在 interval 与 numerical 两条路径中具有一致解释
