# SafeBoxForest (SBF) 算法详述

> 三层架构: **AABB** (区间 FK) → **SafeBoxForest** (box 图) → **SBFPlanner** (规划管线)

---

## 1. 问题定义

给定 $D$-DOF 机器人，关节空间 $\mathcal{C} = \prod_{i=1}^D [l_i, u_i]$，障碍物集合 $\mathcal{O}$，起点 $q_s$、终点 $q_g$，求：

$$
\pi^* = \arg\min_\pi \text{cost}(\pi),\quad \pi(0)=q_s,\ \pi(1)=q_g,\ \pi(t)\in\mathcal{C}_{free}\ \forall t
$$

SBF 将问题分解为三层，自底向上逐层解决。

---

## 2. AABB 层（`aabb/`）

### 2.1 区间正运动学 (Interval FK)

给定关节区间 $[l_i, u_i]$，通过区间算术计算每条连杆的保守 AABB。

**核心步骤**：
1. 区间三角函数封闭 — `_isin(lo, hi)`, `_icos(lo, hi)`
2. 单关节区间 DH 矩阵构建 — 上下界齐次变换矩阵
3. 链式乘法 — 区间矩阵连乘得累积变换
4. 连杆端点提取 — 从累积变换提取 AABB

**增量 FK 优化**：树节点分裂时，复用父节点 FK 前缀矩阵，仅重算变化维度后续的矩阵乘法，平均节省 ~43% 运算。

### 2.2 AABB Calculator

`AABBCalculator` 支持多种策略：
- **interval** (默认)：区间 FK，保守但快速
- **critical**：关键点采样 + L-BFGS-B 优化，紧致
- **random**：随机采样统计包络

### 2.3 HCACHE 持久化

`HierAABBTree` 将区间 FK 结果以 **HCACHE02** 格式持久化：
- 二次运行直接加载，跳过全部 FK 计算
- 增量保存仅写入 dirty/new 节点
- Cache 仅绑定机器人运动学，不绑定障碍物 — 可跨场景复用

---

## 3. Forest 层（`forest/`）

### 3.1 SafeBoxForest 数据结构

无重叠 box 集合 $\{B_1, \ldots, B_N\}$ 及邻接图 $G = (V, E)$。

**三个不变量**：
- **I1** 非重叠：任意两个 box 内部不相交
- **I2** 邻接对称：$B_i \leftrightarrow B_j$ 当且仅当 $B_j \leftrightarrow B_i$
- **I3** 缓存一致：区间缓存与 box 集合同步

### 3.2 HierAABBTree — 层级区间树

- **SoA 存储**：`NodeStore` 使用 Structure-of-Arrays 布局
- **find_free_box (FFB)**：从 seed 配置出发，top-down 搜索+分裂+提升，找到最大无碰撞 box
- **active split dims**：根据机器人运动学自动排除无关维度（如 Panda 第 7 关节）

### 3.3 邻接检测

- **向量化全量**：$O(N^2 D)$ 分块上三角
- **共享面附加**：邻接 box 间的共享面用于 waypoint 优化

### 3.4 碰撞检测层

`CollisionChecker` 统一封装：
- 单配置检测：`check_config_collision(q)`
- 区间 box 检测：`check_box_collision(intervals)`
- 线段检测：`check_segment_collision(q1, q2, resolution)`
- 批量检测：`check_config_collision_batch(Q)`
- 空间索引：障碍物 > 20 时自动启用空间哈希加速

### 3.5 连通性管理

- **UnionFind**：并查集检测连通分量
- **bridge_islands**：三步策略 (dry-run FFB → overlap check → formal FFB) 桥接孤岛
- **coarsen**：维度扫描合并相邻 box，减少图规模

### 3.6 增量更新（优势 D）

```
invalidate_against_obstacle(new_obs)  # 标记碰撞 box
remove_invalidated()                   # 移除无效 box
regrow(seeds)                          # 利用旧 seed 回填
```

---

## 4. Planner 层（`planner/`）

### 4.1 SBFPlanner 10-步管线

| 步骤 | 操作 | 复杂度 |
|---|---|---|
| 0 | 加载/构建 HierAABBTree | $O(N \cdot D \cdot L)$ |
| 1 | 三层 seed 采样 (goal-biased / KD-guided / uniform) | $O(K)$ |
| 2 | FFB 扩展 → box 集合 | $O(K \cdot H \cdot L)$ |
| 3 | 波前边界扩展 (BFS) | $O(N_{frontier})$ |
| 4 | 粗化 (coarsen) | $O(N \cdot D)$ |
| 5 | 邻接构建 | $O(N^2 D)$ |
| 6 | 桥接 (bridge_islands) | $O(N_{islands})$ |
| 7 | 端点接入 + Dijkstra | $O((V+E)\log V)$ |
| 8 | 轨迹优化 (GCS / Dijkstra+waypoint) | 场景相关 |
| 9 | 路径平滑 (shortcut + box-aware MA) | $O(P \cdot C)$ |

### 4.2 三层 Seed 采样

1. **Goal-biased**：以概率 $p_g$ 直接使用 $q_g$
2. **KD-guided**：在最需要覆盖的分区内均匀采样
3. **Uniform**：关节空间均匀随机

### 4.3 波前边界扩展（优势 H）

从 start/goal 锚点 box 出发，BFS 波前传播：
1. 队列前端 box 的表面外侧采样 ($\epsilon$ 偏移)
2. FFB 扩展新 box，成功则入队
3. 失败超限则驱逐该 box
4. 队列清空后回退随机采样

### 4.4 轨迹优化

两种模式：
- **Drake GCS**：`GraphOfConvexSets` 求解器 (需要 pydrake)
- **Dijkstra + SOCP**：box 序列中心点 + 共享面约束优化

### 4.5 路径后处理

1. **Shortcut**：随机选取两点尝试直连
2. **Resample**：等距重采样
3. **Box-aware moving average**：加权平均后投影回 box

---

## 5. 核心优势总结

| 优势 | 机制 | 独特性 |
|---|---|---|
| **A. AABB 缓存持久化** | HCACHE02 → 二次运行跳过 FK | RRT/IRIS 不可持久化 |
| **B. 增量 FK 复用** | 父节点前缀复用 ~43% 节省 | 无类比机制 |
| **C. Forest 跨查询复用** | 同场景多查询 O(Dijkstra) | RRT 每次建树 |
| **D. 障碍物增量更新** | invalidate + regrow | RRT/IRIS 必须重建 |
| **E. Forest 单调增长** | bridge/expand 持久化 | 采样树不可复用 |
| **F. 并行分区扩展** | KD 分区 + 多进程 | RRT 并行化困难 |
| **G. 保守碰撞安全** | 区间 FK → 无碰撞可信 | 采样概率性 |
| **H. BFS 波前扩展** | 边界采样 + 队列传播 | RRT 无广度保证 |

---

## 6. 复杂度总结

| 操作 | 复杂度 | 说明 |
|---|---|---|
| 单次 FFB | $O(H \cdot D \cdot L)$ | $H$=树深度, $L$=连杆数 |
| Forest 构建 | $O(N \cdot H \cdot D \cdot L)$ | $N$=目标 box 数 |
| 邻接构建 | $O(N^2 D)$ | 分块向量化 |
| Dijkstra | $O((V+E)\log V)$ | 稀疏图 |
| 跨查询 plan | $O(\text{Dijkstra})$ | Forest 已构建 |
| 增量更新 | $O(N_{invalid} \cdot H \cdot D \cdot L)$ | 仅重建失效区域 |
