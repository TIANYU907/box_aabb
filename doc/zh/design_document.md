# Box-AABB 系统设计文档

> **版本**: 3.2.0 (v5 BoxForest + HierAABBTree)  
> **日期**: 2026-02-11  
> **作者**: TIAN

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心思想：从 AABB 到无碰撞 Box](#2-核心思想从-aabb-到无碰撞-box)
3. [数学基础](#3-数学基础)
4. [系统架构](#4-系统架构)
5. [基础层：box_aabb 核心库](#5-基础层box_aabb-核心库)
6. [规划层：planner 模块](#6-规划层planner-模块)
7. [HierAABBTree：层级自适应 AABB 缓存树](#7-hieraabbtree层级自适应-aabb-缓存树)
8. [BoxForest：无重叠 Box 森林](#8-boxforest无重叠-box-森林)
9. [路径规划流水线](#9-路径规划流水线)
10. [缓存系统](#10-缓存系统)
11. [可视化与基准测试](#11-可视化与基准测试)
12. [已知问题与改进方向](#12-已知问题与改进方向)
13. [附录：关键数据结构速查](#13-附录关键数据结构速查)

---

## 1. 项目概述

### 1.1 解决什么问题

给定一个串联机械臂和一组 AABB 障碍物，在关节空间（C-space）中找到一条从起始配置到目标配置的**无碰撞路径**。

与传统 PRM/RRT 逐点采样不同，本项目的核心理念是：

> **在 C-space 中构建大块的无碰撞矩形区域（Box），将路径规划转化为凸集图（Graph of Convex Sets）上的图搜索问题。**

这一方法的优势在于：
- **安全性保证**：Box 内任意配置均经数学证明无碰撞（区间算术的保守性）
- **路径质量**：Box 内可自由移动，路径天然平滑
- **可复用性**：Box 森林仅绑定机器人型号，跨场景惰性验证即可复用

### 1.2 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.9+ |
| 核心依赖 | NumPy |
| 可选依赖 | SciPy (优化), Matplotlib (可视化), Drake (GCS) |
| 构建 | setuptools + pyproject.toml |
| 测试 | pytest |

### 1.3 支持的机器人

通过 JSON 配置文件定义，内置：

| 配置 | DOF | 说明 |
|------|-----|------|
| `2dof_planar` | 2 | 平面双连杆，L1=L2=1.0m |
| `3dof_planar` | 3 | 平面三连杆 |
| `panda` | 7 | Franka Emika Panda 协作机器人 |

机器人模型使用**修正 DH 约定**（Modified DH Convention），支持旋转关节和移动关节，并可通过 `tool_frame` 提供末端固定连杆参数。

---

## 2. 核心思想：从 AABB 到无碰撞 Box

### 2.1 两个关键空间

```
工作空间 (Workspace)                   关节空间 (C-space)
┌──────────────────┐                  ┌──────────────────┐
│  机械臂 + 障碍物   │  ← FK(q) ←     │  q = (q₁, q₂, …) │
│  3D 笛卡尔坐标    │  → IK →         │  N维超矩形        │
│                   │                  │  Box ⊂ C-free     │
│  连杆 AABB ∩      │                  │                   │
│  障碍物 AABB = ∅?  │                  │                   │
└──────────────────┘                  └──────────────────┘
```

### 2.2 保守碰撞检测链

对于 C-space 中的一块矩形区域 $B = [q_1^-, q_1^+] \times \cdots \times [q_n^-, q_n^+]$：

1. **区间正运动学**：将关节区间代入 DH 变换，使用区间/仿射算术得到每个连杆位置的区间包围 $\text{AABB}_\ell(B)$
2. **AABB 碰撞检测**：$\text{AABB}_\ell(B) \cap \text{Obstacle}_k = \emptyset$（分离轴测试）
3. **判定**：若所有连杆的 AABB 与所有障碍物均不重叠，则 $B$ 中的所有配置一定无碰撞

**保守性**：区间算术天然过估计（$\text{AABB}_\ell(B) \supseteq$ 真实包络），因此：
- `check = False` ⇒ **一定无碰撞**（安全保证）
- `check = True` ⇒ **可能碰撞**（需要细分）

### 2.3 Box 的拓展逻辑

从一个种子配置 $q_{\text{seed}}$ 出发，不断"生长"出最大的无碰撞 Box：

```
         q₂
         ↑
         │  ┌─────────────┐
         │  │             │
         │  │   seed ●    │  ← Box: 区间FK无碰撞
         │  │             │
         │  └─────────────┘
         └──────────────────→ q₁
```

两种拓展策略：
- **Balanced Expansion**（旧版）：各维度试探性扩张 + 二分搜索边界
- **HierAABBTree 切分**（新版）：自顶向下递归二分，碰撞则切分，无碰撞则停止

---

## 3. 数学基础

### 3.1 区间算术

**定义**：区间 $[a, b]$ 表示一个范围，基本运算：

$$
[a, b] + [c, d] = [a+c, \; b+d]
$$
$$
[a, b] \times [c, d] = [\min(ac, ad, bc, bd), \; \max(ac, ad, bc, bd)]
$$

**性质**：
- 封闭性：任何初等运算的结果仍是区间
- 包含性：真实值一定在计算得到的区间内
- 依赖问题（Dependency Problem）：$x - x \neq [0, 0]$（当 $x = [a, b]$，得到 $[a - b, b - a]$）

### 3.2 仿射算术

仿射形式解决了区间算术的依赖问题：

$$
\hat{x} = x_0 + \sum_{i=1}^{n} x_i \varepsilon_i, \quad \varepsilon_i \in [-1, 1]
$$

其中 $x_0$ 为中心值，$x_i$ 为噪声系数，$\varepsilon_i$ 为噪声符号。

**关键优势**：$\hat{x} - \hat{x} = 0$（完全消除），因为共享相同的噪声符号。

**转换为区间**：

$$
\hat{x} \to \left[x_0 - \sum|x_i|, \; x_0 + \sum|x_i|\right]
$$

### 3.3 仿射三角函数（Chebyshev 线性化）

对于 $\sin(\hat{x})$，当区间宽度 $\leq \pi/2$ 时使用一阶泰勒展开：

$$
\sin(\hat{x}) \approx \sin(x_0) + \cos(x_0) \cdot (\hat{x} - x_0) + \delta
$$

余项上界 $|\delta| \leq r^2 / 2$，其中 $r$ 为半径。这使得 $\sin(q_1)$ 保留对 $q_1$ 的噪声符号依赖，后续矩阵乘法中的交叉项可以部分抵消，显著减少过估计。

当区间宽度 $> \pi/2$ 时，回退到标准区间求值（`I_sin` / `I_cos`），精确跟踪极值点位置。

### 3.4 区间正运动学

将 Modified DH 齐次变换矩阵的链式乘法推广到区间/仿射域：

$$
T_i = T_{i-1} \cdot \begin{bmatrix}
\cos\theta_i & -\sin\theta_i & 0 & a_i \\
\sin\theta_i \cos\alpha_i & \cos\theta_i \cos\alpha_i & -\sin\alpha_i & -d_i \sin\alpha_i \\
\sin\theta_i \sin\alpha_i & \cos\theta_i \sin\alpha_i & \cos\alpha_i & d_i \cos\alpha_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

当 $\theta_i = q_i + \theta_{\text{offset}}$ 为仿射形式时，$\cos\theta_i$ 和 $\sin\theta_i$ 由 `smart_cos` / `smart_sin` 计算，矩阵元素成为仿射形式，4×4 矩阵乘法保留所有噪声符号的线性相关性。

最终从 $T_i$ 的平移列（第 4 列前 3 行）提取连杆端点位置的区间范围，合并相邻端点得到连杆 AABB。

### 3.5 AABB 合并的数学性质

对于 C-space 区域 $D = D_1 \cup D_2$，每个连杆 $\ell$ 的 AABB 满足：

$$
\text{AABB}_\ell(D) = \text{BBox}\left(\text{AABB}_\ell(D_1) \cup \text{AABB}_\ell(D_2)\right)
$$

即**逐连杆取各轴 min/max**。这是 HierAABBTree 渐进精化的理论基础：子节点的 AABB union ≤ 父节点的直接 AABB（单调变紧）。

> **已知问题**：当前 `_union_aabb` 实现的是**逐连杆 AABB 的 bounding box**，而非连杆占据空间的真正布尔并集。这会引入假体积（False Volume），是后续改进方向之一（见第 12 节）。

---

## 4. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    应用层                             │
│  examples/  benchmarks/                              │
│  visualize_box_forest.py  bench_hier_cache.py        │
├─────────────────────────────────────────────────────┤
│                  规划层 (planner/)                     │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────┐ │
│  │ HierAABBTree │  │  BoxForest    │  │  BoxRRT   │ │
│  │ (缓存树)      │  │  (无重叠森林)  │  │  (路径规划)│ │
│  └──────┬───────┘  └───────┬───────┘  └─────┬─────┘ │
│         │                  │                │        │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌─────┴─────┐ │
│  │ CollisionChk │  │  deoverlap   │  │ Connector │ │
│  │ (碰撞检测)    │  │  (去重叠)     │  │ (图搜索)  │ │
│  └──────┬───────┘  └──────────────┘  └─────┬─────┘ │
│         │                                   │        │
│  ┌──────┴───────┐                    ┌─────┴─────┐ │
│  │ BoxExpander  │                    │ GCS/scipy │ │
│  │ (Box扩张)    │                    │ (路径优化) │ │
│  └──────────────┘                    └───────────┘ │
├─────────────────────────────────────────────────────┤
│                基础层 (box_aabb/)                     │
│  ┌───────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │  Robot     │  │ interval_math │  │ interval_fk │ │
│  │  (DH运动学) │  │ (区间/仿射)    │  │ (区间FK)    │ │
│  └───────────┘  └───────────────┘  └─────────────┘ │
│  ┌───────────┐  ┌───────────────┐                   │
│  │  models   │  │ aabb_calcul.  │                   │
│  │  (数据模型) │  │ (策略调度)     │                   │
│  └───────────┘  └───────────────┘                   │
└─────────────────────────────────────────────────────┘
```

**两层分离**：
- `box_aabb/`：纯运动学 + AABB 计算库，不依赖场景或规划概念
- `planner/`：基于 box_aabb 构建的路径规划系统

---

## 5. 基础层：box_aabb 核心库

### 5.1 Robot (`robot.py`)

**职责**：基于 DH 参数的串联机械臂运动学模型。

```python
class Robot:
    dh_params: List[Dict]        # [{alpha, a, d, theta, type}, ...]
    joint_limits: List[Tuple]    # [(lo, hi), ...]
    coupled_pairs: List[Tuple]   # 耦合关节对 (用于采样策略)
    coupled_triples: List[Tuple] # 耦合三元组
    tool_frame: Optional[Dict]   # 末端固定连杆
```

**关键方法**：
- `forward_kinematics(q)` → 4×4 齐次变换矩阵
- `get_link_positions(q)` → 各连杆端点的 3D 坐标列表
- `fingerprint()` → 基于 DH 参数 + 关节限制的 SHA256 哈希，用于缓存索引
- `zero_length_links` → 零长度连杆集合（跳过碰撞检测）

**工具坐标系（tool_frame）**：在 Modified DH 中，最后一行 DH 参数的 `a` 不体现在最后一帧的位置上（它在"下一帧"才体现），因此用 `tool_frame` 补充末端连杆。

### 5.2 interval_math.py

**职责**：区间算术 + 仿射算术的基础实现。

| 类/函数 | 说明 |
|---------|------|
| `Interval` | 基础区间 `[lo, hi]`，支持 `+, -, *, neg` |
| `AffineForm` | 仿射形式 $x_0 + \sum x_i \varepsilon_i$，保留线性相关性 |
| `I_sin` / `I_cos` | 标准区间三角函数，精确跟踪极值点 |
| `smart_sin` / `smart_cos` | Chebyshev 线性化仿射三角函数 |

**全局噪声计数器** `_AFFINE_NOISE_COUNTER`：每次新的 FK 计算前通过 `reset_affine_noise_counter()` 重置为 0。每个关节/乘法余项分配唯一索引，保证不同变量的噪声符号不混淆。

### 5.3 interval_fk.py

**职责**：使用区间/仿射算术计算各连杆的保守 AABB。

**输入**：`Robot` + 关节区间列表 `[(lo, hi), ...]`  
**输出**：`List[LinkAABBInfo]`，每个元素包含某连杆在 3D 工作空间中的 AABB 范围

**算法流程**：
1. 将关节区间转为 `AffineForm`（点区间不创建噪声符号）
2. 逐关节计算 DH 变换矩阵（仿射域中的 4×4 乘法）
3. 从各变换矩阵提取连杆端点位置的区间范围
4. 合并相邻端点得到连杆 AABB
5. 可选：连杆子分段（`n_sub > 1` 时线性插值端点区间）

### 5.4 models.py

**核心数据类**：

| 数据类 | 说明 |
|--------|------|
| `LinkAABBInfo` | 单连杆的 AABB：`min_point[3]`, `max_point[3]`, `link_index`, `is_zero_length` |
| `BoundaryConfig` | 使 AABB 某面达到极值的关节配置 |
| `AABBEnvelopeResult` | 完整 AABB 计算结果：所有连杆 + 元数据 |

### 5.5 AABBCalculator (`aabb_calculator.py`)

**职责**：策略调度器，4 种计算方法：

| 方法 | 策略类 | 说明 |
|------|--------|------|
| `numerical/critical` | `CriticalStrategy` | 关键点枚举 + 约束优化 + 流形随机采样 |
| `numerical/random` | `RandomStrategy` | 纯随机采样 + 局部优化 |
| `numerical/hybrid` | `HybridStrategy` | 关键点 + 流形 + 随机 |
| `interval` | 直接调用 `interval_fk` | 区间/仿射算术保守估计 |

在 Box-RRT 规划中主要使用 `interval` 方法（快速、保守）。数值方法用于对比分析和论文实验。

---

## 6. 规划层：planner 模块

### 6.1 障碍物与场景 (`obstacles.py`)

```python
class Scene:
    def add_obstacle(min_point, max_point, name) → Obstacle
    def get_obstacles() → List[Obstacle]
    def to_json(filepath)  # 序列化
```

**2D 自动扩展**：当 `min_point` 只有 2 维时，自动扩展 z 范围为 `[-1e3, 1e3]`，使 2D 场景可直接复用 3D 碰撞检测逻辑。

### 6.2 碰撞检测 (`collision.py`)

```python
class CollisionChecker:
    def check_config_collision(q) → bool       # 点碰撞（FK+AABB）
    def check_box_collision(intervals) → bool   # Box碰撞（区间FK+AABB）
    def check_config_collision_batch(qs) → array  # 批量点碰撞
```

**保守性**：`check_box_collision` 使用区间 FK，返回 `False` 时**保证**安全。可选的 AABB 缓存避免重复 FK 计算。

### 6.3 Box 扩张 (`box_expansion.py`)

`BoxExpander` 提供两种策略将种子配置扩展为最大无碰撞 Box：

#### Greedy 策略（旧版）
1. 计算各维度的 Jacobian 范数，按升序排列（范数小 = 对末端影响小 = 可安全大幅扩展）
2. 逐维度正/负方向二分搜索碰撞边界
3. 迭代多轮直到体积增长 < 0.1%

#### Balanced 策略（改进版）
1. 维护 $2n$ 个候选方向（每维度 ±）
2. 每步评估所有候选：自适应步长 + 二分搜索安全边界
3. 选择**最大体积增益**的方向（减去重叠惩罚）
4. 当增长 < 0.5% 时早停

**`hard_overlap_reject`**：新 Box 被已有 Box 的边界截止（不允许重叠进入已有空间），保证后续 deoverlap 不会碎片化。

### 6.4 去重叠 (`deoverlap.py`)

**职责**：将可能重叠的 BoxNode 集合转化为互不重叠的碎片集。

**核心算法**：`subtract_box(base, cut)` — 超矩形减法

```
对每个维度 d：
  if base[d] 超出 cut[d] 的左侧:
    切出左碎片 (base[d].lo, cut[d].lo)
    收缩 base[d].lo = cut[d].lo
  if base[d] 超出 cut[d] 的右侧:
    切出右碎片 (cut[d].hi, base[d].hi)
    收缩 base[d].hi = cut[d].hi
结果: 最多 2D 个碎片
```

**deoverlap 策略**：先来先得（First-Come-First-Served），每个新 Box 被所有已提交 Box 切割。微小重叠（< `min_fragment_volume`）容忍为邻接。

**邻接计算**：`compute_adjacency(boxes, tol)` — 向量化 $O(N^2 \cdot D)$，两 Box 邻接当且仅当：
- 恰好 1 个维度面接触（overlap ≤ tol）
- 其余维度有正投影重叠

### 6.5 TreeConnector (`connector.py`)

**v5.0 邻接模式**：直接使用 deoverlap 的邻接表建图，共享面中心作为过渡路点。

```python
def build_adjacency_graph(boxes, adjacency, q_start, q_goal) → graph
```

图的节点 = Box ID + `'start'` + `'goal'`，边权 = 欧氏距离。

### 6.6 GCS 优化器 (`gcs_optimizer.py`)

| 后端 | 说明 |
|------|------|
| Drake GCS | `HPolyhedron` 凸集 + L2 代价 + 连续性约束 |
| SciPy fallback | L-BFGS-B 最小化路径长度，变量 = 共享面上的路点坐标 |

**v5.0 共享面优化**：对 Box 序列中相邻 Box 的共享面，以面上的坐标为优化变量，面的边界为约束，最小化总路径长度。固定维度（接触面法向）锁定，其余维度为自由变量。

### 6.7 路径平滑 (`path_smoother.py`)

- **随机快捷（shortcut）**：随机选两点，若直线段无碰撞则删除中间点
- **Box 感知快捷（v5.0）**：段 $p_i \to p_j$ 必须被 Box 序列 $[i, j]$ 的并集包含
- **滑动窗口平滑**：邻域均值，碰撞则回退
- **Box 感知平滑**：均值点被裁剪到对应 Box 范围内

---

## 7. HierAABBTree：层级自适应 AABB 缓存树

### 7.1 动机

传统 `BoxExpander` 的问题：
1. **细长 Box**：Jacobian 引导优先扩展"安全"维度，导致先扩展的维度远大于后扩展的维度
2. **连接性差**：贪心扩张不考虑全局拓扑
3. **重复计算**：每次扩张独立调用区间 FK，无法复用

### 7.2 设计思想

**KD-tree 式二叉空间切分**，将整个 C-space 递归二分：

```
                [root: 全 C-space]
               /                    \
     [q₁ < π, q₂ full]    [q₁ ≥ π, q₂ full]
        /        \              /        \
   [q₂ < 0]  [q₂ ≥ 0]    [q₂ < 0]  [q₂ ≥ 0]
      ...
```

- **切分维度**：`depth % n_dims`（轮转维度，保证各维度均匀切分）
- **切分点**：区间中点（二等分）
- **惰性求值**：只在查询路径上创建节点和计算 AABB
- **渐进精化**：`refined_aabb = union(left.aabb, right.aabb) ≤ raw_aabb`

### 7.3 核心数据结构

```python
@dataclass
class HierAABBNode:
    intervals: List[Tuple[float, float]]  # C-space 超矩形
    depth: int
    raw_aabb: List[LinkAABBInfo]          # 直接 interval FK（松）
    refined_aabb: List[LinkAABBInfo]      # 子节点 union（更紧）
    split_dim: int                        # 切分维度
    split_val: float                      # 切分值
    left: HierAABBNode                    # 左子节点
    right: HierAABBNode                   # 右子节点
    parent: HierAABBNode                  # 父节点
```

### 7.4 find_free_box 算法

给定种子配置 `seed` 和障碍物列表，找到包含 seed 的最大无碰撞 Box：

```
算法: find_free_box(seed, obstacles, max_depth)

Phase 1 — 下行（切分直到无碰撞）:
    node ← root
    path ← []
    while True:
        path.append(node)
        aabb ← node.refined_aabb or node.raw_aabb
        
        if not collides(aabb, obstacles):
            break                    # 整个节点无碰撞
        
        if node.depth ≥ max_depth:
            return None              # 达到最大深度仍碰撞
        
        if edge_width < min_edge_length × 2:
            return None              # 再分就低于最小边长
        
        split(node)                  # 惰性二分裂
        node ← child containing seed

Phase 2 — 上行（尝试合并更大 Box）:
    result ← current node
    for parent in path (reverse):
        if not collides(parent.refined_aabb, obstacles):
            result ← parent          # 父节点也无碰撞
        else:
            break
    
    return result.intervals
```

**时间复杂度**：$O(D \cdot L)$ 次碰撞检测 + $O(L)$ 次 FK，其中 $D$ 为关节维数，$L$ 为沿路径的深度。

### 7.5 切分与精化

```python
def _split(node):
    dim = node.depth % n_dims
    mid = (lo + hi) / 2
    
    node.left  = HierAABBNode(intervals[dim] = (lo, mid))
    node.right = HierAABBNode(intervals[dim] = (mid, hi))
    
    # 两个子节点都计算 AABB（多 1 次 FK，但启用精化）
    compute_aabb(node.left)
    compute_aabb(node.right)
    
    # 精化：union(children) ≤ raw_aabb
    node.refined_aabb = union(node.left.aabb, node.right.aabb)
    
    # 向上传播精化
    propagate_up(node.parent)
```

**精化的单调性**：每次新的切分只会使 `refined_aabb` 变紧或不变，永远不会变松。这保证了缓存的正确性——旧的"通过"判定在新精化后仍然有效。

### 7.6 与 BoxForest 的协同

HierAABBTree 负责"找 Box"，BoxForest 负责"存 Box"：

```
HierAABBTree.find_free_box(seed)
    → intervals
    → BoxNode(intervals, seed)
    → BoxForest.add_boxes_incremental([node])
        → deoverlap（去掉与已有 Box 重叠的部分）
        → compute_adjacency_incremental（更新邻接关系）
```

### 7.7 min_edge_length 与早停

- **min_edge_length**：当待分割维度宽度 < `2 × min_edge_length` 时停止切分，防止产生过于微小的 Box。默认 0.01 rad（约 0.57°）。
- **早停滑动窗口**：当最近 N 个 Box 的体积都 < 阈值时，判定空间已饱和，停止拓展。

---

## 8. BoxForest：无重叠 Box 森林

### 8.1 设计目标

维护一组**互不重叠**的 BoxNode 及其**邻接关系**，形成 C-free 的覆盖和连通图。

### 8.2 不变量

1. **零重叠**：`∀ i ≠ j, overlap(box_i, box_j) < min_fragment_volume`
2. **邻接正确**：两 Box 相邻 ⟺ 有且仅有 1 维面接触 + 其余维投影重叠
3. **ID 唯一**：每个 Box 有唯一 `node_id`

### 8.3 增量添加

```python
def add_boxes_incremental(new_boxes):
    for box in new_boxes:
        fragments = [box.intervals]
        for committed in existing_boxes:
            fragments = [subtract(frag, committed) 
                        for frag in fragments 
                        if overlap(frag, committed) ≥ threshold]
        for frag in fragments:
            if volume(frag) ≥ min_fragment_volume:
                add_to_forest(frag)
    # 增量邻接更新 O(K·N·D)
    compute_adjacency_incremental(added, all_boxes)
```

### 8.4 查询接口

| 方法 | 说明 | 复杂度 |
|------|------|--------|
| `find_containing(q)` | 找包含 q 的 Box | O(N) |
| `find_nearest(q)` | 找离 q 最近的 Box | O(N) |
| `validate_boxes(checker)` | 惰性碰撞验证 | O(N) × FK |
| `get_uncovered_seeds(n, rng)` | 采样未覆盖点 | O(N) per sample |

---

## 9. 路径规划流水线

### 9.1 BoxRRT 10 步流水线

```
1. 验证 start/goal 无碰撞
2. 尝试直接连接
3. 加载/创建 BoxForest
4. 验证已有 Box 对当前场景的安全性
5. 扩展新 Box（start/goal seed + 随机采样）
6. Deoverlap + 邻接更新
7. 构建搜索图（邻接边 + 端点连接边）
8. Dijkstra 搜索（断连时尝试桥接修复）
9. GCS/scipy 路点优化
10. Box感知快捷 + 平滑后处理
```

### 9.2 HierAABBTree 可视化模式的拓展策略

在 `visualize_box_forest.py` 的 `grow_forest_hier_animated` 中，使用 DFS 边界优先的拓展策略：

```
主循环:
    if 上一个 Box 不为空:
        ── DFS 边界采样 ──
        在上一个 Box 的各面上均匀采样若干 seed
        对每个 seed 尝试 find_free_box + 加入 forest
        成功 → 继续从新 Box 边界拓展（DFS 深入）
        全部失败 → 切换到远处新 seed
    
    ── 最远点采样 ──
    随机生成候选点，选离已有 Box 最远的
    尝试 find_free_box + 加入 forest
    成功 → 下一轮 DFS 从新 Box 开始
    失败 → 计数器+1，连续失败过多则停止
```

这种策略使得 Box 像"藤蔓"一样从一个区域向外生长，相邻 Box 天然有邻接关系，同时周期性切换到远处填补空白区域。

---

## 10. 缓存系统

### 10.1 两级缓存

| 缓存 | 绑定对象 | 作用域 | 文件格式 |
|------|---------|--------|---------|
| **AABBCache** | Robot + 关节区间 → `LinkAABBInfo` | 区间 FK 结果 | pickle |
| **HierAABBTree** | Robot → KD-tree | C-space 全局切分结构 | pickle |

### 10.2 AABBCache 细节

```
.cache/aabb/{robot_name}_{fingerprint[:16]}.pkl
```

内部使用两个 `IntervalStore`（区间方法 + 数值方法），每个 Store 维护：
- **精确哈希索引**：完全相同的区间 → $O(1)$ 查找
- **NumPy 向量化层**：`(M, N, 2)` 数组，支持子集/包含关系的批量查询
- **合并查询**：找到覆盖查询区间的多个已缓存子区间，逐连杆合并它们的 3D AABB

**替换逻辑**：
- 区间方法：更小体积 = 更精确 → 保留较小的
- 数值方法：更大体积 = 更完整 → 保留较大的

### 10.3 HierAABBTree 全局缓存

```
.cache/hier_aabb/{robot_name}_{fingerprint[:16]}.pkl
```

**零配置自动机制**：
- `auto_load(robot)` → 若缓存存在且指纹匹配则加载，否则新建空树
- `auto_save()` → save 到全局缓存目录

**持久化处理**：
- 序列化前：剥离 `parent` 引用（避免循环）和 `robot` 对象
- 反序列化后：`_rebuild_parents()` 恢复 parent 链
- 指纹校验：加载时比对 `robot.fingerprint()`，不匹配则拒绝

**缓存增长特性**（来自基准测试）：

| Run | 缓存节点 | 新增 FK | find_free_box 时间 |
|-----|---------|---------|-------------------|
| 1 (cold) | 0→1609 | 1609 | 0.80s |
| 2 (warm) | 1609→2509 | 900 | 0.54s |
| 3 | 2509→3085 | 576 | 0.35s |
| ... | ... | 递减 | 递减 |
| 10 | 5523→5755 | 232 | 0.23s |

每次运行探索不同区域（不同随机种子），新增节点逐渐减少，`find_free_box` 持续加速。

---

## 11. 可视化与基准测试

### 11.1 visualize_box_forest.py

**功能**：在 2DOF 随机障碍物场景中，逐步拓展 BoxForest 并生成动画。

**输出**：
- `frames/step_NNNN.png` — 每步快照
- `forest_growth.gif` / `.mp4` — 生长动画
- `final_forest.png` — 最终状态（邻接度着色+邻接边）
- `collision_map.png` — C-space 碰撞地图
- `overlay.png` — BoxForest 叠加碰撞地图
- `workspace.png` — 工作空间 arm 姿态
- `stats.txt` — 统计摘要 + 计时报告 + 拓展日志

**分环节计时**：

| 环节 | 典型占比 (有渲染) | 典型占比 (无渲染) |
|------|------------------|------------------|
| render | ~94% | 0% |
| find_free_box | ~4% | ~50-75% |
| deoverlap | ~1% | ~20-40% |
| collision_check | ~0.5% | ~10-20% |
| sampling | ~0.5% | ~5-10% |

> **重要发现**：matplotlib 渲染占 94% 的时间。算法本身在无渲染模式下仅需 ~0.3s（300 boxes，2DOF）。

### 11.2 bench_hier_cache.py

**功能**：无渲染模式下重复运行 N 次，第 1 次无缓存，后续使用缓存，统计各项指标。

**关键发现**（10 次运行，2DOF，5 障碍物，300 boxes）：

| 指标 | 冷启动 | 热缓存 (9次均值) | 加速比 |
|------|--------|-----------------|--------|
| 总耗时 | 1.03s | 0.53s ± 0.11 | 1.94× |
| find_free_box | 0.80s | 0.29s ± 0.10 | 2.74× |
| total_volume | 5.57 | 6.84 ± 0.31 | +23% |
| adj_edges | 458 | 468 ± 7 | +2% |

热缓存不仅更快，覆盖体积也更大（已有的切分节点帮助快速定位更好的 Box）。

---

## 12. 已知问题与改进方向

### 12.1 _union_aabb 的假体积问题

**现状**：`HierAABBTree._union_aabb()` 合并两组 `List[LinkAABBInfo]` 时，对每个连杆取各轴 min/max。这得到的是连杆 AABB 的 bounding box，而非连杆实际扫掠区域的真实并集。

**影响**：
- **碰撞误判**：高层节点的 `refined_aabb` 可能覆盖远大于真实包络的区域
- **精化效果有限**：即使子节点很紧，union 后可能仍然很松
- **根节点几乎必碰**：L 形或弧形运动范围的 bounding box 远大于真实包络

**计划方案**：用 3D 体素占据网格替代 `List[LinkAABBInfo]`：

```python
# 替代 List[LinkAABBInfo]
node.occupancy: np.ndarray  # bool 3D grid (Nx, Ny, Nz)

# 精确 union
parent.occupancy = left.occupancy | right.occupancy  # 布尔并集

# 碰撞检测
np.any(node.occupancy & obstacle_grid)  # 向量化
```

分辨率 2cm，2DOF 退化为 2D（nz=1）。已制定 15 步实施计划，尚未执行。

### 12.2 高维扩展性

当前 2DOF 场景表现优异（~1s / 300 boxes），但 7DOF Panda 机器人面临：
- 区间 FK 过估计随维度指数增长
- Box 体积随维度指数缩小
- 碰撞地图无法直接可视化

**方向**：子空间切分、重要性采样、自适应深度。

### 12.3 与主规划管线的集成

HierAABBTree 目前仅在 `visualize_box_forest.py` 中使用。下一步应替换 `BoxRRT` 中的 `BoxExpander`，使主规划流水线也能受益于层级缓存。

### 12.4 渲染性能

matplotlib 占 94% 运行时间。对于纯算法评估应使用无渲染模式（`bench_hier_cache.py`）。若需要可视化，考虑：
- 减少帧数（增大 `step_interval`）
- 仅渲染最终状态
- 使用更快的渲染后端

---

## 13. 附录：关键数据结构速查

### 文件 → 类/函数 映射

| 文件 | 主要导出 |
|------|---------|
| `box_aabb/robot.py` | `Robot`, `load_robot` |
| `box_aabb/interval_math.py` | `Interval`, `AffineForm`, `smart_sin`, `smart_cos` |
| `box_aabb/interval_fk.py` | `compute_interval_aabb()` |
| `box_aabb/models.py` | `LinkAABBInfo`, `AABBEnvelopeResult` |
| `box_aabb/aabb_calculator.py` | `AABBCalculator` |
| `planner/models.py` | `BoxNode`, `Obstacle`, `PlannerConfig`, `PlannerResult` |
| `planner/obstacles.py` | `Scene` |
| `planner/collision.py` | `CollisionChecker`, `aabb_overlap` |
| `planner/box_expansion.py` | `BoxExpander` |
| `planner/deoverlap.py` | `deoverlap`, `subtract_box`, `compute_adjacency` |
| `planner/box_forest.py` | `BoxForest` |
| `planner/hier_aabb_tree.py` | `HierAABBTree`, `HierAABBNode` |
| `planner/connector.py` | `TreeConnector` |
| `planner/gcs_optimizer.py` | `GCSOptimizer` |
| `planner/path_smoother.py` | `PathSmoother` |
| `planner/aabb_cache.py` | `AABBCache`, `IntervalStore` |
| `planner/box_rrt.py` | `BoxRRT` |

### 缓存文件路径

```
.cache/
├── aabb/
│   ├── 2DOF-Planar_6abf5a8e1a704654.pkl
│   └── Panda_a1b2c3d4e5f6g7h8.pkl
└── hier_aabb/
    ├── 2DOF-Planar_6abf5a8e1a704654.pkl
    └── Panda_a1b2c3d4e5f6g7h8.pkl
```

### 典型调用链

```
用户: python -m examples.visualize_box_forest --mode hier

main()
 ├── load_robot("2dof_planar")
 ├── random_scene_2d(robot, n_obs, rng)
 ├── scan_collision_map(robot, scene, ...)
 └── grow_forest_hier_animated(...)
      ├── HierAABBTree.auto_load(robot, joint_limits)
      ├── BoxForest(robot_fp, joint_limits, config)
      └── for seed in sampling_loop:
           ├── _try_add_box(seed)
           │    ├── checker.check_config_collision(seed)
           │    ├── forest.find_containing(seed)
           │    ├── hier_tree.find_free_box(seed, obstacles)
           │    │    ├── _ensure_aabb(node)
           │    │    │    └── compute_interval_aabb(robot, intervals, ...)
           │    │    ├── _link_aabbs_collide(aabb, obstacles)
           │    │    ├── _split(node)       ← 碰撞则切分
           │    │    └── 上行合并
           │    └── forest.add_boxes_incremental([box])
           │         ├── subtract_box(new, existing)
           │         └── compute_adjacency_incremental(added, all)
           └── render_frame(...)
```
