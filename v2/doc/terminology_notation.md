# 术语与符号表（Notation & Glossary）

## 1. 术语表（Glossary）

| 术语 | 英文 | 含义（在 v2 中的具体语义） |
|---|---|---|
| 关节空间 | Configuration Space / C-space | 由关节变量组成的 $\mathbb{R}^D$ 空间，规划在此进行。 |
| 工作空间 | Workspace | 连杆端点和障碍物所在的笛卡尔空间（通常是 3D）。 |
| box / 区间盒 | Axis-aligned hyperrectangle | 在 C-space 中每维用区间表示的超矩形。 |
| BoxNode | BoxNode | Forest 中的节点对象，包含区间、体积、seed、邻接关系。 |
| box forest | Box forest | 无重叠 box 集合及其邻接图。 |
| 邻接 | Adjacency | 两个 box 在容差意义下共享边界/面，可用于拓扑连通。 |
| 共享面 | Shared face | 相邻 box 的公共超平面区域，用于 waypoint 约束。 |
| seed | Seed sample | 用于触发 `find_free_box` 扩展的候选配置点。 |
| 吸收提升 | Promotion / Absorb | 层级树上行时用更大无碰撞节点替代子节点并回收旧 box。 |
| 子空间分区 | Partitioned subspace | KD 切分得到的互不重叠 C-space 子区间。 |
| 跨区补边 | Cross-partition connection | 仅在相邻子空间之间补充可行过渡边的过程。 |
| 严格不变量校验 | Strict invariants validation | 对重叠、邻接对称、引用有效性做强约束检查，失败即抛错。 |
| 保守碰撞 | Conservative collision | “无碰撞”结论可信；“碰撞”可能含误报。 |
| 直连 | Straight-line connection | 起终点在关节空间线段上离散检测无碰撞。 |
| shortcut | Shortcutting | 尝试删除中间路径点以缩短路径。 |
| box-aware smoothing | Box-aware smoothing | 平滑后将点投影回对应 box，保持路径在 box 集合中。 |
| fallback | Fallback | 高级求解不可用时的降级可行路径（如 Dijkstra + waypoint）。 |
| hcache | Hierarchical cache | `HierAABBTree` 的持久化缓存格式（HCACHE02）。 |

---

## 2. 符号表（Notation）

### 2.1 基础集合与变量

| 符号 | 含义 |
|---|---|
| $D$ | 关节自由度（维度） |
| $q \in \mathbb{R}^D$ | 一个关节配置向量 |
| $q_s, q_g$ | 起点与终点配置 |
| $\mathcal{Q}$ | 一个 C-space 区间盒（box） |
| $\mathcal{O}$ | 障碍物集合 |
| $\mathcal{C}_{free}$ | 自由空间 |
| $B_i$ | 第 $i$ 个 box 节点 |
| $V,E$ | 图的节点集与边集 |
| $G=(V,E)$ | Forest/Planner 使用的图结构 |

### 2.2 区间与包络

| 符号 | 含义 |
|---|---|
| $[l_i,u_i]$ | 第 $i$ 个关节维度区间 |
| $\prod_{i=1}^{D}[l_i,u_i]$ | D 维超矩形 |
| $T_{lo}, T_{hi}$ | 区间齐次变换上下界矩阵 |
| $A_{lo}, A_{hi}$ | 单关节区间 DH 变换上下界 |
| $AABB_\ell$ | 第 $\ell$ 条连杆的轴对齐包围盒 |
| $\mathbf{b}^{min},\mathbf{b}^{max}$ | AABB 对角最小/最大点 |

### 2.3 邻接与距离

| 符号 | 含义 |
|---|---|
| $w_{ij}^{(d)}$ | box $i,j$ 在维度 $d$ 的投影重叠宽度 |
| $tol$ | 邻接/接触判定容差 |
| $\|x-y\|_2$ | 关节空间欧氏距离 |
| $R$ | 从 `start` 可达的节点集合 |
| $\{\mathcal{Q}_k\}_{k=1}^{K}$ | KD 切分得到的子空间集合 |
| $K$ | 子空间数量（通常为 $2^{depth}$） |
| $P_a,P_b$ | 两个分区 ID |
| $\partial \mathcal{Q}$ | 子空间边界超平面 |

### 2.4 复杂度记号

| 符号 | 含义 |
|---|---|
| $N$ | box 数量 |
| $M$ | 障碍数量 |
| $L$ | 连杆段数量 |
| $K$ | 采样迭代次数 |
| $O(N^2D)$ | 全量邻接构建主项 |
| $O((|V|+|E|)\log|V|)$ | Dijkstra 复杂度 |

---

## 3. 缩写对照

| 缩写 | 全称 | 在项目中的作用 |
|---|---|---|
| FK | Forward Kinematics | 由关节配置计算连杆位姿/端点。 |
| AABB | Axis-Aligned Bounding Box | 统一碰撞粗检几何表示。 |
| GCS | Graph of Convex Sets | 可选路径优化分支。 |
| KDTree | k-dimensional tree | 近邻查询加速。 |
| SoA | Structure of Arrays | `NodeStore` 的缓存友好数据布局。 |
| mmap | Memory-mapped file | hcache 增量读写加速。 |
| FFB | Find Free Box | `find_free_box` 扩展过程的简称。 |

---

## 4. 名词到代码对象映射

| 名词 | 代码对象/函数 |
|---|---|
| 规划主流程 | `planner.box_rrt.BoxRRT.plan` |
| seed 采样 | `BoxRRT._sample_seed` |
| 层级扩展 | `forest.hier_aabb_tree.HierAABBTree.find_free_box` |
| 邻接构建 | `forest.deoverlap.compute_adjacency` |
| 端点接入 | `planner.connector.TreeConnector.connect_endpoints_to_forest` |
| box 序列优化 | `planner.gcs_optimizer.GCSOptimizer.optimize_box_sequence` |
| 路径平滑 | `planner.path_smoother.PathSmoother.smooth_in_boxes` |
| 区间 FK | `aabb.interval_fk.compute_interval_aabb` |

---

## 5. 使用建议

- 阅读分册前先通读本页，可统一符号口径。
- 写实验报告时建议直接复用本页符号，避免同一概念多种命名。
- 若后续新增模块（如 learning-based seed policy），请先在本页补充术语与符号再扩写正文。