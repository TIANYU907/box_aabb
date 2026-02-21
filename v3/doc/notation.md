# 术语与符号表（Notation & Glossary）

> SafeBoxForest Planner (SBF) — v3 统一符号参考

---

## 1. 术语表（Glossary）

| 术语 | 英文 | 含义 |
|---|---|---|
| 关节空间 | Configuration Space / C-space | 由关节变量组成的 $\mathbb{R}^D$ 空间，规划在此进行 |
| 工作空间 | Workspace | 连杆端点和障碍物所在的笛卡尔空间（通常 3D） |
| box / 区间盒 | Axis-aligned hyperrectangle | C-space 中每维用区间表示的超矩形 |
| BoxNode | `forest.models.BoxNode` | Forest 中的节点，包含区间、体积、seed、邻接关系 |
| SafeBoxForest | `forest.safe_box_forest.SafeBoxForest` | 无重叠 box 集合及其邻接图 |
| 邻接 | Adjacency | 两个 box 在容差下共享边界/面，可用于拓扑连通 |
| strict 邻接 | Strict face-touching | 某维度精确接触且其余维度有重叠 |
| loose 邻接 | Loose overlap | 所有维度均有重叠（带容差），用于 bridge/Dijkstra |
| 共享面 | Shared face | 相邻 box 的公共超平面区域，用于 waypoint 约束 |
| seed | Seed sample | 触发 `find_free_box` 扩展的候选配置点 |
| 吸收提升 | Promotion / Absorb | 层级树上行时用更大无碰撞节点替代子节点 |
| 保守碰撞 | Conservative collision | "无碰撞"结论可信；"碰撞"可能含误报 |
| wavefront / 波前 | Wavefront (BFS) | 从锚点向外做 BFS 边界采样，生成紧密相邻 box 簇 |
| shortcut | Shortcutting | 删除中间路径点以缩短路径 |
| box-aware smoothing | Box-aware smoothing | 平滑后将点投影回对应 box |
| fallback | Fallback | 高级求解不可用时的降级路径（Dijkstra + waypoint） |
| hcache | Hierarchical cache | `HierAABBTree` 持久化格式（HCACHE02） |
| coarsen | Coarsen / 粗化 | 维度扫描合并相邻 box 减少图节点 |
| bridge | Bridge / 桥接 | 在孤立连通分量间扩展桥接 box 或边 |
| island | Island / 孤岛 | box 邻接图中的一个连通分量 |
| boundary expansion | Boundary expansion | 在已有 box 外表面附近采样新 seed |

---

## 2. 符号表（Notation）

### 2.1 基础集合与变量

| 符号 | 含义 |
|---|---|
| $D$ | 关节自由度（维度） |
| $q \in \mathbb{R}^D$ | 关节配置向量 |
| $q_s, q_g$ | 起点与终点配置 |
| $\mathcal{Q}$ | C-space 区间盒（box） |
| $\mathcal{O}$ | 障碍物集合 |
| $\mathcal{C}_{free}$ | 自由空间 |
| $B_i$ | 第 $i$ 个 box 节点 |
| $V,E$ | 图的节点集与边集 |
| $G=(V,E)$ | SafeBoxForest 图结构 |

### 2.2 区间与包络

| 符号 | 含义 |
|---|---|
| $[l_i,u_i]$ | 第 $i$ 个关节维度区间 |
| $\prod_{i=1}^{D}[l_i,u_i]$ | D 维超矩形 |
| $T_{lo}, T_{hi}$ | 区间齐次变换上下界矩阵 |
| $AABB_\ell$ | 第 $\ell$ 条连杆的轴对齐包围盒 |

### 2.3 邻接与距离

| 符号 | 含义 |
|---|---|
| $w_{ij}^{(d)}$ | box $i,j$ 在维度 $d$ 的重叠宽度 |
| $tol$ | 邻接容差 |
| $\|x-y\|_2$ | 关节空间欧氏距离 |

### 2.4 复杂度记号

| 符号 | 含义 |
|---|---|
| $N$ | box 数量 |
| $M$ | 障碍数量 |
| $L$ | 连杆段数量 |
| $O(N^2D)$ | 全量邻接构建 |
| $O((|V|+|E|)\log|V|)$ | Dijkstra 复杂度 |

---

## 3. 缩写对照

| 缩写 | 全称 | 作用 |
|---|---|---|
| SBF | SafeBoxForest | 算法名称 |
| FK | Forward Kinematics | 关节配置 → 连杆位姿 |
| AABB | Axis-Aligned Bounding Box | 碰撞粗检几何表示 |
| GCS | Graph of Convex Sets | 轨迹优化（Drake） |
| FFB | Find Free Box | box 扩展过程 |
| UF | UnionFind | 并查集 |
| BFS | Breadth-First Search | 波前扩展 |
| SOCP | Second-Order Cone Program | waypoint 优化 |

---

## 4. 名词 → 代码对象映射

| 名词 | v3 代码对象 |
|---|---|
| 规划主流程 | `planner.sbf_planner.SBFPlanner.plan` |
| 只读查询规划 | `planner.sbf_query.SBFQuery.plan` |
| 管线入口 | `planner.pipeline.grow_and_prepare` |
| seed 采样 | `SBFPlanner._sample_seed` |
| 边界扩展采样 | `SBFPlanner._sample_boundary_seed` |
| 层级扩展 | `forest.hier_aabb_tree.HierAABBTree.find_free_box` |
| forest 构建 | `forest.safe_box_forest.SafeBoxForest` |
| 邻接构建 | `forest.deoverlap.compute_adjacency` |
| 端点接入 | `planner.connector.TreeConnector.connect_endpoints_to_forest` |
| GCS 优化 | `planner.gcs_optimizer.GCSOptimizer` |
| 路径平滑 | `planner.path_smoother.PathSmoother` |
| 粗化 | `forest.coarsen.coarsen_forest` |
| 桥接 | `forest.connectivity.bridge_islands` |
| 碰撞检测 | `forest.collision.CollisionChecker` |
| 区间 FK | `aabb.interval_fk.compute_interval_aabb` |
| 增量 FK | `aabb.interval_fk.compute_fk_incremental` |
| 机器人加载 | `aabb.robot.load_robot` |
| 场景 | `forest.scene.Scene` |
| 计时器 | `utils.timing.Timer` |
| Baseline 接口 | `baselines.base.BasePlanner` |
| RRT 族 | `baselines.rrt_family.RRTPlanner` |
| OMPL 桥接 | `baselines.ompl_adapter.OMPLPlanner` |
| IRIS-GCS | `baselines.iris_gcs.IRISGCSPlanner` |
| 实验运行器 | `experiments.runner.ExperimentRunner` |
