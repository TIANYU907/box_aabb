# Forest 模块说明

## 1. 模块定位

- 实现路径：`v2/src/forest/`
- 核心职责：无重叠 box 集合维护、邻接关系、碰撞核心、层级 AABB 树
- 设计目标：
  1. 几何正确（无重叠不变量）
  2. 增量高效（add/remove + 缓存一致性）
  3. 可复用（跨场景持久化）

---

## 2. 子模块划分

| 文件 | 职责 |
|---|---|
| `box_forest.py` | `BoxForest` — 无重叠 box 集合 + 邻接图 + 区间缓存 + KDTree 加速。支持增量操作 (`add_box_direct`, `add_box_no_adjacency`, `remove_boxes`, `remove_boxes_no_adjacency`, `rebuild_adjacency`)、分区合并 (`merge_partition_forests`, `dedup_boundary_boxes`)、不变量校验 (`validate_invariants`)、持久化 (`save`/`load`) |
| `collision.py` | `CollisionChecker` — 单点 (`check_config_collision`)、区间 (`check_box_collision`)、线段 (`check_segment_collision`)、批量 (`check_config_collision_batch`) 碰撞检测。内置 `SpatialIndex` 自动加速。独立函数 `aabb_overlap` |
| `hier_aabb_tree.py` | `HierAABBTree` — KD 层级 AABB 树。核心方法 `find_free_box(seed, obstacles, ...)`，支持 `active_split_dims`、`constrained_intervals`、增量 FK 切分、promotion 吸收、HCACHE02 持久化 |
| `deoverlap.py` | 向量化邻接检测 `compute_adjacency`（分块上三角策略）、`compute_adjacency_incremental`、`shared_face`、`shared_face_center` |
| `connectivity.py` | `UnionFind`（支持动态 `add`/`remove_keys`）、`find_islands`、`bridge_islands`（支持周期关节空间、dry-run FFB + 正式 FFB 三步策略、向量化批量线段碰撞、目标岛对定向桥接、小岛丢弃） |
| `coarsen.py` | `coarsen_forest` — 维度扫描合并（两阶段检测+执行架构、NumPy 向量化分组、批量合并）。返回 `CoarsenStats` |
| `parallel_collision.py` | `ParallelCollisionChecker`（多进程批量碰撞检测）+ `SpatialIndex`（空间哈希索引） |
| `scene.py` | `Scene` — 障碍物管理，支持 JSON 序列化 (`to_json`/`from_json`) |
| `models.py` | `Obstacle`、`BoxNode`（含缓存 `center`、方法 `contains`/`distance_to_config`/`nearest_point_to`/`overlap_with`）、`PlannerConfig`（forest 侧仅含 `adjacency_tolerance`） |
| `configs/` | 预置配置（`default.json`） |
| `_hier_core.pyx` | Cython `NodeStore`（SoA 数据布局，含 `forest_ids_array()` 批量接口）、HCACHE02 持久化 |
| `_hier_layout.py` | NodeStore 辅助布局函数 |

---

## 3. 关键公开接口

### 3.1 包公开导出（`__init__.py`）

```python
Obstacle, BoxNode, PlannerConfig, Scene, CollisionChecker, aabb_overlap
BoxForest, compute_adjacency, compute_adjacency_incremental
UnionFind, find_islands, bridge_islands
ParallelCollisionChecker, SpatialIndex
coarsen_forest, CoarsenStats
```

### 3.2 核心操作

| 操作 | 接口 | 复杂度 |
|---|---|---|
| 增量添加 box | `BoxForest.add_box_direct(box)` | ~O(ND) |
| 批量添加（无邻接） | `BoxForest.add_box_no_adjacency(box)` | O(1) |
| 全量重建邻接 | `BoxForest.rebuild_adjacency()` | O(N²D) |
| 碰撞检测（单点） | `CollisionChecker.check_config_collision(q)` | O(L·M) |
| 碰撞检测（批量） | `CollisionChecker.check_config_collision_batch(configs)` | O(N·L·M) 向量化 |
| 碰撞检测（区间） | `CollisionChecker.check_box_collision(intervals)` | O(L·M) |
| 自由盒搜索 | `HierAABBTree.find_free_box(seed, obstacles)` | O(depth·C_FK) |
| 维度扫描合并 | `coarsen_forest(store, forest)` | O(N·D·logN) |
| 孤岛桥接 | `bridge_islands(boxes, collision_checker, ...)` | 变长 |
| 不变量校验 | `BoxForest.validate_invariants(strict=True)` | O(N²D) |

---

## 4. 核心不变量

1. **(I1) 非重叠**：除容差尺度外，任意两 box 不应有正体积重叠
2. **(I2) 邻接对称**：`j ∈ adjacency[i]` ⟺ `i ∈ adjacency[j]`
3. **(I3) 缓存一致**：`_intervals_arr`、`_interval_id_to_index` 与 `boxes` 同步更新

---

## 5. Benchmark

```bash
python -m v2.benchmarks.forest.bench_panda_forest
python -m v2.benchmarks.forest.bench_panda_multi
python -m v2.benchmarks.forest.bench_adjacency_vectorized
python -m v2.benchmarks.forest.bench_nearest_kdtree
python -m v2.benchmarks.forest.bench_box_forest_incremental_add
python -m v2.benchmarks.forest.bench_promotion_sweep
python -m v2.benchmarks.forest.bench_promotion_depth
```

---

## 6. 测试

```bash
python -m pytest v2/tests/forest/ -v
```

测试文件：

- `test_box_forest_basic.py` — BoxForest 增删与邻接
- `test_adjacency_vectorized.py` — 向量化邻接检测
- `test_collision_basic.py` — 碰撞检测
- `test_hier_aabb_tree_split_dims.py` — 层级树 active_split_dims

---

## 7. 输出路径

- 可视化/缓存：`v2/output/visualizations/...`
- 报告：`v2/output/reports/...`
- hcache：`v2/output/*.hcache`
