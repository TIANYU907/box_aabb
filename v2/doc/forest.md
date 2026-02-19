# Forest 模块说明（详细版）

## 1. 模块定位

- 实现路径：`v2/src/forest`
- 核心职责：
  1. 维护无重叠 box 集合（几何层）
  2. 维护邻接关系（拓扑层）
  3. 提供单点/区间/线段碰撞判定（验证层）
  4. 提供层级扩展树（生成层）

---

## 2. 核心组件

1. `box_forest.py`
  - `BoxForest`：box 字典、邻接表、缓存、合并与校验。
2. `deoverlap.py`
  - `deoverlap`、`subtract_box`、`compute_adjacency`。
3. `collision.py`
  - `CollisionChecker`：配置/区间/线段碰撞 API。
4. `hier_aabb_tree.py`
  - `HierAABBTree`：层级切分、free-box 搜索、占用标记。
5. `models.py`
  - `BoxNode` 及相关结构。

---

## 3. 不变量与一致性

Forest 持续维护以下结构性不变量：

1. 无正体积重叠（容差外）
2. 邻接引用有效
3. 邻接对称性
4. 缓存与 `boxes` 一致

并行合并后通过 `validate_invariants(strict=True)` 强制校验，发现结构问题会立即抛错而不是静默容忍。

---

## 4. 与 Planner 的接口边界

1. Planner 只应通过 `BoxForest` 提供的 box/adjacency 读视图构图。
2. 新盒提交必须走 Forest 的增量接口，避免绕过缓存维护。
3. 并行模式下各 worker 不共享写入，统一在主进程调用 Forest 合并。

---

## 5. 并行相关行为（当前实现）

1. 分区 worker 产出局部 box 列表。
2. 主进程调用 `merge_partition_forests` 写入全局 forest。
3. 执行 `dedup_boundary_boxes` 清理边界重复。
4. 严格不变量校验后，交由 Planner 做跨区补边与图搜索。

---

## 6. 基准与验证

```bash
python -m v2.benchmarks.forest.bench_panda_forest
python -m v2.benchmarks.forest.bench_panda_multi
```

建议同时记录：

1. box 总数
2. 邻接边数
3. collision checks 次数
4. 不变量校验通过率

---

## 7. 常见排障

1. 校验失败（重叠）：先检查分区合并输入是否重复扩展同一区域。
2. 邻接异常少：检查 `adjacency_tolerance` 是否过严。
3. 碰撞耗时高：优先确认是否启用 batch 路径与 spatial index。
