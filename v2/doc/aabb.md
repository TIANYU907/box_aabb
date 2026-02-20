# AABB 模块说明

## 1. 模块定位

- 实现路径：`v2/src/aabb/`
- 核心职责：给定关节区间或离散样本，计算连杆 AABB 包络并输出可复用元数据。
- 设计目标：
  1. 保守正确（安全性优先）
  2. 计算可扩展（高维可用）
  3. 上层可复用（Forest/Planner 可直接消费）

---

## 2. 子模块划分

| 文件 | 职责 |
|---|---|
| `robot.py` | DH 模型、FK、批量 FK (`get_link_positions_batch`)、Cython 标量核可选加载、`load_robot("2dof_planar"/"panda"/...)` |
| `interval_fk.py` | 区间三角函数包络 (`_isin`/`_icos`)、区间 DH 矩阵、全量/增量 interval AABB (`compute_interval_aabb`/`compute_fk_full`/`compute_fk_incremental`/`_split_fk_pair`) |
| `calculator.py` | `AABBCalculator` 统一调度入口 (`compute_envelope`) |
| `strategies/base.py` | 策略公共骨架 |
| `strategies/critical.py` | `CriticalStrategy` — 关键点优先高效采样 |
| `strategies/random.py` | `RandomStrategy` — 随机覆盖优先 |
| `optimization.py` | `optimize_extremes` — L-BFGS-B 边界极值精化 |
| `models.py` | `LinkAABBInfo`、`BoundaryConfig`、`AABBEnvelopeResult` |
| `report.py` | `ReportGenerator` — Markdown 报告输出 |
| `visualizer.py` | `Visualizer` / `visualize_envelope_result` — 包络可视化 |
| `configs/` | 预置机器人参数（`2dof_planar.json`、`3dof_planar.json`、`panda.json`） |
| `_fk_scalar_core.pyx` | Cython FK 标量核（可选编译） |
| `_interval_fk_core.pyx` | Cython 区间 FK 核（可选编译） |

---

## 3. 关键公开接口

### 3.1 包公开导出（`__init__.py`）

```python
Robot, create_panda_robot, load_robot, PANDA_JOINT_LIMITS
AABBEnvelopeResult, LinkAABBInfo, BoundaryConfig
AABBCalculator
ReportGenerator, Visualizer, visualize_envelope_result
compute_interval_aabb, compute_fk_full, compute_fk_incremental, _split_fk_pair
get_link_positions_batch
```

### 3.2 对外入口

- `AABBCalculator.compute_envelope(...)` — 统一调度
- `compute_interval_aabb(...)` — 区间法直接调用
- `compute_fk_full(...)` / `compute_fk_incremental(...)` — 全量/增量 FK
- `Robot.forward_kinematics(q)` / `Robot.get_link_position(q, link_idx)`
- `get_link_positions_batch(robot, configs)` — 批量 FK

### 3.3 语义约定

1. "无碰撞"结论可信（保守性保证）
2. "碰撞"是保守警报，可能包含误报
3. 返回结构包含边界来源，便于回溯与报告

---

## 4. 与上层模块关系

1. **Forest**：`CollisionChecker.check_box_collision` 调用 AABB 区间 FK 获取 link 包络执行快速否证
2. **HierAABBTree**：节点切分时复用 `compute_fk_incremental`，避免重复全量计算
3. **Planner**：通过 Forest 间接依赖 AABB 的保守性假设

---

## 5. 配置建议

| 场景 | 推荐方法 |
|---|---|
| 求稳/安全证明 | `method=interval` |
| 求精度 | `numerical + critical`，提高采样规模 |
| 求吞吐 | 优先启用批量 FK（`get_link_positions_batch`）与 Cython 加速 |

---

## 6. 输出与产物

- 报告：`v2/output/reports/...`
- 基准：`v2/output/benchmarks/...`
- 可视化（示例）：`v2/examples/output/...`

---

## 7. Benchmark

```bash
python -m v2.benchmarks.aabb.bench_interval_fk
python -m v2.benchmarks.aabb.bench_fk_batch
python -m v2.benchmarks.aabb.bench_fk_scalar_cython
```

---

## 8. 测试

```bash
python -m pytest v2/tests/aabb/ -v
```

测试文件：

- `test_calculator_no_hybrid.py` — 计算器无混合模式测试
- `test_interval_fk_fast_alias.py` — 区间 FK 快速别名测试
- `test_robot_fk_batch.py` — 批量 FK 测试

---

## 9. 常见排障

1. 包络"过大"：先检查区间跨度是否过宽
2. 耗时"过高"：检查是否走到逐样本路径而非批量路径
3. 结果"看似异常"：优先核对零长度连杆与 relevant joints 过滤配置
4. Cython 未编译：自动降级到 Python 计算路径，性能略低但结果一致
