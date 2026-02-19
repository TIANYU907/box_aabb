# AABB 模块说明（详细版）

## 1. 模块定位

- 实现路径：`v2/src/aabb`
- 核心职责：给定关节区间或离散样本，计算连杆 AABB 包络并输出可复用元数据。
- 设计目标：
  1. 保守正确（安全性优先）
  2. 计算可扩展（高维可用）
  3. 上层可复用（Forest/Planner 可直接消费）

---

## 2. 子模块划分

1. `robot.py`
  - 机器人 DH 模型、FK、批量 FK、指纹。
2. `interval_fk.py`
  - 区间三角函数、区间变换链、区间 AABB 计算。
3. `calculator.py`
  - `AABBCalculator` 统一调度入口。
4. `strategies/`
  - `critical.py`：关键点优先高效采样。
  - `random.py`：随机覆盖优先。
  - `base.py`：策略公共骨架。
5. `optimization.py`
  - 边界极值精化（L-BFGS-B）。
6. `models.py`
  - `LinkAABBInfo`、`BoundaryConfig`、`AABBEnvelopeResult`。

---

## 3. 关键接口

### 3.1 对外入口

- `AABBCalculator.compute_envelope(...)`
- `compute_interval_aabb(...)`
- `compute_fk_full(...)`
- `compute_fk_incremental(...)`

### 3.2 语义约定

1. “无碰撞”结论可信；
2. “碰撞”是保守警报，可能包含误报；
3. 返回结构包含边界来源，便于回溯与报告。

---

## 4. 与上层模块关系

1. Forest 在 `CollisionChecker.check_box_collision` 中调用 AABB 结果执行快速否证。
2. HierAABBTree 在节点切分时复用 FK/区间能力，避免重复全量计算。
3. Planner 通过 Forest 间接依赖 AABB 的保守性假设。

---

## 5. 配置建议

1. 先求稳：`method=interval`。
2. 求精度：`numerical + critical`，并提高采样规模。
3. 求吞吐：优先启用批量 FK 与 relevant joints 过滤。

---

## 6. 输出与产物

- 报告：`v2/output/reports/...`
- 基准：`v2/output/benchmarks/...`
- 可视化（示例）：`v2/examples/output/...`

---

## 7. 常见排障

1. 包络“过大”：先检查区间跨度是否过宽；
2. 耗时“过高”：检查是否走到逐样本路径而非批量路径；
3. 结果“看似异常”：优先核对零长度连杆与 relevant joints 过滤配置。
