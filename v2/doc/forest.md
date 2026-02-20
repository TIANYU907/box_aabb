# Forest 模块说明

- 位置: `v2/forest`
- 核心职责: 无重叠 box 集合维护、邻接关系、碰撞核心
- 关键接口:
  - `BoxForest`
  - `CollisionChecker`
  - `compute_adjacency`
- 输出路径:
  - 可视化/缓存: `v2/output/visualizations/...`, `v2/output/reports/...`

## Benchmark 用法

```bash
python -m v2.benchmarks.forest.bench_panda_forest
python -m v2.benchmarks.forest.bench_panda_multi
```
