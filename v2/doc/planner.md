# Planner 模块说明

- 位置: `v2/planner`
- 核心职责: 规划主流程、连接、平滑、质量评估、报告
- 关键接口:
  - `BoxRRT.plan`
  - `BoxForestQuery.plan`
  - `evaluate_result`
  - `PlannerReportGenerator.generate`
- 输出路径:
  - 路径与报告: `v2/output/plans/...`, `v2/output/reports/...`
