"""
planner/report.py - 规划结果报告生成器

为 Box-RRT 规划结果生成结构化 Markdown 报告，
涵盖机器人信息、场景、规划器参数、规划结果、
路径质量指标、Box 树详情、AABB 缓存统计等。

用法:
    from planner.report import PlannerReportGenerator
    gen = PlannerReportGenerator()
    md = gen.generate(
        robot=robot, scene=scene, config=config,
        q_start=q_start, q_goal=q_goal,
        result=result, metrics=metrics,
    )
"""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from box_aabb.robot import Robot
from .models import PlannerConfig, PlannerResult, BoxTree, Edge
from .obstacles import Scene
from .metrics import PathMetrics

logger = logging.getLogger(__name__)


class PlannerReportGenerator:
    """Box-RRT 规划结果 Markdown 报告生成器"""

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(
        self,
        robot: Robot,
        scene: Scene,
        config: PlannerConfig,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        result: PlannerResult,
        metrics: Optional[PathMetrics] = None,
        rng_seed: Optional[int] = None,
        saved_files: Optional[List[str]] = None,
        extra_sections: Optional[Dict[str, str]] = None,
    ) -> str:
        """生成完整 Markdown 报告

        Args:
            robot: 机器人模型
            scene: 障碍物场景
            config: 规划器参数
            q_start / q_goal: 始末关节配置
            result: 规划结果
            metrics: 路径质量指标 (可选)
            rng_seed: 随机种子
            saved_files: 输出产物文件路径列表
            extra_sections: 额外的自定义区段 {标题: 内容}

        Returns:
            Markdown 文本
        """
        L: List[str] = []

        self._sec_header(L, robot, rng_seed)
        self._sec_robot(L, robot)
        self._sec_scene(L, scene)
        self._sec_endpoints(L, q_start, q_goal, robot)
        self._sec_config(L, config)
        self._sec_result(L, result)
        self._sec_box_trees(L, result)
        self._sec_forest(L, result)
        self._sec_graph(L, result)
        if metrics is not None and result.success:
            self._sec_metrics(L, metrics)
        if result.success and result.path:
            self._sec_path(L, result)
        if saved_files:
            self._sec_files(L, saved_files)
        if extra_sections:
            for title, body in extra_sections.items():
                L.append(f"## {title}")
                L.append("")
                L.append(body)
                L.append("")

        return "\n".join(L)

    # ------------------------------------------------------------------
    # 各区段
    # ------------------------------------------------------------------

    @staticmethod
    def _sec_header(L: List[str], robot: Robot, seed: Optional[int]) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        L.append("# Box-RRT 规划报告")
        L.append("")
        L.append(f"> 生成时间: {ts}  ")
        if seed is not None:
            L.append(f"> 随机种子: {seed}  ")
        L.append(f"> 机器人: {robot.name} ({robot.n_joints} DOF)")
        L.append("")

    @staticmethod
    def _sec_robot(L: List[str], robot: Robot) -> None:
        L.append("## 1. 机器人")
        L.append("")
        L.append("| 属性 | 值 |")
        L.append("|------|----|")
        L.append(f"| 名称 | {robot.name} |")
        L.append(f"| 自由度 | {robot.n_joints} |")
        L.append(f"| 连杆数 | {len(robot.dh_params)} |")
        if robot.joint_limits:
            for i, (lo, hi) in enumerate(robot.joint_limits):
                L.append(
                    f"| q{i} 范围 | [{lo:+.4f}, {hi:+.4f}] "
                    f"(宽 {hi - lo:.4f} rad) |"
                )
        L.append("")
        # DH 参数表
        L.append("**DH 参数:**")
        L.append("")
        L.append("| Link | α (rad) | a (m) | d (m) | type |")
        L.append("|------|---------|-------|-------|------|")
        for i, p in enumerate(robot.dh_params):
            L.append(
                f"| {i} | {p['alpha']:+.4f} | {p['a']:.4f} "
                f"| {p['d']:.4f} | {p.get('type', 'revolute')} |"
            )
        L.append("")

    @staticmethod
    def _sec_scene(L: List[str], scene: Scene) -> None:
        L.append(f"## 2. 场景 ({scene.n_obstacles} 个障碍物)")
        L.append("")
        if scene.n_obstacles == 0:
            L.append("_无障碍物_")
            L.append("")
            return
        L.append("| 名称 | min | max | 尺寸 | 体积 |")
        L.append("|------|-----|-----|------|------|")
        total_vol = 0.0
        for obs in scene.get_obstacles():
            vol = obs.volume
            total_vol += vol
            ndim = len(obs.min_point)
            if ndim >= 3 and abs(obs.min_point[2]) < 900:
                mn = (f"({obs.min_point[0]:+.3f}, {obs.min_point[1]:+.3f}, "
                      f"{obs.min_point[2]:+.3f})")
                mx = (f"({obs.max_point[0]:+.3f}, {obs.max_point[1]:+.3f}, "
                      f"{obs.max_point[2]:+.3f})")
                sz = f"{obs.size[0]:.3f}×{obs.size[1]:.3f}×{obs.size[2]:.3f}"
            else:
                mn = f"({obs.min_point[0]:+.3f}, {obs.min_point[1]:+.3f})"
                mx = f"({obs.max_point[0]:+.3f}, {obs.max_point[1]:+.3f})"
                sz = f"{obs.size[0]:.3f}×{obs.size[1]:.3f}"
            L.append(f"| {obs.name} | {mn} | {mx} | {sz} | {vol:.4f} |")
        L.append(f"| **合计** | | | | **{total_vol:.4f}** |")
        L.append("")

    @staticmethod
    def _sec_endpoints(
        L: List[str],
        q_start: np.ndarray,
        q_goal: np.ndarray,
        robot: Robot,
    ) -> None:
        def qfmt(q: np.ndarray) -> str:
            return ", ".join(f"{v:+.4f}" for v in q)

        dist = float(np.linalg.norm(q_goal - q_start))
        L.append("## 3. 始末点")
        L.append("")
        L.append(f"- **起点**: [{qfmt(q_start)}]")
        L.append(f"- **终点**: [{qfmt(q_goal)}]")
        L.append(f"- **C-space 直线距离**: {dist:.4f} rad")
        L.append("")

    @staticmethod
    def _sec_config(L: List[str], config: PlannerConfig) -> None:
        L.append("## 4. 规划器参数")
        L.append("")
        L.append("| 参数 | 值 |")
        L.append("|------|----|")
        for k, v in sorted(vars(config).items()):
            L.append(f"| {k} | {v} |")
        L.append("")

    @staticmethod
    def _sec_result(L: List[str], result: PlannerResult) -> None:
        L.append("## 5. 规划结果")
        L.append("")
        ok = "✓ 成功" if result.success else "✗ 失败"
        L.append("| 指标 | 值 |")
        L.append("|------|----|")
        L.append(f"| 状态 | {ok} |")
        L.append(f"| 消息 | {result.message} |")
        L.append(f"| 计算时间 | {result.computation_time:.3f} s |")
        L.append(f"| Box 总数 | {result.n_boxes_created} |")
        L.append(f"| 碰撞检测次数 | {result.n_collision_checks} |")
        L.append(f"| 路径点数 | {len(result.path)} |")
        if result.success:
            L.append(f"| 路径长度 (L2) | {result.path_length:.4f} rad |")
        L.append(f"| Box tree 数 | {len(result.box_trees)} |")
        L.append("")

    @staticmethod
    def _sec_box_trees(L: List[str], result: PlannerResult) -> None:
        if not result.box_trees:
            return
        L.append("### 5.1 Box Tree 明细")
        L.append("")
        L.append("| 树 ID | 节点数 | 总体积 | 叶节点数 | 最大 box 体积 | 最小 box 体积 |")
        L.append("|-------|--------|--------|----------|---------------|---------------|")
        for tree in result.box_trees:
            leaves = tree.get_leaf_nodes()
            vols = [n.volume for n in tree.nodes.values()]
            v_max = max(vols) if vols else 0
            v_min = min(vols) if vols else 0
            L.append(
                f"| {tree.tree_id} | {tree.n_nodes} "
                f"| {tree.total_volume:.4f} | {len(leaves)} "
                f"| {v_max:.4f} | {v_min:.6f} |"
            )
        L.append("")

        # Top-20 largest boxes
        all_boxes = []
        for tree in result.box_trees:
            for node in tree.nodes.values():
                all_boxes.append(node)
        all_boxes.sort(key=lambda b: b.volume, reverse=True)
        show_n = min(20, len(all_boxes))
        if all_boxes:
            L.append(f"### 5.2 Box 详情（前 {show_n}/{len(all_boxes)} 个，按体积降序）")
            L.append("")
            L.append("| # | 树 | ID | 体积 | 各维宽度 |")
            L.append("|---|----|----|------|----------|")
            for idx, box in enumerate(all_boxes[:show_n]):
                widths_s = ", ".join(f"{w:.3f}" for w in box.widths)
                L.append(
                    f"| {idx} | {box.tree_id} | {box.node_id} "
                    f"| {box.volume:.4f} | [{widths_s}] |"
                )
            L.append("")

    @staticmethod
    def _sec_graph(L: List[str], result: PlannerResult) -> None:
        if not result.edges:
            return
        n_edges = len(result.edges)
        n_intra = sum(
            1 for e in result.edges if e.source_tree_id == e.target_tree_id
        )
        n_inter = n_edges - n_intra
        costs = [e.cost for e in result.edges]
        L.append("### 5.3 图连接统计")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|------|----|")
        L.append(f"| 总边数 | {n_edges} |")
        L.append(f"| 树内边 | {n_intra} |")
        L.append(f"| 树间边 | {n_inter} |")
        L.append(f"| 平均边代价 | {np.mean(costs):.4f} |")
        L.append(f"| 最大边代价 | {np.max(costs):.4f} |")
        L.append("")

    @staticmethod
    def _sec_metrics(L: List[str], m: PathMetrics) -> None:
        L.append("## 6. 路径质量指标")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|------|----|")
        L.append(f"| 路径长度 (L2) | {m.path_length:.4f} |")
        L.append(f"| 直线距离 | {m.direct_distance:.4f} |")
        L.append(f"| 路径效率 (比值) | {m.length_ratio:.4f} |")
        L.append(f"| 平滑度 (均值角变) | {m.smoothness:.4f} rad |")
        L.append(f"| 最大曲率 | {m.max_curvature:.4f} rad |")
        L.append(f"| 最小安全裕度 | {m.min_clearance:.6f} |")
        L.append(f"| 平均安全裕度 | {m.avg_clearance:.6f} |")
        L.append(f"| Box 总体积 | {m.box_coverage:.4f} |")
        L.append(f"| Box 数量 | {m.n_boxes} |")
        L.append("")

        if m.joint_range_usage is not None:
            L.append("### 6.1 关节使用率")
            L.append("")
            L.append("| 关节 | 使用率 | 可视化 |")
            L.append("|------|--------|--------|")
            for i, u in enumerate(m.joint_range_usage):
                bar = "█" * int(u * 20) + "░" * (20 - int(u * 20))
                L.append(f"| q{i} | {u * 100:.1f}% | {bar} |")
            L.append("")

    @staticmethod
    def _sec_path(L: List[str], result: PlannerResult) -> None:
        path = result.path
        n = len(path)
        ndim = len(path[0])
        L.append(f"## 8. 路径详情 ({n} 点, {ndim} DOF)")
        L.append("")
        hdr = "| # | " + " | ".join(f"q{j}" for j in range(ndim)) + " |"
        sep = "|---" + "|---" * ndim + "|"
        L.append(hdr)
        L.append(sep)
        for idx, q in enumerate(path):
            vals = " | ".join(f"{v:+.4f}" for v in q)
            L.append(f"| {idx} | {vals} |")
        L.append("")

    @staticmethod
    def _sec_files(L: List[str], files: List[str]) -> None:
        import os
        L.append("## 9. 产物文件")
        L.append("")
        for f in files:
            name = os.path.basename(f)
            L.append(f"- `{name}`")
        L.append("")

    # ------------------------------------------------------------------
    # BoxForest v5 报告区段
    # ------------------------------------------------------------------

    @staticmethod
    def _sec_forest(L: List[str], result: 'PlannerResult') -> None:
        """BoxForest 无重叠 box 集合报告"""
        forest = getattr(result, 'forest', None)
        if forest is None:
            return

        n_boxes = len(forest.boxes)
        total_vol = sum(b.volume for b in forest.boxes.values())
        n_adj_edges = sum(len(v) for v in forest.adjacency.values()) // 2

        L.append("### 5.2 BoxForest 统计（无重叠 box 集合）")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|------|----|")
        L.append(f"| Box 总数 | {n_boxes} |")
        L.append(f"| 总体积 | {total_vol:.4f} |")
        L.append(f"| 邻接边数 | {n_adj_edges} |")

        if n_boxes > 0:
            vols = [b.volume for b in forest.boxes.values()]
            L.append(f"| 最大 box 体积 | {max(vols):.4f} |")
            L.append(f"| 最小 box 体积 | {min(vols):.6f} |")
            L.append(f"| 平均 box 体积 | {np.mean(vols):.4f} |")
            L.append(f"| 体积中位数 | {np.median(vols):.4f} |")

            # 邻接度统计
            degrees = [len(forest.adjacency.get(bid, set()))
                       for bid in forest.boxes]
            L.append(f"| 平均邻接度 | {np.mean(degrees):.2f} |")
            L.append(f"| 最大邻接度 | {max(degrees)} |")
            iso = sum(1 for d in degrees if d == 0)
            L.append(f"| 孤立 box | {iso} |")
        L.append("")

        # 是否自流体积守恒：任意两 box 重叠检测
        boxes_list = list(forest.boxes.values())
        max_overlap = 0.0
        for i in range(min(len(boxes_list), 50)):
            for j in range(i + 1, min(len(boxes_list), 50)):
                ov = boxes_list[i].overlap_volume(boxes_list[j])
                if ov > max_overlap:
                    max_overlap = ov
        L.append(f"**零重叠验证**（前 50 box 抽样）: 最大重叠体积 = {max_overlap:.2e}")
        L.append("")

        # Top-20 box 明细
        sorted_boxes = sorted(boxes_list, key=lambda b: b.volume, reverse=True)
        show_n = min(20, len(sorted_boxes))
        if sorted_boxes:
            L.append(f"**Box 明细（前 {show_n}/{n_boxes} 个，按体积降序）:**")
            L.append("")
            L.append("| # | ID | 体积 | 邻接度 | 各维宽度 |")
            L.append("|---|----|------|--------|----------|")
            for idx, box in enumerate(sorted_boxes[:show_n]):
                deg = len(forest.adjacency.get(box.node_id, set()))
                widths_s = ", ".join(f"{w:.3f}" for w in box.widths)
                L.append(
                    f"| {idx} | {box.node_id} "
                    f"| {box.volume:.4f} | {deg} | [{widths_s}] |"
                )
            L.append("")
