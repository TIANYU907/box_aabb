"""
report.py - AABB分析报告生成器

从 aabb_calculator.py 中提取，负责将 AABBEnvelopeResult 转换为 Markdown 格式报告。
"""

import logging
from itertools import groupby
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AABBEnvelopeResult, LinkAABBInfo

logger = logging.getLogger(__name__)


class ReportGenerator:
    """AABB 分析报告生成器
    
    将 AABBEnvelopeResult 转换为格式化的 Markdown 报告，
    包含算法配置、关节区间、各连杆 AABB 信息和边界配置分析。
    """

    @staticmethod
    def generate(result: 'AABBEnvelopeResult') -> str:
        """生成完整的 Markdown 分析报告
        
        Args:
            result: AABB 计算结果
            
        Returns:
            Markdown 格式的报告字符串
        """
        lines: List[str] = []
        _a = lines.append

        _a("# AABB包络分析报告")
        _a("")
        _a(f"生成时间: {result.timestamp}")
        _a("")

        # 算法配置
        _a("## 算法配置")
        _a("")
        _a(f"- **机器人**: {result.robot_name}")
        _a(f"- **关节数**: {result.n_joints}")
        _a(f"- **计算方法**: {result.method}")
        if result.method.startswith('numerical'):
            _a(f"- **采样模式**: {result.sampling_mode}")
        _a(f"- **采样点数**: {result.n_samples_evaluated}")
        if result.n_subdivisions > 1:
            _a(f"- **连杆等分段数**: {result.n_subdivisions}")
        _a(f"- **计算耗时**: {result.computation_time:.4f} 秒")
        _a("")

        # 关节区间表
        _a("## 关节区间")
        _a("")
        _a("| 关节 | 最小值 | 最大值 | 宽度 | 包含0 |")
        _a("|------|--------|--------|------|-------|")
        for i, (lo, hi) in enumerate(result.joint_intervals):
            z = "✓" if lo <= 0 <= hi else ""
            _a(f"| q{i} | {lo:.4f} | {hi:.4f} | {hi - lo:.4f} | {z} |")
        _a("")

        # 连杆 AABB 信息
        _a("## 连杆AABB信息")
        _a("")

        sorted_aabbs = sorted(result.link_aabbs,
                               key=lambda a: (a.link_index, a.segment_index))
        for li, group in groupby(sorted_aabbs, key=lambda a: a.link_index):
            segs = list(group)
            first = segs[0]
            if first.is_zero_length:
                _a(f"### {first.link_name} (零长度连杆，已跳过)")
                _a("")
                continue

            if len(segs) == 1 and segs[0].n_segments <= 1:
                aabb = segs[0]
                _a(f"### {aabb.link_name}")
                _a("")
                ReportGenerator._report_aabb(lines, aabb, result.joint_intervals)
            else:
                _a(f"### {first.link_name} ({len(segs)} 段)")
                _a("")
                link_vol = sum(s.volume for s in segs)
                _a(f"- **段数**: {len(segs)}, **总体积**: {link_vol:.6f} m³")
                _a("")
                for seg in segs:
                    _a(f"#### 段 {seg.segment_index} "
                       f"(t=[{seg.t_start:.3f}, {seg.t_end:.3f}])")
                    _a("")
                    ReportGenerator._report_aabb(lines, seg, result.joint_intervals)

        # 整体包围盒
        _a("## 整体包围盒")
        _a("")
        mn, mx = result.get_robot_aabb()
        sz = [mx[i] - mn[i] for i in range(3)]
        _a(f"- **最小点**: ({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})")
        _a(f"- **最大点**: ({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})")
        _a(f"- **尺寸**: ({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f})")
        _a(f"- **总体积**: {result.total_volume():.6f} m³")
        _a("")

        # 符号说明
        _a("## 符号说明")
        _a("")
        _a("- ⭐ : AABB顶点（多个边界同时达到极值）")
        _a("- 📍 : 关节在区间边界值")
        _a("- ◯ : 关节值为零")
        _a("- Σ : 满足角度组合条件")
        _a("")
        return '\n'.join(lines)

    @staticmethod
    def _report_aabb(lines: List[str], aabb: 'LinkAABBInfo',
                     intervals: List[Tuple[float, float]]) -> None:
        """生成单个 AABB 的报告段落"""
        _a = lines.append
        _a(f"- **最小点**: ({aabb.min_point[0]:.4f}, {aabb.min_point[1]:.4f}, "
           f"{aabb.min_point[2]:.4f})")
        _a(f"- **最大点**: ({aabb.max_point[0]:.4f}, {aabb.max_point[1]:.4f}, "
           f"{aabb.max_point[2]:.4f})")
        _a(f"- **尺寸**: ({aabb.size[0]:.4f}, {aabb.size[1]:.4f}, {aabb.size[2]:.4f})")
        _a(f"- **体积**: {aabb.volume:.6f} m³")
        _a("")
        if aabb.boundary_configs:
            _a("**边界配置**:")
            _a("")
            for bt, cfg in sorted(aabb.boundary_configs.items()):
                formatted = cfg.format_joint_values(intervals)
                symbols = []
                if cfg.is_aabb_vertex:
                    symbols.append('⭐')
                symbols.extend(f'Σ({c})' for c in cfg.angle_constraints)
                axis, minmax = bt.split('_')
                _a(f"  - **{axis.upper()} {minmax}**: `{formatted}` → "
                   f"{cfg.boundary_value:.4f} m {' '.join(symbols)}")
            _a("")
