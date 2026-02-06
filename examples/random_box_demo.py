"""
examples/random_box_demo.py - 随机关节区间 Box 演示

随机生成关节区间 (box)，运行数值关键点包络计算，
生成 Markdown 报告并可视化。

用法:
    python examples/random_box_demo.py                  # 默认使用 Panda
    python examples/random_box_demo.py --robot panda    # 指定机器人
    python examples/random_box_demo.py --width 0.3      # 设置区间半宽
    python examples/random_box_demo.py --seed 42        # 固定随机种子
    python examples/random_box_demo.py --no-viz         # 不弹出可视化窗口
"""

import sys
import os
import argparse
import random
import math
import datetime

# 确保 box_aabb 包可导入
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_pkg_dir))

from box_aabb import (
    load_robot, AABBCalculator,
    visualize_envelope_result,
)


def random_joint_intervals(joint_limits, half_width=0.3, rng=None):
    """在关节限制范围内随机生成关节区间 box。

    对每个关节：
      1. 在 [lo + hw, hi - hw] 范围内随机选取中心 c
      2. 区间为 [c - hw, c + hw]
    其中 hw = min(half_width, (hi - lo)/2)，保证区间不超限。
    固定关节 (lo == hi) 区间保持 (lo, hi)。

    Args:
        joint_limits: [(lo, hi), ...] 每个关节的弧度限制
        half_width: 期望的区间半宽 (弧度)
        rng: random.Random 实例

    Returns:
        intervals: [(lo_i, hi_i), ...] 随机生成的关节区间
    """
    if rng is None:
        rng = random.Random()

    intervals = []
    for lo, hi in joint_limits:
        span = hi - lo
        if span <= 1e-9:
            # 固定关节
            intervals.append((lo, hi))
            continue

        hw = min(half_width, span / 2.0)
        center_lo = lo + hw
        center_hi = hi - hw
        center = rng.uniform(center_lo, center_hi)
        intervals.append((center - hw, center + hw))

    return intervals


def format_intervals_table(intervals, joint_names=None):
    """将关节区间格式化为可读表格字符串。"""
    lines = []
    lines.append(f"{'关节':<12} {'下限 (rad)':>12} {'上限 (rad)':>12} {'宽度 (rad)':>12} {'宽度 (deg)':>12}")
    lines.append("-" * 64)
    for i, (lo, hi) in enumerate(intervals):
        name = joint_names[i] if joint_names and i < len(joint_names) else f"q{i}"
        width = hi - lo
        lines.append(f"{name:<12} {lo:>12.4f} {hi:>12.4f} {width:>12.4f} {math.degrees(width):>12.2f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="随机生成关节区间 box，计算 AABB 包络并可视化")
    parser.add_argument("--robot", type=str, default="panda",
                        help="机器人配置名称 (默认: panda)")
    parser.add_argument("--width", type=float, default=0.5,
                        help="关节区间半宽，弧度 (默认: 0.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (默认: 无)")
    parser.add_argument("--sampling", type=str, default="critical",
                        choices=["critical", "random", "hybrid"],
                        help="采样策略 (默认: critical)")
    parser.add_argument("--no-viz", action="store_true",
                        help="不弹出可视化窗口")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="报告和图片保存目录 (默认: examples/output/)")
    args = parser.parse_args()

    # ── 1. 加载机器人 ──────────────────────────────────────────
    print("=" * 60)
    print("  随机关节区间 Box — AABB 包络计算演示")
    print("=" * 60)

    robot = load_robot(args.robot)
    print(f"\n✅ 已加载机器人: {robot.name}  (关节数: {robot.n_joints})")

    if robot.joint_limits is None:
        print("❌ 该机器人未定义 joint_limits，无法随机生成区间。")
        sys.exit(1)

    # ── 2. 随机生成关节区间 ────────────────────────────────────
    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    rng = random.Random(seed)
    print(f"🎲 随机种子: {seed}")

    intervals = random_joint_intervals(robot.joint_limits, args.width, rng)

    # 获取关节名（如果有）
    joint_names = None
    try:
        # 尝试从配置中读取
        import json
        cfg_path = os.path.join(_pkg_dir, "configs", f"{args.robot.lower()}.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            joint_names = cfg.get("joint_names")
    except Exception:
        pass

    print(f"\n📐 随机生成的关节区间 (half_width={args.width:.2f} rad ≈ {math.degrees(args.width):.1f}°):\n")
    print(format_intervals_table(intervals, joint_names))

    # ── 3. 计算 AABB 包络 ────────────────────────────────────
    print(f"\n⏳ 正在计算包络 (method=numerical, sampling={args.sampling}) ...")
    calc = AABBCalculator(robot)
    result = calc.compute_envelope(
        joint_intervals=intervals,
        method="numerical",
        sampling=args.sampling,
    )

    print(f"✅ 计算完成!")
    print(f"   采样点数: {result.n_samples_evaluated}")
    print(f"   计算耗时: {result.computation_time:.3f} 秒")
    print(f"   总体积:   {result.total_volume():.6f} m³")

    # 打印各连杆概要
    print(f"\n📦 各连杆 AABB:")
    print(f"{'连杆':<12} {'体积 (m³)':>14} {'dx':>8} {'dy':>8} {'dz':>8}  {'备注'}")
    print("-" * 70)
    for aabb in result.link_aabbs:
        note = "⊘ 零长度" if aabb.is_zero_length else ""
        dims = aabb.size
        print(f"{aabb.link_name:<12} {aabb.volume:>14.6f} {dims[0]:>8.4f} {dims[1]:>8.4f} {dims[2]:>8.4f}  {note}")

    # ── 4. 保存报告 ───────────────────────────────────────────
    save_dir = args.save_dir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"random_box_{args.robot}_{timestamp}.md"
    report_path = os.path.join(save_dir, report_filename)

    report_text = result.generate_report(save_path=report_path)
    print(f"\n📝 报告已保存: {report_path}")

    # ── 5. 可视化 ─────────────────────────────────────────────
    if not args.no_viz:
        print("\n🎨 正在生成 3D 可视化 ...")
        fig_filename = f"random_box_{args.robot}_{timestamp}.png"
        fig_path = os.path.join(save_dir, fig_filename)

        viz = visualize_envelope_result(
            result=result,
            robot=robot,
            show_boundary_configs=True,
            show_samples=True,
            show_aabbs=True,
            title=f"{robot.name} — Random Box (seed={seed}, hw={args.width})",
            save_path=fig_path,
            interactive=True,
        )
        print(f"📊 可视化图片已保存: {fig_path}")
        viz.show()
    else:
        print("\n(已跳过可视化)")

    print("\n✅ 全部完成!")
    return result


if __name__ == "__main__":
    main()
