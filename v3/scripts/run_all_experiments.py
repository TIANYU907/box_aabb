"""
scripts/run_all_experiments.py — 批量运行全部实验 + 生成报告

用法:
    # quick 模式 (烟雾测试, 几分钟)
    python scripts/run_all_experiments.py --quick

    # 完整模式 (论文级数据, 可能 1-2 小时)
    python scripts/run_all_experiments.py --full

    # 指定实验编号 (逗号分隔)
    python scripts/run_all_experiments.py --quick --only 1,2,5

    # 跳过报告生成
    python scripts/run_all_experiments.py --quick --no-report

功能:
  1. 依次运行 exp1 ~ exp8
  2. 每个实验结果保存到 experiments/output/raw/expN_*.json
  3. 运行结束后, 将本次所有 JSON 复制到带时间戳的归档目录
  4. 调用 reporting.py 为每个结果生成 LaTeX 表格 + 图表
  5. 汇总日志保存到 experiments/output/batch_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────────────────
_SCRIPT = Path(__file__).resolve()
_ROOT = _SCRIPT.parents[1]          # v3/
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUTPUT_BASE = _ROOT / "experiments" / "output"
RAW_DIR     = OUTPUT_BASE / "raw"
ARCHIVE_DIR = OUTPUT_BASE / "archive"
TABLES_DIR  = OUTPUT_BASE / "tables"
FIGURES_DIR = OUTPUT_BASE / "figures"

# ── 实验注册表 ────────────────────────────────────────────────────────────

EXPERIMENTS = {
    1: ("exp1_main_comparison",    "experiments.exp1_main_comparison"),
    2: ("exp2_forest_reuse",       "experiments.exp2_forest_reuse"),
    3: ("exp3_obstacle_change",    "experiments.exp3_obstacle_change"),
    4: ("exp4_cache_warmstart",    "experiments.exp4_cache_warmstart"),
    5: ("exp5_ablation",           "experiments.exp5_ablation"),
    6: ("exp6_config_sweep",       "experiments.exp6_config_sweep"),
    7: ("exp7_scalability",        "experiments.exp7_scalability"),
    8: ("exp8_aabb_tightness",     "experiments.exp8_aabb_tightness"),
}

# ── 辅助 ──────────────────────────────────────────────────────────────────

class _TeeWriter:
    """同时写入多个流."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def _load_json_safe(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量运行 SBF 全部实验并保存结果")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true",
                      help="Quick 模式: 少量 seed, 快速验证")
    mode.add_argument("--full", action="store_true",
                      help="Full 模式: 论文级统计数据")
    parser.add_argument("--only", type=str, default=None,
                        help="只运行指定实验, 逗号分隔 (如 1,2,5)")
    parser.add_argument("--no-report", action="store_true",
                        help="跳过报告生成步骤")
    args = parser.parse_args()

    quick = args.quick

    # 确定要运行的实验
    if args.only:
        exp_ids = sorted(set(int(x.strip()) for x in args.only.split(",")))
        for eid in exp_ids:
            if eid not in EXPERIMENTS:
                print(f"ERROR: 实验 {eid} 不存在 (可用: 1-8)")
                sys.exit(1)
    else:
        exp_ids = list(range(1, 9))

    # ── 时间戳 + 日志 ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_label = "quick" if quick else "full"

    # 归档目录: output/archive/batch_quick_20260301_120000/
    archive_run_dir = ARCHIVE_DIR / f"batch_{mode_label}_{ts}"
    archive_run_dir.mkdir(parents=True, exist_ok=True)

    log_path = archive_run_dir / "batch.log"
    log_file = open(log_path, "w", encoding="utf-8")

    # stdout 同时输出到终端和日志
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeWriter(original_stdout, log_file)
    sys.stderr = _TeeWriter(original_stderr, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # ── 打印运行信息 ─────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  SBF Experiment Batch Runner")
    print(f"  Mode:       {mode_label}")
    print(f"  Timestamp:  {ts}")
    print(f"  Experiments: {exp_ids}")
    print(f"  Archive:    {archive_run_dir}")
    print("=" * 70)
    print()

    # ── 依次运行实验 ─────────────────────────────────────────────────────
    results_map: dict[int, dict] = {}   # exp_id → {status, path, duration, error}
    batch_t0 = time.perf_counter()

    for eid in exp_ids:
        exp_label, module_name = EXPERIMENTS[eid]
        print("\n" + "─" * 70)
        print(f"  [{eid}/8] {exp_label}  (mode={mode_label})")
        print("─" * 70)

        t0 = time.perf_counter()
        try:
            mod = __import__(module_name, fromlist=["run"])
            out_path = mod.run(quick=quick)
            elapsed = time.perf_counter() - t0

            results_map[eid] = {
                "status": "OK",
                "path": str(out_path),
                "duration": elapsed,
            }
            print(f"\n  ✓ {exp_label} completed in {_fmt_duration(elapsed)}")
            print(f"    → {out_path}")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            results_map[eid] = {
                "status": "FAIL",
                "error": str(exc),
                "traceback": tb,
                "duration": elapsed,
            }
            print(f"\n  ✗ {exp_label} FAILED after {_fmt_duration(elapsed)}")
            print(f"    Error: {exc}")
            print(tb)

    batch_elapsed = time.perf_counter() - batch_t0

    # ── 归档: 复制 raw JSON 到本次运行目录 ────────────────────────────────
    print("\n" + "=" * 70)
    print("  Archiving results...")
    print("=" * 70)

    archive_raw = archive_run_dir / "raw"
    archive_raw.mkdir(exist_ok=True)

    for eid, info in results_map.items():
        if info["status"] == "OK" and info.get("path"):
            src = Path(info["path"])
            if src.exists():
                dst = archive_raw / src.name
                shutil.copy2(src, dst)
                print(f"  Archived: {src.name}")

    # 保存运行元信息
    run_meta = {
        "timestamp": ts,
        "mode": mode_label,
        "experiments": exp_ids,
        "total_duration": batch_elapsed,
        "results": {str(k): v for k, v in results_map.items()},
    }
    meta_path = archive_run_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta_path}")

    # ── 生成报告 ─────────────────────────────────────────────────────────
    if not args.no_report:
        print("\n" + "=" * 70)
        print("  Generating reports...")
        print("=" * 70)

        try:
            from experiments.reporting import generate_report
            for eid, info in results_map.items():
                if info["status"] == "OK" and info.get("path"):
                    try:
                        generate_report(info["path"])
                    except Exception as exc:
                        print(f"  WARNING: report for exp{eid} failed: {exc}")
        except ImportError as ie:
            print(f"  WARNING: could not import reporting: {ie}")

        # 复制生成的 tables/ 和 figures/ 到归档
        for subdir_name in ("tables", "figures"):
            src_dir = OUTPUT_BASE / subdir_name
            if src_dir.exists() and any(src_dir.iterdir()):
                dst_dir = archive_run_dir / subdir_name
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                n_files = len(list(dst_dir.iterdir()))
                print(f"  Archived {n_files} files from {subdir_name}/")

    # ── 最终汇总 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BATCH SUMMARY")
    print("=" * 70)
    print(f"  Total time: {_fmt_duration(batch_elapsed)}")
    print()

    n_ok = sum(1 for v in results_map.values() if v["status"] == "OK")
    n_fail = sum(1 for v in results_map.values() if v["status"] == "FAIL")
    print(f"  Passed: {n_ok}/{len(results_map)}    Failed: {n_fail}")
    print()

    for eid in exp_ids:
        info = results_map.get(eid, {})
        label = EXPERIMENTS[eid][0]
        status = info.get("status", "SKIP")
        dur = info.get("duration", 0)
        marker = "✓" if status == "OK" else "✗"
        line = f"  {marker} exp{eid}: {label:<30s} {status:>5s}  {_fmt_duration(dur):>10s}"
        if status == "FAIL":
            line += f"  [{info.get('error', '')}]"
        print(line)

    print()
    print(f"  Archive: {archive_run_dir}")
    print(f"  Log:     {log_path}")
    print("=" * 70)

    # ── JSON 结果索引 ────────────────────────────────────────────────────
    # 方便后续一键加载本次全部结果
    index = {
        "timestamp": ts,
        "mode": mode_label,
        "passed": n_ok,
        "failed": n_fail,
        "total_duration": batch_elapsed,
        "result_files": {},
    }
    for eid in exp_ids:
        info = results_map.get(eid, {})
        if info.get("status") == "OK":
            index["result_files"][f"exp{eid}"] = info["path"]
    index_path = archive_run_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    # 恢复 stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
