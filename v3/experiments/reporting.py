"""
experiments/reporting.py — 结果报告生成

- LaTeX 表格
- Matplotlib 图表
- 汇总打印

用法:
    from experiments.reporting import generate_report
    generate_report("output/raw/exp1_main_comparison.json")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_results(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _group_by(results: List[dict],
              keys: Sequence[str]) -> Dict[str, List[dict]]:
    """Group result dicts by composite key."""
    groups = defaultdict(list)
    for r in results:
        key = " | ".join(str(r.get(k, "")) for k in keys)
        groups[key].append(r)
    return dict(groups)


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX table generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_latex_table(data: dict,
                         group_keys: Sequence[str] = ("scene", "planner"),
                         metrics: Sequence[str] = ("planning_time",
                                                    "cost", "success"),
                         caption: str = "",
                         label: str = "") -> str:
    """Generate LaTeX tabular from experiment results."""
    results = data["results"]
    groups = _group_by(results, group_keys)

    # header
    n_metrics = len(metrics)
    col_spec = "l" * len(group_keys) + "r" * n_metrics
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # header row
    header_parts = list(group_keys) + [m.replace("_", " ") for m in metrics]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header_parts)
                 + r" \\")
    lines.append(r"\midrule")

    # data rows
    for key, trials in sorted(groups.items()):
        key_parts = key.split(" | ")
        row_parts = list(key_parts)

        for metric in metrics:
            values = [t.get(metric) for t in trials
                      if t.get(metric) is not None]
            if metric == "success":
                n_success = sum(1 for v in values if v)
                row_parts.append(f"{n_success}/{len(values)}")
            elif values:
                s = _stats([float(v) for v in values
                            if not (isinstance(v, float) and np.isnan(v))])
                row_parts.append(f"{s['mean']:.3f} $\\pm$ {s['std']:.3f}")
            else:
                row_parts.append("--")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(line for line in lines if line)


def save_latex_table(data: dict, filename: str, **kwargs) -> Path:
    """Generate and save LaTeX table."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex = generate_latex_table(data, **kwargs)
    out = TABLES_DIR / filename
    out.write_text(tex, encoding="utf-8")
    logger.info("Saved LaTeX table → %s", out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure generation (matplotlib)
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison_violin(data: dict,
                           metric: str = "planning_time",
                           output_name: str = "comparison_violin.pdf"):
    """Violin plot comparing planners across scenes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = data["results"]
    groups = _group_by(results, ["planner"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = sorted(groups.keys())
    plot_data = []
    for name in names:
        vals = [r.get(metric, float("nan")) for r in groups[name]]
        vals = [v for v in vals if not np.isnan(v)]
        plot_data.append(vals)

    if all(len(d) > 0 for d in plot_data):
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=30, ha="right")

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{data.get('experiment', 'Experiment')} — {metric}")
    ax.grid(True, alpha=0.3)

    out = FIGURES_DIR / output_name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", out)
    return out


def plot_reuse_curves(data: dict,
                      output_name: str = "forest_reuse_curves.pdf"):
    """Plot cumulative time vs K for forest reuse experiment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = data["results"]
    groups = _group_by(results, ["planner"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name in sorted(groups.keys()):
        trials = groups[name]
        # group by K
        k_groups = defaultdict(list)
        for t in trials:
            k = t.get("K")
            if k is not None:
                k_groups[k].append(t.get("cumulative_time", 0))

        ks = sorted(k_groups.keys())
        means_cum = [np.mean(k_groups[k]) for k in ks]
        means_avg = [np.mean(k_groups[k]) / k for k in ks]

        ax1.plot(ks, means_cum, "o-", label=name)
        ax2.plot(ks, means_avg, "s-", label=name)

    ax1.set_xlabel("K (number of queries)")
    ax1.set_ylabel("Cumulative time (s)")
    ax1.set_title("Cumulative Planning Time vs K")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("K (number of queries)")
    ax2.set_ylabel("Avg time per query (s)")
    ax2.set_title("Average Per-Query Time vs K")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    out = FIGURES_DIR / output_name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", out)
    return out


def plot_scalability(data: dict,
                     output_name: str = "scalability.pdf"):
    """Plot time vs obstacle count and vs DOF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # time vs obstacle count (grouped by planner, for Panda only)
    panda_results = [r for r in results if r.get("robot") == "panda"]
    groups = _group_by(panda_results, ["planner"])
    for name in sorted(groups.keys()):
        trials = groups[name]
        obs_groups = defaultdict(list)
        for t in trials:
            n_obs = t.get("n_obstacles")
            if n_obs is not None:
                obs_groups[n_obs].append(t.get("planning_time", 0))

        xs = sorted(obs_groups.keys())
        ys = [np.mean(obs_groups[x]) for x in xs]
        ax1.plot(xs, ys, "o-", label=name)

    ax1.set_xlabel("Number of obstacles")
    ax1.set_ylabel("Planning time (s)")
    ax1.set_title("Time vs Obstacles (Panda 7-DOF)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # time vs DOF (fixed obstacle count)
    target_obs = 8
    dof_results = [r for r in results if r.get("n_obstacles") == target_obs]
    groups = _group_by(dof_results, ["planner"])
    for name in sorted(groups.keys()):
        trials = groups[name]
        dof_groups = defaultdict(list)
        for t in trials:
            dof = t.get("n_dof")
            if dof is not None:
                dof_groups[dof].append(t.get("planning_time", 0))

        xs = sorted(dof_groups.keys())
        ys = [np.mean(dof_groups[x]) for x in xs]
        ax2.bar([str(x) for x in xs], ys, label=name, alpha=0.7)

    ax2.set_xlabel("Degrees of Freedom")
    ax2.set_ylabel("Planning time (s)")
    ax2.set_title(f"Time vs DOF ({target_obs} obstacles)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    out = FIGURES_DIR / output_name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Unified report generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(result_path: str | Path) -> None:
    """Generate all applicable tables and figures for a result file."""
    data = load_results(result_path)
    exp_name = data.get("experiment", "")

    print(f"\n=== Generating report for {exp_name} ===")
    print(f"  Trials: {data.get('n_trials', len(data.get('results', [])))}")

    if "main_comparison" in exp_name:
        save_latex_table(data, "table1_main_comparison.tex",
                         caption="Main comparison of planning methods",
                         label="tab:main_comparison")
        plot_comparison_violin(data, output_name="fig2_main_comparison.pdf")

    elif "forest_reuse" in exp_name:
        save_latex_table(data, "table2_forest_reuse.tex",
                         group_keys=("planner",),
                         metrics=("cumulative_time", "success"),
                         caption="Forest reuse performance",
                         label="tab:forest_reuse")
        plot_reuse_curves(data, output_name="fig3_forest_reuse.pdf")

    elif "obstacle_change" in exp_name:
        save_latex_table(data, "table3_obstacle_change.tex",
                         group_keys=("scene", "planner"),
                         metrics=("planning_time", "success"),
                         caption="Obstacle change incremental update",
                         label="tab:obstacle_change")

    elif "cache_warmstart" in exp_name:
        save_latex_table(data, "table4_cache_warmstart.tex",
                         group_keys=("planner",),
                         metrics=("planning_time",),
                         caption="Cache warmstart comparison",
                         label="tab:cache_warmstart")

    elif "ablation" in exp_name:
        save_latex_table(data, "table5_ablation.tex",
                         group_keys=("scene", "planner"),
                         metrics=("planning_time", "success", "cost"),
                         caption="Ablation study",
                         label="tab:ablation")

    elif "scalability" in exp_name:
        plot_scalability(data, output_name="fig8_scalability.pdf")

    elif "aabb_tightness" in exp_name:
        save_latex_table(data, "table6_aabb_tightness.tex",
                         group_keys=("scene",),
                         metrics=("tightness_ratio",
                                  "time_interval_fk"),
                         caption="AABB tightness analysis",
                         label="tab:aabb_tightness")

    print("  Done.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.reporting <result_json>")
        sys.exit(1)
    generate_report(sys.argv[1])
