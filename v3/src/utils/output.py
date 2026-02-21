"""Output path helpers for SBF artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_output_dir(category: str, name: str) -> Path:
    """Create a timestamped output directory.

    Args:
        category: Top-level category (e.g. "benchmarks", "visualizations")
        name: Sub-name for this run

    Returns:
        Path to the created directory
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(__file__).resolve().parents[2] / "output" / category / f"{name}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path
