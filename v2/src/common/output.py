"""Output path helpers for v2 artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_output_dir(category: str, name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(__file__).resolve().parents[2] / "output" / category / f"{name}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path
