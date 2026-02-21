"""aabb/report.py — 向后兼容 shim, 实际实现位于 viz/aabb_report.py"""

import sys
from pathlib import Path

# viz/ 位于 v3 根目录，不在 src/ 内
_VIZ = str(Path(__file__).resolve().parents[2] / "viz")
if _VIZ not in sys.path:
    sys.path.insert(0, _VIZ)

from aabb_report import ReportGenerator  # noqa: F401, E402

__all__ = ["ReportGenerator"]
