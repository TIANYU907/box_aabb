"""
baselines/ — 统一规划器接口 + Baseline 适配器

- base: BasePlanner ABC, PlanningResult dataclass
- sbf_adapter: SafeBoxForest pipeline adapter (supports_reuse=True)
- rrt_family: RRT / RRTConnect / RRT* / Informed-RRT* / BiRRT*
- ompl_adapter: OMPL C++ family (subprocess via WSL)
- iris_gcs: Drake IRIS + GCS (Marcucci)
"""

from .base import BasePlanner, PlanningResult
from .sbf_adapter import SBFAdapter
from .rrt_family import RRTPlanner
from .ompl_adapter import OMPLPlanner
from .iris_gcs import IRISGCSPlanner

__all__ = [
    "BasePlanner",
    "PlanningResult",
    "SBFAdapter",
    "RRTPlanner",
    "OMPLPlanner",
    "IRISGCSPlanner",
]

