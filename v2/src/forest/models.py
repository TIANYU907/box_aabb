"""v2 forest data models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Obstacle:
    """AABB obstacle in workspace coordinates."""

    min_point: np.ndarray
    max_point: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.min_point, np.ndarray):
            self.min_point = np.array(self.min_point, dtype=np.float64)
        if not isinstance(self.max_point, np.ndarray):
            self.max_point = np.array(self.max_point, dtype=np.float64)
        if self.min_point.shape != self.max_point.shape:
            raise ValueError("min_point and max_point shape mismatch")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": self.min_point.tolist(),
            "max": self.max_point.tolist(),
            "name": self.name,
        }


@dataclass
class PlannerConfig:
    """Forest-side config subset used by BoxForest.

    Kept in forest layer to preserve dependency direction: planner -> forest.
    """

    adjacency_tolerance: float = 1e-8


@dataclass
class BoxNode:
    """Non-overlapping C-space box."""

    node_id: int
    joint_intervals: List[Tuple[float, float]]
    seed_config: np.ndarray
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    volume: float = 0.0
    tree_id: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.seed_config, np.ndarray):
            self.seed_config = np.array(self.seed_config, dtype=np.float64)
        if self.volume == 0.0:
            self.volume = self._compute_volume()

    @property
    def n_dims(self) -> int:
        return len(self.joint_intervals)

    @property
    def center(self) -> np.ndarray:
        return np.array([(lo + hi) / 2.0 for lo, hi in self.joint_intervals], dtype=np.float64)

    def _compute_volume(self) -> float:
        vol = 1.0
        has_nonzero = False
        for lo, hi in self.joint_intervals:
            width = hi - lo
            if width > 0:
                vol *= width
                has_nonzero = True
        return vol if has_nonzero else 0.0

    def contains(self, config: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self.joint_intervals):
            if config[i] < lo - 1e-10 or config[i] > hi + 1e-10:
                return False
        return True

    def distance_to_config(self, config: np.ndarray) -> float:
        d2 = 0.0
        for i, (lo, hi) in enumerate(self.joint_intervals):
            if config[i] < lo:
                d2 += (lo - config[i]) ** 2
            elif config[i] > hi:
                d2 += (config[i] - hi) ** 2
        return float(np.sqrt(d2))

    def nearest_point_to(self, config: np.ndarray) -> np.ndarray:
        nearest = np.empty(self.n_dims, dtype=np.float64)
        for i, (lo, hi) in enumerate(self.joint_intervals):
            nearest[i] = np.clip(config[i], lo, hi)
        return nearest

    def overlap_with(self, other: 'BoxNode') -> bool:
        for (lo1, hi1), (lo2, hi2) in zip(self.joint_intervals, other.joint_intervals):
            if hi1 < lo2 - 1e-10 or hi2 < lo1 - 1e-10:
                return False
        return True
