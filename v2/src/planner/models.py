"""v2 planner data models."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from forest.models import BoxNode


def gmean_edge_length(volume: float, ndim: int) -> float:
    if volume <= 0 or ndim <= 0:
        return 0.0
    return volume ** (1.0 / ndim)


@dataclass
class BoxTree:
    tree_id: int
    nodes: Dict[int, BoxNode] = field(default_factory=dict)
    root_id: int = -1

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_volume(self) -> float:
        return sum(n.volume for n in self.nodes.values())

    def get_leaf_nodes(self) -> List[BoxNode]:
        return [n for n in self.nodes.values() if not n.children_ids]

    def get_all_configs_in_tree(self) -> List[np.ndarray]:
        return [n.seed_config for n in self.nodes.values()]


@dataclass
class Edge:
    edge_id: int
    source_box_id: int
    target_box_id: int
    source_config: np.ndarray
    target_config: np.ndarray
    source_tree_id: int = -1
    target_tree_id: int = -1
    cost: float = 0.0
    is_collision_free: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_config, np.ndarray):
            self.source_config = np.array(self.source_config, dtype=np.float64)
        if not isinstance(self.target_config, np.ndarray):
            self.target_config = np.array(self.target_config, dtype=np.float64)
        if self.cost == 0.0:
            self.cost = float(np.linalg.norm(self.source_config - self.target_config))


@dataclass
class PlannerConfig:
    max_iterations: int = 500
    max_box_nodes: int = 200
    seed_batch_size: int = 5
    min_box_size: float = 0.001
    goal_bias: float = 0.1
    guided_sample_ratio: float = 0.6
    expansion_resolution: float = 0.01
    max_expansion_rounds: int = 3
    jacobian_delta: float = 0.01
    min_initial_half_width: float = 0.001
    expansion_strategy: str = 'balanced'
    balanced_step_fraction: float = 0.5
    balanced_max_steps: int = 200
    use_sampling: Optional[bool] = None
    sampling_n: int = 80
    segment_collision_resolution: float = 0.05
    connection_max_attempts: int = 50
    connection_radius: float = 2.0
    path_shortcut_iters: int = 100
    use_gcs: bool = False
    gcs_bezier_degree: int = 3
    verbose: bool = False
    build_n_seeds: int = 200
    query_expand_budget: int = 10
    forest_path: Optional[str] = None
    interval_width_threshold: float = 1.0
    overlap_weight: float = 1.0
    adjacency_tolerance: float = 1e-8
    hard_overlap_reject: bool = True
    parallel_expand: bool = False
    parallel_workers: int = 0
    parallel_batch_size: int = 32
    parallel_partition_depth: int = 2
    parallel_partition_dims: Optional[List[int]] = None
    parallel_cross_partition_connect: bool = True

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import fields as dc_fields
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    def to_json(self, filepath: str | Path) -> str:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return str(filepath)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlannerConfig':
        from dataclasses import fields as dc_fields
        compat = dict(data)
        if 'min_box_volume' in compat and 'min_box_size' not in compat:
            compat['min_box_size'] = compat.pop('min_box_volume')
        else:
            compat.pop('min_box_volume', None)
        compat.pop('min_fragment_volume', None)
        compat.pop('min_fragment_size', None)
        valid_fields = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in compat.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: str | Path) -> 'PlannerConfig':
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PlannerResult:
    success: bool = False
    path: List[np.ndarray] = field(default_factory=list)
    box_trees: List[BoxTree] = field(default_factory=list)
    forest: Any = None
    edges: List[Edge] = field(default_factory=list)
    computation_time: float = 0.0
    path_length: float = 0.0
    n_boxes_created: int = 0
    n_collision_checks: int = 0
    message: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))

    def compute_path_length(self) -> float:
        if len(self.path) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(self.path)):
            length += float(np.linalg.norm(self.path[i] - self.path[i - 1]))
        self.path_length = length
        return length
