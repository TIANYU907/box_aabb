"""
planner/models.py - 规划器数据模型

定义 Box-RRT 规划器使用的核心数据结构：BoxNode、BoxTree、Obstacle、
PlannerConfig、PlannerResult、Edge。
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Obstacle:
    """AABB 障碍物

    在工作空间 (Cartesian) 中定义的轴对齐包围盒。

    Attributes:
        min_point: AABB 最小角点 [x, y, z]
        max_point: AABB 最大角点 [x, y, z]
        name: 障碍物名称（可选）
    """
    min_point: np.ndarray
    max_point: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.min_point, np.ndarray):
            self.min_point = np.array(self.min_point, dtype=np.float64)
        if not isinstance(self.max_point, np.ndarray):
            self.max_point = np.array(self.max_point, dtype=np.float64)
        if self.min_point.shape != self.max_point.shape:
            raise ValueError("min_point 和 max_point 维度不匹配")

    @property
    def center(self) -> np.ndarray:
        """障碍物中心点"""
        return (self.min_point + self.max_point) / 2.0

    @property
    def size(self) -> np.ndarray:
        """各轴尺寸"""
        return self.max_point - self.min_point

    @property
    def volume(self) -> float:
        """体积"""
        s = self.size
        return float(np.prod(np.maximum(s, 0.0)))

    def to_dict(self) -> Dict[str, Any]:
        """转为与 Visualizer.plot_obstacles 兼容的 dict 格式"""
        return {
            'min': self.min_point.tolist(),
            'max': self.max_point.tolist(),
            'name': self.name,
        }

    def contains_point(self, point: np.ndarray) -> bool:
        """检查点是否在障碍物 AABB 内"""
        return bool(np.all(point >= self.min_point) and np.all(point <= self.max_point))


@dataclass
class BoxNode:
    """C-space 中的一个无碰撞 box 节点

    每个 BoxNode 代表关节空间中的一个超矩形区域，保证该区域内
    所有配置对应的机械臂不与障碍物碰撞（保守检测下）。

    Attributes:
        node_id: 节点唯一标识
        joint_intervals: 关节区间列表 [(lo_0, hi_0), ..., (lo_n, hi_n)]
        seed_config: 用于生成该 box 的种子配置
        parent_id: 父节点 ID（根节点为 None）
        children_ids: 子节点 ID 列表
        volume: box 体积（关节空间中）
        tree_id: 所属树的 ID
    """
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

    def _compute_volume(self) -> float:
        """计算 box 体积（各维宽度之积，忽略固定关节）"""
        vol = 1.0
        has_nonzero = False
        for lo, hi in self.joint_intervals:
            width = hi - lo
            if width > 0:
                vol *= width
                has_nonzero = True
            # 跳过宽度=0 的维度（固定关节），不让它把体积乘为 0
        return vol if has_nonzero else 0.0

    @property
    def center(self) -> np.ndarray:
        """box 中心点（关节空间）"""
        return np.array([(lo + hi) / 2.0 for lo, hi in self.joint_intervals])

    @property
    def widths(self) -> np.ndarray:
        """各维宽度"""
        return np.array([hi - lo for lo, hi in self.joint_intervals])

    @property
    def n_dims(self) -> int:
        """关节空间维度"""
        return len(self.joint_intervals)

    def contains(self, config: np.ndarray) -> bool:
        """检查配置是否在 box 内"""
        for i, (lo, hi) in enumerate(self.joint_intervals):
            if config[i] < lo - 1e-10 or config[i] > hi + 1e-10:
                return False
        return True

    def distance_to_config(self, config: np.ndarray) -> float:
        """计算配置到 box 的最小 L2 距离（在 box 内返回 0）"""
        d = 0.0
        for i, (lo, hi) in enumerate(self.joint_intervals):
            if config[i] < lo:
                d += (lo - config[i]) ** 2
            elif config[i] > hi:
                d += (config[i] - hi) ** 2
        return np.sqrt(d)

    def overlap_with(self, other: 'BoxNode') -> bool:
        """检查是否与另一个 box 有交集"""
        for (lo1, hi1), (lo2, hi2) in zip(self.joint_intervals, other.joint_intervals):
            if hi1 < lo2 - 1e-10 or hi2 < lo1 - 1e-10:
                return False
        return True

    def overlap_volume(self, other: 'BoxNode') -> float:
        """计算与另一个 box 的交集体积

        对每个维度取区间交集 [max(lo1,lo2), min(hi1,hi2)]，
        若任意维度交集为空则返回 0。

        Args:
            other: 另一个 BoxNode

        Returns:
            交集区域的体积（超矩形体积）
        """
        vol = 1.0
        for (lo1, hi1), (lo2, hi2) in zip(self.joint_intervals, other.joint_intervals):
            lo = max(lo1, lo2)
            hi = min(hi1, hi2)
            if lo >= hi:
                return 0.0
            vol *= (hi - lo)
        return vol

    def nearest_point_to(self, config: np.ndarray) -> np.ndarray:
        """返回 box 内离给定配置最近的点"""
        nearest = np.empty(self.n_dims)
        for i, (lo, hi) in enumerate(self.joint_intervals):
            nearest[i] = np.clip(config[i], lo, hi)
        return nearest

    def is_adjacent_to(self, other: 'BoxNode', tol: float = 1e-8) -> bool:
        """检查是否与另一个 box 邻接（面相接或微小重叠）

        邻接条件：恰好一个维度面相接，其余维度投影有正面积重叠。
        微小重叠也视为邻接。
        """
        n_contact = 0
        n_overlap = 0
        for (a_lo, a_hi), (b_lo, b_hi) in zip(
            self.joint_intervals, other.joint_intervals
        ):
            overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
            if overlap < -tol:
                return False  # 分离
            elif overlap <= tol:
                n_contact += 1
            else:
                n_overlap += 1
        if n_contact == 1 and n_overlap == self.n_dims - 1:
            return True
        # 微小全重叠也视为邻接
        if n_overlap == self.n_dims:
            return self.overlap_volume(other) < tol * 100
        return False


@dataclass
class BoxTree:
    """Box 树结构

    以一个根 box 为起点，通过边界采样和拓展生长的树。

    Attributes:
        tree_id: 树的唯一标识
        nodes: 节点字典 {node_id: BoxNode}
        root_id: 根节点 ID
    """
    tree_id: int
    nodes: Dict[int, BoxNode] = field(default_factory=dict)
    root_id: int = -1

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_volume(self) -> float:
        """所有节点 box 的总体积"""
        return sum(n.volume for n in self.nodes.values())

    def get_leaf_nodes(self) -> List[BoxNode]:
        """获取叶子节点（无子节点的节点）"""
        return [n for n in self.nodes.values() if not n.children_ids]

    def get_all_configs_in_tree(self) -> List[np.ndarray]:
        """获取树中所有节点的 seed 配置"""
        return [n.seed_config for n in self.nodes.values()]


@dataclass
class Edge:
    """Graph 中的边

    连接两个 box 之间的线段或两个 box 内的连接点。

    Attributes:
        edge_id: 边的唯一标识
        source_box_id: 起始 box 节点 ID
        target_box_id: 目标 box 节点 ID
        source_config: 起始 box 内的连接点配置
        target_config: 目标 box 内的连接点配置
        source_tree_id: 起始 box 所属树 ID
        target_tree_id: 目标 box 所属树 ID
        cost: 边的代价（默认为两点间的 L2 距离）
        is_collision_free: 边是否经过碰撞检测验证
    """
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
    """Box-RRT 规划器参数配置

    Attributes:
        max_iterations: 最大采样迭代次数
        max_box_nodes: 生成 box 节点的最大数量
        seed_batch_size: 每次边界重采样的 seed 数量
        min_box_volume: box 体积下限（太小的 box 丢弃）
        goal_bias: 朝目标方向采样的概率 [0, 1]
        expansion_resolution: box 拓展时二分搜索精度
        max_expansion_rounds: box 拓展最大迭代轮数
        jacobian_delta: 计算 Jacobian 范数的数值差分步长
        min_initial_half_width: box 初始半宽 (seed 两侧)
        expansion_strategy: box 拓展策略 ('greedy' / 'balanced')
        balanced_step_fraction: balanced 策略的初始比例步长 (0,1]
        balanced_max_steps: balanced 策略最大步数
        use_sampling: 是否使用采样辅助碰撞检测 (None=自动)
        sampling_n: 采样辅助碰撞检测的采样数
        segment_collision_resolution: 线段碰撞检测采样间隔 (rad)
        connection_max_attempts: 树间连接最大尝试次数
        connection_radius: 树间连接最大搜索半径 (rad)
        path_shortcut_iters: 路径 shortcut 优化最大迭代次数
        use_gcs: 是否使用 GCS 优化路径（需要 Drake）
        gcs_bezier_degree: GCS Bézier 曲线阶数
        verbose: 是否输出详细日志
        build_n_seeds: BoxForest 构建阶段的采样种子数量
        query_expand_budget: 查询阶段始末点附近拓展的最大 box 数
        forest_path: 预构建 BoxForest 文件路径（可选）
        interval_width_threshold: 区间/数值方法切换阈值 (rad)
        use_aabb_cache: 是否启用 AABB 包络缓存
    """
    max_iterations: int = 500
    max_box_nodes: int = 200
    seed_batch_size: int = 5
    min_box_volume: float = 1e-6
    goal_bias: float = 0.1
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
    # v4.0 新增：BoxForest 复用
    build_n_seeds: int = 200
    query_expand_budget: int = 10
    forest_path: Optional[str] = None
    # v4.0 新增：自适应阈值与缓存
    interval_width_threshold: float = 1.0
    use_aabb_cache: bool = True
    # v4.2 新增：重叠惩罚
    overlap_weight: float = 1.0
    # v5.0 新增：无重叠 BoxForest
    min_fragment_volume: float = 1e-6
    adjacency_tolerance: float = 1e-8
    hard_overlap_reject: bool = True

    # ── JSON 序列化 ──

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        from dataclasses import fields as dc_fields
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    def to_json(self, filepath: str | Path) -> str:
        """保存到 JSON 文件

        Args:
            filepath: 输出路径

        Returns:
            保存的文件路径字符串
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return str(filepath)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlannerConfig':
        """从字典创建（忽略未知字段，缺失字段用默认值）"""
        from dataclasses import fields as dc_fields
        valid_fields = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: str | Path) -> 'PlannerConfig':
        """从 JSON 文件加载

        Args:
            filepath: JSON 配置文件路径

        Returns:
            PlannerConfig 实例
        """
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PlannerResult:
    """路径规划结果

    Attributes:
        success: 是否成功找到路径
        path: 关节空间路径点序列 [q_start, ..., q_goal]
        box_trees: 构建的 box tree 列表
        forest: BoxForest 实例（v5 无重叠 box 集合）
        edges: 连接边列表
        computation_time: 总计算时间 (s)
        path_length: 路径总长度 (L2 norm in joint space)
        n_boxes_created: 创建的 box 总数
        n_collision_checks: 碰撞检测调用总次数
        message: 描述信息
        timestamp: 时间戳
    """
    success: bool = False
    path: List[np.ndarray] = field(default_factory=list)
    box_trees: List[BoxTree] = field(default_factory=list)
    forest: Any = None  # BoxForest (avoid circular import)
    edges: List[Edge] = field(default_factory=list)
    computation_time: float = 0.0
    path_length: float = 0.0
    n_boxes_created: int = 0
    n_collision_checks: int = 0
    message: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))

    def compute_path_length(self) -> float:
        """计算路径总长度"""
        if len(self.path) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(self.path)):
            length += float(np.linalg.norm(self.path[i] - self.path[i - 1]))
        self.path_length = length
        return length

    # ── 路径序列化 ─────────────────────────────────────────

    def save_path(
        self,
        filepath: str | Path,
        robot_config: str = "",
        scene_json: str = "",
        q_start: Optional[np.ndarray] = None,
        q_goal: Optional[np.ndarray] = None,
    ) -> str:
        """将规划路径保存为 JSON 文件

        Args:
            filepath: 输出 JSON 路径
            robot_config: 机器人配置名称 (如 'panda')
            scene_json: 关联的场景 JSON 路径
            q_start: 起点配置 (可选, 默认取 path[0])
            q_goal: 终点配置 (可选, 默认取 path[-1])

        Returns:
            保存的文件路径字符串
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "robot_config": robot_config,
            "scene_json": str(scene_json) if scene_json else "",
            "success": self.success,
            "path": [q.tolist() for q in self.path],
            "n_waypoints": len(self.path),
            "path_length": self.path_length,
            "computation_time": self.computation_time,
            "n_boxes_created": self.n_boxes_created,
            "n_collision_checks": self.n_collision_checks,
            "message": self.message,
            "timestamp": self.timestamp,
        }
        if q_start is not None:
            data["q_start"] = np.asarray(q_start).tolist()
        elif self.path:
            data["q_start"] = self.path[0].tolist()
        if q_goal is not None:
            data["q_goal"] = np.asarray(q_goal).tolist()
        elif len(self.path) >= 2:
            data["q_goal"] = self.path[-1].tolist()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    @staticmethod
    def load_path(filepath: str | Path) -> Dict[str, Any]:
        """从 JSON 文件加载规划路径

        Args:
            filepath: 路径 JSON 文件

        Returns:
            字典, 包含:
              - robot_config (str)
              - scene_json (str)
              - path (List[np.ndarray])
              - path_length (float)
              - 及其它元数据
        """
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['path'] = [np.array(q, dtype=np.float64) for q in data['path']]
        if 'q_start' in data:
            data['q_start'] = np.array(data['q_start'], dtype=np.float64)
        if 'q_goal' in data:
            data['q_goal'] = np.array(data['q_goal'], dtype=np.float64)
        return data
