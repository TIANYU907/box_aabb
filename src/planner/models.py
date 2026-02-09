"""
planner/models.py - 规划器数据模型

定义 Box-RRT 规划器使用的核心数据结构：BoxNode、BoxTree、Obstacle、
PlannerConfig、PlannerResult、Edge。
"""

import numpy as np
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

    def nearest_point_to(self, config: np.ndarray) -> np.ndarray:
        """返回 box 内离给定配置最近的点"""
        nearest = np.empty(self.n_dims)
        for i, (lo, hi) in enumerate(self.joint_intervals):
            nearest[i] = np.clip(config[i], lo, hi)
        return nearest


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
        segment_collision_resolution: 线段碰撞检测采样间隔 (rad)
        connection_max_attempts: 树间连接最大尝试次数
        connection_radius: 树间连接最大搜索半径 (rad)
        path_shortcut_iters: 路径 shortcut 优化最大迭代次数
        use_gcs: 是否使用 GCS 优化路径（需要 Drake）
        gcs_bezier_degree: GCS Bézier 曲线阶数
        verbose: 是否输出详细日志
    """
    max_iterations: int = 500
    max_box_nodes: int = 200
    seed_batch_size: int = 5
    min_box_volume: float = 1e-6
    goal_bias: float = 0.1
    expansion_resolution: float = 0.01
    max_expansion_rounds: int = 3
    jacobian_delta: float = 0.01
    segment_collision_resolution: float = 0.05
    connection_max_attempts: int = 50
    connection_radius: float = 2.0
    path_shortcut_iters: int = 100
    use_gcs: bool = False
    gcs_bezier_degree: int = 3
    verbose: bool = False


@dataclass
class PlannerResult:
    """路径规划结果

    Attributes:
        success: 是否成功找到路径
        path: 关节空间路径点序列 [q_start, ..., q_goal]
        box_trees: 构建的 box tree 列表
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
