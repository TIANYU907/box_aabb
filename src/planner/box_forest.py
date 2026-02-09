"""
planner/box_forest.py - 可复用 Box 森林

在场景中维护可复用的 box trees，当始末点变化时也能快速重新规划。

核心思想：
- build 阶段：无目标偏向地均匀采样 seed 点，构建覆盖 C-free 的 box 森林
- 森林可持久化到磁盘（pickle），换始末点时无需重建
- query 阶段：在已有森林基础上连接新的始末点、图搜索、路径平滑

使用方式：
    forest = BoxForest.build(robot, scene, config)
    forest.save("forest.pkl")
    # 换始末点
    forest = BoxForest.load("forest.pkl")
    query = BoxForestQuery(forest)
    result = query.plan(q_start, q_goal)
"""

import time
import pickle
import logging
from typing import List, Tuple, Optional

import numpy as np

from box_aabb.robot import Robot
from .models import PlannerConfig, PlannerResult, BoxNode
from .obstacles import Scene
from .collision import CollisionChecker
from .box_expansion import BoxExpander
from .box_tree import BoxTreeManager

logger = logging.getLogger(__name__)


class BoxForest:
    """可复用的 Box 森林

    预先构建覆盖 C-free 的 box 集合，可在多次规划中复用。

    Attributes:
        robot: 机器人模型
        scene: 障碍物场景
        tree_manager: box 树管理器
        joint_limits: 关节限制
        build_time: 构建耗时
    """

    def __init__(
        self,
        robot: Robot,
        scene: Scene,
        tree_manager: BoxTreeManager,
        joint_limits: List[Tuple[float, float]],
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.tree_manager = tree_manager
        self.joint_limits = joint_limits
        self.config = config or PlannerConfig()
        self.build_time: float = 0.0

    @classmethod
    def build(
        cls,
        robot: Robot,
        scene: Scene,
        config: Optional[PlannerConfig] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> 'BoxForest':
        """构建 Box 森林

        无目标偏向地均匀采样 seed 点，在 C-free 中构建 box 覆盖。

        Args:
            robot: 机器人模型
            scene: 障碍物场景
            config: 规划参数
            joint_limits: 关节限制
            seed: 随机数种子

        Returns:
            构建好的 BoxForest 实例
        """
        t0 = time.time()
        config = config or PlannerConfig()
        rng = np.random.default_rng(seed)

        if joint_limits is None:
            if robot.joint_limits is not None:
                joint_limits = list(robot.joint_limits)
            else:
                joint_limits = [(-np.pi, np.pi)] * robot.n_joints

        n_dims = len(joint_limits)

        checker = CollisionChecker(robot=robot, scene=scene)
        _use_sampling = robot.n_joints > 4
        expander = BoxExpander(
            robot=robot,
            collision_checker=checker,
            joint_limits=joint_limits,
            expansion_resolution=config.expansion_resolution,
            max_rounds=config.max_expansion_rounds,
            jacobian_delta=config.jacobian_delta,
            use_sampling=_use_sampling,
        )
        tree_mgr = BoxTreeManager()

        n_seeds = config.build_n_seeds
        n_boxes_created = 0
        max_box_nodes = config.max_box_nodes * 3  # 森林模式允许更多 box

        for i in range(n_seeds):
            if n_boxes_created >= max_box_nodes:
                break

            # 均匀随机采样（无目标偏向）
            q_seed = _sample_uniform(joint_limits, rng)
            if checker.check_config_collision(q_seed):
                continue

            # 跳过已被现有 box 覆盖的 seed
            if tree_mgr.find_containing_box(q_seed) is not None:
                continue

            node_id = tree_mgr.allocate_node_id()
            box = expander.expand(q_seed, node_id=node_id, rng=rng)
            if box is None or box.volume < config.min_box_volume:
                continue

            # 尝试加入现有树
            _add_box_to_forest(tree_mgr, box, config.connection_radius)
            n_boxes_created += 1

            # 边界拓展
            if box.tree_id >= 0 and n_boxes_created < max_box_nodes:
                _boundary_expand_forest(
                    tree_mgr, expander, checker, box.tree_id,
                    config, rng, max_box_nodes - n_boxes_created)
                n_boxes_created = tree_mgr.total_nodes

            if config.verbose and (i + 1) % 50 == 0:
                logger.info(
                    "Forest build: %d/%d seeds, %d trees, %d boxes",
                    i + 1, n_seeds, tree_mgr.n_trees, tree_mgr.total_nodes)

        build_time = time.time() - t0
        logger.info(
            "BoxForest 构建完成: %d 棵树, %d 个 box, 总体积 %.4f, 耗时 %.2fs",
            tree_mgr.n_trees, tree_mgr.total_nodes,
            tree_mgr.get_total_volume(), build_time)

        forest = cls(robot, scene, tree_mgr, joint_limits, config)
        forest.build_time = build_time
        return forest

    @property
    def n_trees(self) -> int:
        return self.tree_manager.n_trees

    @property
    def n_boxes(self) -> int:
        return self.tree_manager.total_nodes

    @property
    def total_volume(self) -> float:
        return self.tree_manager.get_total_volume()

    def save(self, filepath: str) -> None:
        """持久化到文件"""
        data = {
            'robot_fingerprint': self.robot.fingerprint(),
            'tree_manager': self.tree_manager,
            'joint_limits': self.joint_limits,
            'config': self.config,
            'build_time': self.build_time,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BoxForest 已保存到 %s", filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        robot: Robot,
        scene: Scene,
    ) -> 'BoxForest':
        """从文件加载

        Args:
            filepath: 森林文件路径
            robot: 机器人模型（需与构建时一致）
            scene: 障碍物场景（需与构建时一致）

        Returns:
            加载的 BoxForest 实例

        Raises:
            ValueError: 机器人指纹不匹配
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if data['robot_fingerprint'] != robot.fingerprint():
            raise ValueError(
                f"机器人指纹不匹配: "
                f"文件中为 {data['robot_fingerprint'][:16]}..., "
                f"当前为 {robot.fingerprint()[:16]}...")

        forest = cls(
            robot=robot,
            scene=scene,
            tree_manager=data['tree_manager'],
            joint_limits=data['joint_limits'],
            config=data.get('config', PlannerConfig()),
        )
        forest.build_time = data.get('build_time', 0.0)
        logger.info(
            "BoxForest 从 %s 加载: %d 棵树, %d 个 box",
            filepath, forest.n_trees, forest.n_boxes)
        return forest


def _sample_uniform(
    joint_limits: List[Tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """在关节限制内均匀随机采样"""
    return np.array([rng.uniform(lo, hi) for lo, hi in joint_limits])


def _add_box_to_forest(
    tree_mgr: BoxTreeManager,
    box: BoxNode,
    connection_radius: float,
) -> None:
    """将 box 加入森林（最近树或新建树）"""
    seed = box.seed_config

    containing = tree_mgr.find_containing_box(seed)
    if containing is not None:
        tree_mgr.add_box(containing.tree_id, box, parent_id=containing.node_id)
        return

    nearest = tree_mgr.find_nearest_box(seed)
    if nearest is not None:
        dist = nearest.distance_to_config(seed)
        if dist < connection_radius:
            tree_mgr.add_box(nearest.tree_id, box, parent_id=nearest.node_id)
            return

    tree_mgr.create_tree(box)


def _boundary_expand_forest(
    tree_mgr: BoxTreeManager,
    expander: BoxExpander,
    checker: CollisionChecker,
    tree_id: int,
    config: PlannerConfig,
    rng: np.random.Generator,
    budget: int,
) -> None:
    """在森林中对指定树做边界拓展"""
    n_added = 0
    samples = tree_mgr.get_boundary_samples(tree_id, n_samples=3, rng=rng)

    for q_seed in samples:
        if n_added >= budget:
            break
        if checker.check_config_collision(q_seed):
            continue

        node_id = tree_mgr.allocate_node_id()
        new_box = expander.expand(q_seed, node_id=node_id, rng=rng)
        if new_box is None or new_box.volume < config.min_box_volume:
            continue

        nearest = tree_mgr.find_nearest_box_in_tree(tree_id, q_seed)
        if nearest is not None:
            tree_mgr.add_box(tree_id, new_box, parent_id=nearest.node_id)
            n_added += 1
