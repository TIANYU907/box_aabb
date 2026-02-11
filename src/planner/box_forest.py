"""
planner/box_forest.py - 扁平无重叠 Box 森林

在 C-free 空间中维护一组互不重叠（容忍微小角落重叠）的 axis-aligned
hyperrectangle（BoxNode），以及它们之间的邻接关系。

核心特性：
- **零重叠不变量**：所有存储的 box 之间无实质性重叠
  （微小角落重叠 < min_fragment_volume 视为邻接）
- **扁平邻接图**：无树层级，只有 box 和邻接边
- **跨场景复用**：森林绑定机器人型号而非场景，不同场景加载后
  惰性验证碰撞（AABB 缓存避免重复 FK）
- **增量增密**：每次规划可添加新 box，自动 deoverlap + 邻接更新

使用方式：
    # 构建
    forest = BoxForest.build(robot, scene, config)
    forest.save("forest.pkl")

    # 跨场景复用
    forest = BoxForest.load("forest.pkl", robot)
    valid_ids = forest.validate_boxes(collision_checker)
    # 规划时使用 forest.boxes, forest.adjacency
"""

import time
import pickle
import logging
from typing import List, Tuple, Dict, Set, Optional

import numpy as np

from box_aabb.robot import Robot
from .models import PlannerConfig, BoxNode
from .obstacles import Scene
from .collision import CollisionChecker
from .box_expansion import BoxExpander
from .aabb_cache import AABBCache
from .deoverlap import (
    deoverlap,
    compute_adjacency,
    compute_adjacency_incremental,
)

logger = logging.getLogger(__name__)


class BoxForest:
    """扁平无重叠 Box 森林

    维护一组互不重叠的 BoxNode 和它们的邻接关系。
    所有 box 存储在同一个字典中，无树层级。

    Attributes:
        robot_fingerprint: 机器人指纹标识
        boxes: {node_id: BoxNode} 所有无重叠 box
        adjacency: {node_id: set of adjacent node_ids}
        joint_limits: 关节限制
        config: 规划参数
        build_time: 累计构建耗时
    """

    def __init__(
        self,
        robot_fingerprint: str,
        joint_limits: List[Tuple[float, float]],
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.robot_fingerprint = robot_fingerprint
        self.joint_limits = joint_limits
        self.config = config or PlannerConfig()
        self.boxes: Dict[int, BoxNode] = {}
        self.adjacency: Dict[int, Set[int]] = {}
        self._next_id: int = 0
        self.build_time: float = 0.0

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    @property
    def total_volume(self) -> float:
        return sum(b.volume for b in self.boxes.values())

    def allocate_id(self) -> int:
        """分配下一个可用的 box ID"""
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_boxes(
        self,
        new_boxes: List[BoxNode],
        min_frag_vol: Optional[float] = None,
    ) -> List[BoxNode]:
        """将新 box 集合并入森林

        对新 box 与已有 box 执行 deoverlap（先来先得：已有 box 优先），
        然后增量更新邻接关系。

        Args:
            new_boxes: 新扩展的 BoxNode 列表
            min_frag_vol: 碎片最小体积阈值

        Returns:
            实际添加的 box 列表（去重叠后的碎片）
        """
        if not new_boxes:
            return []

        if min_frag_vol is None:
            min_frag_vol = self.config.min_fragment_volume

        # 合并列表：已有 box 在前（优先），新 box 在后
        existing_list = list(self.boxes.values())
        all_input = existing_list + new_boxes

        # 执行全量 deoverlap
        all_deoverlapped = deoverlap(
            all_input,
            min_fragment_volume=min_frag_vol,
            id_start=self._next_id,
        )

        # 分离：前 len(existing_list) 个对应已有 box（可能未变），
        # 后面的是新碎片。但 deoverlap 的 id 已重新分配。
        # 简单起见：清空重建（N 通常不大，O(N²D) 可接受）
        self.boxes.clear()
        self.adjacency.clear()

        for box in all_deoverlapped:
            self.boxes[box.node_id] = box
            if box.node_id >= self._next_id:
                self._next_id = box.node_id + 1

        # 全量邻接重算（向量化，快速）
        self.adjacency = compute_adjacency(
            all_deoverlapped, tol=self.config.adjacency_tolerance)

        # 返回新增的 box（不在 existing_list id 中的）
        old_ids = {b.node_id for b in existing_list}
        added = [b for b in all_deoverlapped if b.node_id not in old_ids]

        logger.info(
            "BoxForest.add_boxes: %d new input → %d deoverlapped total "
            "(%d added), %d adjacency edges",
            len(new_boxes), len(all_deoverlapped), len(added),
            sum(len(v) for v in self.adjacency.values()) // 2,
        )
        return added

    def add_boxes_incremental(
        self,
        new_boxes: List[BoxNode],
        min_frag_vol: Optional[float] = None,
    ) -> List[BoxNode]:
        """增量添加新 box（仅对新 box 做 deoverlap）

        更快的版本：只让新 box 被已有 box 切分（已有 box 不变），
        然后增量更新邻接。适用于每次添加少量 box 的场景。

        Args:
            new_boxes: 新扩展的 BoxNode 列表
            min_frag_vol: 碎片最小体积阈值

        Returns:
            实际添加的 box 列表
        """
        if not new_boxes:
            return []

        if min_frag_vol is None:
            min_frag_vol = self.config.min_fragment_volume

        existing_list = list(self.boxes.values())

        # 仅对新 box 做切分（已有 box 作为 committed 不变）
        added: List[BoxNode] = []
        for box in new_boxes:
            fragments = [list(box.joint_intervals)]
            for committed_box in existing_list:
                from .deoverlap import subtract_box, _overlap_volume_intervals
                new_frags = []
                for frag in fragments:
                    ovlp = _overlap_volume_intervals(
                        frag, list(committed_box.joint_intervals))
                    if ovlp < min_frag_vol:
                        new_frags.append(frag)
                    else:
                        pieces = subtract_box(
                            frag, list(committed_box.joint_intervals))
                        new_frags.extend(pieces)
                fragments = new_frags

            for frag in fragments:
                from .deoverlap import _interval_volume
                vol = _interval_volume(frag)
                if vol < min_frag_vol:
                    continue
                nid = self.allocate_id()
                new_node = BoxNode(
                    node_id=nid,
                    joint_intervals=frag,
                    seed_config=box.seed_config.copy(),
                    parent_id=box.node_id,
                    volume=vol,
                    tree_id=box.tree_id,
                )
                self.boxes[nid] = new_node
                self.adjacency[nid] = set()
                added.append(new_node)

        # 增量邻接更新：O(K·N·D)
        if added:
            all_boxes_list = list(self.boxes.values())
            compute_adjacency_incremental(
                added, all_boxes_list, self.adjacency,
                tol=self.config.adjacency_tolerance,
            )

        return added

    def remove_boxes(self, box_ids: Set[int]) -> None:
        """移除指定 box 及其邻接边"""
        for bid in box_ids:
            if bid in self.boxes:
                del self.boxes[bid]
            if bid in self.adjacency:
                neighbors = self.adjacency.pop(bid)
                for nb in neighbors:
                    if nb in self.adjacency:
                        self.adjacency[nb].discard(bid)

    def find_containing(self, config: np.ndarray) -> Optional[BoxNode]:
        """找到包含 config 的 box"""
        for box in self.boxes.values():
            if box.contains(config):
                return box
        return None

    def find_nearest(self, config: np.ndarray) -> Optional[BoxNode]:
        """找到离 config 最近的 box"""
        best_box = None
        best_dist = float('inf')
        for box in self.boxes.values():
            d = box.distance_to_config(config)
            if d < best_dist:
                best_dist = d
                best_box = box
        return best_box

    def get_uncovered_seeds(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        """采样不在任何 box 内的 seed 点

        Args:
            n: 期望采样数
            rng: 随机数生成器

        Returns:
            无覆盖种子列表（最多 n 个）
        """
        seeds: List[np.ndarray] = []
        max_attempts = n * 10

        for _ in range(max_attempts):
            if len(seeds) >= n:
                break
            q = np.array([
                rng.uniform(lo, hi) for lo, hi in self.joint_limits
            ])
            if self.find_containing(q) is None:
                seeds.append(q)

        return seeds

    def validate_boxes(
        self,
        collision_checker: CollisionChecker,
    ) -> Set[int]:
        """惰性碰撞验证：检查每个 box 在当前场景是否安全

        利用 AABB 缓存避免重复 FK 计算。

        Args:
            collision_checker: 当前场景的碰撞检测器

        Returns:
            碰撞的 box ID 集合
        """
        colliding_ids: Set[int] = set()
        for box in self.boxes.values():
            if collision_checker.check_box_collision(
                box.joint_intervals, skip_merge=True
            ):
                colliding_ids.add(box.node_id)

        if colliding_ids:
            logger.info(
                "BoxForest.validate_boxes: %d/%d boxes collide in current scene",
                len(colliding_ids), len(self.boxes),
            )
        return colliding_ids

    # ── 构建 ──

    @classmethod
    def build(
        cls,
        robot: Robot,
        scene: Scene,
        config: Optional[PlannerConfig] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        aabb_cache: Optional[AABBCache] = None,
    ) -> 'BoxForest':
        """构建 Box 森林

        均匀采样 seed，扩展 box，deoverlap，构建邻接图。

        Args:
            robot: 机器人模型
            scene: 障碍物场景
            config: 规划参数
            joint_limits: 关节限制
            seed: 随机数种子
            aabb_cache: AABB 包络缓存

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

        # AABB 缓存
        if config.use_aabb_cache:
            if aabb_cache is not None:
                _cache = aabb_cache
            else:
                _cache = AABBCache.auto_load(robot)
        else:
            _cache = None
        _owns_cache = (aabb_cache is None and _cache is not None)

        checker = CollisionChecker(robot=robot, scene=scene, aabb_cache=_cache)

        # 自动采样模式
        if config.use_sampling is not None:
            _use_sampling = config.use_sampling
        else:
            _use_sampling = robot.n_joints > 4

        expander = BoxExpander(
            robot=robot,
            collision_checker=checker,
            joint_limits=joint_limits,
            expansion_resolution=config.expansion_resolution,
            max_rounds=config.max_expansion_rounds,
            jacobian_delta=config.jacobian_delta,
            use_sampling=_use_sampling,
            sampling_n=config.sampling_n,
            min_initial_half_width=config.min_initial_half_width,
            strategy=config.expansion_strategy,
            balanced_step_fraction=config.balanced_step_fraction,
            balanced_max_steps=config.balanced_max_steps,
            overlap_weight=config.overlap_weight,
            hard_overlap_reject=config.hard_overlap_reject,
        )

        forest = cls(robot.fingerprint(), joint_limits, config)

        # 采样扩展
        raw_boxes: List[BoxNode] = []
        n_seeds = config.build_n_seeds
        max_box_nodes = config.max_box_nodes * 3

        for i in range(n_seeds):
            if len(raw_boxes) >= max_box_nodes:
                break

            q_seed = np.array([
                rng.uniform(lo, hi) for lo, hi in joint_limits
            ])
            if checker.check_config_collision(q_seed):
                continue

            # 跳过已被覆盖的 seed
            already_covered = any(b.contains(q_seed) for b in raw_boxes)
            if already_covered:
                continue

            nid = forest.allocate_id()
            box = expander.expand(
                q_seed, node_id=nid, rng=rng,
                existing_boxes=raw_boxes,
            )
            if box is None or box.volume < config.min_box_volume:
                continue

            raw_boxes.append(box)

            if config.verbose and (i + 1) % 50 == 0:
                logger.info(
                    "Forest build: %d/%d seeds, %d raw boxes",
                    i + 1, n_seeds, len(raw_boxes),
                )

        # deoverlap + 邻接
        forest.add_boxes(raw_boxes)

        forest.build_time = time.time() - t0
        logger.info(
            "BoxForest 构建完成: %d 个无重叠 box, 总体积 %.4f, "
            "%d 条邻接边, 耗时 %.2fs",
            forest.n_boxes, forest.total_volume,
            sum(len(v) for v in forest.adjacency.values()) // 2,
            forest.build_time,
        )

        # 自动保存缓存
        if _owns_cache and _cache is not None:
            try:
                _cache.auto_save(robot)
            except Exception as e:
                logger.warning("AABB 缓存自动保存失败: %s", e)

        return forest

    # ── 持久化 ──

    def save(self, filepath: str) -> None:
        """保存到文件（pickle）"""
        data = {
            'robot_fingerprint': self.robot_fingerprint,
            'boxes': self.boxes,
            'adjacency': self.adjacency,
            'joint_limits': self.joint_limits,
            'config': self.config,
            'build_time': self.build_time,
            '_next_id': self._next_id,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BoxForest 已保存到 %s (%d boxes)", filepath, self.n_boxes)

    @classmethod
    def load(
        cls,
        filepath: str,
        robot: Robot,
    ) -> 'BoxForest':
        """从文件加载

        不绑定场景 —— 加载后需调用 validate_boxes() 做惰性碰撞验证。

        Args:
            filepath: 森林文件路径
            robot: 机器人模型（需与构建时一致）

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
                f"当前为 {robot.fingerprint()[:16]}...",
            )

        forest = cls(
            robot_fingerprint=data['robot_fingerprint'],
            joint_limits=data['joint_limits'],
            config=data.get('config', PlannerConfig()),
        )
        forest.boxes = data['boxes']
        forest.adjacency = data['adjacency']
        forest.build_time = data.get('build_time', 0.0)
        forest._next_id = data.get('_next_id', 0)

        # 确保 _next_id 不与已有 box 冲突
        if forest.boxes:
            max_existing = max(forest.boxes.keys())
            if forest._next_id <= max_existing:
                forest._next_id = max_existing + 1

        logger.info(
            "BoxForest 从 %s 加载: %d 个 box, %d 条邻接边",
            filepath, forest.n_boxes,
            sum(len(v) for v in forest.adjacency.values()) // 2,
        )
        return forest
