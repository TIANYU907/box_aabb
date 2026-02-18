"""
planner/box_forest.py - 扁平无重叠 Box 森林

在 C-free 空间中维护一组互不重叠（容忍微小角落重叠）的 axis-aligned
hyperrectangle（BoxNode），以及它们之间的邻接关系。

核心特性：
- **零重叠不变量**：HierAABBTree 保证所有 box 之间无重叠
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
try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from aabb.robot import Robot
from .models import PlannerConfig, BoxNode
from .scene import Scene
from .collision import CollisionChecker
from .deoverlap import (
    deoverlap,
    compute_adjacency,
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
        self.hier_tree = None   # Optional[HierAABBTree] 用于 O(depth) 空间查询
        self._kdtree = None
        self._kdtree_ids: List[int] = []
        self._kdtree_dirty: bool = True
        self._intervals_arr = np.empty((0, 0, 2), dtype=np.float64)
        self._interval_ids: List[int] = []
        self._interval_id_to_index: Dict[int, int] = {}

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
    ) -> List[BoxNode]:
        """将新 box 集合并入森林

        对新 box 与已有 box 执行 deoverlap（先来先得：已有 box 优先），
        然后增量更新邻接关系。

        Args:
            new_boxes: 新扩展的 BoxNode 列表

        Returns:
            实际添加的 box 列表（去重叠后的碎片）
        """
        if not new_boxes:
            return []

        # 合并列表：已有 box 在前（优先），新 box 在后
        existing_list = list(self.boxes.values())
        all_input = existing_list + new_boxes

        # 执行全量 deoverlap（当前 HierAABBTree 保证不重叠，仅安全网）
        all_deoverlapped = deoverlap(
            all_input,
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
        self._rebuild_interval_cache()
        self._kdtree_dirty = True

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
    ) -> List[BoxNode]:
        """增量添加新 box（仅对新 box 做 deoverlap）

        更快的版本：只让新 box 被已有 box 切分（已有 box 不变），
        然后增量更新邻接。适用于每次添加少量 box 的场景。

        注意：当前 HierAABBTree 保证不重叠，此方法仅作为安全网。

        Args:
            new_boxes: 新扩展的 BoxNode 列表

        Returns:
            实际添加的 box 列表
        """
        if not new_boxes:
            return []

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
                    if ovlp <= 0:
                        new_frags.append(frag)
                    else:
                        pieces = subtract_box(
                            frag, list(committed_box.joint_intervals))
                        new_frags.extend(pieces)
                fragments = new_frags

            for frag in fragments:
                from .deoverlap import _interval_volume
                vol = _interval_volume(frag)
                if vol <= 0:
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
                self.add_box_direct(new_node)
                added.append(new_node)

        return added

    def add_box_direct(self, box: BoxNode) -> None:
        """直接添加 box（跳过 deoverlap）

        调用方须保证 box 与已有 box 无重叠（如通过
        HierAABBTree 的占用跟踪）。仅执行增量邻接更新。
        """
        neighbor_ids = self._adjacent_existing_ids_from_cache(
            box,
            tol=self.config.adjacency_tolerance,
        )

        self.boxes[box.node_id] = box
        self.adjacency[box.node_id] = set(neighbor_ids)
        for nb in neighbor_ids:
            self.adjacency.setdefault(nb, set()).add(box.node_id)

        self._append_interval_cache(box)

        if box.node_id >= self._next_id:
            self._next_id = box.node_id + 1
        self._kdtree_dirty = True

    def remove_boxes(self, box_ids: Set[int]) -> None:
        """移除指定 box 及其邻接边"""
        for bid in box_ids:
            if bid in self.boxes:
                del self.boxes[bid]
                self._remove_interval_cache(bid)
            if bid in self.adjacency:
                neighbors = self.adjacency.pop(bid)
                for nb in neighbors:
                    if nb in self.adjacency:
                        self.adjacency[nb].discard(bid)
        self._kdtree_dirty = True

    @staticmethod
    def _box_intervals_array(box: BoxNode) -> np.ndarray:
        ivs = np.asarray(box.joint_intervals, dtype=np.float64)
        if ivs.ndim != 2 or ivs.shape[1] != 2:
            raise ValueError("box.joint_intervals 必须是 (D,2)")
        return ivs

    def _rebuild_interval_cache(self) -> None:
        self._interval_ids = list(self.boxes.keys())
        if not self._interval_ids:
            self._intervals_arr = np.empty((0, 0, 2), dtype=np.float64)
            self._interval_id_to_index = {}
            return

        self._intervals_arr = np.stack(
            [self._box_intervals_array(self.boxes[bid]) for bid in self._interval_ids],
            axis=0,
        )
        self._interval_id_to_index = {
            bid: i for i, bid in enumerate(self._interval_ids)
        }

    def _append_interval_cache(self, box: BoxNode) -> None:
        ivs = self._box_intervals_array(box)
        if self._intervals_arr.size == 0:
            self._intervals_arr = ivs[None, :, :]
            self._interval_ids = [box.node_id]
            self._interval_id_to_index = {box.node_id: 0}
            return

        self._intervals_arr = np.concatenate([self._intervals_arr, ivs[None, :, :]], axis=0)
        self._interval_ids.append(box.node_id)
        self._interval_id_to_index[box.node_id] = len(self._interval_ids) - 1

    def _remove_interval_cache(self, box_id: int) -> None:
        idx = self._interval_id_to_index.get(box_id)
        if idx is None:
            return

        self._intervals_arr = np.delete(self._intervals_arr, idx, axis=0)
        self._interval_ids.pop(idx)
        self._interval_id_to_index = {
            bid: i for i, bid in enumerate(self._interval_ids)
        }

    def _adjacent_existing_ids_from_cache(
        self,
        box: BoxNode,
        tol: float,
    ) -> List[int]:
        if not self._interval_ids:
            return []

        ivs = self._box_intervals_array(box)  # (D,2)
        lo_all = self._intervals_arr[:, :, 0]  # (N,D)
        hi_all = self._intervals_arr[:, :, 1]  # (N,D)
        lo_new = ivs[:, 0][None, :]            # (1,D)
        hi_new = ivs[:, 1][None, :]            # (1,D)

        overlap_width = np.minimum(hi_all, hi_new) - np.maximum(lo_all, lo_new)

        separated = overlap_width < -tol
        touching = (overlap_width >= -tol) & (overlap_width <= tol)
        overlapping = overlap_width > tol

        any_separated = np.any(separated, axis=1)
        n_touching = np.sum(touching, axis=1)
        n_overlapping = np.sum(overlapping, axis=1)
        n_dims = ivs.shape[0]

        is_adjacent = (~any_separated) & (n_touching >= 1) & (n_overlapping >= n_dims - 1)

        return [
            self._interval_ids[i]
            for i in np.flatnonzero(is_adjacent)
            if self._interval_ids[i] != box.node_id
        ]

    def _rebuild_kdtree(self) -> None:
        if cKDTree is None or not self.boxes:
            self._kdtree = None
            self._kdtree_ids = []
            self._kdtree_dirty = False
            return
        self._kdtree_ids = list(self.boxes.keys())
        centers = np.array([self.boxes[bid].center for bid in self._kdtree_ids])
        self._kdtree = cKDTree(centers)
        self._kdtree_dirty = False

    def find_containing(self, config: np.ndarray) -> Optional[BoxNode]:
        """找到包含 config 的 box

        若已设置 hier_tree，利用 HierAABBTree 的 O(depth) 查询加速；
        否则回退到 O(N) 线性扫描。
        """
        if self.hier_tree is not None:
            box_id = self.hier_tree.find_containing_box_id(config)
            if box_id is not None and box_id in self.boxes:
                return self.boxes[box_id]
            return None
        for box in self.boxes.values():
            if box.contains(config):
                return box
        return None

    def find_nearest(self, config: np.ndarray) -> Optional[BoxNode]:
        """找到离 config 最近的 box"""
        if self._kdtree_dirty:
            self._rebuild_kdtree()

        if self._kdtree is not None and self._kdtree_ids:
            _, idx = self._kdtree.query(config)
            bid = self._kdtree_ids[int(idx)]
            return self.boxes.get(bid)

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
                box.joint_intervals
            ):
                colliding_ids.add(box.node_id)

        if colliding_ids:
            logger.info(
                "BoxForest.validate_boxes: %d/%d boxes collide in current scene",
                len(colliding_ids), len(self.boxes),
            )
        return colliding_ids

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
        forest._rebuild_interval_cache()

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
