"""
planner/box_forest.py - 扁平无重叠 Box 森林

在 C-free 空间中维护一组互不重叠（容忍微小角落重叠）的 axis-aligned
hyperrectangle（BoxNode），以及它们之间的邻接关系。

核心特性：
- **零重叠不变量**：HierAABBTree 保证所有 box 之间无重叠
- **扁平邻接图**：无树层级，只有 box 和邻接边
- **跨场景复用**：森林绑定机器人型号而非场景，不同场景加载后
  惰性验证碰撞（AABB 缓存避免重复 FK）
- **增量增密**：每次规划可添加新 box，自动邻接更新

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
from .deoverlap import compute_adjacency

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
        period: Optional[float] = None,
    ) -> None:
        self.robot_fingerprint = robot_fingerprint
        self.joint_limits = joint_limits
        self.config = config or PlannerConfig()
        self.period = period  # 关节空间周期（例如 2π），用于邻接检测
        self.boxes: Dict[int, BoxNode] = {}
        self.adjacency: Dict[int, Set[int]] = {}
        self._next_id: int = 0
        self.build_time: float = 0.0
        self.hier_tree = None   # Optional[HierAABBTree] 用于 O(depth) 空间查询
        self._kdtree = None
        self._kdtree_ids: List[int] = []
        self._kdtree_dirty: bool = True
        self._intervals_arr = np.empty((0, 0, 2), dtype=np.float64)
        self._intervals_len: int = 0           # 已使用的行数
        self._intervals_cap: int = 0           # 预分配容量
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

    def add_box_direct(self, box: BoxNode) -> None:
        """添加 box 并增量更新邻接

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

    def add_box_no_adjacency(self, box: BoxNode) -> None:
        """添加 box, 仅更新 boxes dict + interval cache, 跳过邻接计算.

        用于 coarsen 批量合并期间 — 合并完成后调用
        rebuild_adjacency() 一次性重建邻接.
        """
        self.boxes[box.node_id] = box
        self._append_interval_cache(box)
        if box.node_id >= self._next_id:
            self._next_id = box.node_id + 1
        self._kdtree_dirty = True

    def remove_boxes_no_adjacency(self, box_ids: Set[int]) -> None:
        """移除指定 box, 仅更新 boxes dict + interval cache, 跳过邻接清理.

        用于 coarsen 批量合并期间 — 合并完成后调用
        rebuild_adjacency() 一次性重建邻接.
        """
        for bid in box_ids:
            if bid in self.boxes:
                del self.boxes[bid]
                self._remove_interval_cache(bid)
        self._kdtree_dirty = True

    def rebuild_adjacency(self) -> None:
        """从 interval cache 一次性重建全部邻接关系.

        使用向量化 _adjacent_existing_ids_from_cache 逻辑,
        比逐个 add_box_direct 的增量更新更高效 (一次 O(N²) 向量化).
        """
        from .deoverlap import compute_adjacency
        box_list = list(self.boxes.values())
        self.adjacency = compute_adjacency(
            box_list, tol=self.config.adjacency_tolerance)

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
            self._intervals_len = 0
            self._intervals_cap = 0
            self._interval_id_to_index = {}
            return

        stacked = np.stack(
            [self._box_intervals_array(self.boxes[bid]) for bid in self._interval_ids],
            axis=0,
        )
        n = stacked.shape[0]
        n_dims = stacked.shape[1]
        cap = max(64, n)
        self._intervals_arr = np.empty((cap, n_dims, 2), dtype=np.float64)
        self._intervals_arr[:n] = stacked
        self._intervals_len = n
        self._intervals_cap = cap
        self._interval_id_to_index = {
            bid: i for i, bid in enumerate(self._interval_ids)
        }

    def _append_interval_cache(self, box: BoxNode) -> None:
        ivs = self._box_intervals_array(box)          # (D, 2)
        n_dims = ivs.shape[0]

        if self._intervals_len == 0:
            # 首次插入: 预分配 64 行
            cap = 64
            self._intervals_arr = np.empty((cap, n_dims, 2), dtype=np.float64)
            self._intervals_cap = cap
            self._intervals_len = 0

        # 容量不足时 2× 扩容
        if self._intervals_len >= self._intervals_cap:
            new_cap = max(64, self._intervals_cap * 2)
            new_arr = np.empty((new_cap, n_dims, 2), dtype=np.float64)
            new_arr[:self._intervals_len] = self._intervals_arr[:self._intervals_len]
            self._intervals_arr = new_arr
            self._intervals_cap = new_cap

        idx = self._intervals_len
        self._intervals_arr[idx] = ivs
        self._intervals_len += 1
        self._interval_ids.append(box.node_id)
        self._interval_id_to_index[box.node_id] = idx

    def _remove_interval_cache(self, box_id: int) -> None:
        idx = self._interval_id_to_index.get(box_id)
        if idx is None:
            return

        last = self._intervals_len - 1
        if idx != last:
            # swap-with-last: O(1)
            self._intervals_arr[idx] = self._intervals_arr[last]
            moved_id = self._interval_ids[last]
            self._interval_ids[idx] = moved_id
            self._interval_id_to_index[moved_id] = idx

        self._intervals_len -= 1
        self._interval_ids.pop()            # 移除尾部
        del self._interval_id_to_index[box_id]

    def _adjacent_existing_ids_from_cache(
        self,
        box: BoxNode,
        tol: float,
    ) -> List[int]:
        if self._intervals_len == 0:
            return []

        ivs = self._box_intervals_array(box)  # (D,2)
        n = self._intervals_len
        lo_all = self._intervals_arr[:n, :, 0]  # (N,D)
        hi_all = self._intervals_arr[:n, :, 1]  # (N,D)
        lo_new = ivs[:, 0][None, :]            # (1,D)
        hi_new = ivs[:, 1][None, :]            # (1,D)

        # 直接 overlap width
        overlap_width = np.minimum(hi_all, hi_new) - np.maximum(lo_all, lo_new)

        # 周期边界支持：取直接 / 左移 / 右移中的最大 overlap_width
        if self.period is not None:
            p = self.period
            ow_right = np.minimum(hi_all, hi_new + p) - np.maximum(lo_all, lo_new + p)
            ow_left  = np.minimum(hi_all, hi_new - p) - np.maximum(lo_all, lo_new - p)
            overlap_width = np.maximum(overlap_width, np.maximum(ow_right, ow_left))

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

    def invalidate_against_obstacle(
        self,
        obstacle,
        robot: Robot,
        safety_margin: float = 0.0,
    ) -> Set[int]:
        """检测与单个障碍物碰撞的 box (增量式场景变化检测).

        构建仅含该障碍物的临时场景, 对每个 box 做 interval FK AABB
        碰撞检测. 复杂度 O(N × links × 1) — 远低于全量 validate.

        Args:
            obstacle: 要检测的障碍物 (Obstacle 实例或含有 min_point/max_point)
            robot: 机器人实例
            safety_margin: 安全边距

        Returns:
            与该障碍物碰撞的 box ID 集合
        """
        temp_scene = Scene()
        temp_scene.add_obstacle(
            obstacle.min_point, obstacle.max_point,
            name=getattr(obstacle, 'name', 'delta'),
        )
        temp_checker = CollisionChecker(
            robot=robot, scene=temp_scene, safety_margin=safety_margin,
        )
        colliding_ids: Set[int] = set()
        for box in self.boxes.values():
            if temp_checker.check_box_collision(box.joint_intervals):
                colliding_ids.add(box.node_id)
        if colliding_ids:
            logger.info(
                "invalidate_against_obstacle: %d/%d boxes collide with '%s'",
                len(colliding_ids), len(self.boxes),
                getattr(obstacle, 'name', '?'),
            )
        return colliding_ids

    def remove_invalidated(
        self,
        box_ids: Set[int],
    ) -> List[BoxNode]:
        """清除失效 box: 从 forest (dict+adjacency) 和 hier_tree (占用) 中移除.

        Args:
            box_ids: 需要移除的 box ID 集合

        Returns:
            被移除的 BoxNode 列表 (供调用方收集 seed 补种)
        """
        if not box_ids:
            return []
        # 保存被删 box 的副本 (补种用)
        removed: List[BoxNode] = [
            self.boxes[bid] for bid in box_ids if bid in self.boxes
        ]
        # 清除 forest dict + adjacency + interval cache
        self.remove_boxes(box_ids)
        # 清除 hier_tree 占用 (若有挂接的 tree)
        if self.hier_tree is not None:
            self.hier_tree.unoccupy_boxes(box_ids)
        return removed

    def merge_partition_forests(
        self,
        local_forests: List[Dict],
        dedup_rule: str = "partition_id",
    ) -> Dict[int, Set[int]]:
        """将分区 worker 的局部结果合并到当前 forest。

        Args:
            local_forests: [{'partition_id': int, 'boxes': [{'joint_intervals', 'seed_config', 'volume'}, ...]}, ...]
            dedup_rule: 边界去重规则，当前支持 'partition_id'

        Returns:
            分区到全局 box id 的映射 {partition_id: {box_ids...}}
        """
        partition_box_ids: Dict[int, Set[int]] = {}

        for item in sorted(local_forests, key=lambda x: int(x.get('partition_id', 0))):
            pid = int(item.get('partition_id', -1))
            boxes = item.get('boxes', [])
            partition_box_ids.setdefault(pid, set())
            for box_data in boxes:
                ivs = box_data.get('joint_intervals')
                seed = box_data.get('seed_config')
                vol = float(box_data.get('volume', 0.0))
                if ivs is None or seed is None:
                    continue
                nid = self.allocate_id()
                node = BoxNode(
                    node_id=nid,
                    joint_intervals=ivs,
                    seed_config=np.asarray(seed, dtype=np.float64),
                    volume=vol,
                    tree_id=pid,
                )
                self.add_box_direct(node)
                partition_box_ids[pid].add(nid)

        if dedup_rule == "partition_id":
            self.dedup_boundary_boxes(partition_box_ids)

        return partition_box_ids

    def dedup_boundary_boxes(
        self,
        partition_box_ids: Dict[int, Set[int]],
        tol: float = 1e-10,
    ) -> int:
        """按分区优先级去重边界重复 box（低 partition_id 优先）。"""
        if not partition_box_ids:
            return 0

        all_ids = sorted({bid for ids in partition_box_ids.values() for bid in ids})
        to_remove: Set[int] = set()

        for i in range(len(all_ids)):
            a_id = all_ids[i]
            if a_id in to_remove or a_id not in self.boxes:
                continue
            a = self.boxes[a_id]
            a_pid = int(a.tree_id)
            a_arr = np.asarray(a.joint_intervals, dtype=np.float64)
            for j in range(i + 1, len(all_ids)):
                b_id = all_ids[j]
                if b_id in to_remove or b_id not in self.boxes:
                    continue
                b = self.boxes[b_id]
                b_pid = int(b.tree_id)
                if a_pid == b_pid:
                    continue
                b_arr = np.asarray(b.joint_intervals, dtype=np.float64)
                if a_arr.shape != b_arr.shape:
                    continue
                if np.all(np.abs(a_arr - b_arr) <= tol):
                    # 小分区号优先保留
                    if a_pid <= b_pid:
                        to_remove.add(b_id)
                    else:
                        to_remove.add(a_id)
                        break

        if to_remove:
            self.remove_boxes(to_remove)
            for ids in partition_box_ids.values():
                ids.difference_update(to_remove)
        return len(to_remove)

    def validate_invariants(
        self,
        tol: float = 1e-8,
        strict: bool = True,
    ) -> None:
        """校验 forest 关键不变量：邻接对称、引用有效、无正体积重叠。"""
        def _handle(msg: str) -> None:
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        # 1) 邻接引用有效 + 对称
        for bid, neighbors in list(self.adjacency.items()):
            if bid not in self.boxes:
                _handle(f"adjacency 包含不存在 box: {bid}")
                continue
            for nb in list(neighbors):
                if nb not in self.boxes:
                    _handle(f"邻接引用不存在 box: {bid}->{nb}")
                    continue
                if bid not in self.adjacency.get(nb, set()):
                    _handle(f"邻接非对称: {bid}<->{nb}")

        # 2) 无正体积重叠
        box_ids = list(self.boxes.keys())
        for i in range(len(box_ids)):
            a = self.boxes[box_ids[i]]
            a_ivs = np.asarray(a.joint_intervals, dtype=np.float64)
            for j in range(i + 1, len(box_ids)):
                b = self.boxes[box_ids[j]]
                b_ivs = np.asarray(b.joint_intervals, dtype=np.float64)
                overlap_width = np.minimum(a_ivs[:, 1], b_ivs[:, 1]) - np.maximum(a_ivs[:, 0], b_ivs[:, 0])
                if np.all(overlap_width > tol):
                    _handle(f"检测到正体积重叠 box: {a.node_id} 与 {b.node_id}")

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
