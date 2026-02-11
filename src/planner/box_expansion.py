"""
planner/box_expansion.py - 启发式 Box 拓展算法

从一个无碰撞 seed 配置出发，在关节空间中拓展出尽可能大的
无碰撞 box（超矩形区间）。

拓展策略：

'greedy' (旧版):
  1. 在 seed 处计算 Jacobian 范数，按从小到大排序维度
  2. 对排序后的每个维度执行二分搜索，分别向正/负方向找碰撞边界
  3. 迭代优化：拓展完一轮后可再做一轮

'balanced' (新版, 默认):
  1. Jacobian 分析同上，但不再一次性贪心拓展单维度到极限
  2. 维护 2n 个候选方向（每个维度 +/-），每轮对每个候选尝试
     一个自适应步长（剩余空间 × step_fraction），取体积增益最大者执行
  3. 防止任一维度独占区间宽度预算，减少区间 FK 过估计导致的细长 box
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from box_aabb.robot import Robot
from .collision import CollisionChecker
from .models import BoxNode

logger = logging.getLogger(__name__)


@dataclass
class DimSearchLog:
    """记录单次方向二分搜索的详情"""
    dim: int
    direction: str              # '+' or '-'
    start_bound: float          # 搜索起始边界
    limit: float                # 关节极限
    result_bound: float         # 最终安全边界
    expansion_amount: float     # 实际拓展量 (绝对值)
    reached_limit: bool         # 是否直接到达关节极限
    n_bisect_steps: int         # 二分搜索步数 (0 = 一步到位或跳过)
    stop_reason: str            # 'reached_limit' / 'resolution_converged' / 'max_iterations'


@dataclass
class BalancedStepLog:
    """记录 balanced 策略中单步选择的详情"""
    step_index: int
    dim: int
    direction: str              # '+' or '-'
    tried_step: float           # 尝试的步长 (绝对值)
    actual_step: float          # 实际安全步长 (二分后)
    volume_before: float
    volume_after: float
    volume_gain: float
    chosen_reason: str          # 'best_gain' / 'only_candidate'
    n_bisect_steps: int
    stop_reason: str            # 同 DimSearchLog
    overlap_increase: float = 0.0   # 与已有 box 的重叠增量
    novel_gain: float = 0.0         # 净新增体积 = volume_gain - overlap_weight * overlap_increase


@dataclass
class RoundLog:
    """记录一轮拓展的详情"""
    round_index: int
    volume_before: float
    volume_after: float
    dim_searches: List[DimSearchLog] = field(default_factory=list)


@dataclass
class ExpansionLog:
    """记录一次完整 box 拓展过程的详情"""
    seed: Optional[np.ndarray] = None
    node_id: int = -1
    strategy: str = 'greedy'
    # Jacobian 分析
    jacobian_norms: Optional[Dict[int, float]] = None
    dim_order: Optional[List[int]] = None
    dim_order_reason: str = ''
    # greedy 策略各轮拓展
    rounds: List[RoundLog] = field(default_factory=list)
    # balanced 策略各步
    balanced_steps: List[BalancedStepLog] = field(default_factory=list)
    # 最终结果
    final_intervals: Optional[List[Tuple[float, float]]] = None
    final_volume: float = 0.0
    total_rounds: int = 0
    total_steps: int = 0
    early_stop: bool = False
    early_stop_reason: str = ''

    def to_text(self, box_index: int = -1) -> str:
        """生成人类可读的文本报告"""
        lines = []
        header = f"=== Box #{box_index}" if box_index >= 0 else "=== Box"
        lines.append(f"{header}  (node_id={self.node_id}, strategy={self.strategy}) ===")
        if self.seed is not None:
            seed_str = ', '.join(f'{v:.4f}' for v in self.seed)
            lines.append(f"  Seed: [{seed_str}]")

        # Jacobian 分析
        if self.jacobian_norms:
            lines.append(f"  Jacobian analysis (delta for numerical diff):")
            for dim in (self.dim_order or sorted(self.jacobian_norms.keys())):
                norm = self.jacobian_norms[dim]
                lines.append(f"    dim {dim}: ||dP/dq{dim}|| = {norm:.4f}")
        if self.dim_order is not None:
            order_str = ' -> '.join(f'q{d}' for d in self.dim_order)
            lines.append(f"  Expansion order: {order_str}")
            lines.append(f"    Reason: {self.dim_order_reason}")

        # greedy 各轮
        for rnd in self.rounds:
            lines.append(f"  --- Round {rnd.round_index + 1} ---")
            lines.append(f"    Volume before: {rnd.volume_before:.6f}")
            for ds in rnd.dim_searches:
                dir_sym = '+' if ds.direction == '+' else '-'
                lines.append(
                    f"    dim q{ds.dim} ({dir_sym}): "
                    f"{ds.start_bound:.4f} -> {ds.result_bound:.4f}  "
                    f"(Δ={ds.expansion_amount:.4f}, "
                    f"steps={ds.n_bisect_steps}, "
                    f"stop: {ds.stop_reason})")
            lines.append(f"    Volume after:  {rnd.volume_after:.6f}")
            gain = rnd.volume_after - rnd.volume_before
            lines.append(f"    Volume gain:   {gain:.6f}")

        # balanced 各步
        if self.balanced_steps:
            lines.append(f"  --- Balanced expansion: {len(self.balanced_steps)} steps ---")
            for bs in self.balanced_steps:
                dir_sym = '+' if bs.direction == '+' else '-'
                ovlp_str = ''
                if bs.overlap_increase > 1e-10:
                    ovlp_str = f', ovlp_inc={bs.overlap_increase:.6f}, novel={bs.novel_gain:.6f}'
                lines.append(
                    f"    step {bs.step_index}: q{bs.dim}({dir_sym}) "
                    f"Δ={bs.actual_step:.4f} (tried {bs.tried_step:.4f}), "
                    f"vol {bs.volume_before:.6f} -> {bs.volume_after:.6f} "
                    f"(+{bs.volume_gain:.6f}{ovlp_str}), "
                    f"bisect={bs.n_bisect_steps}, "
                    f"reason: {bs.chosen_reason}, stop: {bs.stop_reason}")

        # 最终
        if self.strategy == 'greedy':
            lines.append(f"  Total rounds: {self.total_rounds}")
        else:
            lines.append(f"  Total steps: {self.total_steps}")
        if self.early_stop:
            lines.append(f"  Early stop: {self.early_stop_reason}")
        if self.final_intervals:
            for i, (lo, hi) in enumerate(self.final_intervals):
                lines.append(f"  Final q{i}: [{lo:.4f}, {hi:.4f}]  width={hi-lo:.4f}")
        lines.append(f"  Final volume: {self.final_volume:.6f}")
        lines.append('')
        return '\n'.join(lines)


class BoxExpander:
    """Box 拓展器

    从 seed 配置出发，启发式地拓展无碰撞 box。

    Args:
        robot: 机器人模型
        collision_checker: 碰撞检测器
        joint_limits: 关节限制列表
        expansion_resolution: 二分搜索精度 (rad)
        max_rounds: 最大迭代轮数 (greedy 策略用)
        jacobian_delta: Jacobian 数值差分步长
        min_initial_half_width: 初始半宽（seed 两侧的最小初始区间）
        strategy: 拓展策略 ('greedy' / 'balanced')
        balanced_step_fraction: balanced 策略的比例步长
        balanced_max_steps: balanced 策略的最大步数

    Example:
        >>> expander = BoxExpander(robot, checker, limits)
        >>> box = expander.expand(seed_config, node_id=0)
    """

    def __init__(
        self,
        robot: Robot,
        collision_checker: CollisionChecker,
        joint_limits: List[Tuple[float, float]],
        expansion_resolution: float = 0.01,
        max_rounds: int = 3,
        jacobian_delta: float = 0.01,
        min_initial_half_width: float = 0.001,
        use_sampling: bool = False,
        sampling_n: int = 80,
        strategy: str = 'balanced',
        balanced_step_fraction: float = 0.5,
        balanced_max_steps: int = 200,
        overlap_weight: float = 1.0,
        hard_overlap_reject: bool = True,
    ) -> None:
        self.robot = robot
        self.collision_checker = collision_checker
        self.joint_limits = joint_limits
        self.resolution = expansion_resolution
        self.max_rounds = max_rounds
        self.jacobian_delta = jacobian_delta
        self.min_initial_half_width = min_initial_half_width
        self._n_dims = len(joint_limits)
        self.use_sampling = use_sampling
        self.sampling_n = sampling_n
        self.strategy = strategy
        self.balanced_step_fraction = balanced_step_fraction
        self.balanced_max_steps = balanced_max_steps
        self.overlap_weight = overlap_weight
        self.hard_overlap_reject = hard_overlap_reject
        self._rng: Optional[np.random.Generator] = None
        self._current_log: Optional[ExpansionLog] = None
        self._existing_boxes: List[BoxNode] = []

    def expand(
        self,
        seed: np.ndarray,
        node_id: int = 0,
        tree_id: int = -1,
        rng: Optional[np.random.Generator] = None,
        enable_log: bool = False,
        existing_boxes: Optional[List[BoxNode]] = None,
    ) -> Optional[BoxNode]:
        """从 seed 配置拓展无碰撞 box

        Args:
            seed: 无碰撞 seed 配置 (n_joints,)
            node_id: 分配给该 box 节点的 ID
            tree_id: 所属树 ID
            rng: 随机数生成器（采样模式使用）
            enable_log: 是否启用详细拓展日志
            existing_boxes: 已有 box 列表（用于重叠惩罚，减少冗余重叠）

        Returns:
            BoxNode 实例，或 None（若 seed 本身就碰撞）
        """
        self._existing_boxes = existing_boxes or []
        self._rng = rng or np.random.default_rng()
        self._current_log = ExpansionLog(
            seed=seed.copy(), node_id=node_id,
            strategy=self.strategy,
        ) if enable_log else None

        if self.collision_checker.check_config_collision(seed):
            logger.debug("seed 配置碰撞，跳过: %s", seed)
            return None

        # 初始区间：以 seed 为中心、极小宽度
        intervals = []
        for i in range(self._n_dims):
            lo = max(self.joint_limits[i][0], seed[i] - self.min_initial_half_width)
            hi = min(self.joint_limits[i][1], seed[i] + self.min_initial_half_width)
            intervals.append((lo, hi))

        # 确认初始 box 无碰撞
        if self._check_collision(intervals):
            logger.debug("初始极小 box 碰撞（过估计），使用点区间")
            intervals = [(seed[i], seed[i]) for i in range(self._n_dims)]

        # 计算每个维度的探索优先级
        dim_order = self._compute_dimension_order(seed)

        # 按策略拓展
        if self.strategy == 'balanced':
            intervals = self._expand_balanced(seed, intervals, dim_order)
        else:
            intervals = self._expand_greedy(seed, intervals, dim_order)

        if self._current_log:
            self._current_log.final_intervals = list(intervals)
            self._current_log.final_volume = self._volume(intervals)

        box = BoxNode(
            node_id=node_id,
            joint_intervals=intervals,
            seed_config=seed.copy(),
            tree_id=tree_id,
        )
        self._existing_boxes = []  # 释放引用
        return box

    def get_last_log(self) -> Optional[ExpansionLog]:
        """获取最近一次 expand() 的详细日志（仅当 enable_log=True 时可用）"""
        return self._current_log

    # ── greedy 策略 ──────────────────────────────────────

    def _expand_greedy(
        self,
        seed: np.ndarray,
        intervals: List[Tuple[float, float]],
        dim_order: List[int],
    ) -> List[Tuple[float, float]]:
        """旧版贪心策略：逐维度一次性搜索到边界，多轮迭代"""
        prev_volume = self._volume(intervals)
        actual_rounds = 0

        for round_idx in range(self.max_rounds):
            round_log = RoundLog(
                round_index=round_idx, volume_before=prev_volume,
                volume_after=0.0,
            ) if self._current_log else None
            intervals = self._expand_one_round(seed, intervals, dim_order,
                                               round_log=round_log)
            new_volume = self._volume(intervals)
            actual_rounds = round_idx + 1

            if round_log is not None:
                round_log.volume_after = new_volume
                self._current_log.rounds.append(round_log)

            if new_volume <= prev_volume * 1.001:
                logger.debug("第 %d 轮拓展后体积未增长，停止", round_idx + 1)
                if self._current_log:
                    self._current_log.early_stop = True
                    self._current_log.early_stop_reason = (
                        f'volume not growing after round {round_idx + 1} '
                        f'({prev_volume:.6f} -> {new_volume:.6f}, gain < 0.1%)')
                break
            prev_volume = new_volume
            logger.debug("第 %d 轮拓展: 体积 = %.6f", round_idx + 1, new_volume)

        if self._current_log:
            self._current_log.total_rounds = actual_rounds
            if not self._current_log.early_stop and actual_rounds == self.max_rounds:
                self._current_log.early_stop = True
                self._current_log.early_stop_reason = (
                    f'reached max_rounds={self.max_rounds}')

        return intervals

    # ── balanced 策略 ─────────────────────────────────────

    def _evaluate_candidate(
        self,
        dim: int,
        direction: int,
        intervals: List[Tuple[float, float]],
        current_vol: float,
    ) -> Optional[tuple]:
        """评估单个候选方向的拓展结果

        Returns:
            评估结果元组 (dim, direction, new_bound, gain, step_size,
            actual_step, n_bisect, stop_reason, trial_vol, raw_gain,
            ovlp_increase) 或 None
        """
        lo_limit, hi_limit = self.joint_limits[dim]
        current_lo, current_hi = intervals[dim]
        frac = self.balanced_step_fraction

        if direction > 0:
            remaining = hi_limit - current_hi
            current_bound = current_hi
            limit = hi_limit
        else:
            remaining = current_lo - lo_limit
            current_bound = current_lo
            limit = lo_limit

        if remaining < self.resolution:
            return None

        # ── 边界截断：将已有 box 边界视为“虚拟障碍物”──
        clamp = limit  # 默认不截断
        if self.hard_overlap_reject and self._existing_boxes:
            clamp = self._clamp_to_existing(
                intervals, dim, direction, limit)
            if direction > 0:
                clamped_remaining = clamp - current_hi
            else:
                clamped_remaining = current_lo - clamp
            if clamped_remaining < self.resolution:
                return None  # 该方向已被已有 box 堵死
            remaining = clamped_remaining
            limit = clamp

        # 自适应步长：剩余空间 × fraction
        step_size = remaining * frac
        if step_size < self.resolution:
            step_size = remaining  # 小于精度就尝试一步到位

        # 目标边界
        if direction > 0:
            target = min(current_hi + step_size, limit)
        else:
            target = max(current_lo - step_size, limit)

        # 测试目标边界是否安全
        test_intervals = list(intervals)
        if direction > 0:
            test_intervals[dim] = (current_lo, target)
        else:
            test_intervals[dim] = (target, current_hi)

        if not self._check_collision(test_intervals):
            # 整步安全
            new_bound = target
            n_bisect = 0
            stop_reason = ('reached_limit'
                           if abs(target - limit) < 1e-10
                           else 'step_safe')
        else:
            # 二分搜索在 [current_bound, target] 之间找安全边界
            safe = current_bound
            test = target
            n_bisect = 0
            for bi in range(30):
                if abs(test - safe) < self.resolution:
                    break
                mid = (safe + test) / 2.0
                test_intervals = list(intervals)
                if direction > 0:
                    test_intervals[dim] = (current_lo, mid)
                else:
                    test_intervals[dim] = (mid, current_hi)
                if self._check_collision(test_intervals):
                    test = mid
                else:
                    safe = mid
                n_bisect = bi + 1
            new_bound = safe
            stop_reason = 'resolution_converged'

        actual_step = abs(new_bound - current_bound)
        if actual_step < self.resolution * 0.5:
            return None  # 这个候选本步无法有效拓展

        # 计算体积增益（含重叠惩罚）
        trial_intervals = list(intervals)
        if direction > 0:
            trial_intervals[dim] = (intervals[dim][0], new_bound)
        else:
            trial_intervals[dim] = (new_bound, intervals[dim][1])
        trial_vol = self._volume(trial_intervals)
        raw_gain = trial_vol - current_vol

        # 计算与已有 box 的重叠增量
        if self._existing_boxes:
            if self.hard_overlap_reject:
                # 硬截断模式下重叠应该已被 clamp 避免，但仍可能有微小角落重叠
                ovlp_increase = 0.0
            elif self.overlap_weight > 0:
                overlap_before = self._compute_overlap_with_existing(intervals)
                overlap_after = self._compute_overlap_with_existing(trial_intervals)
                ovlp_increase = overlap_after - overlap_before
            else:
                ovlp_increase = 0.0
        else:
            ovlp_increase = 0.0

        # 净新增体积 = 原始增益 - 重叠权重 × 重叠增量
        novel_gain = raw_gain - self.overlap_weight * ovlp_increase
        gain = novel_gain

        return (
            dim, direction, new_bound, gain,
            step_size, actual_step, n_bisect, stop_reason,
            trial_vol, raw_gain, ovlp_increase,
        )

    def _expand_balanced(
        self,
        seed: np.ndarray,
        intervals: List[Tuple[float, float]],
        dim_order: List[int],
    ) -> List[Tuple[float, float]]:
        """平衡拓展策略：交替自适应步进，优化体积增益

        维护 2n 个候选方向（每个维度的 +/-），每步：
        1. 对每个活跃候选计算自适应步长 = 剩余空间 × step_fraction
        2. 通过二分搜索找到该步长内的安全边界
        3. 计算执行该步后的体积增益
        4. 选择体积增益最大的候选执行
        5. 移除已耗尽的候选（剩余空间 < resolution）

        优化：
        - 全评估与快速轮换交替：每 2n 步做一次全候选评估，
          其余步骤只评估上次最佳候选所在维度 + 1 个随机候选（减少碰撞检测）
        - 体积增长率早停：连续多步增长 <0.01% 则停止
        """
        intervals = list(intervals)

        # 候选池：(dim, direction)，direction = +1 or -1
        candidates = []
        for dim in range(self._n_dims):
            lo_limit, hi_limit = self.joint_limits[dim]
            if abs(hi_limit - lo_limit) < 1e-10:
                continue  # 固定关节
            candidates.append((dim, +1))
            candidates.append((dim, -1))

        step_count = 0
        consecutive_zero = 0
        vol_at_checkpoint = self._volume(intervals)
        checkpoint_step = 0
        full_eval_period = max(2 * self._n_dims, 4)

        for step_i in range(self.balanced_max_steps):
            if not candidates:
                break

            current_vol = self._volume(intervals)

            # 评估每个候选
            eval_results = []
            for dim, direction in candidates:
                result = self._evaluate_candidate(
                    dim, direction, intervals, current_vol)
                if result is not None:
                    eval_results.append(result)

            if not eval_results:
                # 所有候选都无法拓展
                if self._current_log:
                    self._current_log.early_stop = True
                    self._current_log.early_stop_reason = (
                        f'all candidates exhausted at step {step_i}')
                break

            # 选择体积增益最大的
            eval_results.sort(key=lambda x: x[3], reverse=True)
            (best_dim, best_dir, best_bound, best_gain,
             tried_step, actual_step, n_bisect, stop_reason,
             new_vol, best_raw_gain, best_ovlp_increase) = eval_results[0]

            if best_gain < 1e-12:
                consecutive_zero += 1
                if consecutive_zero >= 2 * self._n_dims:
                    if self._current_log:
                        self._current_log.early_stop = True
                        self._current_log.early_stop_reason = (
                            f'no volume gain for {consecutive_zero} steps')
                    break
            else:
                consecutive_zero = 0

            # ── 体积增长率早停 ──
            if step_count > 0 and (step_count - checkpoint_step) >= full_eval_period:
                if vol_at_checkpoint > 0:
                    growth_rate = (current_vol - vol_at_checkpoint) / vol_at_checkpoint
                    if growth_rate < 0.005:  # < 0.5% growth in last period
                        if self._current_log:
                            self._current_log.early_stop = True
                            self._current_log.early_stop_reason = (
                                f'marginal growth {growth_rate:.4%} over '
                                f'{full_eval_period} steps at step {step_i}')
                        break
                vol_at_checkpoint = current_vol
                checkpoint_step = step_count

            # 执行
            current_lo, current_hi = intervals[best_dim]
            if best_dir > 0:
                intervals[best_dim] = (current_lo, best_bound)
            else:
                intervals[best_dim] = (best_bound, current_hi)
            step_count += 1

            # 日志
            if self._current_log:
                chosen_reason = 'best_gain'
                if len(eval_results) == 1:
                    chosen_reason = 'only_candidate'
                self._current_log.balanced_steps.append(BalancedStepLog(
                    step_index=step_i,
                    dim=best_dim,
                    direction='+' if best_dir > 0 else '-',
                    tried_step=tried_step,
                    actual_step=actual_step,
                    volume_before=current_vol,
                    volume_after=new_vol,
                    volume_gain=best_raw_gain,
                    overlap_increase=best_ovlp_increase,
                    novel_gain=best_gain,
                    chosen_reason=chosen_reason,
                    n_bisect_steps=n_bisect,
                    stop_reason=stop_reason,
                ))

            # 更新候选池：移除已耗尽的
            new_candidates = []
            for dim, direction in candidates:
                lo_limit, hi_limit = self.joint_limits[dim]
                clo, chi = intervals[dim]
                if direction > 0:
                    remaining = hi_limit - chi
                else:
                    remaining = clo - lo_limit
                if remaining >= self.resolution:
                    new_candidates.append((dim, direction))
            candidates = new_candidates

        if self._current_log:
            self._current_log.total_steps = step_count
            if not self._current_log.early_stop and step_count >= self.balanced_max_steps:
                self._current_log.early_stop = True
                self._current_log.early_stop_reason = (
                    f'reached balanced_max_steps={self.balanced_max_steps}')

        return intervals

    def _compute_overlap_with_existing(
        self,
        trial_intervals: List[Tuple[float, float]],
    ) -> float:
        """计算 trial_intervals 与所有已有 box 的重叠体积之和

        Args:
            trial_intervals: 待评估的区间

        Returns:
            重叠体积总和
        """
        if not self._existing_boxes:
            return 0.0
        trial = BoxNode(
            node_id=-1,
            joint_intervals=trial_intervals,
            seed_config=np.zeros(self._n_dims),
        )
        total = 0.0
        for b in self._existing_boxes:
            total += trial.overlap_volume(b)
        return total

    def _clamp_to_existing(
        self,
        intervals: List[Tuple[float, float]],
        dim: int,
        direction: int,
        limit: float,
    ) -> float:
        """将扩展目标截断到最近的已有 box 边界

        沿 (dim, direction) 扩展时，找到该方向上所有与当前 box
        在其余维度有投影重叠的已有 box，取它们在 dim 维的最近边界。

        Args:
            intervals: 当前 box 区间
            dim: 扩展维度
            direction: 扩展方向 (+1 或 -1)
            limit: 原始关节限制边界

        Returns:
            截断后的目标边界（不超过 limit）
        """
        clamped = limit
        current_lo, current_hi = intervals[dim]

        for box in self._existing_boxes:
            # 检查其余维度是否有投影重叠
            has_overlap = True
            for d in range(self._n_dims):
                if d == dim:
                    continue
                b_lo, b_hi = box.joint_intervals[d]
                i_lo, i_hi = intervals[d]
                if i_hi <= b_lo or b_hi <= i_lo:
                    has_overlap = False
                    break

            if not has_overlap:
                continue

            # 该 box 在 dim 维的区间
            b_lo, b_hi = box.joint_intervals[dim]

            if direction > 0:
                # 向正方向扩展：已有 box 的 lo 边界是障碍
                # 仅考虑在当前 hi 正方向上的 box
                if b_lo > current_hi + self.resolution:
                    clamped = min(clamped, b_lo)
                elif b_lo <= current_hi and b_hi > current_hi:
                    # 已有 box 与当前位置已经部分重叠
                    # 不能再往这个方向扩展了
                    clamped = min(clamped, current_hi)
            else:
                # 向负方向扩展：已有 box 的 hi 边界是障碍
                if b_hi < current_lo - self.resolution:
                    clamped = max(clamped, b_hi)
                elif b_hi >= current_lo and b_lo < current_lo:
                    clamped = max(clamped, current_lo)

        return clamped

    def _compute_dimension_order(self, config: np.ndarray) -> List[int]:
        """计算拓展维度的优先级排序

        按 Jacobian 列向量范数从小到大排序。范数越小说明该关节变化
        对末端执行器位置的影响越小，可以更大胆地拓展。

        Args:
            config: 当前关节配置

        Returns:
            维度索引的排序列表（按优先级从高到低）
        """
        n_joints = self._n_dims
        delta = self.jacobian_delta

        # 基准末端位置
        base_pos = self.robot.get_link_positions(config)[-1]  # 末端

        jacobian_norms = []
        for i in range(n_joints):
            # 限制：固定关节（区间宽度为 0）直接给极大范数
            lo, hi = self.joint_limits[i]
            if abs(hi - lo) < 1e-10:
                jacobian_norms.append(float('inf'))
                continue

            q_plus = config.copy()
            q_plus[i] += delta
            pos_plus = self.robot.get_link_positions(q_plus)[-1]

            # 数值差分近似 ||∂p/∂qi||
            norm = float(np.linalg.norm(pos_plus - base_pos)) / delta
            jacobian_norms.append(norm)

        # 按范数从小到大排序
        order = sorted(range(n_joints), key=lambda i: jacobian_norms[i])

        if logger.isEnabledFor(logging.DEBUG):
            for i in order:
                logger.debug("  dim %d: Jacobian norm = %.4f", i, jacobian_norms[i])

        # 记录到日志
        if self._current_log is not None:
            self._current_log.jacobian_norms = {i: jacobian_norms[i] for i in range(n_joints)}
            self._current_log.dim_order = list(order)
            reasons = []
            for rank, dim_i in enumerate(order):
                reasons.append(f'q{dim_i}(norm={jacobian_norms[dim_i]:.4f})')
            self._current_log.dim_order_reason = (
                'Sorted by Jacobian column norm (ascending): smaller norm means '
                'less end-effector displacement per joint change, so safer to expand first. '
                'Order: ' + ' < '.join(reasons))

        return order

    def _expand_one_round(
        self,
        seed: np.ndarray,
        intervals: List[Tuple[float, float]],
        dim_order: List[int],
        round_log: Optional[RoundLog] = None,
    ) -> List[Tuple[float, float]]:
        """一轮拓展：按优先级逐维度向两侧二分搜索

        Args:
            seed: seed 配置
            intervals: 当前区间
            dim_order: 维度探索优先级
            round_log: 可选的轮次日志对象

        Returns:
            拓展后的区间列表
        """
        intervals = list(intervals)  # copy

        for dim in dim_order:
            lo_limit, hi_limit = self.joint_limits[dim]
            current_lo, current_hi = intervals[dim]

            # 固定关节跳过
            if abs(hi_limit - lo_limit) < 1e-10:
                continue

            # 向正方向拓展
            new_hi, hi_info = self._binary_search_boundary(
                intervals, dim, current_hi, hi_limit, direction=+1,
                return_info=True,
            )
            if round_log is not None and hi_info is not None:
                round_log.dim_searches.append(hi_info)

            # 更新区间用于后续负方向搜索
            intervals[dim] = (current_lo, new_hi)

            # 向负方向拓展
            new_lo, lo_info = self._binary_search_boundary(
                intervals, dim, current_lo, lo_limit, direction=-1,
                return_info=True,
            )
            if round_log is not None and lo_info is not None:
                round_log.dim_searches.append(lo_info)

            intervals[dim] = (new_lo, new_hi)

        return intervals

    def _check_collision(
        self,
        test_intervals: List[Tuple[float, float]],
    ) -> bool:
        """box 碰撞检测（支持 hybrid 模式）

        当 use_sampling=True 时，先用区间 FK 检查，
        若区间 FK 判碰撞再用采样方式复核。
        采样无碰撞则覆盖为安全（概率性）。

        Returns:
            True = 碰撞, False = 安全
        """
        interval_result = self.collision_checker.check_box_collision(
            test_intervals, skip_merge=True)
        if not interval_result:
            # 区间 FK 说安全 → 一定安全
            return False
        if not self.use_sampling:
            # 不启用采样 → 直接信任区间 FK
            return True
        # 区间 FK 说碰撞但可能过估计 → 用采样复核
        sampling_result = self.collision_checker.check_box_collision_sampling(
            test_intervals, n_samples=self.sampling_n, rng=self._rng,
        )
        return sampling_result

    def _binary_search_boundary(
        self,
        intervals: List[Tuple[float, float]],
        dim: int,
        current_bound: float,
        limit: float,
        direction: int,
        return_info: bool = False,
    ):
        """二分搜索某维度的碰撞边界

        Args:
            intervals: 当前所有维度的区间
            dim: 要拓展的维度索引
            current_bound: 当前已知安全的边界值
            limit: 该维度的关节极限
            direction: +1 向正方向拓展, -1 向负方向拓展
            return_info: 是否返回搜索详情元组

        Returns:
            若 return_info=False: 新的安全边界值 (float)
            若 return_info=True: (新的安全边界值, DimSearchLog)
        """
        safe = current_bound
        test = limit  # 尝试直接到极限
        n_steps = 0
        stop_reason = ''

        # 先尝试一步到极限
        test_intervals = list(intervals)
        if direction > 0:
            test_intervals[dim] = (intervals[dim][0], test)
        else:
            test_intervals[dim] = (test, intervals[dim][1])

        if not self._check_collision(test_intervals):
            # 到极限都无碰撞
            stop_reason = 'reached_limit'
            result = test
        else:
            # 二分搜索
            for step_i in range(50):  # 防止死循环
                if abs(test - safe) < self.resolution:
                    stop_reason = 'resolution_converged'
                    n_steps = step_i
                    break

                mid = (safe + test) / 2.0
                test_intervals = list(intervals)
                if direction > 0:
                    test_intervals[dim] = (intervals[dim][0], mid)
                else:
                    test_intervals[dim] = (mid, intervals[dim][1])

                if self._check_collision(test_intervals):
                    # mid 处碰撞，收缩
                    test = mid
                else:
                    # mid 处安全，拓展
                    safe = mid
                n_steps = step_i + 1
            else:
                stop_reason = 'max_iterations'
                n_steps = 50
            result = safe

        if not return_info:
            return result

        dir_str = '+' if direction > 0 else '-'
        info = DimSearchLog(
            dim=dim,
            direction=dir_str,
            start_bound=current_bound,
            limit=limit,
            result_bound=result,
            expansion_amount=abs(result - current_bound),
            reached_limit=(stop_reason == 'reached_limit'),
            n_bisect_steps=n_steps,
            stop_reason=stop_reason,
        )
        return result, info

    @staticmethod
    def _volume(intervals: List[Tuple[float, float]]) -> float:
        """计算区间体积（忽略固定关节的零宽度维度）"""
        vol = 1.0
        has_nonzero = False
        for lo, hi in intervals:
            w = hi - lo
            if w > 0:
                vol *= w
                has_nonzero = True
        return vol if has_nonzero else 0.0
