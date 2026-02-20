"""
robot.py - 机器人运动学模型

基于DH参数的串联机械臂运动学计算。
包含正运动学、连杆位置计算、相关关节检测、零长度连杆识别
以及耦合关节约束声明。
"""

import math
import json
import hashlib
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from functools import cached_property

logger = logging.getLogger(__name__)

try:
    from ._fk_scalar_core import link_position_core as _link_position_core_cy
except Exception:  # pragma: no cover - optional Cython extension
    _link_position_core_cy = None


class Robot:
    """基于DH参数的串联机械臂

    使用修正DH约定 (Modified DH Convention)。
    可选地声明 ``coupled_pairs`` / ``coupled_triples``，
    供采样策略模块用于生成特定关节耦合约束上的关键点。

    Example:
        >>> dh = [
        ...     {"alpha": 0, "a": 0, "d": 0.333, "theta": 0, "type": "revolute"},
        ...     {"alpha": -math.pi/2, "a": 0, "d": 0, "theta": 0, "type": "revolute"},
        ... ]
        >>> robot = Robot(dh, name="MyRobot")
        >>> T = robot.forward_kinematics([0.1, 0.2])

    Attributes:
        name: 机器人名称
        joint_limits: 关节限制列表，每个元素为 (lo, hi) 弧度
        coupled_pairs: 耦合关节对列表，如 [(0,2), (1,3)]
        coupled_triples: 耦合关节三元组列表，如 [(0,2,4), (1,3,5)]
    """

    def __init__(
        self,
        dh_params: List[Dict],
        coupled_pairs: Optional[List[Tuple[int, int]]] = None,
        coupled_triples: Optional[List[Tuple[int, int, int]]] = None,
        name: str = "Robot",
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        tool_frame: Optional[Dict] = None,
    ):
        """初始化机器人

        Args:
            dh_params: DH参数列表，每个元素是包含以下键的字典：
                - alpha: 连杆扭转角 (rad)
                - a: 连杆长度
                - d: 连杆偏移
                - theta: 关节角偏移 (rad)
                - type: 'revolute' 或 'prismatic'
            coupled_pairs: 耦合关节对（用于采样策略 4），默认 None
            coupled_triples: 耦合关节三元组（用于采样策略 5-6），默认 None
            name: 机器人名称，默认 "Robot"
            joint_limits: 关节限制列表 [(lo, hi), ...]，默认 None
            tool_frame: 末端工具坐标系 DH 参数 {alpha, a, d}，默认 None
                        用于在最后一个关节之后附加一段固定连杆。
                        在 Modified DH 中，最后一个 DH 行的 `a` 参数提供的
                        平移在最后一帧位置中不可见（它将在下一帧才体现），
                        因此需要 tool_frame 来表示末端连杆。
        """
        self.name = name
        self.dh_params = [self._normalize_param(p) for p in dh_params]
        self.n_joints = len(self.dh_params)
        self.coupled_pairs = coupled_pairs or []
        self.coupled_triples = coupled_triples or []
        self.joint_limits = (
            [(float(lo), float(hi)) for lo, hi in joint_limits]
            if joint_limits else None
        )
        # 末端工具坐标系（可选，提供最后一段固定连杆的 DH 参数）
        self.tool_frame: Optional[Dict] = None
        if tool_frame is not None:
            self.tool_frame = {
                'alpha': float(tool_frame.get('alpha', 0.0)),
                'a': float(tool_frame.get('a', 0.0)),
                'd': float(tool_frame.get('d', 0.0)),
            }

        # 预打包 DH 参数（供批量/可选 Cython 路径使用）
        self._dh_alpha = np.array([p['alpha'] for p in self.dh_params], dtype=np.float64)
        self._dh_a = np.array([p['a'] for p in self.dh_params], dtype=np.float64)
        self._dh_d = np.array([p['d'] for p in self.dh_params], dtype=np.float64)
        self._dh_theta = np.array([p['theta'] for p in self.dh_params], dtype=np.float64)
        self._dh_joint_type = np.array(
            [0 if p['type'] == 'revolute' else 1 for p in self.dh_params],
            dtype=np.int32,
        )
        self._tool_alpha = float(self.tool_frame['alpha']) if self.tool_frame is not None else 0.0
        self._tool_a = float(self.tool_frame['a']) if self.tool_frame is not None else 0.0
        self._tool_d = float(self.tool_frame['d']) if self.tool_frame is not None else 0.0
    
    @staticmethod
    def _normalize_param(p: Dict) -> Dict:
        """标准化DH参数"""
        param = {
            'alpha': float(p.get('alpha', 0.0)),
            'a': float(p.get('a', 0.0)),
            'd': float(p.get('d', 0.0)) if p.get('d') is not None else 0.0,
            'theta': float(p.get('theta', 0.0)) if p.get('theta') is not None else 0.0,
            'type': p.get('type', 'revolute')
        }
        if param['type'] not in ('revolute', 'prismatic'):
            raise ValueError("关节类型必须是 'revolute' 或 'prismatic'")
        return param
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Robot':
        """从JSON配置文件加载机器人

        支持两种格式：
        1. 完整配置 (推荐)::

            {
                "name": "Panda",
                "dh_params": [...],
                "joint_limits": [[lo, hi], ...],
                "coupled_pairs": [[0, 2], ...],
                "coupled_triples": [[0, 2, 4], ...]
            }

        2. 旧版兼容 — 纯 DH 列表或 ``{"dh": [...]}``
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls._from_dict(data, source=filepath)
    
    @classmethod
    def _from_dict(cls, data, source: str = '<dict>') -> 'Robot':
        """从字典构建 Robot（内部方法，from_json / from_config 共用）"""
        if isinstance(data, list):
            # 旧版: 纯 DH 列表
            return cls(data)

        if not isinstance(data, dict):
            raise ValueError(f'{source}: JSON 必须是 dict 或 list')

        # DH 参数
        dh_list = data.get('dh_params') or data.get('dh')
        if dh_list is None:
            raise ValueError(f'{source}: 缺少 "dh_params" 或 "dh" 字段')

        name = data.get('name', 'Robot')
        joint_limits = (
            [tuple(lim) for lim in data['joint_limits']]
            if 'joint_limits' in data else None
        )
        coupled_pairs = (
            [tuple(p) for p in data['coupled_pairs']]
            if 'coupled_pairs' in data else None
        )
        coupled_triples = (
            [tuple(t) for t in data['coupled_triples']]
            if 'coupled_triples' in data else None
        )
        tool_frame = data.get('tool_frame', None)
        return cls(
            dh_params=dh_list,
            coupled_pairs=coupled_pairs,
            coupled_triples=coupled_triples,
            name=name,
            joint_limits=joint_limits,
            tool_frame=tool_frame,
        )

    @classmethod
    def from_config(cls, name: str) -> 'Robot':
        """按名称加载内置机器人配置

        在 ``configs/`` 目录中查找 ``<name>.json`` 文件。
        名称匹配不区分大小写。

        Args:
            name: 配置名称，如 ``"panda"``

        Returns:
            Robot 实例

        Raises:
            FileNotFoundError: 找不到指定配置文件
        """
        import pathlib
        configs_dir = pathlib.Path(__file__).parent / 'configs'
        # 不区分大小写搜索
        target = name.lower()
        for f in configs_dir.iterdir():
            if f.suffix == '.json' and f.stem.lower() == target:
                return cls.from_json(str(f))
        available = [f.stem for f in configs_dir.glob('*.json')]
        raise FileNotFoundError(
            f'找不到配置 "{name}"。可用配置: {available}'
        )

    @staticmethod
    def list_configs() -> List[str]:
        """列出所有可用的内置机器人配置名称"""
        import pathlib
        configs_dir = pathlib.Path(__file__).parent / 'configs'
        if not configs_dir.exists():
            return []
        return sorted(f.stem for f in configs_dir.glob('*.json'))

    @staticmethod
    def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        """
        计算单个DH变换矩阵 (Modified DH Convention)
        
        Args:
            alpha: 连杆扭转角
            a: 连杆长度  
            d: 连杆偏移
            theta: 关节角
            
        Returns:
            4x4齐次变换矩阵
        """
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        
        return np.array([
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
    
    def forward_kinematics(self, joint_values: List[float], 
                          return_all: bool = False) -> np.ndarray:
        """
        正向运动学计算
        
        Args:
            joint_values: 关节值列表
            return_all: 是否返回所有连杆的变换矩阵
            
        Returns:
            如果return_all=False: 末端执行器的4x4变换矩阵
            如果return_all=True: 所有连杆变换矩阵的列表 [T0, T1, ..., Tn]
        """
        if len(joint_values) != self.n_joints:
            raise ValueError(f'期望 {self.n_joints} 个关节值，得到 {len(joint_values)}')
        
        transforms = [np.eye(4)]  # T0 = I
        T = np.eye(4)
        
        for i, (param, q) in enumerate(zip(self.dh_params, joint_values)):
            alpha = param['alpha']
            a = param['a']
            
            if param['type'] == 'revolute':
                d = param['d']
                theta = q + param['theta']
            else:  # prismatic
                d = param['d'] + q
                theta = param['theta']
            
            A = self.dh_transform(alpha, a, d, theta)
            T = T @ A
            transforms.append(T.copy())

        # 工具坐标系（可选末端固定连杆）
        if self.tool_frame is not None:
            A_tool = self.dh_transform(
                self.tool_frame['alpha'],
                self.tool_frame['a'],
                self.tool_frame['d'],
                0.0,  # 无关节旋转
            )
            T = T @ A_tool
            transforms.append(T.copy())

        return transforms if return_all else transforms[-1]
    
    def get_link_positions(self, joint_values: List[float]) -> List[np.ndarray]:
        """
        获取所有连杆端点的世界坐标
        
        Args:
            joint_values: 关节值列表
            
        Returns:
            位置列表 [p0, p1, ..., pn]，每个pi是(3,)数组
        """
        transforms = self.forward_kinematics(joint_values, return_all=True)
        return [T[:3, 3] for T in transforms]

    def get_link_positions_batch(
        self,
        joint_values_batch: np.ndarray,
        link_idx: int,
    ) -> np.ndarray:
        """批量计算指定连杆末端位置

        Args:
            joint_values_batch: 关节配置矩阵 (B, n_joints) 或 (B, <=n_joints)
            link_idx: 连杆索引 (1-based)

        Returns:
            位置矩阵 (B, 3)
        """
        q_batch = np.asarray(joint_values_batch, dtype=np.float64)
        if q_batch.ndim == 1:
            q_batch = q_batch[None, :]
        if q_batch.ndim != 2:
            raise ValueError("joint_values_batch 必须是 2D 数组")

        B, n_cols = q_batch.shape
        if n_cols < self.n_joints:
            pad = np.zeros((B, self.n_joints - n_cols), dtype=np.float64)
            q_batch = np.hstack([q_batch, pad])
        elif n_cols > self.n_joints:
            q_batch = q_batch[:, :self.n_joints]

        T = np.tile(np.eye(4, dtype=np.float64), (B, 1, 1))
        max_link = min(int(link_idx), len(self.dh_params))

        for i in range(max_link):
            param = self.dh_params[i]
            alpha = param['alpha']
            a = param['a']

            if param['type'] == 'revolute':
                d = np.full(B, param['d'], dtype=np.float64)
                theta = q_batch[:, i] + param['theta']
            else:
                d = param['d'] + q_batch[:, i]
                theta = np.full(B, param['theta'], dtype=np.float64)

            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(theta), np.sin(theta)

            A = np.zeros((B, 4, 4), dtype=np.float64)
            A[:, 0, 0] = ct
            A[:, 0, 1] = -st
            A[:, 0, 3] = a
            A[:, 1, 0] = st * ca
            A[:, 1, 1] = ct * ca
            A[:, 1, 2] = -sa
            A[:, 1, 3] = -d * sa
            A[:, 2, 0] = st * sa
            A[:, 2, 1] = ct * sa
            A[:, 2, 2] = ca
            A[:, 2, 3] = d * ca
            A[:, 3, 3] = 1.0

            T = np.einsum('nij,njk->nik', T, A)

        if link_idx > len(self.dh_params) and self.tool_frame is not None:
            A_tool = self.dh_transform(
                self.tool_frame['alpha'],
                self.tool_frame['a'],
                self.tool_frame['d'],
                0.0,
            )
            T = np.einsum('nij,jk->nik', T, A_tool)

        return T[:, :3, 3].copy()
    
    def end_effector_pose(self, joint_values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器位姿
        
        Returns:
            (position, rotation_matrix): 位置(3,)和旋转矩阵(3,3)
        """
        T = self.forward_kinematics(joint_values)
        return T[:3, 3].copy(), T[:3, :3].copy()
    
    # ==================== 连杆位置计算 ====================

    def _get_link_position_python(self, joint_values: List[float], link_idx: int) -> np.ndarray:
        """Python 标量 FK 实现（作为回退与基准对照）。"""
        q = list(joint_values)
        while len(q) < self.n_joints:
            q.append(0.0)

        T = np.eye(4)
        for i in range(min(link_idx, len(self.dh_params))):
            param = self.dh_params[i]
            if param['type'] == 'revolute':
                theta = q[i] + param['theta']
                d = param['d']
            else:
                theta = param['theta']
                d = param['d'] + q[i]
            T = T @ self.dh_transform(param['alpha'], param['a'], d, theta)

        # 如果 link_idx 超过 DH 参数数量且存在 tool_frame，追加 tool_frame
        if link_idx > len(self.dh_params) and self.tool_frame is not None:
            A_tool = self.dh_transform(
                self.tool_frame['alpha'],
                self.tool_frame['a'],
                self.tool_frame['d'],
                0.0,
            )
            T = T @ A_tool

        return T[:3, 3]
    
    def get_link_position(self, joint_values: List[float], link_idx: int) -> np.ndarray:
        """计算指定连杆末端的世界坐标位置
        
        与 forward_kinematics(return_all=True) 不同，此方法仅计算到
        第 link_idx 个连杆，避免不必要的矩阵乘法。
        
        Args:
            joint_values: 关节值列表（长度不足时自动补零）
            link_idx: 连杆索引 (1-based, 1~n_joints+1 含 tool_frame)
        
        Returns:
            连杆末端位置 (3,) ndarray
        """
        q_arr = np.asarray(joint_values, dtype=np.float64)
        if q_arr.ndim != 1:
            q_arr = q_arr.reshape(-1)
        if q_arr.shape[0] < self.n_joints:
            q_arr = np.pad(q_arr, (0, self.n_joints - q_arr.shape[0]), mode='constant')
        elif q_arr.shape[0] > self.n_joints:
            q_arr = q_arr[:self.n_joints]

        if _link_position_core_cy is not None:
            return _link_position_core_cy(
                q_arr,
                int(link_idx),
                self._dh_alpha,
                self._dh_a,
                self._dh_d,
                self._dh_theta,
                self._dh_joint_type,
                self.tool_frame is not None,
                self._tool_alpha,
                self._tool_a,
                self._tool_d,
            )

        return self._get_link_position_python(q_arr.tolist(), link_idx)
    
    # ==================== 相关关节检测 ====================
    
    def compute_relevant_joints(self, link_idx: int, 
                                delta: float = 0.1, tol: float = 1e-8) -> Set[int]:
        """检测哪些关节真正影响指定连杆的位置
        
        使用数值微分法：对每个关节逐一施加扰动，检测连杆位置
        x/y/z 三个分量是否变化。在多组基准配置上测试以避免特殊姿态。
        
        Args:
            link_idx: 连杆索引 (1-based)
            delta: 数值微分步长
            tol: 位置变化判定阈值
        
        Returns:
            影响该连杆位置的关节索引集合 (0-based)
        """
        relevant = set()
        test_bases = [
            [0.0] * self.n_joints,
            [0.3] * self.n_joints,
            [-0.5] * self.n_joints,
        ]
        for base_q in test_bases:
            base_pos = self.get_link_position(base_q, link_idx)
            for i in range(min(link_idx, self.n_joints)):
                if i in relevant:
                    continue
                q_test = list(base_q)
                q_test[i] = base_q[i] + delta
                test_pos = self.get_link_position(q_test, link_idx)
                if any(abs(test_pos[ax] - base_pos[ax]) > tol for ax in range(3)):
                    relevant.add(i)
        return relevant
    
    # ==================== 零长度连杆识别 ====================
    
    @cached_property
    def zero_length_links(self) -> Set[int]:
        """识别零长度连杆（a=0 且 d=0 的连杆）
        
        Returns:
            零长度连杆索引集合 (1-based)
        """
        result = set()
        for i, param in enumerate(self.dh_params):
            if abs(param['a']) < 1e-10 and abs(param['d']) < 1e-10:
                result.add(i + 1)
        # tool_frame 连杆索引 = n_joints + 1
        if self.tool_frame is not None:
            if abs(self.tool_frame['a']) < 1e-10 and abs(self.tool_frame['d']) < 1e-10:
                result.add(self.n_joints + 1)
        return result

    def fingerprint(self) -> str:
        """生成机器人唯一指纹（SHA256）

        基于 DH 参数和名称计算哈希，用于 AABB 缓存的机器人标识。
        相同运动学参数的机器人产生相同的指纹。

        Returns:
            64 字符的十六进制哈希字符串
        """
        data = json.dumps({
            'name': self.name,
            'dh_params': self.dh_params,
            'n_joints': self.n_joints,
            'tool_frame': self.tool_frame,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()


# ==================== 便捷工厂函数 ====================

def load_robot(name: str) -> Robot:
    """按名称加载内置机器人配置

    在 ``configs/`` 目录中查找 ``<name>.json`` 配置文件。
    这是 ``Robot.from_config(name)`` 的快捷方式。

    Args:
        name: 配置名称（不区分大小写），如 ``"panda"``

    Returns:
        Robot 实例

    Example:
        >>> robot = load_robot('panda')
        >>> robot.name
        'Panda'
    """
    return Robot.from_config(name)


def create_panda_robot() -> Robot:
    """创建Franka Emika Panda机器人

    Panda是7自由度协作机器人，夹爪连杆通过 tool_frame 表示。

    .. deprecated:: 3.0.0
        请使用 ``load_robot('panda')`` 或 ``Robot.from_config('panda')``。
    """
    return load_robot('panda')


def _panda_joint_limits() -> List[Tuple[float, float]]:
    """获取 Panda 关节限制（从配置文件加载）"""
    robot = load_robot('panda')
    if robot.joint_limits:
        return robot.joint_limits
    # 硬编码后备
    return [
        (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
        (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ]


# 保持向后兼容 — 推荐通过 robot.joint_limits 获取
PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),   # q0
    (-1.7628, 1.7628),   # q1
    (-2.8973, 2.8973),   # q2
    (-3.0718, -0.0698),  # q3
    (-2.8973, 2.8973),   # q4
    (-0.0175, 3.7525),   # q5
    (-2.8973, 2.8973),   # q6
]
