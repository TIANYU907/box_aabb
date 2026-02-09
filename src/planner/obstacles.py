"""
planner/obstacles.py - 障碍物与场景管理

管理工作空间中的 AABB 障碍物集合，提供场景配置和序列化。
与现有 Visualizer.plot_obstacles() 接口兼容。
"""

import json
import logging
from typing import List, Dict, Any, Optional

import numpy as np

from .models import Obstacle

logger = logging.getLogger(__name__)


class Scene:
    """工作空间场景管理

    管理一组 AABB 障碍物，提供增删查改和持久化功能。
    兼容现有 ``Visualizer.plot_obstacles(obstacles: List[Dict])`` 接口。

    Example:
        >>> scene = Scene()
        >>> scene.add_obstacle([0.5, -0.3, 0], [0.8, 0.3, 0.5], name="桌子")
        >>> scene.add_obstacle([-0.5, -0.5, 0], [-0.2, 0.5, 0.4], name="货架")
        >>> # 配合 Visualizer 使用
        >>> viz.plot_obstacles(scene.to_dict_list())
    """

    def __init__(self) -> None:
        self._obstacles: List[Obstacle] = []

    @property
    def n_obstacles(self) -> int:
        return len(self._obstacles)

    def add_obstacle(
        self,
        min_point: Any,
        max_point: Any,
        name: str = "",
    ) -> Obstacle:
        """添加一个 AABB 障碍物

        Args:
            min_point: AABB 最小角点 [x, y, z] 或 [x, y]（2D 场景时 z=0）
            max_point: AABB 最大角点
            name: 障碍物名称

        Returns:
            创建的 Obstacle 实例
        """
        min_pt = np.array(min_point, dtype=np.float64)
        max_pt = np.array(max_point, dtype=np.float64)

        # 2D 场景自动扩展为 3D（z 范围设为负无穷到正无穷）
        if min_pt.shape[0] == 2:
            min_pt = np.array([min_pt[0], min_pt[1], -1e3], dtype=np.float64)
            max_pt = np.array([max_pt[0], max_pt[1], 1e3], dtype=np.float64)

        if not name:
            name = f"obstacle_{self.n_obstacles}"

        obs = Obstacle(min_point=min_pt, max_point=max_pt, name=name)
        self._obstacles.append(obs)
        logger.debug("添加障碍物 '%s': min=%s, max=%s", name,
                      min_pt.tolist(), max_pt.tolist())
        return obs

    def remove_obstacle(self, name: str) -> bool:
        """按名称移除障碍物

        Returns:
            是否找到并移除
        """
        for i, obs in enumerate(self._obstacles):
            if obs.name == name:
                self._obstacles.pop(i)
                return True
        return False

    def clear(self) -> None:
        """清空所有障碍物"""
        self._obstacles.clear()

    def get_obstacles(self) -> List[Obstacle]:
        """获取所有障碍物"""
        return list(self._obstacles)

    def get_obstacle(self, name: str) -> Optional[Obstacle]:
        """按名称查找障碍物"""
        for obs in self._obstacles:
            if obs.name == name:
                return obs
        return None

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """转为 Visualizer.plot_obstacles 兼容的格式

        Returns:
            [{'min': [x,y,z], 'max': [x,y,z], 'name': str}, ...]
        """
        return [obs.to_dict() for obs in self._obstacles]

    def to_json(self, filepath: str) -> None:
        """保存场景到 JSON 文件"""
        data = {
            'obstacles': [
                {
                    'min': obs.min_point.tolist(),
                    'max': obs.max_point.tolist(),
                    'name': obs.name,
                }
                for obs in self._obstacles
            ]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str) -> 'Scene':
        """从 JSON 文件加载场景"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scene':
        """从字典加载场景

        Args:
            data: {'obstacles': [{'min': [...], 'max': [...], 'name': ...}, ...]}
        """
        scene = cls()
        for item in data.get('obstacles', []):
            scene.add_obstacle(
                min_point=item['min'],
                max_point=item['max'],
                name=item.get('name', ''),
            )
        return scene

    @classmethod
    def from_obstacle_list(cls, obstacles: List[Dict[str, Any]]) -> 'Scene':
        """从已有的 obstacle dict 列表创建 Scene（兼容旧格式）"""
        scene = cls()
        for item in obstacles:
            scene.add_obstacle(
                min_point=item['min'],
                max_point=item['max'],
                name=item.get('name', ''),
            )
        return scene

    def __repr__(self) -> str:
        return f"Scene(n_obstacles={self.n_obstacles})"
