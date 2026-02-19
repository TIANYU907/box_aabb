#!/usr/bin/env python
"""
examples/view_path.py - 交互式路径回放 CLI

从 path.json 加载已保存的规划路径，在可交互窗口中循环播放。
支持鼠标拖拽旋转 3D 视角、暂停/播放、单步、变速。

用法:
    python -m examples.view_path <path.json>
    python -m examples.view_path examples/output/random_scene_xxx/path.json
    python -m examples.view_path examples/output/random_scene_xxx/   # 自动找 path.json
    python -m examples.view_path <path.json> --fps 30 --n-frames 200 --ghost 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from box_aabb.robot import load_robot
from planner.models import PlannerResult
from planner.obstacles import Scene
from planner.dynamic_visualizer import resample_path
from planner.interactive_viewer import launch_viewer

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("view_path")


def resolve_path_json(target: str) -> Path:
    """将用户输入解析为 path.json 路径（支持目录或文件）"""
    p = Path(target)
    if p.is_dir():
        candidate = p / "path.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"目录 {p} 中未找到 path.json")
    if p.exists():
        return p
    raise FileNotFoundError(f"文件不存在: {p}")


def main():
    parser = argparse.ArgumentParser(
        description="交互式路径回放查看器")
    parser.add_argument("path_json", type=str,
                        help="path.json 文件路径或包含 path.json 的目录")
    parser.add_argument("--fps", type=int, default=20,
                        help="初始帧率 (默认: 20)")
    parser.add_argument("--n-frames", type=int, default=150,
                        help="重采样帧数 (默认: 150)")
    parser.add_argument("--ghost", type=int, default=0,
                        help="残影间隔帧数 (0=关闭, 默认: 0)")
    parser.add_argument("--trail", type=int, default=40,
                        help="末端轨迹尾长度 (默认: 40)")
    args = parser.parse_args()

    # ── 加载 path.json ──
    path_json = resolve_path_json(args.path_json)
    logger.info("加载路径: %s", path_json)
    data = PlannerResult.load_path(path_json)

    robot_config = data.get('robot_config', '')
    scene_json = data.get('scene_json', '')
    path = data['path']
    logger.info("  robot_config = %s", robot_config)
    logger.info("  路径点数     = %d", len(path))
    logger.info("  路径长度     = %.4f rad", data.get('path_length', 0.0))

    # ── 加载机器人 ──
    if not robot_config:
        logger.error("path.json 中缺少 robot_config 字段, 无法确定机器人型号")
        sys.exit(1)
    logger.info("加载机器人: %s", robot_config)
    robot = load_robot(robot_config)

    # ── 加载场景 ──
    scene = None
    if scene_json:
        scene_path = Path(scene_json)
        # 如果路径不存在，尝试相对于 path.json 所在目录解析
        if not scene_path.exists() and not scene_path.is_absolute():
            scene_path = path_json.parent / scene_path.name
        if scene_path.exists():
            logger.info("加载场景: %s", scene_path)
            scene = Scene.from_json(str(scene_path))
            logger.info("  障碍物数 = %d", scene.n_obstacles)
        else:
            logger.warning("场景文件不存在, 跳过障碍物显示: %s", scene_path)

    # ── 重采样路径 ──
    if len(path) < args.n_frames:
        logger.info("重采样: %d → %d 帧", len(path), args.n_frames)
        path = resample_path(path, args.n_frames)

    # ── 启动交互式查看器 ──
    logger.info("启动交互式查看器 (fps=%d, ghost=%d) ...", args.fps, args.ghost)
    launch_viewer(
        robot=robot,
        path=path,
        scene=scene,
        fps=args.fps,
        trail_length=args.trail,
        ghost_interval=args.ghost,
    )


if __name__ == "__main__":
    main()
