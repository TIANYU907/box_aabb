"""
experiments/scenes.py — 场景配置加载工具

load_scenes(names) → List[dict]
load_planners(names) → List[dict]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_DIR = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_scenes() -> Dict[str, dict]:
    """加载全部标准场景配置."""
    return _load_json(_DIR / "configs" / "scenes" / "standard_scenes.json")


def load_scenes(names: Optional[List[str]] = None) -> List[dict]:
    """按名称加载场景配置; names=None 则加载全部."""
    all_scenes = load_all_scenes()
    if names is None:
        return list(all_scenes.values())
    return [all_scenes[n] for n in names]


def load_all_planners() -> Dict[str, dict]:
    """加载全部标准 planner 配置."""
    return _load_json(_DIR / "configs" / "planners" / "standard_planners.json")


def load_planners(names: Optional[List[str]] = None) -> List[dict]:
    """按名称加载 planner 配置; names=None 则加载全部."""
    all_planners = load_all_planners()
    if names is None:
        return list(all_planners.values())
    return [all_planners[n] for n in names]
