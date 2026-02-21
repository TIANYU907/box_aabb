"""
utils/seed.py — 随机种子管理

统一管理可复现性种子。
"""

import time

import numpy as np


def make_seed(seed: int = 0) -> int:
    """如果 seed == 0, 用当前时间戳生成; 否则原样返回."""
    if seed == 0:
        return int(time.time()) % (2**31)
    return seed


def make_rng(seed: int = 0) -> np.random.Generator:
    """返回 numpy Generator, seed==0 时自动分配."""
    return np.random.default_rng(make_seed(seed))
