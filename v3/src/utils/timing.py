"""
utils/timing.py — 阶段计时器

用于精确记录管线各阶段耗时。
"""

import time
from contextlib import contextmanager


class Timer:
    """阶段计时器，用于精确记录管线各阶段耗时。"""

    def __init__(self):
        self.records: dict[str, float] = {}
        self._stack: list[tuple[str, float]] = []

    @contextmanager
    def phase(self, name: str):
        """记录 name 阶段的耗时 (秒)."""
        t0 = time.perf_counter()
        yield
        self.records[name] = time.perf_counter() - t0

    @property
    def total(self) -> float:
        return sum(self.records.values())

    def to_dict(self) -> dict:
        return {**self.records, "total": self.total}

    def summary(self, unit: str = "ms") -> str:
        """返回格式化汇总字符串."""
        mul = 1000.0 if unit == "ms" else 1.0
        lines = []
        for name, sec in self.records.items():
            lines.append(f"  {name:20s}: {sec * mul:8.1f} {unit}")
        lines.append(f"  {'TOTAL':20s}: {self.total * mul:8.1f} {unit}")
        return "\n".join(lines)
