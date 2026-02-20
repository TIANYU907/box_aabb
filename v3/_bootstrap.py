"""Runtime path bootstrap for v3 scripts.

Allows running scripts via ``python -m v3.examples.*`` without installing the
package, while keeping imports from ``v3/src`` stable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def add_v3_paths() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    for path in (root, src):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
