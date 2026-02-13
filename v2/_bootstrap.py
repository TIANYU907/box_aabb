"""Runtime path bootstrap for v2 scripts.

Allows running scripts via ``python -m v2.examples.*`` without installing the
package, while keeping imports from ``v2/src`` stable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def add_v2_paths() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    for path in (src, root):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
