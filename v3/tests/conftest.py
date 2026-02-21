import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

for prefix in ("aabb", "forest", "planner", "baselines", "utils", "common"):
    for mod in [m for m in list(sys.modules.keys()) if m == prefix or m.startswith(prefix + ".")]:
        sys.modules.pop(mod, None)

os.environ.setdefault("MPLBACKEND", "Agg")
