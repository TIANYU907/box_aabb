"""gcs_planner_panda.py - backwards compatibility shim.

Moved to panda_planner.py. Usage:
    python -m v2.examples.panda_planner
"""
from v2.examples.panda_planner import *  # noqa: F401,F403

if __name__ == "__main__":
    from v2.examples.panda_planner import main
    main()
