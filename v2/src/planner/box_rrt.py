"""box_rrt.py - backwards compatibility shim. Moved to box_planner.py."""
from .box_planner import *  # noqa: F401,F403
from .box_planner import BoxPlanner as BoxRRT  # noqa: F401
