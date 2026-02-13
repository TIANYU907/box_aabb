# box-aabb v2 (WIP)

This folder contains the refactored implementation split into three layers:

- `aabb/`: interval AABB computation
- `forest/`: box forest construction and collision core
- `planner/`: path planning on top of forest

Current status:
- `aabb/` migrated baseline is available
- `forest/` and `planner/` scaffolds are created and pending module migration

Quick test:

```bash
python -m pytest v2/tests/aabb -q
```
