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

## Benchmark 用法

```bash
python -m v2.benchmarks.forest.bench_panda_forest
python -m v2.benchmarks.forest.bench_panda_multi
```

Planner benchmark (strict deps, no fallback):

```bash
python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot 2dof_planar --trials 8
```

One-command workflow (install -> run -> latest summary alias):

```powershell
./v2/scripts/run_benchmark_oneclick.ps1 -Mode conda -EnvName box-rrt -Trials 8
```

This command will:
- setup dependencies (via `setup_benchmark_env.ps1`)
- run benchmark module
- auto-generate latest summary alias:
	- `v2/output/benchmarks/planner_rrt_vs_marcucci_latest_summary.md`

One-click setup (PowerShell, conda mode):

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -RunSmokeTest
```

If you need to create environment first:

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode conda -EnvName box-rrt -CreateEnv -PythonVersion 3.10 -RunSmokeTest
```

One-click setup (PowerShell, pip mode in current Python env):

```powershell
./v2/scripts/setup_benchmark_env.ps1 -Mode pip -RunSmokeTest
```

Manual command sets:

```powershell
# conda
conda install -n box-rrt -c conda-forge numpy scipy matplotlib pytest ompl drake -y
conda run -n box-rrt python -m pip install -e ./v2

# pip (current env)
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pytest ompl drake
python -m pip install -e ./v2
```

Required external libraries:
- `ompl` (for `RRT`/`RRTConnect`/`RRTstar` baselines)
- `pydrake` (for Marcucci-style `GraphOfConvexSets` baseline)

If either dependency is missing, benchmark exits with ImportError directly.

Detailed benchmark docs:
- `v2/doc/benchmark_rrt_vs_marcucci.md`
- `v2/doc/planner.md`
