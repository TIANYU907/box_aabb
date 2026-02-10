# BOX-AABB: 机器人连杆包围盒计算库

一个高效的机器人AABB（轴对齐包围盒）计算库，提供多种计算方法。

## 特性

- **多机器人配置**: 各机器人保存为独立JSON配置，按名称加载
- **关键点采样 (Critical)**: 基于梯度分析的精确枚举 + 约束点优化
- **随机采样 (Random)**: 大量随机采样 + scipy 局部优化
- **混合策略 (Hybrid)**: 关键枚举 + 约束优化 + 少量随机补充（推荐）
- **区间方法 (Interval)**: 使用仿射/区间算术，保守但保证安全
- **连杆等分**: 将连杆划分为多段获得更紧凑的包围盒
- **机器人无关**: 支持任意串联机器人（通过DH参数 + JSON配置定义）

## 安装

```bash
pip install numpy
# 可选依赖
pip install scipy      # L-BFGS-B 局部优化
pip install matplotlib # 3D 可视化
```

或通过 pyproject.toml:

```bash
pip install .                   # 核心
pip install ".[optimization]"   # 含 scipy
pip install ".[visualization]"  # 含 matplotlib
pip install ".[dev]"            # 含 pytest
```

## 快速开始

```python
from box_aabb import load_robot, AABBCalculator, Robot

# 1. 按名称加载内置机器人配置
robot = load_robot('panda')
print(robot.name)          # "Panda"
print(robot.joint_limits)  # [(-2.8973, 2.8973), ...]

# 2. 定义关节区间
joint_intervals = [
    (-0.5, 0.5),    # q0
    (-0.3, 0.3),    # q1
    (-0.5, 0.5),    # q2
    (-2.0, -1.0),   # q3
    (-0.5, 0.5),    # q4
    (0.5, 1.5),     # q5
    (-0.5, 0.5),    # q6
    (0, 0),         # q7 (夹爪，固定)
]

# 3. 计算AABB（robot_name 自动从 robot.name 读取）
calc = AABBCalculator(robot)
result = calc.compute_envelope(
    joint_intervals,
    method='numerical',
    sampling='hybrid',
)

# 4. 查看结果
for aabb in result.link_aabbs:
    if aabb.is_zero_length:
        continue
    print(f"{aabb.link_name}: 体积={aabb.volume:.6f}")

# 5. 生成报告
report = result.generate_report(save_path="aabb_report.md")

# 查看所有可用配置
print(Robot.list_configs())  # ['2dof_planar', 'panda']
```

## API参考

### Robot

机器人运动学模型，基于修正DH参数 (Modified DH Convention)。

```python
from box_aabb import Robot, load_robot

# 方式1: 按名称加载内置配置（推荐）
robot = load_robot('panda')
# 等价于：
robot = Robot.from_config('panda')

# 查看所有内置配置
print(Robot.list_configs())  # ['2dof_planar', 'panda']

# 方式2: 从自定义JSON配置文件加载
robot = Robot.from_json('my_robot.json')

# 方式3: 直接构造（适合临时测试）
dh_params = [
    {"alpha": 0, "a": 0, "d": 0.333, "theta": 0, "type": "revolute"},
    # ...
]
robot = Robot(
    dh_params,
    name="MyRobot",                         # 可选：机器人名称
    joint_limits=[(-2.89, 2.89), ...],       # 可选：关节限制
    coupled_pairs=[(0, 2), (1, 3)],          # 可选：耦合关节对
    coupled_triples=[(0, 2, 4), (1, 3, 5)],  # 可选：耦合关节三元组
)

# Robot 属性
print(robot.name)            # "Panda"
print(robot.n_joints)        # 8
print(robot.joint_limits)    # [(-2.8973, 2.8973), ...]
print(robot.coupled_pairs)   # [(0, 2), (1, 3)]
print(robot.coupled_triples) # [(0, 2, 4), (1, 3, 5)]
```

### 机器人配置文件格式

将JSON文件放入 `configs/` 目录即可通过 `load_robot('名称')` 加载：

```json
{
    "name": "MyRobot",
    "description": "My custom 6-DOF robot",
    "dh_convention": "modified",
    "dh_params": [
        {"alpha": 0, "a": 0, "d": 0.333, "theta": 0, "type": "revolute"},
        ...
    ],
    "joint_limits": [
        [-2.89, 2.89],
        ...
    ],
    "coupled_pairs": [[0, 2], [1, 3]],
    "coupled_triples": [[0, 2, 4], [1, 3, 5]]
}
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | 否 | 机器人名称，默认 "Robot" |
| `dh_params` | list | **是** | DH参数列表 |
| `joint_limits` | list | 否 | 关节限制 `[[lo, hi], ...]` |
| `coupled_pairs` | list | 否 | 耦合关节对 `[[i, j], ...]` |
| `coupled_triples` | list | 否 | 耦合关节三元组 `[[i, j, k], ...]` |

### AABBCalculator

AABB计算器——瘦调度层，委托给各策略模块。

```python
calc = AABBCalculator(robot)  # robot_name 自动从 robot.name 读取

# 完整版 API（推荐）
result = calc.compute_envelope(
    joint_intervals,
    method='numerical',                # 'numerical' 或 'interval'
    sampling='hybrid',                 # 'critical', 'random', 'hybrid'
    n_random_samples=10000,            # random 模式采样数
    n_refine_samples=200,              # hybrid 模式随机补充采样数
    critical_proximity_threshold=0.05, # 随机采样避开关键点的半径
    n_subdivisions=1,                  # 连杆等分段数
    skip_zero_length=True,
    visualize=False,
    save_report=None,
)

# 简化版 API（兼容旧接口）
aabbs = calc.compute_link_aabbs(joint_intervals)
ee_aabb = calc.compute_end_effector_aabb(joint_intervals)
robot_aabb = calc.compute_robot_aabb(joint_intervals)
```

**采样策略对比:**

| 采样策略 | 参数 | 说明 |
|---------|------|------|
| `critical` | — | 纯关键点枚举 + 约束点优化，最快 |
| `hybrid` | `n_refine_samples`, `critical_proximity_threshold` | 关键枚举 + 少量随机补充，推荐 |
| `random` | `n_random_samples`, `critical_proximity_threshold` | 大量随机 + 优化，最慢 |

### Visualizer

3D可视化器。

```python
from box_aabb import Visualizer

viz = Visualizer(figsize=(10, 8))
viz.plot_robot(robot, joint_values)
viz.plot_aabbs(aabbs, alpha=0.3)
viz.show()
```

或快速可视化计算结果:

```python
from box_aabb import visualize_envelope_result

fig = visualize_envelope_result(result, robot, show_boundary_configs=True)
fig.show()
```

## 架构设计

```
box_aabb/
├── pyproject.toml
├── README.md
├── src/
│   ├── box_aabb/             # 核心库 (src layout)
│   │   ├── __init__.py           # 包入口 & 公开API
│   │   ├── robot.py              # Robot 类: DH运动学, FK, 配置加载
│   │   ├── aabb_calculator.py    # AABBCalculator: 瘦调度层
│   │   ├── models.py             # 数据类: BoundaryConfig, LinkAABBInfo, AABBEnvelopeResult
│   │   ├── report.py             # ReportGenerator: Markdown 报告生成
│   │   ├── optimization.py       # optimize_extremes: L-BFGS-B 局部优化
│   │   ├── interval_math.py      # Interval & AffineForm 算术
│   │   ├── interval_fk.py        # 区间正运动学 (保守AABB)
│   │   ├── visualizer.py         # Matplotlib 3D 可视化
│   │   ├── configs/              # 机器人配置文件目录
│   │   └── strategies/           # 采样策略 (critical/random/hybrid)
│   └── planner/              # ★ 路径规划包 (v4.1)
│       ├── box_rrt.py            # 主规划器入口
│       ├── collision.py          # 碰撞检测 (集成 AABB 缓存)
│       ├── box_expansion.py      # Box 拓展 (balanced/greedy 策略)
│       ├── box_tree.py           # Box 树管理
│       ├── aabb_cache.py         # AABB 缓存系统
│       ├── box_forest.py         # 可复用 BoxForest
│       ├── box_query.py          # Forest 查询规划
│       ├── configs/              # 规划器预设 JSON 配置
│       ├── dynamic_visualizer.py # 动态可视化动画
│       ├── free_space_tiler.py   # 自由空间瓦片化
│       └── ...                   # connector, smoother, GCS, metrics 等
├── test/
│   ├── conftest.py       # pytest fixtures
│   ├── test_robot.py     # Robot + 配置系统测试
│   ├── test_interval_math.py  # 区间/仿射算术测试
│   └── planner/          # 规划器测试
├── benchmarks/
│   ├── threshold_experiment.py  # 区间宽度阈值实验 (v4.0)
│   └── ...
└── examples/
    ├── basic_usage.py
    ├── planner_demo_animation.py  # 动态可视化 demo (v4.0)
    └── ...
```

**设计原则：**

- **配置驱动**: 各机器人参数保存为独立JSON文件，`load_robot('name')` 一行加载
- **策略模式 (Strategy Pattern)**: `CriticalStrategy` / `RandomStrategy` / `HybridStrategy` 继承 `BaseStrategy`，共享采样逻辑
- **瘦调度层**: `AABBCalculator` 仅做参数验证和策略分发，不含采样逻辑
- **关注点分离**: 运动学 (`robot.py`)、数据模型 (`models.py`)、优化 (`optimization.py`)、报告 (`report.py`) 各自独立
- **机器人无关**: 耦合关节约束通过 `Robot.coupled_pairs` / `coupled_triples` 声明，策略从 robot 对象读取

## 算法原理

### 关键点分类

正向运动学中，连杆末端位置关于关节角的梯度为零的条件分为两类：

#### 梯度完全为零（精确极值，无需优化）

1. **边界组合**（策略1）：所有关节角在区间端点
   - $q_i \in \{q_i^{min},\; q_i^{max}\}$

2. **单关节关键值**（策略2）：某个关节在 $k\pi/2$ 处
   - $q_i = k \cdot \frac{\pi}{2}$（在区间内）

#### 部分导数项为零（约束点，需局部优化）

3. **两关节和约束**（策略3）：任意两关节和
   - $q_i + q_j = k \cdot \frac{\pi}{2}$

4. **耦合关节对**（策略4）：通过 `Robot.coupled_pairs` 声明
   - 同策略3，但非约束关节取 lo/hi/mid 三种值

5. **三关节和约束**（策略5-6）：通过 `Robot.coupled_triples` 声明
   - $q_a + q_b + q_c = k \cdot \frac{\pi}{2}$

#### 约束流形随机采样（弥补网格点不足）

6. **约束流形随机采样**（策略7）：在约束子流形上连续随机取点
   - 对两关节约束：$q_i$ 随机取值，$q_j = k\pi/2 - q_i$，剩余关节随机
   - 对三关节约束：$q_a, q_b$ 随机取值，$q_c = k\pi/2 - q_a - q_b$

> **关键洞察**：约束点仅使梯度的部分项为零，并非全局极值。
> 以这些约束点为种子做 L-BFGS-B 局部优化，才能找到真正极值。

### 连杆等分 (n_subdivisions)

将每根连杆等分为 $n$ 段，每段独立跟踪 AABB：

$$p(t) = (1-t) \cdot p_\text{start} + t \cdot p_\text{end}$$

FK 只计算两次（始端 + 末端），等分点通过 $O(1)$ 线性插值得到。

```python
result = calc.compute_envelope(
    joint_intervals, sampling='hybrid', n_subdivisions=5,
)
for aabb in result.link_aabbs:
    if not aabb.is_zero_length:
        print(f"{aabb.link_name}: t=[{aabb.t_start:.2f},{aabb.t_end:.2f}] "
              f"V={aabb.volume:.6f}")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_random_samples` | 10000 | random 模式每连杆采样数 |
| `n_refine_samples` | 200 | hybrid 模式每连杆随机补充数 |
| `critical_proximity_threshold` | 0.05 | 随机采样避开关键点的欧氏距离阈值 |
| `n_subdivisions` | 1 | 连杆等分段数 |
| `skip_zero_length` | True | 跳过零长度连杆 |

## 测试

```bash
pytest test/ -v
```

## Planner 模块 (v4.1)

独立路径规划包 `planner`，基于 Box-RRT 算法实现碰撞-free 路径规划。

### 新功能

| 模块 | 功能 | 说明 |
|------|------|------|
| `box_expansion` | **Balanced 拓展策略** | v4.1 默认交替步进策略，box 宽度比中位数从 48 降至 4 |
| `models` | **JSON 配置化** | `PlannerConfig.to_json()` / `from_json()`，预设 2dof/panda 配置文件 |
| `aabb_cache` | AABB 缓存系统 | 双层存储（interval/numerical），v4.1 全局接入 BoxRRT/BoxForest |
| `box_forest` | 可复用 BoxForest | 预构建碰撞-free box 树，支持保存/加载、robot 指纹校验 |
| `box_query` | Forest 查询规划 | 复用 forest 快速规划新查询，局部扩展 + 图搜索 |
| `dynamic_visualizer` | 动态可视化 | FuncAnimation 动画、EE 轨迹、ghost frames、等弧长重采样 |
| `free_space_tiler` | 自由空间瓦片化 | 自适应单维二分切分，识别碰撞-free C-space 区域 |

### 快速示例

```python
from box_aabb import load_robot
from planner import BoxRRT, PlannerConfig, Scene, AABBCache

robot = load_robot('2dof_planar')
scene = Scene()

# 从 JSON 加载配置（或使用默认值）
config = PlannerConfig.from_json("src/planner/configs/2dof_planar.json")
# config = PlannerConfig(expansion_strategy='balanced', use_aabb_cache=True)

# 带缓存的规划
cache = AABBCache()
planner = BoxRRT(robot, scene, config, aabb_cache=cache)
result = planner.plan(start, goal)

# 保存缓存供下次使用
cache.save("cache.pkl")
```

```python
# 动态可视化
from planner import animate_robot_path
anim = animate_robot_path(robot, result.path, scene=scene, fps=30)
anim.save("motion.gif", writer='pillow')
```

详细设计文档见 [doc/zh/planner_design.md](doc/zh/planner_design.md)。

## 性能

| 方法 | 紧凑度 | 安全性 | 速度 |
|------|--------|--------|------|
| critical | 高 | 理论保证 | 最快（~0.015s） |
| hybrid | 最高 | 理论保证 | 较快（~0.1s） |
| random | 高 | 统计保证 | 慢（~0.4s） |
| interval | 较低 | 保守保证 | 最快 |

