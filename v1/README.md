# BOX-AABB v1

这是BOX-AABB库的初始版本，包含完整的AABB计算和路径规划功能。

## 使用方法

### 进入v1目录

```bash
cd v1
```

### 安装依赖

```bash
pip install -e .
```

或安装开发依赖：

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest test/
```

### 运行示例

```bash
python examples/basic_usage.py
python examples/random_box_demo.py
```

### 运行基准测试

```bash
python benchmarks/bench_panda_multi.py
python benchmarks/threshold_experiment.py
```

### 对比v1和v2性能

```bash
python benchmarks/compare_v1_v2_current.py
```

对比报告会生成在根目录的 `comparison_reports/` 文件夹中。

## 项目结构

```
v1/
├── src/              # 源代码
│   ├── box_aabb/     # AABB计算核心
│   └── planner/      # 路径规划
├── test/             # 测试代码
├── examples/         # 示例脚本
├── benchmarks/       # 性能基准测试
├── doc/              # 文档
├── build/            # 构建输出
├── pyproject.toml    # 项目配置
└── setup_cython.py   # Cython扩展配置
```

## 注意事项

- v1中的所有脚本都假定从v1目录内运行
- 如需编译Cython扩展：`python setup_cython.py build_ext --inplace`
- 对比测试会同时访问v1和v2的代码，确保两个目录都存在
