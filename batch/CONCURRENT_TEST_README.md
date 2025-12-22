# 并发测试说明

## 概述

本测试脚本实现了终端设备上多应用并发实例的性能测试，包括两个测试场景：

1. **测试1：MobileNet多实例测试**
   - 同时启动N=1,2,4,8...个MobileNet容器（每个容器运行1个实例）
   - **所有容器共享总资源限制**（例如：总共2核2GB内存）
   - 测试到内存满为止

2. **测试2：混合应用测试**
   - 同时运行多个不同的应用容器（MobileNet, ResNet, ShuffleNet等）
   - **所有容器共享总资源限制**
   - 不同应用进程的指标分开记录

**资源限制说明**：
- 所有容器共享总CPU和总内存限制，而不是每个容器独立限制
- CPU通过Docker的`--cpus`参数分配（每个容器平均分配）
- 内存通过监控所有容器的总PSS来判断是否达到限制

## 收集的指标

- **容器启动时间**：从docker run到容器启动完成的耗时
- **P95/P99时延**：每次推理时延的95和99百分位数
- **cgroup memory.stat**：容器的cgroup内存统计信息
- **进程PSS**：每个容器进程的实际物理内存使用（Proportional Set Size）
- **抖动检测**：通过变异系数(CV)检测性能抖动

## 文件说明

- `benchmark.py`: MobileNet基准测试脚本（已修改，输出详细时延）
- `benchmark_resnet.py`: ResNet基准测试脚本
- `benchmark_efficientnet.py`: EfficientNet基准测试脚本
- `benchmark_shufflenet.py`: ShuffleNet基准测试脚本
- `concurrent_benchmark.py`: 主并发测试脚本

## 使用方法

### 1. 构建Docker镜像

```bash
cd /home/yingjiaze/experiment/batch
docker build -t torch-cpu .
```

### 2. 运行并发测试

```bash
python concurrent_benchmark.py [选项]
```

#### 选项说明

- `--total-cpus`: 所有容器共享的总CPU核心，如 "0-1" 表示2核，默认 "0-1"
  - 支持格式：`"0-1"`（范围），`"0,1,2"`（列表），`"2"`（单个）
  - 每个容器会平均分配CPU份额（例如：2核/4容器 = 每个容器0.5核）
- `--total-memory-mb`: 所有容器共享的总内存限制(MB)，默认 2048
  - **注意**：不设置单个容器的内存限制，所有容器共享这个总限制
- `--volume`: 卷挂载，默认当前目录:/app
- `--output-dir`: 结果输出基础目录，默认 "results"（会在其下创建时间戳子目录）
- `--max-instances`: 最大实例数，默认 8
- `--warmup-runs`: 预热运行次数，默认 2（实际测试取第3次运行的结果）

**重要说明**：
- 所有容器**共享**总资源限制，而不是每个容器独立限制
- 例如：N=4个容器，`--total-cpus="0-1"`（2核），`--total-memory-mb=2048`（2GB）
  - 每个容器约使用 0.5 核
  - 所有容器总共最多使用 2GB 内存
- CPU使用Docker的`--cpus`参数进行软限制，内存通过监控总PSS来判断是否达到限制

#### 示例

```bash
# 使用默认设置运行
python concurrent_benchmark.py

# 自定义总资源限制（所有容器共享）
python concurrent_benchmark.py --total-cpus "0-3" --total-memory-mb 4096

# 指定输出目录
python concurrent_benchmark.py --output-dir my_results
```

## 输出结果

### 输出目录结构

每次运行实验会在 `results/` 目录下创建一个基于时间的子目录，格式为：
```
results/experiment_YYYYMMDD_HHMMSS/
```

这样可以避免覆盖之前的实验结果。

### 1. 单个实验结果文件

每个测试完成后会立即保存详细的原始数据到单独的文件：
- `test1_mobilenet_n1_result.json`
- `test1_mobilenet_n2_result.json`
- `test2_mixed_*_result.json`
- ...

每个结果文件包含：
- 测试配置信息
- 每个容器的详细信息：
  - 容器ID和名称
  - 启动时间（毫秒）
  - **所有推理的原始时延数据**（`all_latencies_ms`）
  - P95/P99时延
  - 平均/最小/最大时延
  - PSS内存使用
  - PID和memory.stat

### 2. 汇总文件

- `summary.json`: 所有测试的汇总信息
- `experiment_config.json`: 实验配置信息

### 3. 可视化图表

- `test1_mobilenet_multiple_instances.png`: 测试1的可视化结果
  - 容器启动时间 vs 实例数
  - P95/P99时延 vs 实例数
  - PSS内存使用 vs 实例数
  - 抖动检测（CV）vs 实例数
  - 时延分布箱线图

- `test2_mixed_applications.png`: 测试2的可视化结果
  - 各应用的P95/P99时延对比
  - 各应用的PSS内存对比
  - 各应用的启动时间对比
  - 各应用的时延分布对比
  - 各应用的抖动检测

## PSS内存获取说明

PSS (Proportional Set Size) 是通过读取 `/proc/<pid>/smaps` 文件计算得到的。
该方法参考了 [这篇博客](https://blog.csdn.net/m0_51504545/article/details/119685325)。

## 抖动检测

抖动检测使用变异系数(Coefficient of Variation, CV)来衡量：
- CV = 标准差 / 平均值
- CV > 0.3 认为存在明显抖动

## 预热机制

每个测试在正式记录结果前会先进行预热运行（默认2次），第3次运行的结果才会被记录。这可以：
- 消除冷启动的影响
- 确保系统和Docker容器已经稳定
- 获得更准确的性能数据

可以通过 `--warmup-runs` 参数调整预热次数。

## 数据保存策略

- **立即保存**：每个测试完成后立即保存详细的原始数据到单独的文件
- **原始数据**：保存所有推理的时延数据（`all_latencies_ms`），便于后续分析
- **时间戳目录**：每次运行创建独立的输出目录，避免覆盖之前的实验结果

## 注意事项

1. 需要root权限或docker权限来访问 `/proc/<pid>/smaps` 和cgroup文件
2. 确保Docker镜像 `torch-cpu` 已构建
3. 测试过程会创建和删除多个Docker容器（包括预热容器）
4. 内存满的判断基于所有容器的PSS总和
5. 预热运行会增加测试时间，但能提供更准确的结果

