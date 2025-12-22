# 实验结果绘图脚本说明

## 概述

`plot_results.py` 是一个独立的脚本，用于根据实验数据生成可视化图表。

## 使用方法

### 基本用法

```bash
python plot_results.py <实验结果目录>
```

例如：
```bash
# 绘制指定实验结果目录的图表
python plot_results.py results/experiment_20251222_113639

# 指定输出目录（默认输出到结果目录）
python plot_results.py results/experiment_20251222_113639 --output-dir my_plots
```

### 参数说明

- `result_dir`: **必需**。包含实验结果的目录路径（例如：`results/experiment_YYYYMMDD_HHMMSS`）
- `--output-dir`: **可选**。图表输出目录（默认与结果目录相同）

### 依赖

需要安装 matplotlib：
```bash
pip install matplotlib
```

## 生成的图表

脚本会在输出目录生成以下图表：

### 测试1：MobileNet多实例测试

生成文件：`test1_mobilenet_multiple_instances.png`

包含6个子图：
1. **容器启动时间** vs 实例数
2. **P95时延** vs 实例数
3. **P99时延** vs 实例数
4. **PSS内存使用** vs 实例数（平均和总计）
5. **抖动检测（CV）** vs 实例数
6. **时延分布箱线图** vs 实例数

### 测试2：混合应用测试

生成文件：`test2_mixed_applications.png`

包含6个子图：
1. **各应用的P95时延对比**
2. **各应用的P99时延对比**
3. **各应用的PSS内存对比**
4. **各应用的启动时间对比**
5. **各应用的时延分布对比**（箱线图）
6. **各应用的抖动检测对比**

## 示例

```bash
# 绘制最新的实验结果
python plot_results.py results/experiment_20251222_113639

# 输出：
# Loading results from: results/experiment_20251222_113639
# Found 4 test1 results and 6 test2 results
# 
# Plotting test1 results...
# Test1 plots saved to results/experiment_20251222_113639/test1_mobilenet_multiple_instances.png
# 
# Plotting test2 results...
# Test2 plots saved to results/experiment_20251222_113639/test2_mixed_applications.png
# 
# All plots saved to: results/experiment_20251222_113639
```

## 注意事项

1. 脚本会自动查找结果目录中所有 `*_result.json` 文件
2. `test1_*` 开头的文件用于测试1的图表
3. `test2_*` 开头的文件用于测试2的图表
4. 如果某个测试没有数据，对应的图表将被跳过

