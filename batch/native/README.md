# Native Concurrent Benchmark

不使用 Docker 容器，直接在主机上并发运行多个 PyTorch 推理进程的基准测试。
使用 Systemd Slice 进行资源限制（CPU 亲和性 + 内存上限），与容器版本形成对比。

## 与容器版本的区别

| 特性 | 容器版本 | Native 版本 |
|------|---------|-------------|
| 隔离方式 | Docker 容器 | Systemd Scope/Slice |
| 启动开销 | 较高（容器创建） | 较低（直接进程） |
| 资源限制 | cgroup (via Docker) | cgroup (via systemd) |
| 文件系统 | 容器 overlay | 主机文件系统 |
| 网络 | 容器网络 | 主机网络 |

## 使用方法

```bash
# 基本用法（需要 root 权限）
sudo /path/to/venv/bin/python concurrent_benchmark.py

# 完整参数
sudo /path/to/venv/bin/python concurrent_benchmark.py \
  --allowed-cpus 0,1 \       # 指定允许使用的 CPU 核心
  --mem 2048 \               # 内存上限 (MB)
  --python /path/to/python   # 可选：指定 Python 解释器
```

## 监控

```bash
# 查看 Slice 状态
systemctl status native_bench_*.slice

# 查看内存使用
watch -n 1 'systemctl show native_bench_*.slice -p MemoryCurrent -p MemoryMax'

# 查看 CPU 亲和性
systemctl show native_bench_*.slice -p AllowedCPUs
```

## 清理

```bash
# 使用清理脚本
sudo ./cleanup.sh

# 或手动清理
sudo systemctl list-units --type=slice | grep native_bench | awk '{print $1}' | xargs systemctl stop
```

## 输出

结果保存在 `results/native_experiment_YYMMDD_HHMMSS/` 目录：
- `data.json`: 详细测试数据
- `native_scaling_result.png`: 性能曲线图
