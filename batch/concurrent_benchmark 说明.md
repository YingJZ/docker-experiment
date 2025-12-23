# concurrent_benchmark.py 实现说明

## 问题&解决方法

### CPUQuota vs AllowedCPUs

**问题背景：** 使用 `systemd-run --property=CPUQuota=xx%` 限制 Slice 的 CPU 使用率时，发现其限制的是总体使用率，即线程**仍然可以使用所有 CPU 核心**，只是总使用率被限制了。这会导致线程频繁在核心间切换，影响测试结果的准确性。

**解决方案：** 改用 `AllowedCPUs` (cgroup cpuset) 限制 Slice 可用的 CPU 核心集合，所有容器共享并竞争这些固定核心，避免跨核切换。

**优雅退出机制：**
- 脚本注册了 `SIGINT` (Ctrl+C) 和 `SIGTERM` 信号处理器
- 收到中断信号时，自动清理所有活跃容器和 Slice
- 使用 `atexit` 确保正常退出时也执行清理
- 全局追踪所有创建的资源（`ACTIVE_SLICES`, `ACTIVE_CONTAINERS`）

### 线程数设置


## 命令行用法

```bash
# 基本用法（需要 root 权限）
sudo python3 concurrent_benchmark.py

# 完整参数
sudo python3 concurrent_benchmark.py \
  --allowed-cpus 0,1 \       # 指定允许使用的 CPU 核心（逗号分隔或范围），默认0,1
  --mem 4096 \               # Slice 内存上限（MB），默认2048MB
  --volume /path/to/batch:/app  # 可选：自定义挂载卷,默认当前目录
```

## 监控 Slice 资源使用

### 查看 Slice 内存使用情况

在脚本运行期间，可在另一终端执行以下命令实时监控 Slice 的内存使用：

```bash
# 方法 1：使用 systemctl 查看属性
systemctl show bench_test_n*.slice -p MemoryCurrent -p MemoryMax

# 方法 2：直接读取 cgroup 文件（更精确）
# 找到 Slice 的 cgroup 路径
systemctl show bench_test_n10_*.slice -p ControlGroup

# 查看内存使用（需要 root 权限）
cat /sys/fs/cgroup/memory/<slice_path>/memory.current   # 当前使用量（字节）
cat /sys/fs/cgroup/memory/<slice_path>/memory.max       # 内存上限
cat /sys/fs/cgroup/memory/<slice_path>/memory.stat      # 详细统计

# 方法 3：实时监控（推荐）
watch -n 1 'systemctl show bench_test_n*.slice -p MemoryCurrent -p MemoryMax'

# 方法 4：查看 Slice 状态（含内存和进程信息）
systemctl status bench_test_n*.slice
```

### 手动清理资源

如果脚本异常中断（Ctrl+C 或进程被杀死），可能会留下容器和 Slice：

```bash
# 使用提供的清理脚本（推荐）
sudo ./cleanup_benchmark.sh

# 或手动清理

# 1. 清理测试容器
docker rm -f $(docker ps -a --filter name=test_n -q)

# 2. 停止测试 Slice
systemctl list-units --type=slice | grep bench_test_n | awk '{print $1}' | xargs systemctl stop

# 3. 清理孤儿容器
docker container prune -f
```

### 查看 CPU 亲和性配置

```bash
# 查看 Slice 的 AllowedCPUs 配置
systemctl show bench_test_n*.slice -p AllowedCPUs

# 查看容器实际使用的 CPU
docker inspect <container_name> --format '{{.HostConfig.CpusetCpus}}'

# 查看进程实际运行的 CPU（需要 PID）
taskset -cp <pid>
```

### 查看所有容器状态

```bash
# 列出测试相关的容器
docker ps -a --filter name=test_n

# 实时监控容器资源
docker stats $(docker ps --filter name=test_n -q)

# 查看特定容器日志
docker logs test_n10_c0
```

## 核心函数说明

### `run_concurrent_test`

主要测试逻辑流程：

1. 调用 `create_resource_limited_slice()` 创建资源受限的 Slice
2. 为所有容器生成统一的 CPU 亲和性（共享竞争模式）
3. 启动多个线程，每个线程运行 `run_container_task()` 函数
4. 在所有线程完成后，清理 Slice
5. 收集并返回所有线程的结果


### `create_resource_limited_slice`

创建一个 Systemd Slice，并设置 `AllowedCPUs`（cpuset）与 `MemoryMax`。所有容器将被置于该 Slice 下，实现共享/竞争同一核心集合。

```python
# 创建 Slice 单元
subprocess.run(['systemctl', 'start', slice_name], check=True, capture_output=True)

# 设置 cpuset（CPU 核心限制）
subprocess.run(['systemctl', 'set-property', slice_name, 
                f'AllowedCPUs={allowed_cpus}'], check=True, capture_output=True)

# 设置内存上限
subprocess.run(['systemctl', 'set-property', slice_name, 
                f'MemoryMax={memory_bytes}'], check=True, capture_output=True)
```

**关键变化：**
- **移除了 `CPUQuota`**：不再限制总使用率百分比
- **改用 `AllowedCPUs`**：直接限制可用的 CPU 核心集合
- **使用 `systemctl set-property`**：动态配置 Slice 属性


### `run_container_task`

实际运行的指令 `docker_cmd` 包含以下几部分：

```python
docker_cmd = [
    'docker', 'run', '-d',
    '--name', container_name,
    '-v', volume,
    '-e', f'OMP_NUM_THREADS={effective_threads}',
    '-e', f'MKL_NUM_THREADS={effective_threads}',
    '-e', f'PYTORCH_NUM_THREADS={effective_threads}',
    '--cgroup-parent', slice_name,  # 将容器置于 Slice 下
    '--cpuset-cpus', cpuset_cpus,   # 容器级 CPU 亲和性（与 Slice 相同）
    'torch-cpu', 'python', script_name
]
```

**关键点：**
- `--cgroup-parent`：将容器纳入 Slice 的 cgroup 层级，使其受 Slice 资源限制
- `--cpuset-cpus`：容器级 CPU 绑定，与 Slice 的 `AllowedCPUs` 保持一致
- 环境变量：限制容器内线程数，防止过度订阅

主要逻辑：
1. 启动容器 `subprocess.run(docker_cmd, ...)`
2. 每 0.5 秒通过 `docker inspect` 获取容器 PID，进而读取 `/proc/{pid}/smaps` 查询 PSS
3. 容器运行结束后，通过 `docker logs` 获取输出结果，解析各阶段耗时
