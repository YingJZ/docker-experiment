# Native 版本 init_ms 异常大的原因分析

## 一、init_ms 的计算逻辑

### 代码分析

从 `concurrent_benchmark.py` 可以看出 init_ms 的计算方式：

**步骤1：记录启动命令的时间（第375行）**
```python
start_cmd_time = time.time()  # 主进程调用 systemd-run 的时刻
proc = subprocess.Popen(cmd, ...)  # 启动 systemd-run 命令
```

**步骤2：benchmark.py 开始执行时记录时间（benchmark.py 第53-54行）**
```python
start_time = time.time()  # Python 脚本刚开始执行的时刻
print(f"start time: {start_time:.3f}")
```

**步骤3：计算 init_ms（concurrent_benchmark.py 第532行）**
```python
timing['init_ms'] = (start_time - start_cmd_time) * 1000.0
```

### 时间线示意图

```
时间点A: start_cmd_time 
         (主进程调用 systemd-run)
         │
         ├─ [等待 systemd 处理请求]
         ├─ [systemd 创建 scope unit]
         ├─ [设置 cgroup（AllowedCPUs, MemoryMax）]
         ├─ [taskset 设置 CPU 亲和性]
         ├─ [启动 Python 解释器]
         ├─ [Python 解释器初始化]
         ├─ [加载 benchmark.py 脚本]
         │
时间点B: start_time
         (benchmark.py 开始执行，打印 start time)
         │
         ├─ [import torch - import_ms]
         ├─ [load model - load_ms]
         ├─ [warmup - warmup_ms]
         │
         └─ [inference loop]

init_ms = 时间点B - 时间点A
```

## 二、理论上 init_ms 应该包含什么？

根据代码逻辑，`init_ms` **理论上**应该只包含：

1. ✅ systemd-run 命令启动开销
2. ✅ systemd 创建 scope unit 的时间
3. ✅ 设置 cgroup 限制（CPU、内存）的时间
4. ✅ taskset 设置 CPU 亲和性的时间
5. ✅ Python 解释器启动时间
6. ✅ Python 脚本加载和执行到第53行的时间

**不应该包含**：
- ❌ import torch 的时间（在 start_time 之后）
- ❌ load model 的时间（在 start_time 之后）
- ❌ warmup 的时间（在 start_time 之后）

## 三、实际数据的异常现象

### 单进程情况（n=1）

```
init_ms = 2729.79ms
import_ms + load_ms + warmup_ms = 2345 + 132 + 58 = 2535ms
差值 = 195ms
```

这个结果基本合理：
- 195ms 的差值是 systemd-run + Python 启动 + 脚本加载的开销
- init_ms 确实不包含 import/load/warmup

### 高并发情况（n=16）

```
init_ms = 17815.89ms (平均)
import_ms + load_ms + warmup_ms = 15706 + 1158 + 815 = 17679ms
差值 = 137ms
```

**异常现象：**
1. init_ms 增长了 6.5 倍（2730ms → 17816ms）
2. import_ms 增长了 6.7 倍（2345ms → 15706ms）
3. 两者增长模式几乎一致！

## 四、问题的根本原因：并发启动时的串行化和资源竞争

### 原因1：systemd 处理请求的串行化

当并发启动 16 个进程时：

```python
# concurrent_benchmark.py 第583-589行
for i in range(count):
    t = threading.Thread(target=run_process_task, ...)
    t.start()  # "并发"启动
```

虽然 Python 代码是并发启动的，但实际上：

**systemd 的处理是有限并发或串行的：**
- systemd-run 需要与 systemd daemon 通信
- systemd 需要创建 scope unit
- 设置 cgroup 参数需要文件系统操作
- 这些操作有全局锁或有限的并发能力

**结果：后启动的进程需要排队等待！**

```
进程0: [systemd处理] → 启动
进程1: [等待...systemd处理] → 启动
进程2: [等待...systemd处理] → 启动
...
进程15: [等待...systemd处理] → 启动  ← 等待时间最长！
```

### 原因2：cgroup 文件系统的串行化写入

代码中有大量的 cgroup 设置操作（concurrent_benchmark.py 第252-290行）：

```python
# 创建 slice
subprocess.run(['systemctl', 'start', slice_name], ...)

# 设置 CPU 限制
subprocess.run(['systemctl', 'set-property', slice_name, 
                f'AllowedCPUs={allowed_cpus}'], ...)

# 设置内存限制
subprocess.run(['systemctl', 'set-property', slice_name, 
                f'MemoryMax={memory_bytes}'], ...)

# 手动写入 cgroup 文件
with open(cpuset_path, 'w') as f:
    f.write(allowed_cpus)
```

这些操作涉及：
- 文件系统写入（/sys/fs/cgroup/）
- 内核 cgroup 子系统的处理
- 可能存在内核级别的锁竞争

### 原因3：验证和等待操作

代码中还有多处验证操作会增加延迟：

```python
# 第390行：等待 systemd-run 启动
time.sleep(0.5)

# 第393-402行：通过 systemctl 获取 PID
result = subprocess.run(['systemctl', 'show', scope_name, ...], timeout=5)

# 第417-425行：验证 CPU 亲和性
result = subprocess.run(['taskset', '-p', str(pid)], timeout=2)
```

当 16 个进程并发时，这些操作会累积延迟。

## 五、为什么 init_ms 和 import_ms 增长模式相似？

这是最关键的观察！两者都受到**并发启动时的系统资源竞争**影响：

### init_ms 的增长因素：
1. systemd 处理请求排队
2. cgroup 配置串行化
3. Python 解释器启动时的系统调用竞争

### import_ms 的增长因素：
1. 文件系统缓存竞争（多个进程同时读取 .so 和 .py 文件）
2. 动态链接器的全局锁
3. I/O 带宽限制

两者都在并发场景下遇到了**系统级别的瓶颈**！

## 六、数据验证

让我们看看并发数增加时各个指标的增长倍数：

| 指标 | n=1 | n=16 | 增长倍数 | 说明 |
|------|-----|------|---------|------|
| init_ms | 2730ms | 17816ms | **6.5x** | 受 systemd 串行化影响 |
| import_ms | 2345ms | 15706ms | **6.7x** | 受文件系统竞争影响 |
| load_ms | 132ms | 1158ms | **8.8x** | 模型文件读取竞争 |
| warmup_ms | 58ms | 815ms | **14x** | CPU 竞争（所有进程共享 2 个核心） |

**关键洞察：**
- init_ms 和 import_ms 的增长倍数接近（6.5x vs 6.7x）
- 这不是巧合，而是因为它们都受到**并发启动时的系统资源竞争**影响
- warmup_ms 增长最快（14x），因为 CPU 是最严重的瓶颈（16个进程争抢2个核心）

## 七、与容器版本的对比

### 容器版本的 init_ms 为什么小？

容器版本的 init_ms（486ms @ n=1）只包含：
- Docker 容器创建时间
- 网络命名空间建立
- cgroup 初始化

**不包含容器内的 Python 启动、import、load、warmup！**

### 裸机版本的 init_ms 为什么大？

虽然代码设计上 init_ms 也不应该包含 import/load/warmup，但实际上：

1. **单进程时**：init_ms 比 import+load+warmup 多 195ms，符合预期
2. **高并发时**：init_ms 受到 systemd 串行化影响，等待时间被计入

实际上，init_ms 大的主要原因是：
- **并发启动时的排队等待时间**
- systemd-run 需要串行或半串行处理多个请求

## 八、结论

### Native 的 init_ms 这么大的根本原因：

1. **并发启动的串行化瓶颈**
   - systemd 处理多个 systemd-run 请求时有限并发
   - cgroup 文件系统操作有串行化特性
   - 后启动的进程需要排队等待

2. **系统资源竞争**
   - 多个进程同时启动时竞争系统调用
   - Python 解释器启动时的文件系统访问竞争
   - 动态链接器的全局锁竞争

3. **测量方式包含了等待时间**
   - init_ms 测量的是从"调用 systemd-run"到"Python 脚本开始执行"
   - 这个时间包含了所有的排队和等待时间
   - 不仅是"启动时间"，更是"启动+排队等待时间"

### 为什么看起来 init_ms 应该只包括 Python 解释器初始化？

这是一个误解！从代码可以看出：

```python
init_ms = (benchmark.py 开始执行的时间戳) - (调用 systemd-run 的时间戳)
```

这个时间包含了**从命令提交到进程实际开始运行**的全过程，而不仅仅是 Python 解释器初始化。

在高并发场景下，这个过程中的大部分时间可能都花在了**等待 systemd 处理请求**上！

### 优化建议

如果要减少 native 的 init_ms：

1. **避免使用 systemd-run**
   - 直接启动进程，手动设置 CPU 亲和性（使用 taskset 或 Python 的 os.sched_setaffinity）
   - 手动设置 cgroup（预先创建 cgroup 目录）

2. **预创建资源**
   - 提前创建所有 slice 和 cgroup
   - 减少运行时的动态配置

3. **批量操作**
   - 将多个 systemd 操作合并
   - 减少与 systemd daemon 的通信次数
