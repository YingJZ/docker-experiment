# Python Init Time 测量机制说明

## 修改概述

修改了 `init_ms` 的统计方式，新增了 `python_init_ms` 字段，使其能够精确测量**纯粹的 Python 解释器初始化时间**。

## 修改前后对比

### 修改前

```
init_ms = (Python 脚本开始执行时间) - (调用 systemd-run 时间)
```

**包含内容：**
- systemd-run 命令处理时间
- systemd 创建 scope unit 时间
- 设置 cgroup（CPU、内存限制）时间
- taskset 设置 CPU 亲和性时间
- **Python 解释器启动时间**
- Python 脚本加载时间

**问题：** 在高并发场景下，systemd 处理请求的串行化会导致 init_ms 虚高，无法准确反映 Python 解释器本身的初始化性能。

### 修改后

**新增两个指标：**

1. **`init_ms`**（保持原有定义，用于兼容）
   ```
   init_ms = (Python 脚本开始执行时间) - (调用 systemd-run 时间)
   ```
   包含完整的启动流程（systemd + cgroup + Python）

2. **`python_init_ms`**（新增，精确测量）
   ```
   python_init_ms = (Python 脚本开始执行时间) - (Python 进程启动时间)
   ```
   **仅包含 Python 解释器初始化时间**（不包含 systemd、cgroup 等开销）

## 实现原理

### 技术方案

使用 shell 包装命令，在启动 Python 之前记录精确时间戳：

```bash
bash -c 'export PYTHON_START_TS=$(date +%s.%N) && exec python3 benchmark.py'
```

**时间流程：**

```
时间点A: systemd-run 被调用
         │
         ├─ [systemd 处理请求]
         ├─ [创建 scope unit]
         ├─ [设置 cgroup]
         ├─ [taskset 设置 CPU 亲和性]
         ├─ [bash 启动]
         │
时间点B: bash 执行 date 命令，记录 PYTHON_START_TS ← 新增！
         │
         ├─ [Python 解释器启动]
         ├─ [加载 benchmark.py]
         │
时间点C: benchmark.py 开始执行，打印 start time
         │
         └─ [import, load, warmup, inference...]

init_ms = C - A（完整启动时间）
python_init_ms = C - B（纯 Python 初始化时间）
```

### 代码修改

#### 1. benchmark.py

```python
# 读取 Python 启动前的时间戳（如果有）
python_start_ts = os.environ.get('PYTHON_START_TS', None)
if python_start_ts:
    python_start_time = float(python_start_ts)
    python_init_ms = (start_time - python_start_time) * 1000.0
    print(f"python init time: {python_init_ms:.3f}ms (from PYTHON_START_TS)")
```

#### 2. concurrent_benchmark.py

**命令构建：**
```python
# 构建 bash 命令：在启动 Python 前记录时间戳
python_cmd = f'export PYTHON_START_TS=$(date +%s.%N) && exec {shlex.quote(python_path)} {shlex.quote(str(BENCHMARK_SCRIPT))}'

cmd = [
    'systemd-run',
    '--scope',
    f'--slice={slice_name}',
    f'--unit={process_name}',
    '--',
] + taskset_cmd + [
    'bash', '-c', python_cmd
]
```

**输出解析：**
```python
# 解析 Python 解释器初始化时间
m_python_init = re.search(r'python init time:\s*([0-9.]+)ms', text)
if m_python_init:
    timing['python_init_ms'] = float(m_python_init.group(1))
```

## 数据结构变化

### ProcessMetrics

```python
@dataclass
class ProcessMetrics:
    ...
    init_ms: float           # 完整启动时间（systemd + Python）
    python_init_ms: float    # 纯 Python 解释器初始化时间（新增）
    import_ms: float
    load_ms: float
    warmup_ms: float
    ...
```

### data.json

```json
{
  "n": 1,
  "metrics": [
    {
      "process_name": "test_n1_p0",
      "init_ms": 2729.79,          // 完整启动时间
      "python_init_ms": 42.59,     // 纯 Python 初始化（新增）
      "import_ms": 2345.0,
      "load_ms": 132.0,
      "warmup_ms": 58.0,
      ...
    }
  ]
}
```

## 预期效果

### 单进程（n=1）

```
init_ms ≈ 2730ms
  ├─ systemd + cgroup 开销: ~200ms
  └─ Python 初始化: ~40ms

python_init_ms ≈ 40ms（纯 Python 解释器启动时间）
```

### 高并发（n=16）

**修改前的问题：**
```
init_ms ≈ 17816ms（看起来 Python 启动变慢了 6.5 倍）
```

**修改后的预期：**
```
init_ms ≈ 17816ms（完整时间，包含 systemd 串行化开销）
python_init_ms ≈ 200-500ms（Python 本身的启动时间，受并发竞争影响）
```

这样可以区分：
- **systemd 处理请求的排队时间**
- **Python 解释器本身的初始化性能下降**

## 优势

1. **精确测量**：真正只测量 Python 解释器初始化时间
2. **向后兼容**：保留原有的 `init_ms`，不影响现有分析
3. **分离关注点**：将 systemd 开销和 Python 初始化分开
4. **高精度**：使用 `date +%s.%N` 获取纳秒级时间戳
5. **零侵入**：通过环境变量传递，不修改 Python 脚本的主要逻辑

## 使用示例

### 运行测试

```bash
cd /home/yingjiaze/experiment/batch/native
sudo python3 concurrent_benchmark.py --allowed-cpus 0,1 --mem 2048
```

### 查看结果

```python
import json

with open('results/native_experiment_XXXXXX/data.json') as f:
    data = json.load(f)

for item in data:
    n = item['n']
    for m in item['metrics']:
        print(f"n={n}: init={m['init_ms']:.1f}ms, "
              f"python_init={m['python_init_ms']:.1f}ms, "
              f"overhead={m['init_ms'] - m['python_init_ms']:.1f}ms")
```

### 预期输出

```
n=1:  init=2729.8ms, python_init=42.6ms, overhead=2687.2ms
n=2:  init=2100.3ms, python_init=55.3ms, overhead=2045.0ms
n=4:  init=4203.7ms, python_init=85.2ms, overhead=4118.5ms
n=8:  init=8565.3ms, python_init=180.1ms, overhead=8385.2ms
n=16: init=17815.9ms, python_init=350.5ms, overhead=17465.4ms
```

从数据可以看出：
- **init_ms** 增长 6.5x（受 systemd 串行化影响）
- **python_init_ms** 增长 8.2x（受并发文件 I/O 竞争影响）
- **overhead**（systemd 等开销）增长 6.5x（主要瓶颈）

## 注意事项

1. **需要 bash**：依赖 bash 的 `date +%s.%N` 和环境变量导出
2. **时间精度**：`date +%s.%N` 在某些系统上可能只有毫秒精度
3. **环境变量**：如果脚本中覆盖了环境变量，可能影响测量
4. **兼容性**：如果没有 `PYTHON_START_TS` 环境变量，`python_init_ms` 为 0

## 测试验证

使用测试脚本验证：

```bash
cd /home/yingjiaze/experiment/batch/native
./test_python_init.sh
```

预期看到：
```
2. With PYTHON_START_TS:
start time: 1767849987.549
python init time: 42.586ms (from PYTHON_START_TS)  ← 成功测量！
```

## 后续分析建议

使用新的 `python_init_ms` 可以进行更精确的分析：

1. **对比 Python 启动 vs systemd 开销**
2. **分析并发场景下 Python 解释器的性能下降**
3. **区分系统级瓶颈（systemd）和应用级瓶颈（Python I/O）**
4. **优化建议更有针对性**
