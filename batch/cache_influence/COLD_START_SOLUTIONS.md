# 彻底消除缓存影响的解决方案

## 问题分析

即使执行 `drop_caches()`，第一次运行仍然比后续运行慢很多，原因包括：

1. **`drop_caches` 只清文件系统缓存**：不能清除：
   - CPU 缓存（L1/L2/L3、TLB、分支预测器）
   - PyTorch/MKL 等库的全局状态和懒初始化
   - 进程间共享的代码段（多个 Python 进程共享同一份库代码）
   - `~/.cache/torch` 等磁盘缓存文件

2. **第一次 `import torch` 会做一次性初始化**：
   - 动态加载 `.so` 库、符号重定位
   - 初始化 OpenMP/MKL 线程池
   - 探测 CPU 指令集、构建算子注册表
   - 可能编译/加载 JIT kernel

3. **预热阶段的一次性成本**：
   - 内存池的大块分配
   - MKL/BLAS 内部 warmup
   - Kernel 选择/优化

## 解决方案对比

### 方案 1：Docker 容器隔离（推荐 ⭐⭐⭐⭐⭐）

**原理**：每次运行都用全新的容器，容器销毁后所有状态都被清除。

**优点**：
- ✅ **最彻底**：完全隔离进程状态、内存、文件系统
- ✅ **可重复性强**：每次都是相同的初始环境
- ✅ **易于管理**：可以限制资源（CPU、内存）

**缺点**：
- ⚠️ 容器启动有额外开销（~400ms）
- ⚠️ 需要 Docker 环境

**使用方法**：
```bash
# 使用 Docker 模式（默认）
python auto_benchmark_docker.py --mode docker --image torch-cpu

# 或者直接运行（默认就是 docker 模式）
python auto_benchmark_docker.py
```

### 方案 2：增强的 drop_caches（部分有效 ⭐⭐⭐）

**原理**：清理更多缓存，包括用户级缓存目录。

**优点**：
- ✅ 不需要 Docker
- ✅ 开销小

**缺点**：
- ⚠️ 无法清除库的全局状态
- ⚠️ 无法清除 CPU 缓存
- ⚠️ 无法清除进程间共享的代码段

**实现**：可以清理 `~/.cache/torch` 等目录，但效果有限。

### 方案 3：使用 unshare 创建新 namespace（中等效果 ⭐⭐⭐⭐）

**原理**：使用 Linux `unshare` 创建新的 PID/UTS/Mount namespace。

**优点**：
- ✅ 比 Docker 轻量
- ✅ 可以隔离一些状态

**缺点**：
- ⚠️ 仍然共享内核和部分系统资源
- ⚠️ 配置复杂

**示例**：
```bash
unshare -p -f --mount-proc sh -c "taskset -c 0,1 python benchmark.py"
```

### 方案 4：重启 Python 解释器进程（效果有限 ⭐⭐）

**原理**：确保每次都是新的 Python 进程。

**优点**：
- ✅ 简单

**缺点**：
- ⚠️ 无法清除库的全局状态（如果库在进程间共享）
- ⚠️ 无法清除系统级缓存

## 推荐方案

**使用 Docker 容器隔离**（`auto_benchmark_docker.py`），因为：

1. **最彻底**：每次都是全新的环境
2. **可重复**：结果更可靠
3. **易用**：已有 Dockerfile，一键运行

## 使用示例

### Docker 模式（推荐）
```bash
cd /home/yingjiaze/experiment/batch
python auto_benchmark_docker.py --mode docker --image torch-cpu --runs 5
```

### 主机模式（对比用）
```bash
python auto_benchmark_docker.py --mode host --runs 5
```

### 不清理缓存（测试缓存影响）
```bash
python auto_benchmark_docker.py --mode docker --no-drop-cache --runs 5
```

## 预期效果

使用 Docker 模式后，每次运行应该：
- ✅ `import torch` 时间接近（都在 ~4-5s）
- ✅ 预热时间接近（都在 ~50-200ms）
- ✅ 初始化时间稳定（容器启动开销 ~400ms）

如果仍然有差异，可能是：
- 磁盘 I/O 的随机性
- CPU 调度的影响
- 系统负载的影响

这些是正常的系统级波动，Docker 隔离已经消除了**应用级**的状态影响。

