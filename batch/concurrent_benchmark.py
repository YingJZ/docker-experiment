#!/usr/bin/env python3
"""
优化版并发测试脚本
主要修复：
1. 限制 PyTorch 线程数，防止 CPU 过载和上下文切换风暴。
2. 减少监控循环中的 subprocess 调用频率，降低系统态 CPU 开销。
"""

import subprocess
import time
import json
import os
import re
import statistics
import threading
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- 配置常量 ---
# 默认每个容器内部限制的线程数。
# 对于高并发吞吐测试，建议设为 1，避免线程竞争。
DEFAULT_THREADS_PER_CONTAINER = 2
INSTANCE_COUNTS = [1, 2, 4, 8, 10, 12, 14, 16]


@dataclass
class ContainerMetrics:
    container_id: str
    container_name: str
    app_name: str
    start_time: float
    start_duration: float
    init_ms: float
    import_ms: float
    load_ms: float
    warmup_ms: float
    latencies: List[float]
    p95_latency: float
    p99_latency: float
    memory_stat: Dict[str, int]
    pss_mb: float
    pid: Optional[int] = None

@dataclass
class TestResult:
    test_name: str
    num_instances: int
    app_config: List[str]
    container_metrics: List[ContainerMetrics]
    timestamp: float
    memory_full: bool

def get_process_pss(pid: int) -> float:
    """获取进程 PSS 内存 (MB)"""
    try:
        smaps_path = f"/proc/{pid}/smaps"
        if not os.path.exists(smaps_path):
            return 0.0
        
        pss_total = 0
        with open(smaps_path, 'r') as f:
            for line in f:
                if line.startswith('Pss:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        pss_total += int(parts[1])
        return pss_total / 1024.0
    except Exception:
        return 0.0

def get_container_pid(container_id: str) -> Optional[int]:
    """获取容器主进程 PID"""
    try:
        result = subprocess.run(
            ['docker', 'inspect', '--format', '{{.State.Pid}}', container_id],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except Exception:
        return None

def get_cgroup_memory_stat(container_id: str) -> Dict[str, int]:
    """获取 cgroup memory.stat"""
    try:
        # 简化版：优先尝试直接通过 docker stats 获取摘要，或者寻找常见路径
        # 这里保留原有逻辑的健壮性，但为了代码简洁略微折叠
        result = subprocess.run(
            ['docker', 'inspect', '--format', '{{.Id}}', container_id],
            capture_output=True, text=True, check=True
        )
        container_hash = result.stdout.strip()
        
        possible_paths = [
            f"/sys/fs/cgroup/memory/docker/{container_hash}/memory.stat",
            f"/sys/fs/cgroup/memory/system.slice/docker-{container_hash}.scope/memory.stat",
            f"/sys/fs/cgroup/docker/{container_hash}/memory.stat",
        ]
        
        # 尝试通过 CgroupParent 获取
        try:
            cgroup_res = subprocess.run(
                ['docker', 'inspect', '--format', '{{.HostConfig.CgroupParent}}', container_id],
                capture_output=True, text=True
            )
            if cgroup_res.returncode == 0 and cgroup_res.stdout.strip():
                possible_paths.insert(0, f"{cgroup_res.stdout.strip()}/memory.stat")
        except Exception:
            pass
        
        for path in possible_paths:
            if os.path.exists(path):
                stat = {}
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                stat[parts[0]] = int(parts[1])
                            except ValueError:
                                pass
                return stat
    except Exception:
        pass
    return {}

def parse_benchmark_output(output: str, docker_start_time: float) -> Tuple[float, List[float], Dict[str, float]]:
    """解析输出"""
    lines = output.strip().split('\n')
    text = '\n'.join(lines)
    
    start_time = None
    latencies = []
    timing = {'init_ms': 0.0, 'import_ms': 0.0, 'load_ms': 0.0, 'warmup_ms': 0.0}
    
    m_start = re.search(r'start time:\s*([0-9.]+)', text)
    if m_start:
        start_time = float(m_start.group(1))
        timing['init_ms'] = (start_time - docker_start_time) * 1000.0
    
    patterns = {
        'import_ms': r'Import Torch Done, Time Spent:\s*([0-9.]+)s',
        'load_ms': r'Model Loaded, Time Spent:\s*([0-9.]+)s',
        'warmup_ms': r'Warmup Done, Time Spent:\s*([0-9.]+)s'
    }
    
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            timing[key] = float(m.group(1)) * 1000.0
            
    for line in lines:
        if line.startswith('LATENCIES:'):
            parts = line.replace('LATENCIES:', '').strip().split(',')
            latencies = [float(x) for x in parts if x.strip()]
            break
            
    return start_time, latencies, timing

def create_resource_limited_slice(test_name: str, total_cpus: float, total_memory_mb: int) -> Optional[str]:
    """创建限制资源的 systemd slice"""
    slice_name = f"benchmark-{test_name}-{int(time.time())}"
    slice_unit = f"{slice_name}.slice"
    
    cpu_percent = int(total_cpus * 100)
    memory_bytes = total_memory_mb * 1024 * 1024
    
    try:
        # 使用 systemd-run 启动一个占位 sleep 进程来创建 slice
        subprocess.run(
            ['systemd-run', '--slice', slice_unit,
             f'--property=CPUQuota={cpu_percent}%',
             f'--property=MemoryMax={memory_bytes}',
             'sleep', '3600'],
            capture_output=True, timeout=5
        )
        return slice_name
    except Exception as e:
        print(f"Warning: Failed to create slice: {e}")
        return None

def run_container(app_name: str, container_name: str, 
                  scope_name: Optional[str], volume: str, script_name: str,
                  threads_limit: int = 1) -> Tuple[str, float, str, Dict, float]:
    """
    运行容器并收集指标。
    **优化点**：注入线程限制环境变量，降低轮询频率。
    """
    start_cmd_time = time.time()
    runtime_metrics = {}
    
    # 核心优化：限制容器内 PyTorch/OpenMP 的线程数
    # 防止 N 个容器 x M 个核 造成的线程爆炸
    env_vars = [
        '-e', f'OMP_NUM_THREADS={threads_limit}',
        '-e', f'MKL_NUM_THREADS={threads_limit}',
        '-e', f'PYTORCH_NUM_THREADS={threads_limit}',
        # 也可以限制 Intra/Inter op threads
        '-e', f'TORCH_INTRA_OP_PARALLELISM={threads_limit}',
        '-e', f'TORCH_INTER_OP_PARALLELISM={threads_limit}'
    ]
    
    docker_base = [
        'docker', 'run', '-d',
        '--name', container_name,
        '-v', volume
    ] + env_vars + [
        'torch-cpu',
        'python', script_name
    ]
    
    if scope_name:
        cmd = ['systemd-run', '--scope', '--slice', scope_name.replace('.scope', '.slice'), '--'] + docker_base
    else:
        cmd = docker_base
        
    try:
        # 1. 启动容器
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = res.stdout.strip()
        start_duration = (time.time() - start_cmd_time) * 1000
        
        # 2. 监控循环 (优化后)
        max_wait_time = 300
        wait_start = time.time()
        metrics_collected = False
        
        while time.time() - wait_start < max_wait_time:
            # 检查运行状态
            check_res = subprocess.run(
                ['docker', 'inspect', '--format', '{{.State.Running}}', container_id],
                capture_output=True, text=True
            )
            
            is_running = (check_res.returncode == 0 and check_res.stdout.strip() == 'true')
            
            if not is_running:
                # 容器已退出（可能运行完成，也可能出错）
                break
                
            # 容器正在运行，且尚未收集指标 -> 收集一次
            if is_running and not metrics_collected:
                pid = get_container_pid(container_id)
                if pid:
                    runtime_metrics['pid'] = pid
                    runtime_metrics['pss_mb'] = get_process_pss(pid)
                    runtime_metrics['memory_stat'] = get_cgroup_memory_stat(container_id)
                    metrics_collected = True # 标记已收集，后续循环不再执行昂贵操作
            
            # 增加睡眠时间，减少 subprocess 调用带来的 CPU System 开销
            time.sleep(1.0) 
            
        # 3. 等待容器完全结束并获取日志
        subprocess.run(['docker', 'wait', container_id], capture_output=True, timeout=60)
        logs = subprocess.run(['docker', 'logs', container_id], capture_output=True, text=True)
        
        return container_id, start_duration, logs.stdout, runtime_metrics, start_cmd_time
        
    except Exception as e:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)
        raise e

def save_result_to_file(result: TestResult, output_dir: str):
    """保存结果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)
    data = {
        'test_name': result.test_name,
        'num_instances': result.num_instances,
        'metrics': [
            {
                'container_name': m.container_name,
                'start_duration_ms': m.start_duration,
                'init_ms': m.init_ms,
                'p95_latency_ms': m.p95_latency,
                'pss_mb': m.pss_mb,
                'all_latencies': m.latencies
            } for m in result.container_metrics
        ]
    }
    with open(os.path.join(output_dir, f"{result.test_name}.json"), 'w') as f:
        json.dump(data, f, indent=2)

def parse_cpus(cpus_str: str) -> float:
    if '-' in cpus_str:
        s, e = cpus_str.split('-')
        return float(int(e) - int(s) + 1)
    return float(len(cpus_str.split(','))) if ',' in cpus_str else 1.0

def run_concurrent_test(test_name: str, app_config: List[str], 
                       total_cpus_str: str, volume: str,
                       total_memory_mb: int, threads_per_container: int) -> TestResult:
    
    app_script_map = {
        'mobilenet': 'benchmark.py',
        'resnet': 'benchmark_resnet.py',
        'efficientnet': 'benchmark_efficientnet.py',
        'shufflenet': 'benchmark_shufflenet.py',
    }
    
    total_cpus_count = parse_cpus(total_cpus_str)
    slice_name = create_resource_limited_slice(test_name, total_cpus_count, total_memory_mb)
    
    print(f"\nTest: {test_name} | Instances: {len(app_config)} | Slice Limit: {total_cpus_count} CPUs")
    print(f"Container Thread Limit: {threads_per_container} threads/container")

    containers_data = []
    lock = threading.Lock()
    
    def worker(idx, app_name):
        c_name = f"{test_name}_c{idx}_{int(time.time()*1000)}"
        script = app_script_map.get(app_name, 'benchmark.py')
        try:
            # 传递线程限制参数
            res = run_container(app_name, c_name, slice_name, volume, script, threads_per_container)
            with lock:
                containers_data.append(res)
        except Exception as e:
            print(f"Error in container {idx}: {e}")

    threads = []
    start_all = time.time()
    
    for i, app in enumerate(app_config):
        t = threading.Thread(target=worker, args=(i, app))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print(f"Total duration: {(time.time()-start_all)*1000:.2f} ms")
    
    # 清理 slice 占位进程（如果有）
    if slice_name:
        subprocess.run(['systemctl', 'stop', f"{slice_name}.slice"], capture_output=True)

    # 处理数据
    metrics_list = []
    total_mem = 0
    
    for cid, start_dur, out, runtime, d_start in containers_data:
        start_ts, lats, timings = parse_benchmark_output(out, d_start)
        if not lats: continue
        
        lats.sort()
        p95 = lats[int(len(lats)*0.95)] if lats else 0
        p99 = lats[int(len(lats)*0.99)] if lats else 0
        
        pss = runtime.get('pss_mb', 0)
        total_mem += pss
        
        metrics_list.append(ContainerMetrics(
            container_id=cid, container_name="", app_name="", # 简化填充
            start_time=start_ts or 0, start_duration=start_dur,
            init_ms=timings['init_ms'], import_ms=timings['import_ms'],
            load_ms=timings['load_ms'], warmup_ms=timings['warmup_ms'],
            latencies=lats, p95_latency=p95, p99_latency=p99,
            memory_stat=runtime.get('memory_stat', {}), pss_mb=pss,
            pid=runtime.get('pid')
        ))
        
        # 清理容器
        subprocess.run(['docker', 'rm', '-f', cid], capture_output=True)

    return TestResult(
        test_name=test_name, num_instances=len(app_config),
        app_config=app_config, container_metrics=metrics_list,
        timestamp=time.time(), memory_full=(total_mem >= total_memory_mb * 0.95)
    )

def plot_simple(results: List[TestResult], output_dir: str):
    """简化的绘图逻辑"""
    if not HAS_MATPLOTLIB or not results: return
    
    x = [r.num_instances for r in results]
    y_lat = [statistics.mean([m.p95_latency for m in r.container_metrics]) for r in results]
    y_mem = [sum([m.pss_mb for m in r.container_metrics]) for r in results]
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Instances')
    ax1.set_ylabel('Avg P95 Latency (ms)', color='tab:red')
    ax1.plot(x, y_lat, 'o-', color='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Memory (MB)', color='tab:blue')
    ax2.plot(x, y_mem, 's--', color='tab:blue')
    
    plt.title('Concurrency Scaling Test')
    plt.savefig(os.path.join(output_dir, 'scaling_result.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-cpus', default='0-1', help='CPU range for slice (e.g. 0-1)')
    parser.add_argument('--total-memory-mb', type=int, default=2048)
    parser.add_argument('--volume', default=None)
    parser.add_argument('--threads-per-container', type=int, default=2, 
                        help='OMP/Torch threads per container. Keep at 2 for max throughput.')
    
    args = parser.parse_args()
    
    if args.volume is None:
        args.volume = f"{os.path.abspath(os.path.dirname(__file__))}:/app"
        
    out_dir = f"results/experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    # 简单的递增测试逻辑
    for n in INSTANCE_COUNTS:
        res = run_concurrent_test(
            f"test_n{n}", ['mobilenet'] * n,
            args.total_cpus, args.volume, 
            args.total_memory_mb, args.threads_per_container
        )
        
        save_result_to_file(res, out_dir)
        results.append(res)
        
        print(f"Finished N={n}. Memory Full: {res.memory_full}")
        if res.memory_full:
            print("Memory limit reached.")
            break
            
        n *= 2
        time.sleep(3) # 让系统喘口气

    plot_simple(results, out_dir)
    print(f"Done. Results in {out_dir}")

if __name__ == "__main__":
    main()