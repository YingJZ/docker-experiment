#!/usr/bin/env python3

INSTANCE_COUNTS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
# INSTANCE_COUNTS = [10, 12, 14, 16]

print(f"\033[92mINFO: INSTANCE_COUNTS={INSTANCE_COUNTS}\033[0m")

import subprocess
import time
import json
import os
import re
import statistics
import threading
import argparse
import sys
import signal
import atexit
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

# --- 全局状态追踪，用于清理 ---
ACTIVE_SLICES: Set[str] = set()
ACTIVE_CONTAINERS: Set[str] = set()
CLEANUP_LOCK = threading.Lock()

# --- 检查 Root 权限 (必须有权限才能读取 /proc/pid/smaps) ---
if os.geteuid() != 0:
    print("Error: This script must be run as root (sudo) to read process memory stats.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def cleanup_all_resources():
    """清理所有测试相关的容器和 Slice"""
    with CLEANUP_LOCK:
        print("\n=== Cleaning up resources ===")
        
        # 1. 停止并删除所有测试容器
        for container_id in list(ACTIVE_CONTAINERS):
            try:
                subprocess.run(['docker', 'rm', '-f', container_id], 
                             capture_output=True, timeout=5)
                print(f"Cleaned container: {container_id[:12]}")
            except Exception as e:
                print(f"Failed to clean container {container_id[:12]}: {e}")
        ACTIVE_CONTAINERS.clear()
        
        # 2. 停止所有 Slice
        for slice_name in list(ACTIVE_SLICES):
            try:
                subprocess.run(['systemctl', 'stop', slice_name], 
                             capture_output=True, timeout=5)
                print(f"Stopped slice: {slice_name}")
            except Exception as e:
                print(f"Failed to stop slice {slice_name}: {e}")
        ACTIVE_SLICES.clear()
        
        print("=== Cleanup complete ===")

def signal_handler(signum, frame):
    """信号处理器：优雅退出"""
    print(f"\n\nReceived signal {signum}, cleaning up...")
    cleanup_all_resources()
    sys.exit(1)

# 注册信号处理和退出清理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_all_resources)

@dataclass
class ContainerMetrics:
    container_id: str
    start_duration: float
    init_ms: float
    import_ms: float
    load_ms: float
    warmup_ms: float
    latencies: List[float]
    p95_latency: float
    max_pss_mb: float  # 改为记录峰值
    avg_pss_mb: float  # 记录平均值
    pid: Optional[int] = None

@dataclass
class TestResult:
    test_name: str
    num_instances: int
    container_metrics: List[ContainerMetrics]
    memory_full: bool

def get_process_pss(pid: int) -> float:
    """
    获取进程 PSS 内存 (MB)。
    需要 Root 权限读取 /proc/{pid}/smaps
    """
    try:
        smaps_path = f"/proc/{pid}/smaps"
        if not os.path.exists(smaps_path):
            return 0.0
        
        pss_total_kb = 0
        with open(smaps_path, 'r') as f:
            for line in f:
                # 格式通常是: "Pss:       1024 kB"
                if line.startswith('Pss:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        pss_total_kb += int(parts[1])
        return pss_total_kb / 1024.0
    except PermissionError:
        print(f"Warning: Permission denied reading {smaps_path}. Run with sudo.")
        return 0.0
    except Exception:
        return 0.0

def create_resource_limited_slice(test_name: str, allowed_cpus: str, total_memory_mb: int) -> Optional[str]:
    """
    创建一个 Systemd Slice，并设置 AllowedCPUs（cpuset）与 MemoryMax。
    所有容器将被置于该 Slice 下，实现共享/竞争同一核心集合。
    """
    slice_name = f"bench_{test_name}_{int(time.time())}.slice"
    memory_bytes = int(total_memory_mb * 1024 * 1024)

    print(f"Creating Slice {slice_name}: AllowedCPUs={allowed_cpus}, MemMax={total_memory_mb}MB")

    try:
        # 创建（启动）Slice 单元
        subprocess.run(['systemctl', 'start', slice_name], check=True, capture_output=True)
        # 设置 cpuset 以及内存上限
        subprocess.run(['systemctl', 'set-property', slice_name, f'AllowedCPUs={allowed_cpus}'], check=True, capture_output=True)
        subprocess.run(['systemctl', 'set-property', slice_name, f'MemoryMax={memory_bytes}'], check=True, capture_output=True)
        
        # 注册到全局追踪
        with CLEANUP_LOCK:
            ACTIVE_SLICES.add(slice_name)
        
        time.sleep(0.3)
        return slice_name
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to create/configure slice: {e}")
        return None

def run_container_task(idx: int, app_name: str, test_name: str, 
                       slice_name: str, volume: str, 
                       threads_limit: int, results_list: list,
                       cpuset_cpus: Optional[str] = None):
    """单独的线程任务：运行容器并监控"""
    
    container_name = f"{test_name}_c{idx}"
    script_name = 'benchmark.py' # 根据需要映射
    
    # 使用调用者传入的线程数（已在 run_concurrent_test 中根据实例数计算好）
    effective_threads = max(1, threads_limit)
    env_vars = [
        '-e', f'OMP_NUM_THREADS={effective_threads}',
        '-e', f'MKL_NUM_THREADS={effective_threads}',
        '-e', f'PYTORCH_NUM_THREADS={effective_threads}'
    ]
    
    # --- 关键修改 ---
    # 使用 --cgroup-parent 将容器真正放入 Systemd Slice 中
    # 注意：systemd slice 的路径通常需要完整格式。
    # 对于 docker，如果 driver 是 systemd，直接用 slice 名通常可以，
    # 但有时需要全路径，这里尝试直接传递 slice 名。
    docker_cmd = [
        'docker', 'run', '-d',
        '--name', container_name,
        # 不使用 --rm，手动删除以便先获取日志
        '-v', volume,
    ] + env_vars
    
    if slice_name:
        docker_cmd.extend(['--cgroup-parent', slice_name])
    # 使用 cpuset 限制容器可用 CPU 核心，避免跨核频繁切换
    if cpuset_cpus:
        docker_cmd.extend(['--cpuset-cpus', cpuset_cpus])
        
    docker_cmd.extend(['torch-cpu', 'python', script_name])

    start_cmd_time = time.time()
    container_id = None
    
    try:
        # 1. 启动容器
        res = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        container_id = res.stdout.strip()
        
        # 注册到全局追踪
        with CLEANUP_LOCK:
            ACTIVE_CONTAINERS.add(container_id)
        
        start_duration = (time.time() - start_cmd_time) * 1000
        print(f"[{container_name}] Started, ID: {container_id[:12]}")
        
        # 2. 监控循环 (采样多次取最大值)
        pss_samples = []
        pid = None
        container_finished = False
        wait_iterations = 0
        max_wait = 300  # 最多等待 300 次 (150秒)
        
        for iteration in range(max_wait):
            wait_iterations = iteration
            # 检查存活
            check = subprocess.run(
                ['docker', 'inspect', '--format', '{{.State.Running}}|{{.State.Pid}}|{{.State.ExitCode}}', container_id],
                capture_output=True, text=True
            )
            
            if check.returncode != 0: 
                break # 容器可能消失了
                
            state_str = check.stdout.strip()
            if not state_str: break
            
            parts = state_str.split('|')
            if len(parts) < 3: break
            
            is_running_str, pid_str, exit_code_str = parts
            
            # 如果已经停止，标记完成并退出循环
            if is_running_str != 'true':
                container_finished = True
                break
            
            # 尝试获取内存
            try:
                current_pid = int(pid_str)
                if current_pid > 0:
                    pid = current_pid
                    current_pss = get_process_pss(pid)
                    if current_pss > 0:
                        pss_samples.append(current_pss)
            except ValueError:
                pass
            
            time.sleep(0.5) # 每 0.5 秒采样一次
        
        # 检查是否超时
        if not container_finished and wait_iterations >= max_wait - 1:
            print(f"[{container_name}] WARNING: Timeout after {wait_iterations * 0.5:.1f}s, forcing stop")
            subprocess.run(['docker', 'stop', container_id], capture_output=True, timeout=10)
            
        # 3. 获取日志
        logs = subprocess.run(['docker', 'logs', container_id], capture_output=True, text=True)
        output_text = logs.stdout
        
        if not output_text.strip():
            print(f"[{container_name}] WARNING: No logs captured, checking stderr...")
            stderr_logs = logs.stderr
            if stderr_logs:
                print(f"[{container_name}] stderr: {stderr_logs[:500]}")
        
        # 解析日志
        start_ts, lats, timings = parse_benchmark_output(output_text, start_cmd_time)
        
        if lats:
            lats.sort()
            p95 = lats[int(len(lats)*0.95)]
            max_pss = max(pss_samples) if pss_samples else 0.0
            avg_pss = statistics.mean(pss_samples) if pss_samples else 0.0
            
            metrics = ContainerMetrics(
                container_id=container_id,
                start_duration=start_duration,
                init_ms=timings['init_ms'], import_ms=timings['import_ms'],
                load_ms=timings['load_ms'], warmup_ms=timings['warmup_ms'],
                latencies=lats, p95_latency=p95,
                max_pss_mb=max_pss, avg_pss_mb=avg_pss,
                pid=pid
            )
            results_list.append(metrics)
            print(f"[{container_name}] Completed: P95={p95:.2f}ms, MaxPSS={max_pss:.1f}MB")
        else:
            print(f"[{container_name}] ERROR: No latency data parsed from logs")
            
    except subprocess.CalledProcessError as e:
        print(f"[{container_name}] Failed to start: {e}")
        if e.stderr:
            print(f"[{container_name}] Docker error: {e.stderr[:200]}")
    except Exception as e:
        print(f"[{container_name}] Unexpected error: {e}")
    finally:
        # 确保清理
        if container_id:
            subprocess.run(['docker', 'rm', '-f', container_id], capture_output=True)
            with CLEANUP_LOCK:
                ACTIVE_CONTAINERS.discard(container_id)

def parse_benchmark_output(text: str, docker_start_time: float):
    """保持原有的解析逻辑"""
    start_time = None
    latencies = []
    timing = {'init_ms': 0.0, 'import_ms': 0.0, 'load_ms': 0.0, 'warmup_ms': 0.0}
    
    m_start = re.search(r'start time:\s*([0-9.]+)', text)
    if m_start:
        start_time = float(m_start.group(1))
        # 估算的 Init 时间
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
            
    # 查找 Latencies 行
    for line in text.split('\n'):
        if 'LATENCIES:' in line:
            parts = line.split('LATENCIES:')[1].split(',')
            latencies = [float(x) for x in parts if x.strip()]
            
    return start_time, latencies, timing

def run_concurrent_test(test_name: str, count: int, 
                       allowed_cpus: str, volume: str,
                       total_memory_mb: int, threads_per_container: int):
    
    # 1. 创建资源限制 Slice
    slice_name = create_resource_limited_slice(test_name, allowed_cpus, total_memory_mb)
    
    # 计算可用核心数
    try:
        # 解析 cpuset 格式，如 "0,1" 或 "0-3"
        total_cores = 0
        for part in allowed_cpus.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                total_cores += int(end) - int(start) + 1
            else:
                total_cores += 1
    except Exception:
        total_cores = 2  # 默认
    
    # 动态调整每个容器的线程数，确保总线程数不超过可用核心数
    # 这样可以避免 CPU 过度订阅导致的性能崩溃
    effective_threads_per_container = max(1, total_cores // count)
    total_threads = effective_threads_per_container * count
    
    print(f"\n=== Test: {count} Instances | AllowedCPUs: {allowed_cpus} ({total_cores} cores) | Slice Mem: {total_memory_mb}MB ===")
    print(f"    Threads per container: {effective_threads_per_container} (total: {total_threads})")
    
    threads = []
    results_list = []
    
    # 2. 并发启动
    for i in range(count):
        t = threading.Thread(
            target=run_container_task,
            args=(i, 'mobilenet', test_name, slice_name, volume, effective_threads_per_container, results_list, allowed_cpus)
        )
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    # 3. 清理 Slice
    if slice_name:
        subprocess.run(['systemctl', 'stop', slice_name], capture_output=True)
        with CLEANUP_LOCK:
            ACTIVE_SLICES.discard(slice_name)
        
    # 4. 汇总
    total_pss_max = sum(m.max_pss_mb for m in results_list)
    avg_p95 = statistics.mean([m.p95_latency for m in results_list]) if results_list else 0
    
    print(f"Result N={count}: Avg P95={avg_p95:.2f}ms, Total Peak PSS={total_pss_max:.2f}MB")
    print()
    
    is_mem_full = total_pss_max >= (total_memory_mb * 0.99)
    
    return TestResult(
        test_name=test_name, num_instances=count, 
        container_metrics=results_list, memory_full=is_mem_full
    )

def plot_results(results: List[TestResult], out_dir: str):
    if not HAS_MATPLOTLIB or not results: return
    
    # 过滤掉没有数据的结果
    valid_results = [r for r in results if r.container_metrics]
    if not valid_results:
        print("No valid results to plot")
        return
    
    x = [r.num_instances for r in valid_results]
    y_lat = [statistics.mean([m.p95_latency for m in r.container_metrics]) for r in valid_results]
    y_mem = [sum([m.max_pss_mb for m in r.container_metrics]) for r in valid_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Concurrent Containers')
    ax1.set_ylabel('Avg P95 Latency (ms)', color=color)
    ax1.plot(x, y_lat, 'o-', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Peak PSS Memory (MB)', color=color)
    ax2.plot(x, y_mem, 's--', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Docker Concurrency Scaling Benchmark')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scaling_result.png'))
    print(f"Plot saved to {out_dir}/scaling_result.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allowed-cpus', default='0,1', help='CPU cores to allow (cpuset), e.g., "0,1" or "0-3"') 
    parser.add_argument('--mem', type=int, default=2048, help='Total Memory for slice in MB')
    parser.add_argument('--volume', default=None)
    
    args = parser.parse_args()
    
    if args.volume is None:
        args.volume = f"{os.path.abspath(os.path.dirname(__file__))}:/app"
        print(f"\033[93mWARNING: Using default volume mapping: {args.volume}\033[0m")
    
    allowed_cpus = str(args.allowed_cpus)

    out_dir = f"results/experiment_{time.strftime('%y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    for n in INSTANCE_COUNTS:
        res = run_concurrent_test(
            f"test_n{n}", n, allowed_cpus, args.volume, args.mem, 
            threads_per_container=2
        )
        results.append(res)
        
        # 保存中间数据
        with open(os.path.join(out_dir, 'data.json'), 'w') as f:
            json.dump([
                {
                    'n': r.num_instances,
                    'memory_full': r.memory_full,
                    'metrics': [
                        {
                            'container_id': m.container_id,
                            'start_duration': m.start_duration,
                            'init_ms': m.init_ms,
                            'import_ms': m.import_ms,
                            'load_ms': m.load_ms,
                            'warmup_ms': m.warmup_ms,
                            'p95_latency': m.p95_latency,
                            'avg_latency': statistics.mean(m.latencies) if m.latencies else 0,
                            'min_latency': min(m.latencies) if m.latencies else 0,
                            'max_latency': max(m.latencies) if m.latencies else 0,
                            'max_pss_mb': m.max_pss_mb,
                            'avg_pss_mb': m.avg_pss_mb,
                            'pid': m.pid,
                            'latencies_count': len(m.latencies)
                        } 
                        for m in r.container_metrics
                    ]
                } for r in results
            ], f, indent=2)
            
        # if res.memory_full:
        #     print(f"!!! Warning: Approaching memory limit ({args.mem}MB) at N={n}. Stopping scaling.")
        #     break
            
        # time.sleep(5) # 冷却
        
    plot_results(results, out_dir)

if __name__ == "__main__":
    main()