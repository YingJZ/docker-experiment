#!/usr/bin/env python3
"""
Native Concurrent Benchmark - 不使用容器，直接在主机上并发运行推理
使用 systemd slice 限制 CPU 亲和性和内存，与容器版本对比
"""

# 测试实例数配置
INSTANCE_COUNTS = [1, 2, 4, 6, 8, 10, 12, 14, 16]

print(f"\033[92mINFO: Native benchmark, INSTANCE_COUNTS={INSTANCE_COUNTS}\033[0m")

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
from pathlib import Path

# --- 全局状态追踪，用于清理 ---
ACTIVE_SLICES: Set[str] = set()
ACTIVE_PROCESSES: Dict[str, subprocess.Popen] = {}  # name -> Popen
CLEANUP_LOCK = threading.Lock()

# 脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
BENCHMARK_SCRIPT = SCRIPT_DIR / "benchmark.py"

# --- 检查 Root 权限 ---
if os.geteuid() != 0:
    print("Error: This script must be run as root (sudo) to use systemd-run and read /proc/pid/smaps.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def cleanup_all_resources():
    """清理所有测试相关的进程和 Slice"""
    with CLEANUP_LOCK:
        print("\n=== Cleaning up resources ===")
        
        # 1. 终止所有子进程
        for name, proc in list(ACTIVE_PROCESSES.items()):
            try:
                if proc.poll() is None:  # 进程仍在运行
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                print(f"Terminated process: {name}")
            except Exception as e:
                print(f"Failed to terminate {name}: {e}")
        ACTIVE_PROCESSES.clear()
        
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
class ProcessMetrics:
    process_name: str
    pid: int
    start_duration: float
    init_ms: float
    import_ms: float
    load_ms: float
    warmup_ms: float
    latencies: List[float]
    p95_latency: float
    max_pss_mb: float
    avg_pss_mb: float

@dataclass
class TestResult:
    test_name: str
    num_instances: int
    process_metrics: List[ProcessMetrics]
    memory_full: bool

def get_process_pss(pid: int) -> float:
    """获取进程 PSS 内存 (MB)"""
    try:
        smaps_path = f"/proc/{pid}/smaps"
        if not os.path.exists(smaps_path):
            return 0.0
        
        pss_total_kb = 0
        with open(smaps_path, 'r') as f:
            for line in f:
                if line.startswith('Pss:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        pss_total_kb += int(parts[1])
        return pss_total_kb / 1024.0
    except PermissionError:
        return 0.0
    except Exception:
        return 0.0

def create_resource_limited_slice(test_name: str, allowed_cpus: str, total_memory_mb: int) -> Optional[str]:
    """
    创建一个 Systemd Slice，设置 AllowedCPUs 和 MemoryMax
    """
    slice_name = f"native_bench_{test_name}_{int(time.time())}.slice"
    memory_bytes = int(total_memory_mb * 1024 * 1024)

    print(f"Creating Slice {slice_name}: AllowedCPUs={allowed_cpus}, MemMax={total_memory_mb}MB")

    try:
        subprocess.run(['systemctl', 'start', slice_name], check=True, capture_output=True)
        subprocess.run(['systemctl', 'set-property', slice_name, f'AllowedCPUs={allowed_cpus}'], 
                      check=True, capture_output=True)
        subprocess.run(['systemctl', 'set-property', slice_name, f'MemoryMax={memory_bytes}'], 
                      check=True, capture_output=True)
        
        with CLEANUP_LOCK:
            ACTIVE_SLICES.add(slice_name)
        
        time.sleep(0.3)
        return slice_name
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to create/configure slice: {e}")
        return None

def run_process_task(idx: int, test_name: str, slice_name: str,
                     python_path: str, threads_limit: int, 
                     results_list: list, allowed_cpus: str):
    """在 slice 下启动一个 Python 推理进程并监控"""
    
    process_name = f"{test_name}_p{idx}"
    
    # 构建环境变量
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads_limit)
    env['MKL_NUM_THREADS'] = str(threads_limit)
    env['PYTORCH_NUM_THREADS'] = str(threads_limit)
    env['OPENBLAS_NUM_THREADS'] = str(threads_limit)
    
    # 使用 systemd-run 在 slice 下运行进程
    # --scope: 创建一个 scope unit 而非 service
    # --slice: 指定父 slice
    # -p AllowedCPUs: 设置 CPU 亲和性（cgroup cpuset）
    cmd = [
        'systemd-run',
        '--scope',  # 使用 scope 而非 service，便于直接获取输出
        f'--slice={slice_name}',
        f'--unit={process_name}',
        '-p', f'AllowedCPUs={allowed_cpus}',
        '--',
        python_path,
        str(BENCHMARK_SCRIPT)
    ]
    
    start_cmd_time = time.time()
    proc = None
    pid = None
    
    try:
        # 启动进程
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # 获取实际的 Python 进程 PID（需要等待 systemd-run 启动）
        time.sleep(0.5)
        
        # 通过 systemctl 获取 scope 下的进程
        try:
            scope_name = f"{process_name}.scope"
            result = subprocess.run(
                ['systemctl', 'show', scope_name, '-p', 'MainPID', '--value'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                pid = int(result.stdout.strip())
        except Exception:
            pass
        
        # 如果获取不到，尝试从 /proc 查找
        if not pid or pid == 0:
            pid = proc.pid
        
        with CLEANUP_LOCK:
            ACTIVE_PROCESSES[process_name] = proc
        
        start_duration = (time.time() - start_cmd_time) * 1000
        print(f"[{process_name}] Started, PID: {pid}")
        
        # 监控循环
        pss_samples = []
        max_wait = 300  # 最多等待 150 秒
        
        for iteration in range(max_wait):
            # 检查进程是否结束
            poll_result = proc.poll()
            if poll_result is not None:
                break
            
            # 采样 PSS
            if pid and pid > 0:
                current_pss = get_process_pss(pid)
                if current_pss > 0:
                    pss_samples.append(current_pss)
            
            time.sleep(0.5)
        
        # 检查是否超时
        if proc.poll() is None:
            print(f"[{process_name}] WARNING: Timeout, terminating...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        
        # 获取输出
        stdout, stderr = proc.communicate(timeout=5)
        output_text = stdout
        
        if not output_text.strip():
            print(f"[{process_name}] WARNING: No output, stderr: {stderr[:500] if stderr else 'none'}")
        
        # 解析输出
        start_ts, lats, timings = parse_benchmark_output(output_text, start_cmd_time)
        
        if lats:
            lats.sort()
            p95 = lats[int(len(lats)*0.95)]
            max_pss = max(pss_samples) if pss_samples else 0.0
            avg_pss = statistics.mean(pss_samples) if pss_samples else 0.0
            
            metrics = ProcessMetrics(
                process_name=process_name,
                pid=pid or 0,
                start_duration=start_duration,
                init_ms=timings['init_ms'],
                import_ms=timings['import_ms'],
                load_ms=timings['load_ms'],
                warmup_ms=timings['warmup_ms'],
                latencies=lats,
                p95_latency=p95,
                max_pss_mb=max_pss,
                avg_pss_mb=avg_pss
            )
            results_list.append(metrics)
            print(f"[{process_name}] Completed: P95={p95:.2f}ms, MaxPSS={max_pss:.1f}MB")
        else:
            print(f"[{process_name}] ERROR: No latency data parsed")
            
    except Exception as e:
        print(f"[{process_name}] Unexpected error: {e}")
    finally:
        with CLEANUP_LOCK:
            ACTIVE_PROCESSES.pop(process_name, None)

def parse_benchmark_output(text: str, start_cmd_time: float):
    """解析 benchmark 输出"""
    start_time = None
    latencies = []
    timing = {'init_ms': 0.0, 'import_ms': 0.0, 'load_ms': 0.0, 'warmup_ms': 0.0}
    
    m_start = re.search(r'start time:\s*([0-9.]+)', text)
    if m_start:
        start_time = float(m_start.group(1))
        timing['init_ms'] = (start_time - start_cmd_time) * 1000.0
    
    patterns = {
        'import_ms': r'Import Torch Done, Time Spent:\s*([0-9.]+)s',
        'load_ms': r'Model Loaded, Time Spent:\s*([0-9.]+)s',
        'warmup_ms': r'Warmup Done, Time Spent:\s*([0-9.]+)s'
    }
    
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            timing[key] = float(m.group(1)) * 1000.0
            
    for line in text.split('\n'):
        if 'LATENCIES:' in line:
            parts = line.split('LATENCIES:')[1].split(',')
            latencies = [float(x) for x in parts if x.strip()]
            
    return start_time, latencies, timing

def run_concurrent_test(test_name: str, count: int, 
                       allowed_cpus: str, python_path: str,
                       total_memory_mb: int):
    
    # 创建资源限制 Slice
    slice_name = create_resource_limited_slice(test_name, allowed_cpus, total_memory_mb)
    
    # 计算可用核心数
    try:
        total_cores = 0
        for part in allowed_cpus.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                total_cores += int(end) - int(start) + 1
            else:
                total_cores += 1
    except Exception:
        total_cores = 2
    
    # 动态调整线程数
    effective_threads = max(1, total_cores // count)
    total_threads = effective_threads * count
    
    print(f"\n=== Test: {count} Instances | AllowedCPUs: {allowed_cpus} ({total_cores} cores) | Mem: {total_memory_mb}MB ===")
    print(f"    Threads per process: {effective_threads} (total: {total_threads})")
    
    threads = []
    results_list = []
    
    # 并发启动
    for i in range(count):
        t = threading.Thread(
            target=run_process_task,
            args=(i, test_name, slice_name, python_path, effective_threads, results_list, allowed_cpus)
        )
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    # 清理 Slice
    if slice_name:
        subprocess.run(['systemctl', 'stop', slice_name], capture_output=True)
        with CLEANUP_LOCK:
            ACTIVE_SLICES.discard(slice_name)
        
    # 汇总
    total_pss_max = sum(m.max_pss_mb for m in results_list)
    avg_p95 = statistics.mean([m.p95_latency for m in results_list]) if results_list else 0
    
    print(f"Result N={count}: Avg P95={avg_p95:.2f}ms, Total Peak PSS={total_pss_max:.2f}MB")
    print()
    
    is_mem_full = total_pss_max >= (total_memory_mb * 0.99)
    
    return TestResult(
        test_name=test_name,
        num_instances=count, 
        process_metrics=results_list,
        memory_full=is_mem_full
    )

def plot_results(results: List[TestResult], out_dir: str):
    if not HAS_MATPLOTLIB or not results:
        return
    
    valid_results = [r for r in results if r.process_metrics]
    if not valid_results:
        print("No valid results to plot")
        return
    
    x = [r.num_instances for r in valid_results]
    y_lat = [statistics.mean([m.p95_latency for m in r.process_metrics]) for r in valid_results]
    y_mem = [sum([m.max_pss_mb for m in r.process_metrics]) for r in valid_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Concurrent Processes')
    ax1.set_ylabel('Avg P95 Latency (ms)', color=color)
    ax1.plot(x, y_lat, 'o-', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Peak PSS Memory (MB)', color=color)
    ax2.plot(x, y_mem, 's--', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Native Concurrency Scaling Benchmark (No Docker)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'native_scaling_result.png'))
    print(f"Plot saved to {out_dir}/native_scaling_result.png")

def main():
    parser = argparse.ArgumentParser(description="Native concurrent benchmark without Docker")
    parser.add_argument('--allowed-cpus', default='0,1', 
                       help='CPU cores to allow (cpuset), e.g., "0,1" or "0-3"') 
    parser.add_argument('--mem', type=int, default=2048, 
                       help='Total Memory for slice in MB')
    parser.add_argument('--python', default=None,
                       help='Path to Python interpreter (default: current venv)')
    
    args = parser.parse_args()
    
    # 确定 Python 解释器路径
    if args.python:
        python_path = args.python
    else:
        python_path = sys.executable
    
    print(f"\033[93mINFO: Using Python interpreter: {python_path}\033[0m")
    
    allowed_cpus = str(args.allowed_cpus)

    out_dir = f"results/native_experiment_{time.strftime('%y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    for n in INSTANCE_COUNTS:
        res = run_concurrent_test(
            f"test_n{n}", n, allowed_cpus, python_path, args.mem
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
                            'process_name': m.process_name,
                            'pid': m.pid,
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
                            'latencies_count': len(m.latencies)
                        } 
                        for m in r.process_metrics
                    ]
                } for r in results
            ], f, indent=2)
        
    plot_results(results, out_dir)

if __name__ == "__main__":
    main()
