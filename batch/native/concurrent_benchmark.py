#!/usr/bin/env python3
"""
Native Concurrent Benchmark - 不使用容器，直接在主机上并发运行推理
使用 systemd slice 限制 CPU 亲和性和内存，与容器版本对比
"""

# 测试实例数配置
INSTANCE_COUNTS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

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
import shlex
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
    init_ms: float  # 从 systemd-run 到脚本开始执行（包含 systemd、cgroup、Python 启动）
    python_init_ms: float  # 仅 Python 解释器初始化时间（从 Python 进程启动到脚本开始）
    import_ms: float
    load_ms: float
    warmup_ms: float
    latencies: List[float]
    p95_latency: float
    max_pss_mb: float
    avg_pss_mb: float
    mem_breakdown_max: Dict[str, float]  # 聚合后的 smaps_rollup 最大值
    mem_breakdown_avg: Dict[str, float]  # 聚合后的 smaps_rollup 平均值
    cgroup_mem_stat_max: Dict[str, float]  # cgroup memory.stat 最大值
    cgroup_mem_stat_avg: Dict[str, float]  # cgroup memory.stat 平均值
    cgroup_mem_stat_last: Dict[str, float] # cgroup memory.stat 最后一次采样

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

def get_process_mem_breakdown(pid: int) -> Dict[str, float]:
    """
    从 /proc/<pid>/smaps_rollup 抽取一次性的内存组成（MB）。
    相比逐行 smaps，rollup 代价低，可用于采样。
    """
    rollup_path = f"/proc/{pid}/smaps_rollup"
    fields = {
        'Pss:': 'pss_mb',
        'Rss:': 'rss_mb',
        'Shared_Clean:': 'shared_clean_mb',
        'Shared_Dirty:': 'shared_dirty_mb',
        'Private_Clean:': 'private_clean_mb',
        'Private_Dirty:': 'private_dirty_mb',
        'Swap:': 'swap_mb',
        'AnonHugePages:': 'anon_huge_mb'
    }
    out: Dict[str, float] = {v: 0.0 for v in fields.values()}
    try:
        with open(rollup_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0]
                if key in fields:
                    try:
                        kb = float(parts[1])
                        out[fields[key]] = kb / 1024.0
                    except ValueError:
                        continue
    except FileNotFoundError:
        return {}
    except PermissionError:
        return {}
    except Exception:
        return {}
    return out

def read_cgroup_memory_stat(pid: int) -> Dict[str, float]:
    """
    读取进程所在 cgroup 的 memory.stat，返回 MB。
    同时兼容 cgroup v1 与 v2。
    """
    try:
        cgroup_path = f"/proc/{pid}/cgroup"
        is_unified = os.path.exists("/sys/fs/cgroup/cgroup.controllers")  # cgroup v2 判定
        rel_path = None

        with open(cgroup_path, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) < 3:
                    continue
                subsystems = parts[1]
                candidate = parts[2]

                # cgroup v2: 行形如 "0::/system.slice/xxx.scope"
                if is_unified and subsystems == '':
                    rel_path = candidate
                    break

                # cgroup v1: 找包含 memory 的子系统
                if 'memory' in subsystems.split(','):
                    rel_path = candidate
                    break

        if not rel_path:
            return {}

        root = "/sys/fs/cgroup" if is_unified else "/sys/fs/cgroup/memory"
        stat_path = os.path.join(root, rel_path.lstrip('/'), 'memory.stat')
        if not os.path.exists(stat_path):
            return {}

        stats: Dict[str, float] = {}
        with open(stat_path, 'r') as f:
            for line in f:
                k, v = line.split()
                stats[k] = float(v) / (1024.0 * 1024.0)  # 转 MB
        return stats
    except Exception:
        return {}

def verify_cpuset_controller():
    """检查并尝试启用 cpuset 控制器（如果可能）"""
    try:
        # 检查 cgroup v2
        if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
            controllers_path = "/sys/fs/cgroup/cgroup.subtree_control"
            if os.path.exists(controllers_path):
                with open(controllers_path, 'r') as f:
                    controllers = f.read().strip()
                if 'cpuset' not in controllers:
                    print("Warning: cpuset controller not enabled in cgroup v2")
                    print(f"  Current controllers: {controllers}")
                    print("  You may need to enable it manually:")
                    print(f"    echo '+cpuset' | sudo tee {controllers_path}")
        else:
            # cgroup v1: 检查 cpuset 挂载点
            if not os.path.exists("/sys/fs/cgroup/cpuset"):
                print("Warning: cpuset cgroup not mounted")
    except Exception as e:
        print(f"Warning: Could not check cpuset controller: {e}")

def create_resource_limited_slice(test_name: str, allowed_cpus: str, total_memory_mb: int) -> Optional[str]:
    """
    创建一个 Systemd Slice，设置 AllowedCPUs 和 MemoryMax
    """
    slice_name = f"native_bench_{test_name}_{int(time.time())}.slice"
    memory_bytes = int(total_memory_mb * 1024 * 1024)

    print(f"Creating Slice {slice_name}: AllowedCPUs={allowed_cpus}, MemMax={total_memory_mb}MB")
    
    # 检查 cpuset 控制器
    verify_cpuset_controller()

    try:
        subprocess.run(['systemctl', 'start', slice_name], check=True, capture_output=True)
        subprocess.run(['systemctl', 'set-property', slice_name, f'AllowedCPUs={allowed_cpus}'], 
                      check=True, capture_output=True)
        subprocess.run(['systemctl', 'set-property', slice_name, f'MemoryMax={memory_bytes}'], 
                      check=True, capture_output=True)
        
        # 验证 CPU 限制是否生效
        result = subprocess.run(
            ['systemctl', 'show', slice_name, '-p', 'AllowedCPUs', '--value'],
            capture_output=True, text=True, check=True
        )
        actual_cpus = result.stdout.strip()
        if actual_cpus != allowed_cpus:
            print(f"Warning: Slice AllowedCPUs mismatch. Expected: {allowed_cpus}, Got: {actual_cpus}")
        
        # 在 cgroup v1 下，systemd 的 AllowedCPUs 可能不工作
        # 我们将使用 taskset 来确保 CPU 限制生效
        # 这里尝试直接设置 cgroup cpuset（如果可能）
        cpuset_set = False
        try:
            # 获取 slice 的 cgroup 路径
            cgroup_result = subprocess.run(
                ['systemctl', 'show', slice_name, '-p', 'ControlGroup', '--value'],
                capture_output=True, text=True, check=True
            )
            cgroup_path = cgroup_result.stdout.strip()
            if cgroup_path:
                # 检查是否是 cgroup v2
                if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
                    # cgroup v2: 设置 cpuset.cpus
                    cpuset_dir = f"/sys/fs/cgroup{cgroup_path}"
                    cpuset_path = f"{cpuset_dir}/cpuset.cpus"
                    if os.path.exists(cpuset_dir):
                        try:
                            with open(cpuset_path, 'w') as f:
                                f.write(allowed_cpus)
                            print(f"✓ Set cpuset.cpus={allowed_cpus} via cgroup v2")
                            cpuset_set = True
                        except (IOError, PermissionError) as e:
                            print(f"Warning: Could not write cpuset.cpus: {e}")
                else:
                    # cgroup v1: systemd 可能不使用 cpuset 子系统
                    # 尝试在 /sys/fs/cgroup/cpuset 下创建对应的目录
                    cpuset_base = "/sys/fs/cgroup/cpuset"
                    cpuset_dir = f"{cpuset_base}{cgroup_path}"
                    
                    # 尝试创建目录（如果不存在）
                    if not os.path.exists(cpuset_dir):
                        try:
                            os.makedirs(cpuset_dir, exist_ok=True)
                        except (OSError, PermissionError):
                            pass
                    
                    if os.path.exists(cpuset_dir):
                        cpuset_cpus_path = f"{cpuset_dir}/cpuset.cpus"
                        cpuset_mems_path = f"{cpuset_dir}/cpuset.mems"
                        
                        try:
                            # 读取根 cpuset 的 mems 设置
                            root_mems_path = f"{cpuset_base}/cpuset.mems"
                            if os.path.exists(root_mems_path):
                                with open(root_mems_path, 'r') as f:
                                    mems = f.read().strip()
                            else:
                                mems = "0"
                            
                            # 设置 cpuset.cpus
                            with open(cpuset_cpus_path, 'w') as f:
                                f.write(allowed_cpus)
                            # 设置 cpuset.mems
                            with open(cpuset_mems_path, 'w') as f:
                                f.write(mems)
                            print(f"✓ Set cpuset.cpus={allowed_cpus} via cgroup v1")
                            cpuset_set = True
                        except (IOError, PermissionError) as e:
                            print(f"Warning: Could not set cpuset in cgroup v1: {e}")
        except Exception as e:
            print(f"Warning: Could not set cgroup cpuset: {e}")
        
        if not cpuset_set:
            print(f"Note: Will use taskset to enforce CPU affinity to CPUs: {allowed_cpus}")
        
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
    # 对于 cgroup v1，systemd 的 AllowedCPUs 可能不工作
    # 因此使用 taskset 作为额外的 CPU 限制
    # --scope: 创建一个 scope unit 而非 service
    # --slice: 指定父 slice
    # 使用 taskset 确保进程和所有线程都限制在指定 CPU 上
    # 
    # 为了精确测量 Python 解释器初始化时间，使用 bash 包装：
    # 在启动 Python 之前记录时间戳，并通过环境变量传递给 Python
    taskset_cmd = ['taskset', '-c', allowed_cpus]
    
    # 构建 bash 命令：在启动 Python 前记录时间戳
    python_cmd = f'export PYTHON_START_TS=$(date +%s.%N) && exec {shlex.quote(python_path)} {shlex.quote(str(BENCHMARK_SCRIPT))}'
    
    cmd = [
        'systemd-run',
        '--scope',  # 使用 scope 而非 service，便于直接获取输出
        f'--slice={slice_name}',
        f'--unit={process_name}',
        '--',
    ] + taskset_cmd + [
        'bash', '-c', python_cmd
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
        
        # 验证 CPU 亲和性是否设置成功（使用 taskset 检查）
        if pid and pid > 0:
            try:
                result = subprocess.run(
                    ['taskset', '-p', str(pid)],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    cpu_mask = result.stdout.strip().split()[-1]
                    print(f"[{process_name}] CPU affinity mask: {cpu_mask}")
            except Exception:
                pass
        
        # 监控循环 (采样多次)
        pss_samples = []
        mem_samples = []
        cgroup_samples = []
        max_wait = 300  # 最多等待 150 秒
        
        for iteration in range(max_wait):
            # 检查进程是否结束
            poll_result = proc.poll()
            if poll_result is not None:
                break
            
            # 采样 PSS、内存组成和 cgroup 统计
            if pid and pid > 0:
                current_pss = get_process_pss(pid)
                if current_pss > 0:
                    pss_samples.append(current_pss)
                mem_breakdown = get_process_mem_breakdown(pid)
                if mem_breakdown:
                    mem_samples.append(mem_breakdown)
                cg_stat = read_cgroup_memory_stat(pid)
                if cg_stat:
                    cgroup_samples.append(cg_stat)
            
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
            
            def aggregate_breakdown(samples: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
                """对采样数据求最大值 & 取平均"""
                if not samples:
                    return {}, {}
                keys: Set[str] = set()
                for s in samples:
                    keys.update(s.keys())
                max_out: Dict[str, float] = {}
                avg_out: Dict[str, float] = {}
                for k in keys:
                    vals = [s.get(k, 0.0) for s in samples]
                    max_out[k] = max(vals)
                    avg_out[k] = statistics.mean(vals)
                return max_out, avg_out

            mem_max, mem_avg = aggregate_breakdown(mem_samples)
            cgroup_max, cgroup_avg = aggregate_breakdown(cgroup_samples)
            cgroup_last = cgroup_samples[-1] if cgroup_samples else {}
            
            metrics = ProcessMetrics(
                process_name=process_name,
                pid=pid or 0,
                start_duration=start_duration,
                init_ms=timings['init_ms'],
                python_init_ms=timings.get('python_init_ms', 0.0),
                import_ms=timings['import_ms'],
                load_ms=timings['load_ms'],
                warmup_ms=timings['warmup_ms'],
                latencies=lats,
                p95_latency=p95,
                max_pss_mb=max_pss,
                avg_pss_mb=avg_pss,
                mem_breakdown_max=mem_max,
                mem_breakdown_avg=mem_avg,
                cgroup_mem_stat_max=cgroup_max,
                cgroup_mem_stat_avg=cgroup_avg,
                cgroup_mem_stat_last=cgroup_last
            )
            results_list.append(metrics)
            python_init_info = f", PyInit={metrics.python_init_ms:.1f}ms" if metrics.python_init_ms > 0 else ""
            print(f"[{process_name}] Completed: P95={p95:.2f}ms, MaxPSS={max_pss:.1f}MB{python_init_info}")
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
    timing = {'init_ms': 0.0, 'python_init_ms': 0.0, 'import_ms': 0.0, 'load_ms': 0.0, 'warmup_ms': 0.0}
    
    m_start = re.search(r'start time:\s*([0-9.]+)', text)
    if m_start:
        start_time = float(m_start.group(1))
        # 保留原来的 init_ms（从 systemd-run 到脚本开始）
        timing['init_ms'] = (start_time - start_cmd_time) * 1000.0
    
    # 尝试解析 Python 解释器初始化时间（从 PYTHON_START_TS 到脚本开始）
    m_python_init = re.search(r'python init time:\s*([0-9.]+)ms', text)
    if m_python_init:
        timing['python_init_ms'] = float(m_python_init.group(1))
    
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
    parser.add_argument('--mem', type=int, default=4096, 
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
                            'python_init_ms': m.python_init_ms,
                            'import_ms': m.import_ms,
                            'load_ms': m.load_ms,
                            'warmup_ms': m.warmup_ms,
                            'p95_latency': m.p95_latency,
                            'avg_latency': statistics.mean(m.latencies) if m.latencies else 0,
                            'min_latency': min(m.latencies) if m.latencies else 0,
                            'max_latency': max(m.latencies) if m.latencies else 0,
                            'max_pss_mb': m.max_pss_mb,
                            'avg_pss_mb': m.avg_pss_mb,
                            'mem_breakdown_max': m.mem_breakdown_max,
                            'mem_breakdown_avg': m.mem_breakdown_avg,
                            'cgroup_mem_stat_max': m.cgroup_mem_stat_max,
                            'cgroup_mem_stat_avg': m.cgroup_mem_stat_avg,
                            'cgroup_mem_stat_last': m.cgroup_mem_stat_last,
                            'latencies_count': len(m.latencies)
                        } 
                        for m in r.process_metrics
                    ]
                } for r in results
            ], f, indent=2)
        
    plot_results(results, out_dir)

if __name__ == "__main__":
    main()
