#!/usr/bin/env python3
"""
并发测试脚本：测试多应用并发实例的性能指标
- 测试1：MobileNet模型，同时起N=1,2,4...个实例
- 测试2：多个不同应用混合运行
"""

import subprocess
import time
import json
import os
import re
import statistics
import threading
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


@dataclass
class ContainerMetrics:
    """容器指标"""
    container_id: str
    container_name: str
    app_name: str
    start_time: float
    start_duration: float  # 总启动耗时(ms) - 从docker run到容器完成
    init_ms: float  # Docker启动耗时(ms) = Python进程启动时间 - date +%s%3N 时间戳(开始执行docker run命令时的时间戳)
    import_ms: float  # Import torch耗时(ms)
    load_ms: float  # 模型加载耗时(ms)
    warmup_ms: float  # 预热耗时(ms) - 第一次推理
    latencies: List[float]  # 每次推理的时延(ms)
    p95_latency: float
    p99_latency: float
    memory_stat: Dict[str, int]  # cgroup memory.stat
    pss_mb: float  # 进程PSS (MB)
    pid: Optional[int] = None


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    num_instances: int
    app_config: List[str]  # 应用配置列表，如 ["mobilenet", "mobilenet"] 或 ["mobilenet", "resnet"]
    container_metrics: List[ContainerMetrics]
    timestamp: float
    memory_full: bool  # 是否达到内存上限


def get_process_pss(pid: int) -> float:
    """
    获取进程的PSS (Proportional Set Size) 内存，单位MB
    参考: https://blog.csdn.net/m0_51504545/article/details/119685325
    """
    try:
        smaps_path = f"/proc/{pid}/smaps"
        if not os.path.exists(smaps_path):
            return 0.0
        
        pss_total = 0
        with open(smaps_path, 'r') as f:
            for line in f:
                if line.startswith('Pss:'):
                    # 格式: Pss:    1234 kB
                    parts = line.split()
                    if len(parts) >= 2:
                        pss_kb = int(parts[1])
                        pss_total += pss_kb
        
        return pss_total / 1024.0  # 转换为MB
    except Exception as e:
        print(f"Error reading PSS for PID {pid}: {e}")
        return 0.0


def get_container_pid(container_id: str) -> Optional[int]:
    """获取容器主进程的PID"""
    try:
        result = subprocess.run(
            ['docker', 'inspect', '--format', '{{.State.Pid}}', container_id],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def get_cgroup_memory_stat(container_id: str) -> Dict[str, int]:
    """获取容器的cgroup memory.stat信息"""
    try:
        # 查找容器的cgroup路径
        result = subprocess.run(
            ['docker', 'inspect', '--format', '{{.Id}}', container_id],
            capture_output=True,
            text=True,
            check=True
        )
        container_hash = result.stdout.strip()
        
        # 尝试查找memory.stat文件（可能在不同的cgroup版本路径下）
        possible_paths = [
            f"/sys/fs/cgroup/memory/docker/{container_hash}/memory.stat",
            f"/sys/fs/cgroup/memory/system.slice/docker-{container_hash}.scope/memory.stat",
            f"/sys/fs/cgroup/docker/{container_hash}/memory.stat",
        ]
        
        # 也尝试通过docker inspect获取cgroup路径
        try:
            cgroup_result = subprocess.run(
                ['docker', 'inspect', '--format', '{{.HostConfig.CgroupParent}}', container_id],
                capture_output=True,
                text=True
            )
            if cgroup_result.returncode == 0 and cgroup_result.stdout.strip():
                cgroup_parent = cgroup_result.stdout.strip()
                if cgroup_parent:
                    possible_paths.insert(0, f"{cgroup_parent}/memory.stat")
        except Exception:
            pass
        
        for path in possible_paths:
            if os.path.exists(path):
                memory_stat = {}
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if ' ' in line:
                            key, value = line.split(None, 1)
                            try:
                                memory_stat[key] = int(value)
                            except ValueError:
                                pass
                return memory_stat
        
        # 如果找不到memory.stat，尝试使用docker stats获取基本信息
        try:
            result = subprocess.run(
                ['docker', 'stats', '--no-stream', '--format', 
                 '{{.MemUsage}}\t{{.MemPerc}}', container_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # 解析输出，例如 "1.2GiB / 2GiB   60.00%"
                parts = result.stdout.strip().split()
                if len(parts) >= 1:
                    # 这里我们只返回一个标记，表示获取了docker stats
                    return {"docker_stats_available": 1}
        except Exception:
            pass
        
    except Exception as e:
        print(f"Error reading memory.stat for container {container_id}: {e}")
    
    return {}


def parse_benchmark_output(output: str, docker_start_time: float) -> Tuple[float, List[float], Dict[str, float]]:
    """
    解析benchmark输出，返回启动时间、时延列表和详细时间信息
    
    Args:
        output: benchmark脚本的输出
        docker_start_time: docker run命令开始的时间戳（用于计算初始化时间）
    
    Returns:
        (start_time, latencies, timing_details)
        timing_details包含: init_ms, import_ms, load_ms, warmup_ms
    """
    lines = output.strip().split('\n')
    text = '\n'.join(lines)
    
    start_time = None
    latencies = []
    timing_details = {
        'init_ms': 0.0,
        'import_ms': 0.0,
        'load_ms': 0.0,
        'warmup_ms': 0.0,
    }
    
    # 查找start time（Python进程启动时间）
    m_start = re.search(r'start time:\s*([0-9.]+)', text)
    if m_start:
        start_time = float(m_start.group(1))
        # 初始化时间 = Python进程启动时间 - docker命令开始时间
        timing_details['init_ms'] = (start_time - docker_start_time) * 1000.0
    
    # Import Torch Done, Time Spent: 3.563s
    m_import = re.search(r'Import Torch Done, Time Spent:\s*([0-9.]+)s', text)
    if m_import:
        timing_details['import_ms'] = float(m_import.group(1)) * 1000.0
    
    # Model Loaded, Time Spent: 0.197s
    m_load = re.search(r'Model Loaded, Time Spent:\s*([0-9.]+)s', text)
    if m_load:
        timing_details['load_ms'] = float(m_load.group(1)) * 1000.0
    
    # Warmup Done, Time Spent: 0.195s
    m_warmup = re.search(r'Warmup Done, Time Spent:\s*([0-9.]+)s', text)
    if m_warmup:
        timing_details['warmup_ms'] = float(m_warmup.group(1)) * 1000.0
    
    # 查找LATENCIES行
    for line in lines:
        if line.startswith('LATENCIES:'):
            latencies_str = line.replace('LATENCIES:', '').strip()
            latencies = [float(x) for x in latencies_str.split(',') if x.strip()]
            break
    
    return start_time, latencies, timing_details


def create_resource_limited_slice(test_name: str, total_cpus: str, total_memory_mb: int) -> Optional[str]:
    """
    创建一个 systemd slice 来限制总资源（CPU和内存）
    返回 slice 名称（不含.slice后缀），如果失败返回 None
    
    使用 systemd 的 transient slice 功能，可以动态创建并设置资源限制
    """
    slice_name = f"benchmark-{test_name}-{int(time.time())}"
    
    # 解析CPU设置
    total_cpu_cores = parse_cpus(total_cpus)
    cpu_percent = int(total_cpu_cores * 100)  # CPUQuota使用百分比，200% = 2核
    memory_bytes = total_memory_mb * 1024 * 1024  # 转换为字节
    
    # 使用 systemctl 创建 transient slice 并设置资源限制
    # 方法：创建一个临时 unit 文件，然后启动它
    try:
        # 使用 systemd-run 创建 slice（作为临时解决方案）
        # 或者直接使用 systemctl set-property 创建 transient slice
        # 但实际上，我们需要先启动一个进程在 slice 内，所以用 systemd-run
        
        # 方案：创建一个临时的 systemd service，在 slice 内运行 sleep
        # 然后在这个 slice 内运行所有容器
        slice_unit = f"{slice_name}.slice"
        
        # 尝试直接创建 slice（systemd 支持 transient units）
        # 但更简单的方法是：使用 systemd-run 运行一个临时进程，并指定 slice
        # 这样 slice 会自动创建，然后我们可以在其中运行其他进程
        
        print(f"Creating resource-limited slice: {slice_unit}")
        print(f"  CPU: {total_cpu_cores} cores ({cpu_percent}%)")
        print(f"  Memory: {total_memory_mb} MB")
        
        # 注意：systemd-run 创建的 slice 需要至少有一个进程在其中
        # 我们启动一个 sleep 进程作为占位符
        sleep_result = subprocess.run(
            ['systemd-run', '--slice', slice_unit,
             f'--property=CPUQuota={cpu_percent}%',
             f'--property=MemoryMax={memory_bytes}',
             'sleep', '3600'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if sleep_result.returncode == 0:
            return slice_name
        else:
            # 如果上面的方法失败，尝试使用 systemctl set-property
            # 先创建 slice（通过启动一个临时进程）
            subprocess.run(['systemd-run', '--slice', slice_unit, 'true'],
                         capture_output=True, timeout=5)
            # 然后设置属性
            subprocess.run(['systemctl', 'set-property', slice_unit,
                          f'CPUQuota={cpu_percent}%',
                          f'MemoryMax={memory_bytes}'],
                         capture_output=True, timeout=5)
            return slice_name
            
    except Exception as e:
        print(f"Warning: Failed to create systemd slice: {e}")
        return None


def cleanup_slice(slice_name: Optional[str]):
    """清理 systemd slice"""
    if not slice_name:
        return
    try:
        slice_unit = f"{slice_name}.slice"
        # 停止 slice（会停止其中的所有进程）
        subprocess.run(['systemctl', 'stop', slice_unit], 
                     capture_output=True, timeout=10)
    except Exception:
        pass


def run_container(app_name: str, container_name: str, 
                  scope_name: Optional[str], volume: str, script_name: str) -> Tuple[str, float, str, Dict, float]:
    """
    在指定的 systemd scope 内运行一个容器
    注意：所有容器共享总资源限制，由操作系统动态调度
    
    返回: (container_id, start_duration, output, runtime_metrics, docker_start_time)
    runtime_metrics包含: {'pid': pid, 'pss_mb': pss, 'memory_stat': stat}
    docker_start_time: docker run命令开始的时间戳（用于计算初始化时间）
    
    Args:
        app_name: 应用名称
        container_name: 容器名称
        scope_name: systemd scope 名称，如果为None则不使用scope
        volume: 卷挂载
        script_name: 脚本名称
    """
    start_cmd_time = time.time()
    runtime_metrics = {}
    
    # 构建docker run命令
    # 不设置容器级别的CPU和内存限制，让操作系统在scope的限制内动态分配
    docker_cmd = [
        'docker', 'run', '-d',
        '--name', container_name,
        '-v', volume,
        'torch-cpu',
        'python', script_name
    ]
    
    # 如果指定了scope，使用systemd-run在scope内运行docker命令
    # 注意：systemd-run --scope 需要指定在哪个scope内运行
    # 我们使用 --slice 选项让所有容器在同一个slice下
    if scope_name:
        # 使用 --slice 让容器在指定的scope/slice下运行
        cmd = [
            'systemd-run', '--scope',
            '--slice', scope_name.replace('.scope', '.slice'),  # slice名称
            '--',
        ] + docker_cmd
    else:
        cmd = docker_cmd
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        container_id = result.stdout.strip()
        start_duration = (time.time() - start_cmd_time) * 1000  # ms
        
        # 等待容器启动（检查容器状态）
        time.sleep(1.0)  # 给容器一点时间启动
        
        # 在容器运行期间收集指标（benchmark执行期间）
        # 由于benchmark需要运行一段时间，我们有时间窗口收集指标
        max_wait_time = 300  # 5分钟超时
        wait_start = time.time()
        collected_metrics = False
        is_running = True
        
        # 使用docker logs -f在后台跟踪，同时收集指标
        while time.time() - wait_start < max_wait_time:
            # 检查容器是否还在运行
            check_result = subprocess.run(
                ['docker', 'inspect', '--format', '{{.State.Running}}', container_id],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if check_result.returncode == 0:
                is_running = check_result.stdout.strip() == 'true'
                if is_running and not collected_metrics:
                    # 容器还在运行，收集指标
                    pid = get_container_pid(container_id)
                    if pid:
                        try:
                            pss_mb = get_process_pss(pid)
                            memory_stat = get_cgroup_memory_stat(container_id)
                            runtime_metrics = {
                                'pid': pid,
                                'pss_mb': pss_mb,
                                'memory_stat': memory_stat
                            }
                            collected_metrics = True
                            # 收集到指标后可以继续等待容器完成
                        except Exception as e:
                            print(f"Warning: Failed to collect metrics for {container_name}: {e}")
                
                if not is_running:
                    # 容器已完成
                    break
            else:
                # 检查失败，可能容器已退出
                is_running = False
                break
            
            time.sleep(0.5)  # 每0.5秒检查一次
        
        # 如果容器还在运行，等待完成
        if is_running:
            wait_result = subprocess.run(
                ['docker', 'wait', container_id],
                capture_output=True,
                text=True,
                timeout=max(1, max_wait_time - (time.time() - wait_start))
            )
        
        # 获取容器输出
        logs_result = subprocess.run(
            ['docker', 'logs', container_id],
            capture_output=True,
            text=True
        )
        output = logs_result.stdout
        
        if logs_result.returncode != 0 and not output:
            raise RuntimeError(f"Failed to get container logs: {logs_result.stderr}")
        
        return container_id, start_duration, output, runtime_metrics, start_cmd_time
    except subprocess.TimeoutExpired as e:
        print(f"Container {container_name} timed out")
        subprocess.run(['docker', 'stop', container_name], capture_output=True, timeout=5)
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error running container {container_name}: {e.stderr}")
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=5)
        raise
    except Exception as e:
        print(f"Unexpected error running container {container_name}: {e}")
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=5)
        raise


def save_result_to_file(result: TestResult, output_dir: str):
    """
    将单个测试结果保存到文件，记录详细的原始数据
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细的JSON结果
    result_data = {
        'test_name': result.test_name,
        'num_instances': result.num_instances,
        'app_config': result.app_config,
        'timestamp': result.timestamp,
        'memory_full': result.memory_full,
        'containers': []
    }
    
    for metrics in result.container_metrics:
        container_data = {
            'container_id': metrics.container_id,
            'container_name': metrics.container_name,
            'app_name': metrics.app_name,
            'start_time': metrics.start_time,
            # 总启动耗时
            'start_duration_ms': metrics.start_duration,
            # 细分阶段耗时（与 auto_benchmark.py 含义一致）
            'init_ms': metrics.init_ms,
            'import_ms': metrics.import_ms,
            'load_ms': metrics.load_ms,
            'warmup_ms': metrics.warmup_ms,
            # 推理延迟统计
            'pid': metrics.pid,
            'pss_mb': metrics.pss_mb,
            'p95_latency_ms': metrics.p95_latency,
            'p99_latency_ms': metrics.p99_latency,
            'avg_latency_ms': statistics.mean(metrics.latencies) if metrics.latencies else 0,
            'min_latency_ms': min(metrics.latencies) if metrics.latencies else 0,
            'max_latency_ms': max(metrics.latencies) if metrics.latencies else 0,
            'all_latencies_ms': metrics.latencies,  # 保存所有原始时延数据
            'memory_stat': metrics.memory_stat,
        }
        result_data['containers'].append(container_data)
    
    # 保存到单独的文件
    result_file = os.path.join(output_dir, f"{result.test_name}_result.json")
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    print(f"Result saved to {result_file}")


def parse_cpus(cpus_str: str) -> float:
    """
    解析CPU字符串，转换为CPU核数
    支持格式：
    - "0-1" -> 2.0 (2个CPU)
    - "2" -> 1.0 (1个CPU)
    - "0,1,2" -> 3.0 (3个CPU)
    """
    if '-' in cpus_str:
        # 范围格式，如 "0-1"
        start, end = cpus_str.split('-')
        return float(int(end) - int(start) + 1)
    elif ',' in cpus_str:
        # 列表格式，如 "0,1,2"
        return float(len(cpus_str.split(',')))
    else:
        # 单个CPU
        return 1.0


def run_concurrent_test(test_name: str, app_config: List[str], 
                       total_cpus: str, volume: str,
                       total_memory_mb: int = 2048) -> TestResult:
    """
    运行并发测试
    注意：所有容器共享总资源限制，而不是每个容器独立限制
    
    Args:
        test_name: 测试名称
        app_config: 应用配置列表，如 ["mobilenet", "mobilenet"] 或 ["mobilenet", "resnet"]
        total_cpus: 所有容器共享的总CPU，如 "0-1" 表示2核
        volume: 卷挂载，如 "/path/to/app:/app"
        total_memory_mb: 所有容器共享的总内存限制(MB)，用于判断是否内存满
    """
    app_script_map = {
        'mobilenet': 'benchmark.py',
        'resnet': 'benchmark_resnet.py',
        'efficientnet': 'benchmark_efficientnet.py',
        'shufflenet': 'benchmark_shufflenet.py',
    }
    
    num_instances = len(app_config)
    container_metrics_list = []
    total_used_memory_mb = 0
    
    # 解析总CPU核数
    total_cpu_cores = parse_cpus(total_cpus)
    
    print(f"\n{'='*60}")
    print(f"Running test: {test_name}")
    print(f"Number of instances: {num_instances}")
    print(f"App config: {app_config}")
    print(f"Total CPU cores (shared): {total_cpu_cores}")
    print(f"Total memory limit (shared): {total_memory_mb} MB")
    print(f"{'='*60}\n")
    
    # 创建资源限制的 slice
    slice_name = create_resource_limited_slice(test_name, total_cpus, total_memory_mb)
    if not slice_name:
        print("Warning: Failed to create resource-limited slice, running without resource limits")
    
    # 实际测试运行（第warmup_runs+1次）
    # 并发启动所有容器
    containers_info = []
    containers_info_lock = threading.Lock()
    
    def run_container_thread(idx, app_name):
        """在线程中运行容器"""
        container_name = f"{test_name}_container_{idx}_{int(time.time() * 1000)}"
        script_name = app_script_map.get(app_name, 'benchmark.py')
        
        try:
            container_id, start_duration, output, runtime_metrics, docker_start_time = run_container(
                app_name, container_name, slice_name, volume, script_name
            )
            
            with containers_info_lock:
                containers_info.append({
                    'idx': idx,
                    'app_name': app_name,
                    'container_id': container_id,
                    'container_name': container_name,
                    'start_duration': start_duration,
                    'output': output,
                    'runtime_metrics': runtime_metrics,
                    'docker_start_time': docker_start_time,
                })
        except Exception as e:
            print(f"Failed to run container {idx} ({app_name}): {e}")
            import traceback
            traceback.print_exc()
    
    # 启动所有容器的线程
    start_all_time = time.time()
    threads = []
    for idx, app_name in enumerate(app_config):
        thread = threading.Thread(target=run_container_thread, args=(idx, app_name))
        thread.start()
        threads.append(thread)
    
    # 等待所有容器完成
    for thread in threads:
        thread.join()
    
    total_start_time = (time.time() - start_all_time) * 1000
    print(f"All {len(containers_info)} containers completed in {total_start_time:.2f} ms")
    
    # 收集每个容器的指标
    for container_info in containers_info:
        container_id = container_info['container_id']
        container_name = container_info['container_name']
        app_name = container_info['app_name']
        start_duration = container_info['start_duration']
        output = container_info['output']
        runtime_metrics = container_info.get('runtime_metrics', {})
        docker_start_time = container_info.get('docker_start_time', 0.0)
        
        # 解析输出：启动时间 + 每阶段耗时
        start_time, latencies, timing_details = parse_benchmark_output(output, docker_start_time)
        
        if not latencies:
            print(f"Warning: No latencies found for container {container_name}")
            continue
        
        # 计算P95和P99
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
        
        # 获取PID和PSS（从运行时指标或尝试获取）
        pid = runtime_metrics.get('pid')
        pss_mb = runtime_metrics.get('pss_mb', 0.0)
        memory_stat = runtime_metrics.get('memory_stat', {})
        
        if not pid:
            # 如果运行时没有收集到，尝试获取（可能失败）
            pid = get_container_pid(container_id)
            if pid and os.path.exists(f"/proc/{pid}"):
                pss_mb = get_process_pss(pid)
        
        if not memory_stat:
            memory_stat = get_cgroup_memory_stat(container_id)
        
        total_used_memory_mb += pss_mb
        
        metrics = ContainerMetrics(
            container_id=container_id,
            container_name=container_name,
            app_name=app_name,
            start_time=start_time or time.time(),
            start_duration=start_duration,
            init_ms=timing_details.get('init_ms', 0.0),
            import_ms=timing_details.get('import_ms', 0.0),
            load_ms=timing_details.get('load_ms', 0.0),
            warmup_ms=timing_details.get('warmup_ms', 0.0),
            latencies=latencies,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            memory_stat=memory_stat,
            pss_mb=pss_mb,
            pid=pid
        )
        
        container_metrics_list.append(metrics)
        
        print(f"\nContainer {container_name} ({app_name}):")
        print(f"  Start duration: {start_duration:.2f} ms")
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  P99 latency: {p99_latency:.2f} ms")
        print(f"  PSS: {pss_mb:.2f} MB")
        print(f"  PID: {pid}")
    
    # 判断是否内存满（所有容器的总内存使用）
    memory_full = total_used_memory_mb >= total_memory_mb * 0.95  # 达到95%认为内存满
    
    result = TestResult(
        test_name=test_name,
        num_instances=num_instances,
        app_config=app_config,
        container_metrics=container_metrics_list,
        timestamp=time.time(),
        memory_full=memory_full
    )
    
    # 清理容器
    for container_info in containers_info:
        try:
            subprocess.run(['docker', 'rm', '-f', container_info['container_name']], 
                         capture_output=True)
        except Exception:
            pass
    
    return result


def detect_jitter(latencies_list: List[List[float]], threshold: float = 0.3) -> Dict[str, float]:
    """
    检测抖动
    使用变异系数(CV)来衡量抖动程度
    """
    if not latencies_list:
        return {"jitter_detected": False, "cv": 0.0}
    
    # 计算每个容器的平均时延
    avg_latencies = [statistics.mean(lats) for lats in latencies_list if lats]
    
    if len(avg_latencies) < 2:
        return {"jitter_detected": False, "cv": 0.0}
    
    mean_avg = statistics.mean(avg_latencies)
    std_avg = statistics.stdev(avg_latencies) if len(avg_latencies) > 1 else 0
    cv = std_avg / mean_avg if mean_avg > 0 else 0
    
    jitter_detected = cv > threshold
    
    return {
        "jitter_detected": jitter_detected,
        "cv": cv,
        "mean_latency": mean_avg,
        "std_latency": std_avg
    }


def plot_results(test1_results: List[TestResult], test2_results: List[TestResult],
                 output_dir: str = "results"):
    """绘制测试结果图表"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试1：MobileNet多实例
    if test1_results:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Test 1: MobileNet Multiple Instances', fontsize=16)
        
        num_instances = [r.num_instances for r in test1_results]
        
        # 启动时间
        ax = axes[0, 0]
        start_times = [statistics.mean([m.start_duration for m in r.container_metrics]) 
                      for r in test1_results]
        ax.plot(num_instances, start_times, 'o-', color='blue')
        ax.set_xlabel('Number of Instances')
        ax.set_ylabel('Start Time (ms)')
        ax.set_title('Container Start Time')
        ax.grid(True)
        
        # P95时延
        ax = axes[0, 1]
        p95_latencies = [statistics.mean([m.p95_latency for m in r.container_metrics]) 
                        for r in test1_results]
        ax.plot(num_instances, p95_latencies, 'o-', color='green')
        ax.set_xlabel('Number of Instances')
        ax.set_ylabel('P95 Latency (ms)')
        ax.set_title('P95 Latency')
        ax.grid(True)
        
        # P99时延
        ax = axes[0, 2]
        p99_latencies = [statistics.mean([m.p99_latency for m in r.container_metrics]) 
                        for r in test1_results]
        ax.plot(num_instances, p99_latencies, 'o-', color='red')
        ax.set_xlabel('Number of Instances')
        ax.set_ylabel('P99 Latency (ms)')
        ax.set_title('P99 Latency')
        ax.grid(True)
        
        # PSS内存
        ax = axes[1, 0]
        pss_values = [statistics.mean([m.pss_mb for m in r.container_metrics]) 
                     for r in test1_results]
        total_pss = [sum([m.pss_mb for m in r.container_metrics]) for r in test1_results]
        ax.plot(num_instances, pss_values, 'o-', label='Avg PSS per Container', color='purple')
        ax.plot(num_instances, total_pss, 's-', label='Total PSS', color='orange')
        ax.set_xlabel('Number of Instances')
        ax.set_ylabel('PSS Memory (MB)')
        ax.set_title('Memory Usage (PSS)')
        ax.legend()
        ax.grid(True)
        
        # 抖动检测
        ax = axes[1, 1]
        jitter_cvs = []
        for r in test1_results:
            latencies_list = [m.latencies for m in r.container_metrics]
            jitter_info = detect_jitter(latencies_list)
            jitter_cvs.append(jitter_info['cv'])
        ax.plot(num_instances, jitter_cvs, 'o-', color='brown')
        ax.axhline(y=0.3, color='r', linestyle='--', label='Jitter Threshold')
        ax.set_xlabel('Number of Instances')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Jitter Detection (CV)')
        ax.legend()
        ax.grid(True)
        
        # 时延分布箱线图
        ax = axes[1, 2]
        all_latencies_by_instance = []
        labels = []
        for r in test1_results:
            all_latencies = []
            for m in r.container_metrics:
                all_latencies.extend(m.latencies)
            all_latencies_by_instance.append(all_latencies)
            labels.append(f"N={r.num_instances}")
        if all_latencies_by_instance:
            ax.boxplot(all_latencies_by_instance, labels=labels)
            ax.set_xlabel('Number of Instances')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency Distribution')
            ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test1_mobilenet_multiple_instances.png", dpi=150)
        plt.close()
    
    # 测试2：混合应用
    if test2_results:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Test 2: Mixed Applications', fontsize=16)
        
        # 按应用类型分组
        app_metrics = defaultdict(list)
        for r in test2_results:
            for m in r.container_metrics:
                app_metrics[m.app_name].append(m)
        
        apps = list(app_metrics.keys())
        
        # P95时延对比
        ax = axes[0, 0]
        p95_by_app = [statistics.mean([m.p95_latency for m in app_metrics[app]]) 
                      for app in apps]
        ax.bar(apps, p95_by_app, color=['blue', 'green', 'red', 'purple'][:len(apps)])
        ax.set_xlabel('Application')
        ax.set_ylabel('P95 Latency (ms)')
        ax.set_title('P95 Latency by Application')
        ax.grid(True, axis='y')
        
        # P99时延对比
        ax = axes[0, 1]
        p99_by_app = [statistics.mean([m.p99_latency for m in app_metrics[app]]) 
                      for app in apps]
        ax.bar(apps, p99_by_app, color=['blue', 'green', 'red', 'purple'][:len(apps)])
        ax.set_xlabel('Application')
        ax.set_ylabel('P99 Latency (ms)')
        ax.set_title('P99 Latency by Application')
        ax.grid(True, axis='y')
        
        # PSS内存对比
        ax = axes[0, 2]
        pss_by_app = [statistics.mean([m.pss_mb for m in app_metrics[app]]) 
                     for app in apps]
        ax.bar(apps, pss_by_app, color=['blue', 'green', 'red', 'purple'][:len(apps)])
        ax.set_xlabel('Application')
        ax.set_ylabel('PSS Memory (MB)')
        ax.set_title('Memory Usage (PSS) by Application')
        ax.grid(True, axis='y')
        
        # 启动时间对比
        ax = axes[1, 0]
        start_by_app = [statistics.mean([m.start_duration for m in app_metrics[app]]) 
                       for app in apps]
        ax.bar(apps, start_by_app, color=['blue', 'green', 'red', 'purple'][:len(apps)])
        ax.set_xlabel('Application')
        ax.set_ylabel('Start Time (ms)')
        ax.set_title('Container Start Time by Application')
        ax.grid(True, axis='y')
        
        # 时延分布对比
        ax = axes[1, 1]
        all_latencies_by_app = [sum([m.latencies for m in app_metrics[app]], []) 
                               for app in apps]
        if all_latencies_by_app:
            ax.boxplot(all_latencies_by_app, labels=apps)
            ax.set_xlabel('Application')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency Distribution by Application')
            ax.grid(True, axis='y')
        
        # 抖动检测
        ax = axes[1, 2]
        jitter_by_app = []
        for app in apps:
            latencies_list = [m.latencies for m in app_metrics[app]]
            jitter_info = detect_jitter(latencies_list)
            jitter_by_app.append(jitter_info['cv'])
        ax.bar(apps, jitter_by_app, color=['blue', 'green', 'red', 'purple'][:len(apps)])
        ax.axhline(y=0.3, color='r', linestyle='--', label='Jitter Threshold')
        ax.set_xlabel('Application')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Jitter Detection (CV) by Application')
        ax.legend()
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test2_mixed_applications.png", dpi=150)
        plt.close()


def main():
    """主函数"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Concurrent benchmark test')
    parser.add_argument('--total-cpus', type=str, default='0-1', 
                       help='Total CPU cores shared by all containers, e.g., "0-1" means 2 cores (default: 0-1)')
    parser.add_argument('--total-memory-mb', type=int, default=2048,
                       help='Total memory limit (MB) shared by all containers (default: 2048)')
    parser.add_argument('--volume', type=str, default=None,
                       help='Volume mount (default: current_dir:/app)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Base output directory for results (default: results)')
    parser.add_argument('--max-instances', type=int, default=16,
                       help='Maximum number of instances to test (default: 16)')
    
    args = parser.parse_args()
    
    # 设置默认volume
    if args.volume is None:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        args.volume = f"{current_dir}:/app"
    
    # 根据实验开始时间创建输出目录
    experiment_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"experiment_{experiment_start_time}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # 保存实验配置
    config_file = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'start_time': experiment_start_time,
            'total_cpus': args.total_cpus,
            'total_cpu_cores': parse_cpus(args.total_cpus),
            'total_memory_mb': args.total_memory_mb,
            'volume': args.volume,
            'max_instances': args.max_instances,
            'note': 'All containers share the total resource limits, not per-container limits',
        }, f, indent=2)
    
    test1_results = []
    test2_results = []
    
    # 用于汇总所有结果的列表
    all_results = []
    
    # 测试1：MobileNet多实例，N=1,2,4,8...
    print("\n" + "="*60)
    print("TEST 1: MobileNet Multiple Instances")
    print("="*60)
    
    # 动态增加实例数直到内存满
    num_instances = 1
    while num_instances <= args.max_instances:
        app_config = ['mobilenet'] * num_instances
        test_name = f"test1_mobilenet_n{num_instances}"
        
        try:
            result = run_concurrent_test(
                test_name=test_name,
                app_config=app_config,
                total_cpus=args.total_cpus,
                volume=args.volume,
                total_memory_mb=args.total_memory_mb
            )
            
            current_memory = sum([m.pss_mb for m in result.container_metrics])
            test1_results.append(result)
            all_results.append(result)
            
            # 立即保存单个实验结果
            save_result_to_file(result, args.output_dir)
            
            print(f"Test with {num_instances} instances: Total memory usage = {current_memory:.2f} MB / {args.total_memory_mb} MB")
            
            if result.memory_full or current_memory >= args.total_memory_mb * 0.95:
                print(f"Memory limit reached at {num_instances} instances")
                break
            
            # 如果内存使用较少，可以增加更多实例
            if current_memory < args.total_memory_mb * 0.5:
                num_instances *= 2  # 快速增加
            else:
                num_instances += 1  # 慢速增加
                
            time.sleep(2)  # 等待一下再运行下一个测试
        except Exception as e:
            print(f"Error in test1 with n={num_instances}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 测试2：混合应用
    print("\n" + "="*60)
    print("TEST 2: Mixed Applications")
    print("="*60)
    
    # 测试不同的混合配置（逐渐增加直到内存满）
    base_configs = [
        ['mobilenet', 'resnet'],
        ['mobilenet', 'resnet', 'shufflenet'],
    ]
    
    for base_config in base_configs:
        # 从基础配置开始，逐渐增加实例数
        multiplier = 1
        while True:
            app_config = base_config * multiplier
            if len(app_config) > args.max_instances:
                break
            
            test_name = f"test2_mixed_{'_'.join(base_config)}_x{multiplier}"
            try:
                result = run_concurrent_test(
                    test_name=test_name,
                    app_config=app_config,
                    total_cpus=args.total_cpus,
                    volume=args.volume,
                    total_memory_mb=args.total_memory_mb
                )
                
                current_memory = sum([m.pss_mb for m in result.container_metrics])
                test2_results.append(result)
                all_results.append(result)
                
                # 立即保存单个实验结果
                save_result_to_file(result, args.output_dir)
                
                print(f"Mixed test {base_config} x{multiplier}: Total memory usage = {current_memory:.2f} MB / {args.total_memory_mb} MB")
                
                if result.memory_full or current_memory >= args.total_memory_mb * 0.95:
                    print(f"Memory limit reached for {base_config}")
                    break
                
                multiplier += 1
                time.sleep(2)
            except Exception as e:
                print(f"Error in test2 with config {app_config}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    # 保存汇总结果（所有结果已单独保存，这里保存汇总信息）
    summary_file = os.path.join(args.output_dir, "summary.json")
    summary_data = {
        'experiment_start_time': experiment_start_time,
        'experiment_end_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'total_tests': len(all_results),
        'test1_count': len(test1_results),
        'test2_count': len(test2_results),
        'test1_summary': [
            {
                'test_name': r.test_name,
                'num_instances': r.num_instances,
                'total_memory_mb': sum([m.pss_mb for m in r.container_metrics]),
                'avg_p95_latency_ms': statistics.mean([m.p95_latency for m in r.container_metrics]) if r.container_metrics else 0,
                'avg_p99_latency_ms': statistics.mean([m.p99_latency for m in r.container_metrics]) if r.container_metrics else 0,
            }
            for r in test1_results
        ],
        'test2_summary': [
            {
                'test_name': r.test_name,
                'num_instances': r.num_instances,
                'app_config': r.app_config,
                'total_memory_mb': sum([m.pss_mb for m in r.container_metrics]),
            }
            for r in test2_results
        ],
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_file}")
    
    # 绘制图表（如果matplotlib可用）
    if HAS_MATPLOTLIB:
        plot_results(test1_results, test2_results, args.output_dir)
        print(f"Plots saved to {args.output_dir}/")
    else:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()

