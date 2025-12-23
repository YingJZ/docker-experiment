# concurrent_benchmark.py 实现说明

当前存在的问题：
- 用 `systemd-run --property=CPUQuota=xx%` 限制 Slice 的 CPU 使用率时，发现其限制的是总体使用率，即线程**仍然可以使用所有 CPU 核心**，只是总使用率被限制了。这会导致线程频繁切换，影响测试结果的准确性。

## `run_concurrent_benchmark`

该函数是主要测试逻辑。

1. 调用 `create_resource_limited_slice()` 创建资源受限的 Slice。
2. 启动多个线程，每个线程运行 `run_container_task()` 函数
3. 在所有线程完成后，清理 Slice
4. 收集并返回所有线程的结果


### `create_resource_limited_slice`

该函数用于创建一个资源受限的 Slice，并返回其名称

```python
subprocess.Popen(
    ['systemd-run', '--unit', slice_name, '--slice', slice_name,
     f'--property=CPUQuota={cpu_quota}%',
     f'--property=MemoryMax={memory_bytes}',
     'sleep', '3600'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(0.5)
return slice_name
```


### `run_container_task`

实际运行的指令 `docker_cmd` 包含一下几部分：

```python
    docker_cmd = [
        'docker', 'run', '-d',
        '--name', container_name,
        # 不使用 --rm，手动删除以便先获取日志
        '-v', volume,
    ] + [ # env_vars，用于限制容器内线程，防止上下文切换风暴
        '-e', f'OMP_NUM_THREADS={threads_limit}',
        '-e', f'MKL_NUM_THREADS={threads_limit}',
        '-e', f'PYTORCH_NUM_THREADS={threads_limit}'
    ] + ['--cgroup-parent', slice_name] + 
        ['torch-cpu', 'python', script_name]
]
```

主要逻辑：
1. 启动逻辑 `res = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)`
2. 每 0.5 秒通过 `docker inspect` 命令获取 pid，进而查询 pss
3. 容器运行结束后，通过 `docker logs` 获取输出结果，解析耗时
