## 实验结果

参数说明：

- torch threads: compute=2 interop=16 表示 PyTorch 内部计算线程数为 2，调度线程数为 16
- env OMP_NUM_THREADS=2 表示 OpenMP 使用 2 个线程
- env MKL_NUM_THREADS=2 表示 MKL 使用 2 个线程
- env OPENBLAS_NUM_THREADS=2 表示 OpenBLAS 使用 2 个线程


| 主机/容器 | 指令 | torch_compute | torch_interop | envs | 模型加载时间 | 模型单次推理平均耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 主机 | `systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=2G python benchmark.py` | 16 | 16 | unset | 136.6 ms | 31.78 ms |
| 主机 | `systemd-run --user --scope -p AllowedCPUs=0-1 -p MemoryMax=2G python benchmark.py` | 16 | 16 | unset | 140.3 ms | 29.40 ms |
| 主机 | `systemd-run --user --scope -p AllowedCPUs=0-1 -p MemoryMax=2G --setenv=OMP_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 --setenv=TORCH_NUM_THREADS=2 --setenv=TORCH_NUM_INTEROP_THREADS=2 python benchmark.py` | 2 | 16 | OMP/MKL/OPENBLAS=2, TORCH=2/2 | 136.3 ms | 41.76 ms |
| 主机 | `taskset -c 0,1 python benchmark.py` | 2 | 16 | unset | 120.9 ms | 37.44 ms |
| 容器 | `sudo docker run --rm --cpus="2.0" --memory="2g" --cpuset-cpus="0-1" -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e TORCH_NUM_THREADS=2 -e TORCH_NUM_INTEROP_THREADS=2 -v $(pwd):/app torch-cpu python benchmark.py` | 2 | 16 | OMP/MKL/OPENBLAS=2, TORCH=2/2 | 325.1 ms | 38.46 ms |
| 容器 | `sudo docker run --rm --memory="2g" --cpuset-cpus="0-1" -v $(pwd):/app torch-cpu python benchmark.py` | 2 | 16 | unset | 162.8 ms | 38.92 ms |
| 容器 | `sudo docker run --rm --cpuset-cpus="0-1" -v $(pwd):/app torch-cpu python benchmark.py` | 2 | 16 | unset | 163.7 ms | 40.68 ms |
｜容器 | `sudo docker run --rm --memory="2g" --cpuset-cpus="0-1" -e OMP_NUM_THREADS=16 -e MKL_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 -e TORCH_NUM_THREADS=16 -e TORCH_NUM_INTEROP_THREADS=16 -v $(pwd):/app torch-cpu python benchmark.py` | 16 | 16 | OMP/MKL/OPENBLAS=16, TORCH=16/16 | 347.5 ms | 77.89 ms |


### 主机直接运行（仅保留含 Runtime Info 的结果）

#### 主机受限（含运行时信息）

model load: 136.6ms
inference: 31.78ms

```
(experiment) yingjiaze@haslab4:~/experiment$ systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=2G /home/yingjiaze/experiment/.venv/bin/python /home/yingjiaze/experiment/benchmark.py
Running scope as unit: run-r4a3150dc9b7946a98bb52194427b13a8.scope
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.8.0+cu128
mkldnn: True
mkl: True
openmp: True
torch threads: compute=16 interop=16
env OMP_NUM_THREADS=
env MKL_NUM_THREADS=
env OPENBLAS_NUM_THREADS=
====================
[0.0297s] Loading Model...
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
[0.1366s] Model Loaded
[0.2001s] Warmup Done
[0.2002s] Starting Inference Loop (100 iters)...
[3.3779s] Inference Done. Avg Latency: 31.78 ms
```

#### 主机仅限制 AllowedCPUs=0-1（不使用 CPUQuota），内存=2G；不限制线程

model load: 140.3ms
inference: 29.40ms

```
(experiment) yingjiaze@haslab4:~/experiment$ systemd-run --user --scope -p AllowedCPUs=0-1 -p MemoryMax=2G /home/yingjiaze/experiment/.venv/bin/python /home/yingjiaze/experiment/benchmark.py
Running scope as unit: run-r5cc8dc0d3ad1494db3ac3bac9f6d2a89.scope
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.8.0+cu128
mkldnn: True
mkl: True
openmp: True
torch threads: compute=16 interop=16
env OMP_NUM_THREADS=
env MKL_NUM_THREADS=
env OPENBLAS_NUM_THREADS=
====================
[0.0304s] Loading Model...
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
[0.1403s] Model Loaded
[0.2323s] Warmup Done
[0.2326s] Starting Inference Loop (100 iters)...
[3.1727s] Inference Done. Avg Latency: 29.40 ms
```

#### 主机仅限制 AllowedCPUs=0-1（不使用 CPUQuota），内存=2G；显式 2 线程

model load: 136.3ms
inference: 41.76ms

```
(experiment) yingjiaze@haslab4:~/experiment$ systemd-run --user --scope -p AllowedCPUs=0-1 -p MemoryMax=2G \
  --setenv=OMP_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 \
  --setenv=TORCH_NUM_THREADS=2 --setenv=TORCH_NUM_INTEROP_THREADS=2 \
  /home/yingjiaze/experiment/.venv/bin/python /home/yingjiaze/experiment/benchmark.py
Running scope as unit: run-r77a9883fb01a4107993fd8c697de7262.scope
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.8.0+cu128
mkldnn: True
mkl: True
openmp: True
torch threads: compute=2 interop=16
env OMP_NUM_THREADS=2
env MKL_NUM_THREADS=2
env OPENBLAS_NUM_THREADS=2
====================
[0.0298s] Loading Model...
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
[0.1363s] Model Loaded
[0.2147s] Warmup Done
[0.2147s] Starting Inference Loop (100 iters)...
[4.3909s] Inference Done. Avg Latency: 41.76 ms
```

#### 主机使用 taskset 绑定 CPU 0-1；不限制线程（含运行时信息）

model load: 120.9ms
inference: 37.44ms

```
(experiment) yingjiaze@haslab4:~/experiment$ source ./.venv/bin/activate
(experiment) yingjiaze@haslab4:~/experiment$ taskset -c 0,1 python benchmark.py
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.8.0+cu128
mkldnn: True
mkl: True
openmp: True
torch threads: compute=2 interop=16
env OMP_NUM_THREADS=
env MKL_NUM_THREADS=
env OPENBLAS_NUM_THREADS=
====================
[0.0229s] Loading Model...
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/yingjiaze/experiment/.venv/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
[0.1209s] Model Loaded
[0.1917s] Warmup Done
[0.1917s] Starting Inference Loop (100 iters)...
[3.9354s] Inference Done. Avg Latency: 37.44 ms
```

### 容器内，限制 CPU=2，内存=2G

model load: 325.1ms
inference: 38.46ms

```bash
yingjiaze@haslab4:~/experiment$ sudo docker run --rm \
  --cpus="2.0" --memory="2g" --cpuset-cpus="0-1" \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -e TORCH_NUM_THREADS=2 -e TORCH_NUM_INTEROP_THREADS=2 \
  -v $(pwd):/app torch-cpu python benchmark.py
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.25
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.41
torch: 2.5.1+cpu
mkldnn: True
mkl: True
openmp: True
torch threads: compute=2 interop=16
env OMP_NUM_THREADS=2
env MKL_NUM_THREADS=2
env OPENBLAS_NUM_THREADS=2
====================
[0.2071s] Loading Model...
[0.3251s] Model Loaded
[0.3940s] Warmup Done
[0.3940s] Starting Inference Loop (100 iters)...
[4.2399s] Inference Done. Avg Latency: 38.46 ms
```

#### 容器内，仅限制 cpuset=0-1（不使用 --cpus），内存=2G；不限制线程

model load: 162.8ms
inference: 38.92ms

```
yingjiaze@haslab4:~/experiment$ sudo docker run --rm --memory="2g" --cpuset-cpus="0-1" -v $(pwd):/app torch-cpu python benchmark.py
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.25
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.41
torch: 2.5.1+cpu
mkldnn: True
mkl: True
openmp: True
torch threads: compute=2 interop=16
env OMP_NUM_THREADS=
env MKL_NUM_THREADS=
env OPENBLAS_NUM_THREADS=
====================
[0.0368s] Loading Model...
[0.1628s] Model Loaded
[0.2313s] Warmup Done
[0.2313s] Starting Inference Loop (100 iters)...
[4.1230s] Inference Done. Avg Latency: 38.92 ms
```

#### 容器内，仅限制 cpuset=0-1（无 CPU 配额、无内存限制），不限制线程

model load: 163.7ms
inference: 40.68ms

```
yingjiaze@haslab4:~/experiment$ sudo docker run --rm --cpuset-cpus="0-1" -v $(pwd):/app torch-cpu python benchmark.py
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.25
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.41
torch: 2.5.1+cpu
mkldnn: True
mkl: True
openmp: True
torch threads: compute=2 interop=16
env OMP_NUM_THREADS=
env MKL_NUM_THREADS=
env OPENBLAS_NUM_THREADS=
====================
[0.0370s] Loading Model...
[0.1637s] Model Loaded
[0.2328s] Warmup Done
[0.2328s] Starting Inference Loop (100 iters)...
[4.3009s] Inference Done. Avg Latency: 40.68 ms
```

#### 容器内，手动将线程限制设置为 16，效果反而变差？

```bash
yingjiaze@haslab4:~/experiment$ sudo docker run --rm \
>   --memory="2g" --cpuset-cpus="0-1" \
>   -e OMP_NUM_THREADS=16 -e MKL_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
>   -e TORCH_NUM_THREADS=16 -e TORCH_NUM_INTEROP_THREADS=16 \
>   -v $(pwd):/app torch-cpu python benchmark.py
/usr/local/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
[0.0000s] Process Started
=== Runtime Info ===
python: 3.9.25
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.41
torch: 2.5.1+cpu
mkldnn: True
mkl: True
openmp: True
torch threads: compute=16 interop=16
env OMP_NUM_THREADS=16
env MKL_NUM_THREADS=16
env OPENBLAS_NUM_THREADS=16
====================
[0.1788s] Loading Model...
[0.3475s] Model Loaded
[0.4588s] Warmup Done
[0.4588s] Starting Inference Loop (100 iters)...
[8.2480s] Inference Done. Avg Latency: 77.89 ms
```
