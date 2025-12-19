# Cache 的影响

受操作系统缓存的影响，似乎前两次 `import torch` 会非常慢，从第三次开始会比较快？

现在有两个实验方案：

1. 每次实验都强制冷启动：在每次实验前，用指令清空操作系统的缓存

```python
subprocess.run("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True, check=True)
```

- 缺点：可能会影响服务器上运行的其他程序？

2. 每次实验前都先预热一下，可能先运行 3 次，然后从第 4 次开始取 5～10 次的平均数据

- 缺点：这样会不会就不算冷启动了？可能这种更接近服务器的场景，但是我们的场景是不是在终端上

```bash
(experiment) yingjiaze@haslab4:~/experiment/batch$ date +%s%3N && taskset -c 0,1 python benchmark.py
1766135238901
start time: 1766135238.953
[0.000s] Python Process Started, Importing Torch...
[5.295s] Import Torch Done, Time Spent: 5.295s
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.5.1+cpu
backends mkldnn=True mkl=True openmp=True
torch threads: compute=2 interop=16
envs OMP_NUM_THREADS= MKL_NUM_THREADS= OPENBLAS_NUM_THREADS=
====================
[5.329s] Loading Model...
[5.525s] Loaded weights from mobilenet_v2-imagenet1k-v1.pth
[5.527s] Model Loaded, Time Spent: 0.197s
[5.527s] Starting Warmup...
[5.722s] Warmup Done, Time Spent: 0.195s
[5.722s] Starting Inference Loop (100 iters)...
[9.380s] Inference Done. Avg Latency: 36.58 ms

(experiment) yingjiaze@haslab4:~/experiment/batch$ date +%s%3N && taskset -c 0,1 python benchmark.py
1766135370534
start time: 1766135370.592
[0.000s] Python Process Started, Importing Torch...
[4.970s] Import Torch Done, Time Spent: 4.970s
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.5.1+cpu
backends mkldnn=True mkl=True openmp=True
torch threads: compute=2 interop=16
envs OMP_NUM_THREADS= MKL_NUM_THREADS= OPENBLAS_NUM_THREADS=
====================
[5.007s] Loading Model...
[5.199s] Loaded weights from mobilenet_v2-imagenet1k-v1.pth
[5.201s] Model Loaded, Time Spent: 0.194s
[5.201s] Starting Warmup...
[5.268s] Warmup Done, Time Spent: 0.068s
[5.268s] Starting Inference Loop (100 iters)...
[9.181s] Inference Done. Avg Latency: 39.13 ms

(experiment) yingjiaze@haslab4:~/experiment/batch$ date +%s%3N && taskset -c 0,1 python benchmark.py
1766135381991
start time: 1766135382.046
[0.000s] Python Process Started, Importing Torch...
[3.550s] Import Torch Done, Time Spent: 3.550s
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.5.1+cpu
backends mkldnn=True mkl=True openmp=True
torch threads: compute=2 interop=16
envs OMP_NUM_THREADS= MKL_NUM_THREADS= OPENBLAS_NUM_THREADS=
====================
[3.585s] Loading Model...
[3.777s] Loaded weights from mobilenet_v2-imagenet1k-v1.pth
[3.778s] Model Loaded, Time Spent: 0.193s
[3.778s] Starting Warmup...
[3.846s] Warmup Done, Time Spent: 0.068s
[3.846s] Starting Inference Loop (100 iters)...
[7.610s] Inference Done. Avg Latency: 37.64 ms
(experiment) yingjiaze@haslab4:~/experiment/batch$ date +%s%3N && taskset -c 0,1 python benchmark.py
1766135397073
start time: 1766135397.133
[0.000s] Python Process Started, Importing Torch...
[3.563s] Import Torch Done, Time Spent: 3.563s
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.5.1+cpu
backends mkldnn=True mkl=True openmp=True
torch threads: compute=2 interop=16
envs OMP_NUM_THREADS= MKL_NUM_THREADS= OPENBLAS_NUM_THREADS=
====================
[3.598s] Loading Model...
[3.789s] Loaded weights from mobilenet_v2-imagenet1k-v1.pth
[3.790s] Model Loaded, Time Spent: 0.192s
[3.790s] Starting Warmup...
[3.858s] Warmup Done, Time Spent: 0.068s
[3.858s] Starting Inference Loop (100 iters)...
[7.614s] Inference Done. Avg Latency: 37.56 ms
```