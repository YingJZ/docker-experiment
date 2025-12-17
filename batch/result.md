
| 环境 | 初始化 | import torch | 模型加载 | 预热 | 推理平均延迟 (ms) |
|------|------|--------------|----------|------|------------------|
|主机| 52ms | 4.313s   | 0.206s   | 54ms |28.23 ms         |
|Docker| 431ms | 4.042s   | 0.179s   | 64ms |38.50 ms         |


### Host

```bash
(experiment) yingjiaze@haslab4:~/experiment/batch$ date +%s%3N && python benchmark.py
1765780394871
start time: 1765780394.923
[0.000s] Python Process Started, Importing Torch...
[4.313s] Import Torch Done, Time Spent: 4.313s
=== Runtime Info ===
python: 3.9.5
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.31
torch: 2.5.1+cpu
backends mkldnn=True mkl=True openmp=True
torch threads: compute=16 interop=16
envs OMP_NUM_THREADS= MKL_NUM_THREADS= OPENBLAS_NUM_THREADS=
====================
[4.356s] Loading Model...
[4.561s] Loaded weights from mobilenet_v2-imagenet1k-v1.pth
[4.562s] Model Loaded, Time Spent: 0.206s
[4.563s] Starting Warmup...
[4.617s] Warmup Done, Time Spent: 0.054s
[4.617s] Starting Inference Loop (100 iters)...
[7.441s] Inference Done. Avg Latency: 28.23 ms
```

### Docker

```bash
(experiment) yingjiaze@haslab4:~/experiment/batch$ date +%s%3N && sudo docker run --rm --memory="2g" --cpuset-cpus="0-1" -v $(pwd):/app torch-cpu python benchmark.py
1765779239243
...
start time: 1765779239.674
[0.000s] Python Process Started, Importing Torch...
[4.042s] Import Torch Done, Time Spent: 4.042s
=== Runtime Info ===
python: 3.9.25
platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.41
torch: 2.5.1+cpu
backends mkldnn=True mkl=True openmp=True
torch threads: compute=2 interop=16
envs OMP_NUM_THREADS= MKL_NUM_THREADS= OPENBLAS_NUM_THREADS=
====================
[4.073s] Loading Model...
[4.250s] Loaded weights from mobilenet_v2-imagenet1k-v1.pth
[4.252s] Model Loaded, Time Spent: 0.179s
[4.252s] Starting Warmup...
[4.316s] Warmup Done, Time Spent: 0.064s
[4.316s] Starting Inference Loop (100 iters)...
[8.165s] Inference Done. Avg Latency: 38.50 ms
```