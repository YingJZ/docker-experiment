#!/usr/bin/env python3
"""
Native benchmark script - 直接在主机上运行推理
与容器版本功能相同，但不需要 Docker 环境
"""

import time
import os
import sys
import platform

def print_runtime_info():
    print("=== Runtime Info (Native) ===")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"pid: {os.getpid()}")
    _torch = sys.modules.get('torch')
    print(f"torch: {getattr(_torch, '__version__', 'unknown')}")
    try:
        import torch.backends as tb
        mkldnn_ok = getattr(tb.mkldnn, 'is_available', lambda: None)()
        mkl_ok = getattr(tb, 'mkl', None) and tb.mkl.is_available()
        openmp_ok = getattr(tb, 'openmp', None) and tb.openmp.is_available()
        print(f"backends mkldnn={mkldnn_ok} mkl={mkl_ok} openmp={openmp_ok}")
    except Exception:
        pass
    try:
        print(f"torch threads: compute={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    except Exception:
        print("torch threads: compute=? interop=?")
    env_parts = []
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTORCH_NUM_THREADS"]:
        env_parts.append(f"{k}={os.environ.get(k, '')}")
    print("envs " + " ".join(env_parts))
    print("==============================")

def setup_runtime():
    print_runtime_info()
    try:
        num_threads = int(os.environ.get("TORCH_NUM_THREADS", "0"))
        interop_threads = int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "0"))
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        if interop_threads > 0:
            torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass

def log(msg):
    print(f"[{time.time() - start_time:.3f}s] {msg}")

start_time = time.time()
print(f"start time: {start_time:.3f}")

log("Python Process Started, Importing Torch...")
import_start = time.time()

import torch
import torchvision.models as models

import_end = time.time()
log(f"Import Torch Done, Time Spent: {import_end - import_start:.3f}s")

def main():
    setup_runtime()
    
    # 模型加载
    log("Loading Model...")
    load_start = time.time()
    model = models.mobilenet_v2(weights=None)
    
    # 权重文件路径：支持环境变量或默认路径
    weights_path = os.environ.get("MOBILENET_V2_WEIGHTS", None)
    if weights_path is None:
        # 尝试多个常见路径
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "mobilenet_v2-imagenet1k-v1.pth"),
            os.path.join(os.path.dirname(__file__), "mobilenet_v2-imagenet1k-v1.pth"),
            "mobilenet_v2-imagenet1k-v1.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break
    
    if weights_path and os.path.exists(weights_path):
        try:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        log(f"Loaded weights from {weights_path}")
    else:
        log(f"Weights file not found, using random weights")
    
    model.eval()
    log("Model Loaded, Time Spent: {:.3f}s".format(time.time() - load_start))

    # Warmup
    log("Starting Warmup...")
    warmup_start = time.time()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy_input)
    log("Warmup Done, Time Spent: {:.3f}s".format(time.time() - warmup_start))

    # 推理循环
    num_iters = int(os.environ.get("BENCHMARK_ITERS", "100"))
    log(f"Starting Inference Loop ({num_iters} iters)...")
    infer_start = time.time()
    latencies = []
    for i in range(num_iters):
        iter_start = time.time()
        with torch.no_grad():
            model(dummy_input)
        iter_end = time.time()
        latency_ms = (iter_end - iter_start) * 1000
        latencies.append(latency_ms)
    infer_end = time.time()
    
    avg_latency = (infer_end - infer_start) / num_iters * 1000
    log(f"Inference Done. Avg Latency: {avg_latency:.2f} ms")
    
    # 输出每次推理的时延
    latencies_str = ",".join([f"{l:.2f}" for l in latencies])
    print(f"LATENCIES: {latencies_str}")

if __name__ == "__main__":
    main()
