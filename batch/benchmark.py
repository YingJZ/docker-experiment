import time
import os
import sys
import platform

def print_runtime_info():
    print("=== Runtime Info ===")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
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
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env_parts.append(f"{k}={os.environ.get(k, '')}")
    print("envs " + " ".join(env_parts))
    print("====================")

def setup_runtime():
    # 打印运行时信息以便对比主机与容器差异
    print_runtime_info()
    # 若通过环境或参数指定线程数，则进行设置以便在容器与主机对齐
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
    # 根据环境变量调整环境配置，并输出运行时信息
    setup_runtime()

    # 1. 模拟依赖加载和运行时初始化
    # (import 已经在上面做了，这里主要看模型加载)
    
    # 2. 模型加载 (模拟冷启动中最耗时的 I/O 和 内存映射)
    log("Loading Model...")
    load_start = time.time()
    # 使用 MobileNetV2，因为它常用于端侧
    model = models.mobilenet_v2(weights=None)
    # 若提供本地权重则加载，优先使用安全的 weights_only=True（PyTorch 2.5+）
    weights_path = os.environ.get("MOBILENET_V2_WEIGHTS", "mobilenet_v2-imagenet1k-v1.pth")
    if os.path.exists(weights_path):
        try:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            # 兼容旧版 torch 没有 weights_only 参数
            sd = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        log(f"Loaded weights from {weights_path}")
    else:
        log(f"Weights file not found, skip loading: {weights_path}")
    model.eval()
    log("Model Loaded, Time Spent: {:.3f}s".format(time.time() - load_start))

    # 3. 预热 (Warmup) - 某些框架第一次推理很慢
    log("Starting Warmup...")
    warmup_start = time.time()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy_input)
    log("Warmup Done, Time Spent: {:.3f}s".format(time.time() - warmup_start))

    # 4. 推理性能测试 (Inference Loop)
    log("Starting Inference Loop (100 iters)...")
    infer_start = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(dummy_input)
    infer_end = time.time()
    
    avg_latency = (infer_end - infer_start) / 100 * 1000 # ms
    log(f"Inference Done. Avg Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()