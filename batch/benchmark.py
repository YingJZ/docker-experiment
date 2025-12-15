import time
import os
import sys
import torch
import torchvision.models as models
import platform

def print_runtime_info():
    print("=== Runtime Info ===")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    _torch = sys.modules.get('torch')
    print(f"torch: {getattr(_torch, '__version__', 'unknown')}")
    try:
        import torch.backends as tb
        print(f"mkldnn: {getattr(tb.mkldnn, 'is_available', lambda: None)()}")
        print(f"mkl: {getattr(tb, 'mkl', None) and tb.mkl.is_available()}")
        print(f"openmp: {getattr(tb, 'openmp', None) and tb.openmp.is_available()}")
    except Exception:
        pass
    try:
        print(f"torch threads: compute={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    except Exception:
        print("torch threads: compute=? interop=?")
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        print(f"env {k}={os.environ.get(k, '')}")
    print("====================")

def log(msg):
    print(f"[{time.time() - start_time:.4f}s] {msg}")

def main():
    global start_time
    start_time = time.time()
    log("Process Started")
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
    
    # 1. 模拟依赖加载和运行时初始化
    # (import 已经在上面做了，这里主要看模型加载)
    
    # 2. 模型加载 (模拟冷启动中最耗时的 I/O 和 内存映射)
    log("Loading Model...")
    # 使用 MobileNetV2，因为它常用于端侧
    model = models.mobilenet_v2(pretrained=False) 
    # 可以在这里加载本地 .pth 文件以模拟真实磁盘读取
    model.eval()
    log("Model Loaded")

    # 3. 预热 (Warmup) - 某些框架第一次推理很慢
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy_input)
    log("Warmup Done")

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