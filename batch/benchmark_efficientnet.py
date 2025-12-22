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
        import torch
        print(f"torch threads: compute={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    except Exception:
        print("torch threads: compute=? interop=?")
    env_parts = []
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env_parts.append(f"{k}={os.environ.get(k, '')}")
    print("envs " + " ".join(env_parts))
    print("====================")

def setup_runtime():
    print_runtime_info()
    try:
        import torch
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
    
    log("Loading Model (EfficientNet-B0)...")
    load_start = time.time()
    try:
        # EfficientNet is available in newer torchvision
        model = models.efficientnet_b0(weights=None)
    except AttributeError:
        # Fallback to MobileNetV3 if EfficientNet not available
        log("EfficientNet not available, using MobileNetV3 instead...")
        model = models.mobilenet_v3_small(weights=None)
    model.eval()
    log("Model Loaded, Time Spent: {:.3f}s".format(time.time() - load_start))

    log("Starting Warmup...")
    warmup_start = time.time()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy_input)
    log("Warmup Done, Time Spent: {:.3f}s".format(time.time() - warmup_start))

    log("Starting Inference Loop (100 iters)...")
    infer_start = time.time()
    latencies = []
    for i in range(100):
        iter_start = time.time()
        with torch.no_grad():
            model(dummy_input)
        iter_end = time.time()
        latency_ms = (iter_end - iter_start) * 1000
        latencies.append(latency_ms)
    infer_end = time.time()
    
    avg_latency = (infer_end - infer_start) / 100 * 1000
    log(f"Inference Done. Avg Latency: {avg_latency:.2f} ms")
    
    latencies_str = ",".join([f"{l:.2f}" for l in latencies])
    print(f"LATENCIES: {latencies_str}")

if __name__ == "__main__":
    main()

