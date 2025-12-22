import subprocess
import statistics
import re
import os
from typing import Dict, List


def drop_caches() -> None:
    """Drop system caches before each run to simulate cold start."""
    print("🧹 Dropping system caches to ensure cold start...")
    # sync 确保脏数据写回磁盘，避免数据丢失
    # echo 3 > ... 清除 PageCache, dentries 和 inodes
    subprocess.run(
        "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches",
        shell=True,
        check=True,
    )


def run_once(working_dir: str) -> Dict[str, float]:
    """
    运行一次命令：
        date +%s%3N && taskset -c 0,1 python benchmark.py
    并解析输出，返回一次的关键阶段耗时（单位：ms）。
    """
    cmd = "date +%s%3N && taskset -c 0,1 python benchmark.py"
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=working_dir,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = proc.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError("No output captured from benchmark command.")

    # 第 1 行：date +%s%3N 的毫秒时间戳
    try:
        t0_ms = int(stdout[0].strip())
    except ValueError as e:
        raise RuntimeError(f"Failed to parse first line as ms timestamp: {stdout[0]!r}") from e

    # 解析后续行
    text = "\n".join(stdout[1:])

    # start time: 1766135397.133
    m_start = re.search(r"start time:\s*([0-9.]+)", text)
    if not m_start:
        raise RuntimeError("Cannot find 'start time' line in output.")
    start_time_s = float(m_start.group(1))

    # 初始化时间 = start_time - (date 输出 / 1000)，单位转为 ms
    init_ms = (start_time_s - t0_ms / 1000.0) * 1000.0

    # Import Torch Done, Time Spent: 3.563s
    m_import = re.search(r"Import Torch Done, Time Spent:\s*([0-9.]+)s", text)
    if not m_import:
        raise RuntimeError("Cannot find 'Import Torch Done' line in output.")
    import_ms = float(m_import.group(1)) * 1000.0

    # Model Loaded, Time Spent: 0.197s
    m_load = re.search(r"Model Loaded, Time Spent:\s*([0-9.]+)s", text)
    if not m_load:
        raise RuntimeError("Cannot find 'Model Loaded' line in output.")
    load_ms = float(m_load.group(1)) * 1000.0

    # Warmup Done, Time Spent: 0.195s
    m_warmup = re.search(r"Warmup Done, Time Spent:\s*([0-9.]+)s", text)
    if not m_warmup:
        raise RuntimeError("Cannot find 'Warmup Done' line in output.")
    warmup_ms = float(m_warmup.group(1)) * 1000.0

    # Inference Done. Avg Latency: 36.58 ms
    m_infer = re.search(r"Inference Done\. Avg Latency:\s*([0-9.]+)\s*ms", text)
    if not m_infer:
        raise RuntimeError("Cannot find 'Inference Done. Avg Latency' line in output.")
    infer_avg_ms = float(m_infer.group(1))

    return {
        "init_ms": init_ms,
        "import_ms": import_ms,
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "infer_avg_ms": infer_avg_ms,
    }


def main() -> None:
    # 默认工作目录设为当前脚本所在目录（即包含 benchmark.py 的目录）
    working_dir = os.path.dirname(os.path.abspath(__file__))

    runs: List[Dict[str, float]] = []

    num_runs = 5
    for i in range(1, num_runs + 1):
        print(f"\n===== Run {i}/{num_runs} =====")
        # drop_caches()
        result = run_once(working_dir)
        runs.append(result)

        print(
            "本次结果(ms): "
            f"初始化={result['init_ms']:.1f}, "
            f"import torch={result['import_ms']:.1f}, "
            f"模型加载={result['load_ms']:.1f}, "
            f"预热={result['warmup_ms']:.1f}, "
            f"推理平均延迟={result['infer_avg_ms']:.2f}"
        )

    # 计算平均值
    def avg(key: str) -> float:
        return statistics.mean(r[key] for r in runs)

    avg_init = avg("init_ms")
    avg_import = avg("import_ms")
    avg_load = avg("load_ms")
    avg_warmup = avg("warmup_ms")
    avg_infer = avg("infer_avg_ms")

    print("\n===== 5 次运行平均耗时 (ms) =====")
    print(f"初始化: {avg_init:.1f} ms")
    print(f"import torch: {avg_import:.1f} ms")
    print(f"模型加载: {avg_load:.1f} ms")
    print(f"预热: {avg_warmup:.1f} ms")
    print(f"推理平均延迟: {avg_infer:.2f} ms")


if __name__ == "__main__":
    main()


