import subprocess
import threading
import time
import os
import sys

# === 配置区域 ===
# 限制总资源池：2核 CPU, 2GB 内存
SLICE_NAME = "edge_simulation.slice"
CPU_QUOTA = "200%"  # 200% 代表 2 个核心
MEM_MAX = "2G"
IMAGE_NAME = "torch-cpu"
SCRIPT_CMD = ["python", "benchmark.py"]  # 容器内执行的命令
# ================

def setup_cgroup_slice():
    """
    创建一个 Systemd Slice 来充当资源池。
    """
    print(f"[*] 初始化资源环境: /{SLICE_NAME}")
    print(f"    - 总 CPU 限制: {CPU_QUOTA}")
    print(f"    - 总内存限制:  {MEM_MAX}")

    # --- 改进点 1: 更彻底的清理逻辑 ---
    # 先停止，再清除失败状态，防止 Unit name collision
    subprocess.run(["sudo", "systemctl", "stop", "edge_resource_holder"], stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "systemctl", "reset-failed", "edge_resource_holder"], stderr=subprocess.DEVNULL)

    # 创建新 slice
    cmd = [
        "sudo", "systemd-run",
        "--unit=edge_resource_holder",
        f"--slice={SLICE_NAME}",
        f"--property=CPUQuota={CPU_QUOTA}",
        f"--property=MemoryMax={MEM_MAX}",
        "--property=MemorySwapMax=0", 
        "sleep", "infinity"
    ]
    
    try:
        # 这里的 check_call 会在失败时抛出异常
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(1)
        print("    -> 环境创建成功。")
    except subprocess.CalledProcessError as e:
        # --- 改进点 2: 打印真实错误信息 ---
        print(f"    -> [错误] 创建 Slice 失败！")    
        print(f"    -> {str(e)}")
        print(f"    -> 提示: 如果提示 'Unit name already in use'，请手动执行: sudo systemctl reset-failed edge_resource_holder")
        sys.exit(1)

def run_container(index):
    """
    启动单个容器的工作线程
    """
    container_name = f"edge_worker_{index}"
    cwd = os.getcwd() # 获取当前路径，对应 $(pwd)
    
    # 构造 Docker 命令
    # 变化点：移除了 --memory 和 --cpuset，增加了 --cgroup-parent
    cmd = [
        "sudo", "docker", "run",
        "--rm",
        f"--name={container_name}",
        f"--cgroup-parent=/{SLICE_NAME}",  # 【关键】加入总控 Slice
        "-v", f"{cwd}:/app",               # 挂载当前目录
        IMAGE_NAME
    ] + SCRIPT_CMD

    print(f"[{index}] 启动容器...")
    
    # 记录开始时间 (模拟 date +%s%3N)
    start_time = time.time()
    
    try:
        # 执行命令，捕获输出
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # 分析结果
        if result.returncode == 0:
            print(f"✅ [{index}] 完成 | 耗时: {duration_ms:.0f}ms | 输出: {result.stdout.strip()[:50]}...")
        else:
            # Docker 容器被 OOM Kill 通常返回 137 (128 + SIGKILL 9)
            status = "❌ 失败"
            reason = f"Exit Code {result.returncode}"
            if result.returncode == 137:
                status = "💀 OOM Killed"
                reason = "内存不足被系统杀掉"
            
            print(f"{status} [{index}] | 耗时: {duration_ms:.0f}ms | 原因: {reason}")
            # 如果需要调试，可以取消下面这行的注释打印错误日志
            # print(f"[{index}] Stderr: {result.stderr.strip()}")

    except Exception as e:
        print(f"[{index}] 异常: {e}")

def cleanup():
    """清理资源"""
    print("[*] 测试结束，正在清理资源池...")
    subprocess.run(["sudo", "systemctl", "stop", "edge_resource_holder"], stderr=subprocess.DEVNULL)

def main(n):
    setup_cgroup_slice()
    
    threads = []
    print(f"[*] 开始并发运行 {n} 个容器...\n")
    
    for i in range(n):
        t = threading.Thread(target=run_container, args=(i,))
        threads.append(t)
        t.start()
        
    # 等待所有线程结束
    for t in threads:
        t.join()
        
    print("\n[*] 所有任务已完成。")
    cleanup()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <容器并发数量>")
        print(f"示例: python {sys.argv[0]} 4")
        sys.exit(1)
    
    try:
        num = int(sys.argv[1])
        main(num)
    except KeyboardInterrupt:
        print("\n强制中断...")
        cleanup()