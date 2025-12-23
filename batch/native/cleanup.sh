#!/bin/bash
# 清理 native benchmark 相关资源

echo "=== Cleaning up native benchmark resources ==="

# 1. 停止所有 native_bench_ 开头的 scope
echo "Stopping benchmark scopes..."
systemctl list-units --type=scope --all 2>/dev/null | grep native_bench | awk '{print $1}' | xargs -r systemctl stop 2>/dev/null

# 2. 停止所有 native_bench_ 开头的 Slice
echo "Stopping benchmark slices..."
systemctl list-units --type=slice --all 2>/dev/null | grep native_bench | awk '{print $1}' | xargs -r systemctl stop 2>/dev/null

# 3. 杀死可能残留的 benchmark.py 进程
echo "Killing stray benchmark processes..."
pkill -f "native/benchmark.py" 2>/dev/null || true

echo "=== Cleanup complete ==="
