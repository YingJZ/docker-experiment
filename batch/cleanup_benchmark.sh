#!/bin/bash
# 清理脚本：停止所有测试容器和 Slice

echo "=== Cleaning up benchmark resources ==="

# 1. 停止并删除所有测试容器
echo "Stopping test containers..."
docker ps -a --filter name=test_n --format "{{.Names}}" | xargs -r docker rm -f
echo "Container cleanup done."

# 2. 停止所有 bench_ 开头的 Slice
echo "Stopping benchmark slices..."
systemctl list-units --type=slice --all | grep bench_test_n | awk '{print $1}' | xargs -r systemctl stop
echo "Slice cleanup done."

# 3. 清理孤儿容器
echo "Cleaning orphaned containers..."
docker container prune -f

echo "=== Cleanup complete ==="
