#!/bin/bash

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <命令>"
    echo "示例: $0 python benchmark.py"
    exit 1
fi

# 配置采样间隔 (秒)
INTERVAL=0.5

# 1. 准备日志目录
# 使用时间戳和命令名作为目录名，避免冲突
CMD_NAME=$(basename "$1")
TIMESTAMP_DIR=$(date +%Y%m%d_%H%M%S)
LOG_DIR="smaps_logs/smaps_logs_${CMD_NAME}_${TIMESTAMP_DIR}"

mkdir -p "$LOG_DIR"
echo ">>> 日志将保存在文件夹: $LOG_DIR"

# 2. 启动目标进程
echo ">>> 正在启动目标进程..."
"$@" &
TARGET_PID=$!
echo ">>> 目标 PID: $TARGET_PID"

# 捕获 Ctrl+C，确保脚本退出时如果子进程还在，也能被清理（可选，视需求而定）
trap "kill $TARGET_PID 2>/dev/null; exit" SIGINT SIGTERM

# 3. 循环采样
echo ">>> 开始采样 (间隔: ${INTERVAL}s)..."
echo "    [时间]          [文件名]          [Total RSS (KB)]"

count=0
while kill -0 $TARGET_PID 2>/dev/null; do
    # 生成带时间戳的文件名
    CURRENT_TIME=$(date +%H%M%S_%N) # 时分秒_纳秒
    FILE_NAME="${LOG_DIR}/${CURRENT_TIME}.smaps"
    
    # 尝试捕获 smaps
    if [ -f "/proc/$TARGET_PID/smaps" ]; then
        cat "/proc/$TARGET_PID/smaps" > "$FILE_NAME" 2>/dev/null
        
        # 简单的实时反馈：计算该时刻的 Total RSS (物理内存)
        # 注意：这会稍微增加一点点开销，但对于观察趋势很有用
        if [ -s "$FILE_NAME" ]; then
            RSS_TOTAL=$(grep -F "Rss:" "$FILE_NAME" | awk '{sum+=$2} END {print sum}')
            printf "    %s  ->  %s  ->  %s KB\n" "$(date +%H:%M:%S)" "$(basename $FILE_NAME)" "$RSS_TOTAL"
        fi
    else
        # 如果进程在循环中间退出了，就跳出
        break
    fi

    sleep $INTERVAL
    ((count++))
done

# 4. 等待进程彻底结束并获取退出码
wait $TARGET_PID
EXIT_CODE=$?

echo "---------------------------------------------------"
echo ">>> 进程已结束 (Exit Code: $EXIT_CODE)"
echo ">>> 共采集样本数: $count"
echo ">>> 结果保存在: $LOG_DIR/"