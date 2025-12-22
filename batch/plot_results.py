#!/usr/bin/env python3
"""
根据实验结果数据绘制图表
"""

import json
import os
import glob
import argparse
import statistics
from typing import List, Dict
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib is required for plotting. Please install it with: pip install matplotlib")
    exit(1)


def detect_jitter(latencies_list: List[List[float]], threshold: float = 0.3) -> Dict[str, float]:
    """
    检测抖动
    使用变异系数(CV)来衡量抖动程度
    """
    if not latencies_list:
        return {"jitter_detected": False, "cv": 0.0}
    
    # 计算每个容器的平均时延
    avg_latencies = [statistics.mean(lats) for lats in latencies_list if lats]
    
    if len(avg_latencies) < 2:
        return {"jitter_detected": False, "cv": 0.0}
    
    mean_avg = statistics.mean(avg_latencies)
    std_avg = statistics.stdev(avg_latencies) if len(avg_latencies) > 1 else 0
    cv = std_avg / mean_avg if mean_avg > 0 else 0
    
    jitter_detected = cv > threshold
    
    return {
        "jitter_detected": jitter_detected,
        "cv": cv,
        "mean_latency": mean_avg,
        "std_latency": std_avg
    }


def load_test_results(result_dir: str) -> tuple:
    """
    从结果目录加载所有测试结果
    
    返回: (test1_results, test2_results)
    test1_results: List[Dict] - test1的结果列表
    test2_results: List[Dict] - test2的结果列表
    """
    test1_results = []
    test2_results = []
    
    # 查找所有结果文件
    pattern = os.path.join(result_dir, "*_result.json")
    result_files = glob.glob(pattern)
    
    for result_file in sorted(result_files):
        filename = os.path.basename(result_file)
        if filename.startswith('test1_'):
            with open(result_file, 'r') as f:
                test1_results.append(json.load(f))
        elif filename.startswith('test2_'):
            with open(result_file, 'r') as f:
                test2_results.append(json.load(f))
    
    return test1_results, test2_results


def plot_test1_results(test1_results: List[Dict], output_dir: str):
    """绘制测试1的结果（MobileNet多实例）"""
    if not test1_results:
        print("No test1 results found, skipping test1 plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Test 1: MobileNet Multiple Instances', fontsize=16)
    
    # 按实例数排序
    test1_results = sorted(test1_results, key=lambda x: x['num_instances'])
    num_instances = [r['num_instances'] for r in test1_results]
    
    # 启动时间
    ax = axes[0, 0]
    start_times = [statistics.mean([c['start_duration_ms'] for c in r['containers']]) 
                  for r in test1_results]
    ax.plot(num_instances, start_times, 'o-', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_ylabel('Start Time (ms)', fontsize=12)
    ax.set_title('Container Start Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # P95时延
    ax = axes[0, 1]
    p95_latencies = [statistics.mean([c['p95_latency_ms'] for c in r['containers']]) 
                    for r in test1_results]
    ax.plot(num_instances, p95_latencies, 'o-', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_ylabel('P95 Latency (ms)', fontsize=12)
    ax.set_title('P95 Latency', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # P99时延
    ax = axes[0, 2]
    p99_latencies = [statistics.mean([c['p99_latency_ms'] for c in r['containers']]) 
                    for r in test1_results]
    ax.plot(num_instances, p99_latencies, 'o-', color='red', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax.set_title('P99 Latency', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # PSS内存
    ax = axes[1, 0]
    pss_values = [statistics.mean([c['pss_mb'] for c in r['containers']]) 
                 for r in test1_results]
    total_pss = [sum([c['pss_mb'] for c in r['containers']]) for r in test1_results]
    ax.plot(num_instances, pss_values, 'o-', label='Avg PSS per Container', 
           color='purple', linewidth=2, markersize=8)
    ax.plot(num_instances, total_pss, 's-', label='Total PSS', 
           color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_ylabel('PSS Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage (PSS)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 抖动检测
    ax = axes[1, 1]
    jitter_cvs = []
    for r in test1_results:
        latencies_list = [c['all_latencies_ms'] for c in r['containers'] if c.get('all_latencies_ms')]
        jitter_info = detect_jitter(latencies_list)
        jitter_cvs.append(jitter_info['cv'])
    ax.plot(num_instances, jitter_cvs, 'o-', color='brown', linewidth=2, markersize=8)
    ax.axhline(y=0.3, color='r', linestyle='--', label='Jitter Threshold', linewidth=2)
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('Jitter Detection (CV)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 时延分布箱线图
    ax = axes[1, 2]
    all_latencies_by_instance = []
    labels = []
    for r in test1_results:
        all_latencies = []
        for c in r['containers']:
            if c.get('all_latencies_ms'):
                all_latencies.extend(c['all_latencies_ms'])
        if all_latencies:
            all_latencies_by_instance.append(all_latencies)
            labels.append(f"N={r['num_instances']}")
    if all_latencies_by_instance:
        bp = ax.boxplot(all_latencies_by_instance, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Number of Instances', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency Distribution', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "test1_mobilenet_multiple_instances.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Test1 plots saved to {output_file}")


def plot_test2_results(test2_results: List[Dict], output_dir: str):
    """绘制测试2的结果（混合应用）"""
    if not test2_results:
        print("No test2 results found, skipping test2 plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Test 2: Mixed Applications', fontsize=16)
    
    # 按应用类型分组
    app_metrics = defaultdict(list)
    for r in test2_results:
        for c in r['containers']:
            app_name = c['app_name']
            app_metrics[app_name].append(c)
    
    apps = sorted(app_metrics.keys())
    if not apps:
        print("No container data found in test2 results")
        return
    
    # P95时延对比
    ax = axes[0, 0]
    p95_by_app = [statistics.mean([c['p95_latency_ms'] for c in app_metrics[app]]) 
                  for app in apps]
    colors = plt.cm.Set3(range(len(apps)))
    bars = ax.bar(apps, p95_by_app, color=colors[:len(apps)])
    ax.set_xlabel('Application', fontsize=12)
    ax.set_ylabel('P95 Latency (ms)', fontsize=12)
    ax.set_title('P95 Latency by Application', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # P99时延对比
    ax = axes[0, 1]
    p99_by_app = [statistics.mean([c['p99_latency_ms'] for c in app_metrics[app]]) 
                  for app in apps]
    bars = ax.bar(apps, p99_by_app, color=colors[:len(apps)])
    ax.set_xlabel('Application', fontsize=12)
    ax.set_ylabel('P99 Latency (ms)', fontsize=12)
    ax.set_title('P99 Latency by Application', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # PSS内存对比
    ax = axes[0, 2]
    pss_by_app = [statistics.mean([c['pss_mb'] for c in app_metrics[app]]) 
                 for app in apps]
    bars = ax.bar(apps, pss_by_app, color=colors[:len(apps)])
    ax.set_xlabel('Application', fontsize=12)
    ax.set_ylabel('PSS Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage (PSS) by Application', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 启动时间对比
    ax = axes[1, 0]
    start_by_app = [statistics.mean([c['start_duration_ms'] for c in app_metrics[app]]) 
                   for app in apps]
    bars = ax.bar(apps, start_by_app, color=colors[:len(apps)])
    ax.set_xlabel('Application', fontsize=12)
    ax.set_ylabel('Start Time (ms)', fontsize=12)
    ax.set_title('Container Start Time by Application', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 时延分布对比
    ax = axes[1, 1]
    all_latencies_by_app = []
    for app in apps:
        all_lats = []
        for c in app_metrics[app]:
            if c.get('all_latencies_ms'):
                all_lats.extend(c['all_latencies_ms'])
        all_latencies_by_app.append(all_lats)
    
    if all_latencies_by_app:
        bp = ax.boxplot(all_latencies_by_app, labels=apps, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
        ax.set_xlabel('Application', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency Distribution by Application', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
    
    # 抖动检测
    ax = axes[1, 2]
    jitter_by_app = []
    for app in apps:
        latencies_list = [c['all_latencies_ms'] for c in app_metrics[app] 
                         if c.get('all_latencies_ms')]
        jitter_info = detect_jitter(latencies_list)
        jitter_by_app.append(jitter_info['cv'])
    bars = ax.bar(apps, jitter_by_app, color=colors[:len(apps)])
    ax.axhline(y=0.3, color='r', linestyle='--', label='Jitter Threshold', linewidth=2)
    ax.set_xlabel('Application', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('Jitter Detection (CV) by Application', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "test2_mixed_applications.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Test2 plots saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark test results')
    parser.add_argument('result_dir', type=str, 
                       help='Directory containing experiment results (e.g., results/experiment_YYYYMMDD_HHMMSS)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as result_dir)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.result_dir):
        print(f"Error: Result directory does not exist: {args.result_dir}")
        exit(1)
    
    output_dir = args.output_dir if args.output_dir else args.result_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from: {args.result_dir}")
    test1_results, test2_results = load_test_results(args.result_dir)
    
    print(f"Found {len(test1_results)} test1 results and {len(test2_results)} test2 results")
    
    # 绘制测试1结果
    if test1_results:
        print("\nPlotting test1 results...")
        plot_test1_results(test1_results, output_dir)
    
    # 绘制测试2结果
    if test2_results:
        print("\nPlotting test2 results...")
        plot_test2_results(test2_results, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

