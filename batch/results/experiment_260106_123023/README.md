# 实验说明

2026.1.6

## 本次实验新增内容

- 并发基准脚本新增内存组成采样：基于 `/proc/<pid>/smaps_rollup` 记录 PSS/RSS、Shared/Private Clean/Dirty、Swap、AnonHugePages 的 max/avg。
- 【暂时失败】采集容器所属 cgroup 的 `memory.stat`（anon/file/slab/kernel_stack/shmem/swap 等，单位 MB），便于区分匿名页与 file-backed 占比。
- `ContainerMetrics` 与 `data.json` 增加 `mem_breakdown_max/avg` 与 `cgroup_mem_stat` 字段，用于后续分析权重共享、匿名页膨胀。

## 结果分析

### 关键发现

- 并发延迟随实例数上升显著：P95 从 N=1 的 43ms 增长到 N=16 的 486ms，N≥10 后增长陡峭，说明 CPU 2 核 + 内存 2GB 的瓶颈被明显触发。

- 内存总峰值近似线性增长：单容器峰值 PSS ~235–250MB，N=10 已达 ~2.37GB（超出 2GB Slice，memory_full=true），N=16 达 ~3.79GB。每增 2 个实例约多 470–500MB，总体接近“每实例一份”。

- 内存构成（smaps_rollup）稳定模式：

    - shared_clean_mb 平均 ~78–83MB，最大 ~94–96MB，说明有较大 file-backed 映射（权重等），且多实例间可共享。

    - private_dirty_mb 平均 ~155–180MB，最大 ~223–230MB，随实例线性叠加，表明真正占用内存的主要是私有匿名页（模型加载/中间缓冲）。

    - anon_huge_mb、swap_mb 基本为 0，未出现大页或 swap；shared_dirty_mb 也几乎为 0。

- cgroup 统计缺失：cgroup_mem_stat 全为空，当前采集逻辑未能拿到数据（可能是 cgroup v2 路径或权限/驱动原因），无法用 cgroup 视角拆分 anon/file/slab。

### 粗量化结论

- 共享部分（file-backed clean）约 80–95MB/容器；私有部分（private dirty）约 160–180MB/容器。故即便权重映射共享，主耗还是私有匿名页，实例数上来后几乎线性占用。

- 2GB 限制下，理论可容纳 ≈ 2GB / ~240MB ≈ 8–9 个实例，再往上会逼近或超过限额并触发退化（与观测的 N≥10 延迟劣化、memory_full=true 相符）。

### 建议与后续

- 复核 cgroup 采集：检查宿主是否 cgroup v2，若是需调整 memory.stat 路径（统一在 /sys/fs/cgroup/<slice>/memory.current 等）；或确认 docker cgroup driver（systemd/cgroupfs）与进程对应路径。

- 若目标是提高并发密度：

- 尝试权重量化/剪枝或改为更小模型，减少私有匿名页。

- 复用推理进程（多请求复用线程）而非多容器，避免每实例一份私有状态。

- 探索共享中间缓冲/allocator 复用，或在容器内限制 torch.set_num_threads、禁用 MKL 大缓存。

- 若要在现有数据上画出“共享 vs 私有”趋势，可用 shared_clean_mb 与 private_dirty_mb 的平均/最大叠加，验证线性增长并估算可承载实例上限。

## 运行记录

```bash
(experiment) yingjiaze@haslab4:~/experiment/batch$ sudo /home/yingjiaze/experiment/.venv/bin/python concurrent_benchmark.py 
INFO: INSTANCE_COUNTS=[1, 2, 4, 6, 8, 10, 12, 14, 16]
WARNING: Using default volume mapping: /home/yingjiaze/experiment/batch:/app
Creating Slice bench_test_n1_1767702623.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 1 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 2 (total: 2)
[test_n1_c0] Started, ID: 43137f51e4e7
[test_n1_c0] Completed: P95=43.43ms, MaxPSS=323.0MB
Result N=1: Avg P95=43.43ms, Total Peak PSS=322.97MB

Creating Slice bench_test_n2_1767702634.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 2 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 2)
[test_n2_c1] Started, ID: d6fc97ee41d0
[test_n2_c0] Started, ID: 9e61a66d2bca
[test_n2_c1] Completed: P95=53.17ms, MaxPSS=273.9MB
[test_n2_c0] Completed: P95=56.84ms, MaxPSS=303.6MB
Result N=2: Avg P95=55.01ms, Total Peak PSS=577.51MB

Creating Slice bench_test_n4_1767702645.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 4 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 4)
[test_n4_c0] Started, ID: ee2777a68f02
[test_n4_c3] Started, ID: 33d343cbe44f
[test_n4_c2] Started, ID: 22150b3e8f22
[test_n4_c1] Started, ID: 1c38bd60ad2f
[test_n4_c1] Completed: P95=118.93ms, MaxPSS=251.2MB
[test_n4_c2] Completed: P95=121.50ms, MaxPSS=251.3MB
[test_n4_c3] Completed: P95=122.08ms, MaxPSS=251.2MB
[test_n4_c0] Completed: P95=120.74ms, MaxPSS=250.2MB
Result N=4: Avg P95=120.81ms, Total Peak PSS=1003.85MB

Creating Slice bench_test_n6_1767702668.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 6 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 6)
[test_n6_c2] Started, ID: 489044ea8fe7
[test_n6_c5] Started, ID: c3571885eb40
[test_n6_c4] Started, ID: ccbeab504b71
[test_n6_c3] Started, ID: e645b0425f05
[test_n6_c1] Started, ID: eaa71aa978f4
[test_n6_c0] Started, ID: 30927f00271d
[test_n6_c5] Completed: P95=177.31ms, MaxPSS=260.9MB
[test_n6_c2] Completed: P95=182.68ms, MaxPSS=243.8MB
[test_n6_c0] Completed: P95=172.44ms, MaxPSS=242.0MB
[test_n6_c4] Completed: P95=177.34ms, MaxPSS=245.8MB
[test_n6_c1] Completed: P95=178.53ms, MaxPSS=263.6MB
[test_n6_c3] Completed: P95=184.85ms, MaxPSS=265.0MB
Result N=6: Avg P95=178.86ms, Total Peak PSS=1521.09MB

Creating Slice bench_test_n8_1767702702.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 8 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 8)
[test_n8_c0] Started, ID: 1aaa8a8e7791
[test_n8_c6] Started, ID: eac6670e8e2b
[test_n8_c1] Started, ID: 4df6c0c5f072
[test_n8_c2] Started, ID: 1798a372632c
[test_n8_c3] Started, ID: 87554e862d18
[test_n8_c7] Started, ID: afe0bb210b9b
[test_n8_c4] Started, ID: 49c826840114
[test_n8_c5] Started, ID: 6707f002c747
[test_n8_c0] Completed: P95=178.89ms, MaxPSS=242.1MB
[test_n8_c3] Completed: P95=180.13ms, MaxPSS=241.4MB
[test_n8_c7] Completed: P95=175.29ms, MaxPSS=242.2MB
[test_n8_c6] Completed: P95=305.92ms, MaxPSS=246.0MB
[test_n8_c1] Completed: P95=291.07ms, MaxPSS=242.7MB
[test_n8_c5] Completed: P95=306.54ms, MaxPSS=245.4MB
[test_n8_c4] Completed: P95=291.84ms, MaxPSS=245.9MB
[test_n8_c2] Completed: P95=287.58ms, MaxPSS=313.5MB
Result N=8: Avg P95=252.16ms, Total Peak PSS=2019.23MB

Creating Slice bench_test_n10_1767702747.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 10 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 10)
[test_n10_c0] Started, ID: ee071294a805
[test_n10_c3] Started, ID: 4a9665f650a5
[test_n10_c4] Started, ID: 94aea2147674
[test_n10_c6] Started, ID: a4674fffbab3
[test_n10_c2] Started, ID: 93707a3cdf61
[test_n10_c1] Started, ID: c29c993aa65b
[test_n10_c7] Started, ID: 5dc1ba73c3b2
[test_n10_c8] Started, ID: 6bf9103a778b
[test_n10_c5] Started, ID: 7cc10c83ab6a
[test_n10_c9] Started, ID: 6a6f83dd990a
[test_n10_c3] Completed: P95=288.58ms, MaxPSS=238.5MB
[test_n10_c1] Completed: P95=305.71ms, MaxPSS=236.4MB
[test_n10_c0] Completed: P95=315.21ms, MaxPSS=235.2MB
[test_n10_c4] Completed: P95=318.38ms, MaxPSS=239.0MB
[test_n10_c6] Completed: P95=283.76ms, MaxPSS=236.3MB
[test_n10_c5] Completed: P95=317.80ms, MaxPSS=239.6MB
[test_n10_c7] Completed: P95=280.56ms, MaxPSS=233.6MB
[test_n10_c2] Completed: P95=282.44ms, MaxPSS=235.1MB
[test_n10_c8] Completed: P95=305.67ms, MaxPSS=235.8MB
[test_n10_c9] Completed: P95=296.38ms, MaxPSS=238.7MB
Result N=10: Avg P95=299.45ms, Total Peak PSS=2368.20MB

Creating Slice bench_test_n12_1767702804.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 12 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 12)
[test_n12_c1] Started, ID: 9b054496e1c6
[test_n12_c5] Started, ID: c437d74680c7
[test_n12_c6] Started, ID: 67a358042954
[test_n12_c0] Started, ID: f2872736714d
[test_n12_c3] Started, ID: be1e76b4d423
[test_n12_c4] Started, ID: 978bacbcf0d7
[test_n12_c2] Started, ID: c6184c4dd080
[test_n12_c10] Started, ID: ab92f51768ec
[test_n12_c7] Started, ID: 269404c054cf
[test_n12_c8] Started, ID: 351d02b8cbd9
[test_n12_c11] Started, ID: b4cc80d8d2a2
[test_n12_c9] Started, ID: 1ba5e90160ef
[test_n12_c1] Completed: P95=410.57ms, MaxPSS=243.2MB
[test_n12_c3] Completed: P95=424.76ms, MaxPSS=244.1MB
[test_n12_c4] Completed: P95=430.49ms, MaxPSS=241.1MB
[test_n12_c0] Completed: P95=319.67ms, MaxPSS=238.6MB
[test_n12_c5] Completed: P95=299.02ms, MaxPSS=235.6MB
[test_n12_c2] Completed: P95=301.54ms, MaxPSS=236.7MB
[test_n12_c6] Completed: P95=306.50ms, MaxPSS=238.1MB
[test_n12_c10] Completed: P95=287.12ms, MaxPSS=238.6MB
[test_n12_c9] Completed: P95=387.81ms, MaxPSS=250.5MB
[test_n12_c8] Completed: P95=397.38ms, MaxPSS=241.9MB
[test_n12_c7] Completed: P95=392.28ms, MaxPSS=245.5MB
[test_n12_c11] Completed: P95=391.42ms, MaxPSS=241.7MB
Result N=12: Avg P95=362.38ms, Total Peak PSS=2895.58MB

Creating Slice bench_test_n14_1767702872.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 14 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 14)
[test_n14_c0] Started, ID: 2d6e4afe3cfd
[test_n14_c1] Started, ID: 68ec4006fb7e
[test_n14_c5] Started, ID: acb82ae24f16
[test_n14_c2] Started, ID: 7cca85c7c578
[test_n14_c3] Started, ID: 1c6167c7ecb2
[test_n14_c6] Started, ID: d083d8a03d45
[test_n14_c11] Started, ID: 56ddd57bb89f
[test_n14_c4] Started, ID: 1f02ad32d5e2
[test_n14_c8] Started, ID: 60141d327931
[test_n14_c12] Started, ID: f42e7340fb0f
[test_n14_c7] Started, ID: 9745e3f5e06c
[test_n14_c10] Started, ID: f0753027a003
[test_n14_c13] Started, ID: 31d796cc451d
[test_n14_c9] Started, ID: 4060f2ff3330
[test_n14_c0] Completed: P95=125.73ms, MaxPSS=241.4MB
[test_n14_c1] Completed: P95=122.13ms, MaxPSS=241.2MB
[test_n14_c5] Completed: P95=366.33ms, MaxPSS=236.0MB
[test_n14_c3] Completed: P95=348.52ms, MaxPSS=238.0MB
[test_n14_c6] Completed: P95=365.29ms, MaxPSS=236.0MB
[test_n14_c7] Completed: P95=354.50ms, MaxPSS=235.2MB
[test_n14_c2] Completed: P95=363.19ms, MaxPSS=242.3MB
[test_n14_c8] Completed: P95=353.52ms, MaxPSS=232.1MB
[test_n14_c4] Completed: P95=357.59ms, MaxPSS=235.9MB
[test_n14_c11] Completed: P95=378.32ms, MaxPSS=235.8MB
[test_n14_c12] Completed: P95=379.48ms, MaxPSS=237.4MB
[test_n14_c10] Completed: P95=374.11ms, MaxPSS=237.4MB
[test_n14_c9] Completed: P95=371.60ms, MaxPSS=239.1MB
[test_n14_c13] Completed: P95=382.01ms, MaxPSS=237.8MB
Result N=14: Avg P95=331.59ms, Total Peak PSS=3325.54MB

Creating Slice bench_test_n16_1767702952.slice: AllowedCPUs=0,1, MemMax=2048MB

=== Test: 16 Instances | AllowedCPUs: 0,1 (2 cores) | Slice Mem: 2048MB ===
    Threads per container: 1 (total: 16)
[test_n16_c0] Started, ID: db8c0da3c5f3
[test_n16_c2] Started, ID: 1afcd3e444e7
[test_n16_c1] Started, ID: 048e55fa86af
[test_n16_c4] Started, ID: dd6fb08f9d56
[test_n16_c5] Started, ID: 0b4c559bae88
[test_n16_c7] Started, ID: 2f0d3efe8b1f
[test_n16_c3] Started, ID: e1e2b7201f26
[test_n16_c6] Started, ID: bec7d8c96e92
[test_n16_c10] Started, ID: 4f4defcc1ca3
[test_n16_c9] Started, ID: 051f0176eb8f
[test_n16_c15] Started, ID: eed70448fb6b
[test_n16_c8] Started, ID: c392f8a90591
[test_n16_c14] Started, ID: d9c561dc049d
[test_n16_c13] Started, ID: 717c49a9a7e0
[test_n16_c11] Started, ID: b4defa29fa0a
[test_n16_c12] Started, ID: 1f0584f776cb
[test_n16_c0] Completed: P95=536.13ms, MaxPSS=249.5MB
[test_n16_c15] Completed: P95=440.01ms, MaxPSS=235.7MB
[test_n16_c2] Completed: P95=555.66ms, MaxPSS=235.2MB
[test_n16_c14] Completed: P95=549.60ms, MaxPSS=235.3MB
[test_n16_c1] Completed: P95=446.40ms, MaxPSS=233.9MB
[test_n16_c4] Completed: P95=422.68ms, MaxPSS=234.9MB
[test_n16_c5] Completed: P95=400.52ms, MaxPSS=234.1MB
[test_n16_c6] Completed: P95=421.46ms, MaxPSS=236.6MB
[test_n16_c9] Completed: P95=408.73ms, MaxPSS=234.8MB
[test_n16_c13] Completed: P95=418.73ms, MaxPSS=238.4MB
[test_n16_c3] Completed: P95=510.72ms, MaxPSS=235.1MB
[test_n16_c12] Completed: P95=539.34ms, MaxPSS=236.8MB
[test_n16_c8] Completed: P95=513.81ms, MaxPSS=236.2MB
[test_n16_c11] Completed: P95=522.71ms, MaxPSS=236.1MB
[test_n16_c10] Completed: P95=554.59ms, MaxPSS=239.2MB
[test_n16_c7] Completed: P95=535.38ms, MaxPSS=239.2MB
Result N=16: Avg P95=486.03ms, Total Peak PSS=3791.07MB

Plot saved to results/experiment_260106_123023/scaling_result.png

=== Cleaning up resources ===
=== Cleanup complete ===
```