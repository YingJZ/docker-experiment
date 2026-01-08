# 实验说明

代码修改：当前实验修复了 init_ms 的计算方式，使其能够精确测量纯粹的 Python 解释器初始化时间，排除 systemd、cgroup 等系统级开销的影响。（具体可见 [MODIFICATION_SUMMARY.md](MODIFICATION_SUMMARY.md)）

配置修改：**内存限制改为 4GB** （CPU 限制保持 2 核不变）

## 命令行输出(部分)

```bash
(experiment) yingjiaze@haslab4:~/experiment/batch/native$ sudo /home/yingjiaze/experiment/.venv/bin/python concurrent_benchmark.py --mem 4096
INFO: Native benchmark, INSTANCE_COUNTS=[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
INFO: Using Python interpreter: /home/yingjiaze/experiment/.venv/bin/python
Creating Slice native_bench_test_n1_1767852622.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 1 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 2 (total: 2)
[test_n1_p0] Started, PID: 3737047
[test_n1_p0] CPU affinity mask: 3
[test_n1_p0] Completed: P95=41.72ms, MaxPSS=352.5MB, PyInit=48.3ms
Result N=1: Avg P95=41.72ms, Total Peak PSS=352.52MB

Creating Slice native_bench_test_n2_1767852632.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 2 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 2)
[test_n2_p0] Started, PID: 3737114
[test_n2_p1] Started, PID: 3737115
[test_n2_p0] CPU affinity mask: 3
[test_n2_p1] CPU affinity mask: 3
[test_n2_p0] Completed: P95=50.95ms, MaxPSS=295.3MB, PyInit=49.6ms
[test_n2_p1] Completed: P95=51.38ms, MaxPSS=295.2MB, PyInit=46.9ms
Result N=2: Avg P95=51.17ms, Total Peak PSS=590.52MB

Creating Slice native_bench_test_n4_1767852642.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 4 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 4)
[test_n4_p1] Started, PID: 3737213
[test_n4_p0] Started, PID: 3737215
[test_n4_p1] CPU affinity mask: 3
[test_n4_p2] Started, PID: 3737216
[test_n4_p3] Started, PID: 3737217
[test_n4_p0] CPU affinity mask: 3
[test_n4_p2] CPU affinity mask: 3
[test_n4_p3] CPU affinity mask: 3
[test_n4_p0] Completed: P95=112.78ms, MaxPSS=257.9MB, PyInit=80.7ms
[test_n4_p1] Completed: P95=144.99ms, MaxPSS=255.0MB, PyInit=59.7ms
[test_n4_p2] Completed: P95=145.13ms, MaxPSS=273.7MB, PyInit=79.7ms
[test_n4_p3] Completed: P95=144.33ms, MaxPSS=270.9MB, PyInit=55.1ms
Result N=4: Avg P95=136.81ms, Total Peak PSS=1057.47MB

Creating Slice native_bench_test_n6_1767852662.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 6 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 6)
[test_n6_p0] Started, PID: 3737361
[test_n6_p3] Started, PID: 3737366
[test_n6_p1] Started, PID: 3737362
[test_n6_p2] Started, PID: 3737364
[test_n6_p5] Started, PID: 3737369
[test_n6_p4] Started, PID: 3737367
[test_n6_p3] CPU affinity mask: 3
[test_n6_p1] CPU affinity mask: 3
[test_n6_p0] CPU affinity mask: 3
[test_n6_p5] CPU affinity mask: 3
[test_n6_p2] CPU affinity mask: 3
[test_n6_p4] CPU affinity mask: 3
[test_n6_p2] Completed: P95=169.64ms, MaxPSS=244.5MB, PyInit=103.0ms
[test_n6_p5] Completed: P95=155.78ms, MaxPSS=243.5MB, PyInit=94.9ms
[test_n6_p4] Completed: P95=168.62ms, MaxPSS=241.0MB, PyInit=88.0ms
[test_n6_p3] Completed: P95=171.50ms, MaxPSS=240.8MB, PyInit=81.7ms
[test_n6_p0] Completed: P95=170.48ms, MaxPSS=246.8MB, PyInit=117.9ms
[test_n6_p1] Completed: P95=170.90ms, MaxPSS=245.5MB, PyInit=108.7ms
Result N=6: Avg P95=167.82ms, Total Peak PSS=1462.07MB

Creating Slice native_bench_test_n8_1767852691.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 8 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 8)
[test_n8_p0] Started, PID: 3737518
[test_n8_p2] Started, PID: 3737525
[test_n8_p0] CPU affinity mask: 3
[test_n8_p1] Started, PID: 3737521
[test_n8_p3] Started, PID: 3737528
[test_n8_p4] Started, PID: 3737530
[test_n8_p2] CPU affinity mask: 3
[test_n8_p5] Started, PID: 3737532
[test_n8_p6] Started, PID: 3737535
[test_n8_p1] CPU affinity mask: 3
[test_n8_p7] Started, PID: 3737539
[test_n8_p4] CPU affinity mask: 3
[test_n8_p6] CPU affinity mask: 3
[test_n8_p3] CPU affinity mask: 3
[test_n8_p5] CPU affinity mask: 3
[test_n8_p7] CPU affinity mask: 3
[test_n8_p3] Completed: P95=273.75ms, MaxPSS=237.4MB, PyInit=151.7ms
[test_n8_p6] Completed: P95=264.52ms, MaxPSS=239.7MB, PyInit=149.6ms
[test_n8_p1] Completed: P95=276.11ms, MaxPSS=238.3MB, PyInit=115.2ms
[test_n8_p7] Completed: P95=274.34ms, MaxPSS=238.7MB, PyInit=152.2ms
[test_n8_p0] Completed: P95=280.41ms, MaxPSS=238.8MB, PyInit=146.9ms
[test_n8_p4] Completed: P95=274.40ms, MaxPSS=239.6MB, PyInit=111.3ms
[test_n8_p5] Completed: P95=274.11ms, MaxPSS=248.2MB, PyInit=151.1ms
[test_n8_p2] Completed: P95=276.42ms, MaxPSS=261.2MB, PyInit=161.1ms
Result N=8: Avg P95=274.26ms, Total Peak PSS=1941.87MB

Creating Slice native_bench_test_n10_1767852731.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 10 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 10)
[test_n10_p0] Started, PID: 3737899
[test_n10_p2] Started, PID: 3737903
[test_n10_p3] Started, PID: 3737905
[test_n10_p1] Started, PID: 3737901
[test_n10_p0] CPU affinity mask: 3
[test_n10_p4] Started, PID: 3737908
[test_n10_p6] Started, PID: 3737913
[test_n10_p5] Started, PID: 3737909
[test_n10_p7] Started, PID: 3737916
[test_n10_p2] CPU affinity mask: 3
[test_n10_p8] Started, PID: 3737918
[test_n10_p9] Started, PID: 3737920
[test_n10_p1] CPU affinity mask: 3
[test_n10_p3] CPU affinity mask: 3
[test_n10_p6] CPU affinity mask: 3
[test_n10_p7] CPU affinity mask: 3
[test_n10_p4] CPU affinity mask: 3
[test_n10_p8] CPU affinity mask: 3
[test_n10_p5] CPU affinity mask: 3
[test_n10_p9] CPU affinity mask: 3
[test_n10_p5] Completed: P95=247.04ms, MaxPSS=249.3MB, PyInit=164.5ms
[test_n10_p3] Completed: P95=259.47ms, MaxPSS=246.4MB, PyInit=146.3ms
[test_n10_p4] Completed: P95=275.20ms, MaxPSS=249.7MB, PyInit=187.2ms
[test_n10_p2] Completed: P95=271.75ms, MaxPSS=237.5MB, PyInit=175.0ms
[test_n10_p9] Completed: P95=276.20ms, MaxPSS=238.9MB, PyInit=189.5ms
[test_n10_p8] Completed: P95=276.89ms, MaxPSS=252.0MB, PyInit=197.3ms
[test_n10_p6] Completed: P95=280.43ms, MaxPSS=248.7MB, PyInit=189.7ms
[test_n10_p1] Completed: P95=278.13ms, MaxPSS=249.3MB, PyInit=173.9ms
[test_n10_p7] Completed: P95=278.25ms, MaxPSS=271.3MB, PyInit=125.8ms
[test_n10_p0] Completed: P95=279.74ms, MaxPSS=274.7MB, PyInit=169.9ms
Result N=10: Avg P95=272.31ms, Total Peak PSS=2517.91MB

Creating Slice native_bench_test_n12_1767852782.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 12 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 12)
[test_n12_p0] Started, PID: 3738342
[test_n12_p1] Started, PID: 3738344
[test_n12_p0] CPU affinity mask: 3
[test_n12_p3] Started, PID: 3738349
[test_n12_p1] CPU affinity mask: 3
[test_n12_p2] Started, PID: 3738346
[test_n12_p6] Started, PID: 3738354
[test_n12_p4] Started, PID: 3738351
[test_n12_p5] Started, PID: 3738358
[test_n12_p3] CPU affinity mask: 3
[test_n12_p8] Started, PID: 3738360
[test_n12_p9] Started, PID: 3738366
[test_n12_p10] Started, PID: 3738368
[test_n12_p4] CPU affinity mask: 3
[test_n12_p2] CPU affinity mask: 3
[test_n12_p11] Started, PID: 3738369
[test_n12_p7] Started, PID: 3738362
[test_n12_p5] CPU affinity mask: 3
[test_n12_p6] CPU affinity mask: 3
[test_n12_p10] CPU affinity mask: 3
[test_n12_p8] CPU affinity mask: 3
[test_n12_p9] CPU affinity mask: 3
[test_n12_p11] CPU affinity mask: 3
[test_n12_p7] CPU affinity mask: 3
[test_n12_p9] Completed: P95=333.45ms, MaxPSS=227.6MB, PyInit=229.4ms
[test_n12_p3] Completed: P95=334.45ms, MaxPSS=229.1MB, PyInit=194.9ms
[test_n12_p7] Completed: P95=334.42ms, MaxPSS=231.4MB, PyInit=220.0ms
[test_n12_p1] Completed: P95=338.75ms, MaxPSS=232.5MB, PyInit=193.6ms
[test_n12_p0] Completed: P95=336.26ms, MaxPSS=233.3MB, PyInit=200.0ms
[test_n12_p6] Completed: P95=339.09ms, MaxPSS=232.4MB, PyInit=208.6ms
[test_n12_p11] Completed: P95=336.03ms, MaxPSS=232.1MB, PyInit=225.3ms
[test_n12_p4] Completed: P95=347.09ms, MaxPSS=232.2MB, PyInit=245.5ms
[test_n12_p5] Completed: P95=336.54ms, MaxPSS=229.3MB, PyInit=215.9ms
[test_n12_p2] Completed: P95=339.24ms, MaxPSS=232.9MB, PyInit=207.0ms
[test_n12_p10] Completed: P95=351.57ms, MaxPSS=236.7MB, PyInit=213.2ms
[test_n12_p8] Completed: P95=354.49ms, MaxPSS=233.5MB, PyInit=219.3ms
Result N=12: Avg P95=340.12ms, Total Peak PSS=2782.81MB

Creating Slice native_bench_test_n14_1767852842.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 14 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 14)
[test_n14_p0] Started, PID: 3738642
[test_n14_p2] Started, PID: 3738645
[test_n14_p0] CPU affinity mask: 3
[test_n14_p3] Started, PID: 3738646
[test_n14_p1] Started, PID: 3738643
[test_n14_p2] CPU affinity mask: 3
[test_n14_p4] Started, PID: 3738650
[test_n14_p5] Started, PID: 3738652
[test_n14_p6] Started, PID: 3738653
[test_n14_p1] CPU affinity mask: 3
[test_n14_p10] Started, PID: 3738667
[test_n14_p3] CPU affinity mask: 3
[test_n14_p4] CPU affinity mask: 3
[test_n14_p8] Started, PID: 3738661
[test_n14_p6] CPU affinity mask: 3
[test_n14_p7] Started, PID: 3738658
[test_n14_p10] CPU affinity mask: 3
[test_n14_p11] Started, PID: 3738673
[test_n14_p7] CPU affinity mask: 3
[test_n14_p8] CPU affinity mask: 3
[test_n14_p11] CPU affinity mask: 3
[test_n14_p5] CPU affinity mask: 3
[test_n14_p12] Started, PID: 3738672
[test_n14_p9] Started, PID: 3738663
[test_n14_p13] Started, PID: 3738670
[test_n14_p9] CPU affinity mask: 3
[test_n14_p12] CPU affinity mask: 3
[test_n14_p13] CPU affinity mask: 3
```