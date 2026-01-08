# 实验说明

代码修改：暂时 stash 对 init_ms 的修复,因为init_ms发生了较大变动?设计上应该只是增加了python_init_ms

配置修改：内存限制 4GB，CPU 限制保持 2 核不变

## 命令行输出(部分)

```bash
(experiment) yingjiaze@haslab4:~/experiment/batch/native$ sudo /home/yingjiaze/experiment/.venv/bin/python concurrent_benchmark.py --mem 4096
INFO: Native benchmark, INSTANCE_COUNTS=[1, 2, 4, 6, 8, 10, 12, 14, 16]
INFO: Using Python interpreter: /home/yingjiaze/experiment/.venv/bin/python
Creating Slice native_bench_test_n1_1767853988.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 1 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 2 (total: 2)
[test_n1_p0] Started, PID: 3742796
[test_n1_p0] CPU affinity mask: 3
[test_n1_p0] Completed: P95=42.74ms, MaxPSS=358.7MB
Result N=1: Avg P95=42.74ms, Total Peak PSS=358.70MB

Creating Slice native_bench_test_n2_1767853999.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 2 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 2)
[test_n2_p0] Started, PID: 3742863
[test_n2_p1] Started, PID: 3742864
[test_n2_p0] CPU affinity mask: 3
[test_n2_p1] CPU affinity mask: 3
[test_n2_p1] Completed: P95=52.40ms, MaxPSS=294.8MB
[test_n2_p0] Completed: P95=56.22ms, MaxPSS=329.8MB
Result N=2: Avg P95=54.31ms, Total Peak PSS=624.62MB

Creating Slice native_bench_test_n4_1767854009.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 4 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 4)
[test_n4_p0] Started, PID: 3742985
[test_n4_p1] Started, PID: 3742987
[test_n4_p2] Started, PID: 3742988
[test_n4_p0] CPU affinity mask: 3
[test_n4_p3] Started, PID: 3742989
[test_n4_p1] CPU affinity mask: 3
[test_n4_p2] CPU affinity mask: 3
[test_n4_p3] CPU affinity mask: 3
[test_n4_p2] Completed: P95=120.00ms, MaxPSS=257.1MB
[test_n4_p0] Completed: P95=144.43ms, MaxPSS=277.2MB
[test_n4_p1] Completed: P95=143.97ms, MaxPSS=273.4MB
[test_n4_p3] Completed: P95=145.55ms, MaxPSS=270.9MB
Result N=4: Avg P95=138.49ms, Total Peak PSS=1078.61MB

Creating Slice native_bench_test_n6_1767854029.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 6 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 6)
[test_n6_p2] Started, PID: 3743145
[test_n6_p2] CPU affinity mask: 3
[test_n6_p0] Started, PID: 3743146
[test_n6_p1] Started, PID: 3743148
[test_n6_p3] Started, PID: 3743150
[test_n6_p4] Started, PID: 3743152
[test_n6_p5] Started, PID: 3743153
[test_n6_p0] CPU affinity mask: 3
[test_n6_p1] CPU affinity mask: 3
[test_n6_p3] CPU affinity mask: 3
[test_n6_p4] CPU affinity mask: 3
[test_n6_p5] CPU affinity mask: 3
[test_n6_p2] Completed: P95=169.01ms, MaxPSS=247.4MB
[test_n6_p1] Completed: P95=168.66ms, MaxPSS=241.4MB
[test_n6_p4] Completed: P95=170.68ms, MaxPSS=246.2MB
[test_n6_p5] Completed: P95=168.86ms, MaxPSS=241.5MB
[test_n6_p3] Completed: P95=181.46ms, MaxPSS=251.1MB
[test_n6_p0] Completed: P95=169.31ms, MaxPSS=255.0MB
Result N=6: Avg P95=171.33ms, Total Peak PSS=1482.46MB

Creating Slice native_bench_test_n8_1767854058.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 8 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 8)
[test_n8_p0] Started, PID: 3743304
[test_n8_p1] Started, PID: 3743306
[test_n8_p0] CPU affinity mask: 3
[test_n8_p2] Started, PID: 3743308
[test_n8_p3] Started, PID: 3743310
[test_n8_p4] Started, PID: 3743312
[test_n8_p1] CPU affinity mask: 3
[test_n8_p5] Started, PID: 3743314
[test_n8_p6] Started, PID: 3743315
[test_n8_p2] CPU affinity mask: 3
[test_n8_p4] CPU affinity mask: 3
[test_n8_p7] Started, PID: 3743316
[test_n8_p5] CPU affinity mask: 3
[test_n8_p3] CPU affinity mask: 3
[test_n8_p6] CPU affinity mask: 3
[test_n8_p7] CPU affinity mask: 3
[test_n8_p6] Completed: P95=223.13ms, MaxPSS=244.6MB
[test_n8_p5] Completed: P95=222.65ms, MaxPSS=242.1MB
[test_n8_p1] Completed: P95=218.96ms, MaxPSS=237.4MB
[test_n8_p3] Completed: P95=221.25ms, MaxPSS=235.8MB
[test_n8_p2] Completed: P95=250.57ms, MaxPSS=248.9MB
[test_n8_p4] Completed: P95=223.46ms, MaxPSS=238.1MB
[test_n8_p7] Completed: P95=227.45ms, MaxPSS=246.7MB
[test_n8_p0] Completed: P95=218.90ms, MaxPSS=235.0MB
Result N=8: Avg P95=225.80ms, Total Peak PSS=1928.50MB

Creating Slice native_bench_test_n10_1767854098.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 10 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 10)
[test_n10_p0] Started, PID: 3743529
[test_n10_p1] Started, PID: 3743531
[test_n10_p0] CPU affinity mask: 3
[test_n10_p2] Started, PID: 3743533
[test_n10_p3] Started, PID: 3743535
[test_n10_p1] CPU affinity mask: 3
[test_n10_p6] Started, PID: 3743541
[test_n10_p3] CPU affinity mask: 3
[test_n10_p2] CPU affinity mask: 3
[test_n10_p8] Started, PID: 3743543
[test_n10_p5] Started, PID: 3743539
[test_n10_p9] Started, PID: 3743545
[test_n10_p6] CPU affinity mask: 3
[test_n10_p4] Started, PID: 3743537
[test_n10_p7] Started, PID: 3743544
[test_n10_p8] CPU affinity mask: 3
[test_n10_p5] CPU affinity mask: 3
[test_n10_p9] CPU affinity mask: 3
[test_n10_p4] CPU affinity mask: 3
[test_n10_p7] CPU affinity mask: 3
[test_n10_p0] Completed: P95=263.14ms, MaxPSS=249.2MB
[test_n10_p1] Completed: P95=265.31ms, MaxPSS=254.7MB
[test_n10_p9] Completed: P95=261.09ms, MaxPSS=246.3MB
[test_n10_p8] Completed: P95=269.45ms, MaxPSS=237.9MB
[test_n10_p4] Completed: P95=265.80ms, MaxPSS=237.8MB
[test_n10_p3] Completed: P95=275.87ms, MaxPSS=243.4MB
[test_n10_p2] Completed: P95=277.93ms, MaxPSS=240.7MB
[test_n10_p7] Completed: P95=274.18ms, MaxPSS=251.3MB
[test_n10_p6] Completed: P95=276.66ms, MaxPSS=254.6MB
[test_n10_p5] Completed: P95=274.67ms, MaxPSS=328.2MB
Result N=10: Avg P95=270.41ms, Total Peak PSS=2544.04MB

Creating Slice native_bench_test_n12_1767854148.slice: AllowedCPUs=0,1, MemMax=4096MB
Warning: Slice AllowedCPUs mismatch. Expected: 0,1, Got: 0-1
✓ Set cpuset.cpus=0,1 via cgroup v1

=== Test: 12 Instances | AllowedCPUs: 0,1 (2 cores) | Mem: 4096MB ===
    Threads per process: 1 (total: 12)
[test_n12_p0] Started, PID: 3743864
[test_n12_p1] Started, PID: 3743868
[test_n12_p0] CPU affinity mask: 3
[test_n12_p2] Started, PID: 3743874
[test_n12_p3] Started, PID: 3743873
[test_n12_p1] CPU affinity mask: 3
[test_n12_p6] Started, PID: 3743879
[test_n12_p4] Started, PID: 3743875
[test_n12_p5] Started, PID: 3743877
[test_n12_p3] CPU affinity mask: 3
[test_n12_p7] Started, PID: 3743882
[test_n12_p8] Started, PID: 3743886
[test_n12_p10] Started, PID: 3743892
[test_n12_p9] Started, PID: 3743896
[test_n12_p4] CPU affinity mask: 3
[test_n12_p6] CPU affinity mask: 3
[test_n12_p11] Started, PID: 3743897
[test_n12_p2] CPU affinity mask: 3
[test_n12_p5] CPU affinity mask: 3
[test_n12_p10] CPU affinity mask: 3
[test_n12_p8] CPU affinity mask: 3
[test_n12_p7] CPU affinity mask: 3
[test_n12_p9] CPU affinity mask: 3
[test_n12_p11] CPU affinity mask: 3
```