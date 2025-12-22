在每次实验前，使用 `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` 清空操作系统页缓存还是不能完全解决问题

```bash
(experiment) yingjiaze@haslab4:~/experiment/batch$ python auto_benchmark.py

===== Run 1/5 =====
🧹 Dropping system caches to ensure cold start...
3
本次结果(ms): 初始化=372.0, import torch=28508.0, 模型加载=515.0, 预热=780.0, 推理平均延迟=37.55

===== Run 2/5 =====
🧹 Dropping system caches to ensure cold start...
3
本次结果(ms): 初始化=75.0, import torch=7073.0, 模型加载=382.0, 预热=242.0, 推理平均延迟=38.01

===== Run 3/5 =====
🧹 Dropping system caches to ensure cold start...
3
本次结果(ms): 初始化=68.0, import torch=7048.0, 模型加载=375.0, 预热=241.0, 推理平均延迟=36.99

===== Run 4/5 =====
🧹 Dropping system caches to ensure cold start...
3
本次结果(ms): 初始化=68.0, import torch=7030.0, 模型加载=380.0, 预热=241.0, 推理平均延迟=39.32

===== Run 5/5 =====
🧹 Dropping system caches to ensure cold start...
3
本次结果(ms): 初始化=72.0, import torch=7069.0, 模型加载=387.0, 预热=245.0, 推理平均延迟=37.77

===== 5 次运行平均耗时 (ms) =====
初始化: 131.0 ms
import torch: 11345.6 ms
模型加载: 407.8 ms
预热: 349.8 ms
推理平均延迟: 37.93 ms
```

核心原因概括
即使你每次都 drop_caches，第一次还是最慢，是因为：你只把「文件系统的页缓存」清了，但很多「一次性初始化」和「非页缓存」的东西并没有被重置。
具体拆一下：
drop_caches 只清 PageCache / dentries / inodes：
它不会清：
CPU 的 L1/L2/L3 cache、TLB、分支预测器
任何进程里已经分配的匿名内存（比如 PyTorch 的内存池）
PyTorch / MKL / OpenBLAS 内部的全局状态和懒初始化结果
~/.cache/torch 之类写在磁盘上的缓存文件（它们还是存在，只是读它们时可能变成冷 IO）
第一次 import torch 会做「一堆只需要做一次或很少做的工作」：
动态加载很多 .so，做符号重定位、初始化 OpenMP/MKL 线程池
探测 CPU 指令集、构建算子注册表、加载/初始化后端
可能创建/写入 ~/.cache/torch、编译/加载 JIT kernel 等
后面几次，即使页缓存被清掉了，这些「一次性工作」已经完成或变得非常便宜，所以 导入时间从 28s 掉到 7s。
预热（Warmup）同理：
第一次推理会触发：
内存池的大块分配
MKL/BLAS 的内部 warmup
可能的 kernel 选择 / layout 选择
后面几次在相同模型、相同 batch 上，很多路径已经被热身过，所以从 780ms 变成 ~240ms。
子进程之间仍然共享一些「系统级」状态：
多个 python 进程会共享同一份 Python 解释器、libc、MKL 等代码段的物理页，只要系统里还有进程在用，它们就不会被 drop 掉。
你是用父进程 auto_benchmark.py 循环起子进程，父进程本身、系统里的其他进程也会让一部分库保持常驻。
总结
drop_caches ≠「把系统变回刚开机的样子」，它只能保证「磁盘文件读是冷的」，不能把 PyTorch/MKL 等库内部的一次性初始化、JIT、内存池等恢复成「第一次」。
因此你看到的现象是正常的：第 1 次包含了很多真正的一次性初始化成本，后 4 次在“冷 IO +热 runtime”的状态下运行，自然会明显更快。
