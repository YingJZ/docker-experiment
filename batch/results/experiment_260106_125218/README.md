**当前 commit 修复内容**
- 采集容器所属 cgroup 的 `memory.stat`（anon/file/slab/kernel_stack/shmem/swap 等，单位 MB），便于区分匿名页与 file-backed 占比。

**前一 commit 实现内容**
- 并发基准脚本新增内存组成采样：基于 `/proc/<pid>/smaps_rollup` 记录 PSS/RSS、Shared/Private Clean/Dirty、Swap、AnonHugePages 的 max/avg。
- `ContainerMetrics` 与 `data.json` 增加 `mem_breakdown_max/avg` 与 `cgroup_mem_stat` 字段，用于后续分析权重共享、匿名页膨胀。


## 结果分析

- cgroup 采集已生效：每个容器的 cgroup_mem_stat 已填充，rss ≈ 200MB，与 smaps PSS/Private Dirty 水平一致；cache 很小（≈0.7MB），说明 clean file-backed 部分不计入 RSS（符合 cgroup 语义）。total_* 字段数值很大，代表层级/祖先 cgroup 的累计，不是单容器本地值。

- 内存构成稳定：shared_clean ~80–96MB（权重文件映射，可共享），private_dirty 常见 150–190MB，随实例数近线性叠加；anon_huge、swap 仍为 0。个别样本 shared_dirty 有非零（可能短暂写脏页/日志），但量小。

- 并发上限：memory_full=true 从 N=10 开始，符合 2GB 限额下每实例 ~230–250MB 峰值的线性叠加。

- 延迟趋势：P95 随并发上升显著，N≥10 后延迟抬升，N=16 组内分布两极（部分 180ms 左右，部分 >300ms），仍表现出过度争用。

- 对照 cgroup 与 smaps：

- smaps_rollup 给出 PSS/RSS/共享/私有拆分；cgroup rss 仅统计匿名+文件脏页，不含 clean cache，因此比 PSS 略低是预期的。

- 如果需要“切片总占用”，可在采样时额外读取 slice 路径（当前使用容器自己的路径，total_* 已是层级累加，但包含祖先的系统其他负载）。

## 后续建议

- 若关心 slice 汇总，可在监控线程中解析一次容器的 cgroup 路径，向上定位 slice 目录，读取该目录的 memory.current/memory.stat 作为“全切片”值。

- 如需更强的 file-backed/anon 对比，可在 cgroup 采样里记录 anon、file（v2 有）或用 memory.stat 的 inactive_file/active_file 与 smaps 的 shared_clean 对齐。