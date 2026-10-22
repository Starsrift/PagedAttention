# 服务器小规模实验记录

日期：2026-10-22。上游提交：`8dfd8c916e7d082b9167e35c01a6b44f8c973eb9`。

结论：RTX 3060 12GB 能运行该教学仓库的 CPU/CUDA 数值实验。本次已完成机制验证；生产级分页 Kernel、真实模型 serving、吞吐和延迟实验尚未完成，不能宣称已复现 PagedAttention 论文。

## 1. 实测环境

实验设备：RTX 3060 Linux 服务器（连接信息已脱敏）。

- Ubuntu 18.04.6，Linux 5.4.0-150-generic。
- NVIDIA GeForce RTX 3060，12045 MiB 显存；驱动 470.223.02。
- 系统 nvcc 11.8；实验 PyTorch 为 1.12.1+cu113，运行时 CUDA 11.3。三者不是同一个版本概念。
- 内存约 125GiB，根分区可用约 35GiB。
- Python 3.10.20；实验环境 NumPy 1.26.4。
- 检查期间其他 Python 任务占用约 6310MiB，GPU 利用率持续 100%。没有终止该任务，没有修改系统驱动或原环境。

独立实验目录：`~/workspace/pagedattention-small-20261022`。
目录保存上游 README、Notebook 的源码快照及补充脚本，不是完整 Git clone。
`.venv` 通过 `.pth` 只读复用 `已有 PyTorch 环境的 site-packages`，在自己的 site-packages 安装 NumPy 1.26.4。它依赖原环境路径，并非可独立迁移的环境。

原环境 NumPy 2.2.6 与老 PyTorch 的 NumPy 接口不兼容，Notebook 原样执行曾报 `RuntimeError: Numpy is not available`。独立环境修复后重新执行通过。

## 2. 实现与证据边界

上游 Notebook 的 `KVCacheAllocator` 提供物理块池，`SequenceState.append_token` 用整除/取模完成块定位；`paged_attention` 按表读取后调用 `torch.cat` 重建连续 K/V，再做普通 attention。它没有生产分页 CUDA Kernel，也没有实际 CoW、完整请求调度和 swap 生命周期。

`run_small.py` 的 `Pool`、`Seq` 是本次新增的实验扩展：随机化空闲块顺序、引用计数、分支共享、尾块写入前复制、释放归还及 OOM 状态保持。它不是上游已有能力，也不是高性能实现。扩展生命周期测试均在 CPU 上运行，即使指定 `--device cuda`。

## 3. 已完成实验

| 实验 | 覆盖范围 | 实测结果 |
|---|---|---|
| 上游 CPU 正确性 | B=4/16/32，各 6 种长度、3 个种子，随机物理块映射 | 54/54 通过，最大绝对误差 2.80895e-7 |
| 上游 CUDA 正确性 | 同一矩阵；FP32 输出对独立 FP64 连续参考 | 54/54 通过，最大绝对误差 2.10331e-7 |
| CoW/回收扩展 | 三种块大小，前缀长度 B-1/B/B+1，三个分支写入不同 token | 9/9 通过；历史 K/V 保持，分支独立，全部块回收 |
| OOM 原子性 | 仅一个物理块，共享尾块后写入触发 OOM | 长度、映射及引用计数保持，释放后块回收 |
| 空间核算 | 固定种子，64 请求，prompt 17–256，output 8–128 | 五种块大小均满足每请求尾部浪费 ≤ B-1 |
| 外部碎片构造例 | 两个不相邻 16-slot 空洞，申请 24 slots | first-fit 连续区间放不下；B=8 的分页可容纳 |

正确性测试覆盖块边界和尾部不足一块；不覆盖空序列、FP16/BF16、多头/GQA、并行线程安全、真实模型生成或长时间压力测试。CUDA 测试只确认功能，不将其运行时间用于性能结论。

## 4. 指定请求轨迹的空间结果

每个 tick 请求生成一个 token，到完成时释放；各方法处理完全相同的请求轨迹，无排队/拒绝。利用率 = 各 tick 有效 token slots 之和 / 各 tick 分配 token slots 之和。
静态基线为每活跃请求预留 512 slots；不是所有连续 KV 实现的统一行为。理想精确分配的连续布局只需要有效 slots，这里未计其增长复制和外部碎片成本。

| Block size | 分页利用率 | 静态预留利用率 | 分配 slot-ticks 减少 |
|---|---:|---:|---:|
| 4 | 99.06% | 30.91% | 68.79% |
| 8 | 97.84% | 30.91% | 68.40% |
| 16 | 95.47% | 30.91% | 67.62% |
| 32 | 91.06% | 30.91% | 66.05% |
| 64 | 83.64% | 30.91% | 63.04% |

这些数值是合成负载的逻辑槽位核算，不是 CUDA allocator 实测、模型容量或吞吐提升。块表、引用计数和内存对齐开销未计入。小块减少尾部浪费，但本实验未测它增加的寻址和 Kernel 开销，不能据此断言 B=4 最优。

## 5. 复现已完成部分

```bash
cd ~/workspace/pagedattention-small-20261022
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python experiments/run_small.py --device cpu --output experiments/results/cpu.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python experiments/run_small.py --device cuda --output experiments/results/cuda_correctness.json
```

保留测量数据、移除主机标识后的结果和完整请求轨迹见 `results/cpu.json`、`results/cuda_correctness.json`；环境与日志位于同一目录。脚本失败时抛异常，不输出 passed。

## 6. 完整小规模试验尚缺的部分

1. **空闲 GPU 窗口**：他人的计算负载会污染延迟、吞吐和显存数据。正式测量前确认进程列表和利用率，记录预热与重复运行策略。
2. **生产后端兼容环境**：官方当前 vLLM 文档所列 CUDA 12.x 预编译方案不能直接使用 470 驱动。NVIDIA 给出的 CUDA 12.x minor compatibility 驱动下限为 525；实际选定 wheel/后端可能要求更高。升级驱动会影响正在运行的任务，不在本次工作中自动执行。Ubuntu 18.04 的 glibc 也需对照所选 wheel 检查。
3. **真实分页 Kernel**：用其原生 block table 布局，与同形状连续 attention 做正确性和预热后计时；不能用当前 `torch.cat` 实现代表分页 Kernel。
4. **真实模型 serving**：建议从 Qwen2.5-0.5B-Instruct 开始，确认余量后扩到 1.5B；均为待验证配置，未下载或跑通。固定模型 revision、FP16、输出长度、随机种子；并发 1/2/4/8，输入长度 128/512/1024，输出 64，先预热，再每配置至少三轮。
5. **指标与归因**：记录 output tokens/s、TTFT、TPOT、P50/P95、峰值显存、失败率和 preemption；短小样本不主张稳定 P99。先关闭 prefix cache；共享前缀实验另行控制命中率。vLLM 与其他引擎的对比只能作为系统对比，单独归因分页须保持模型、调度与算子条件一致。

当前未执行安装 vLLM、驱动升级、模型下载或服务压测；缺少这些结果时，本记录只支持“分页寻址和生命周期正确，特定轨迹下槽位浪费降低”。

官方依据（2026-10-22 查询）：

- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- https://arxiv.org/abs/2309.06180 （原论文入口，本次未重新阅读全文）
