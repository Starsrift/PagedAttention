# DeepSeek 1.5B 本机分页缓存实验结果

日期：2026-10-23（Asia/Shanghai）。BF16 正式运行开始时间 09:10。
事前方案及精度修订：[DEEPSEEK_PLAN.md](DEEPSEEK_PLAN.md)。

**结论：DeepSeek-R1-Distill-Qwen-1.5B 已在 RTX 3070 8GB 上完成 BF16 E1–E5，全部通过。首次 FP16 普通生成失败，未标记通过，失败和诊断记录一并保存。**

## 1. 模型与实现范围

| 项目 | 配置与实测 |
|---|---|
| 模型 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| 模型 commit | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| 基础架构 | Qwen2，28 层，12 个 Q 头、2 个 KV 头，head_dim=128 |
| 实际参数量 | 1,777,088,000 |
| BF16 参数字节数 | 3,554,176,000 bytes |
| 权重文件 | 3,554,214,621 bytes，SHA256 与官方 LFS 元数据一致 |
| 环境 | Windows 原生，RTX 3070 8GB，驱动 610.88，Python 3.13.16 |
| 依赖 | PyTorch 2.10.0+cu126，Transformers 4.57.6，沿用仓库 uv 锁定环境 |
| 正式精度与后端 | BF16，eager attention，batch=1，关闭 TF32 |
| 解码 | 固定贪心回归，最多 32 个新 token，使用 DeepSeek 自带模板 |

这是 DeepSeek-R1 的 Qwen 架构蒸馏模型，不是完整 DeepSeek-R1 或 DeepSeek-V3。
持久 K/V 存放于物理块池，通过块表定位，attention 前仍临时重建连续 K/V。
因此本轮验证缓存管理正确性，不代表原生分页 Kernel、服务吞吐或推理能力评测。

模型目录为 `models/DeepSeek-R1-Distill-Qwen-1.5B/`，没有覆盖 Qwen3-0.6B 的模型目录。
DeepSeek 模板分词显式关闭额外特殊 token 添加，确认输入仅含一个 BOS。
Qwen 原三个提示词的 token IDs 在相应分词调整前后完全一致，长度仍为 21/22/20。

## 2. FP16 失败及 BF16 修订依据

第一次按 FP16 运行，E1 普通 DynamicCache 生成即出现 `Nonfinite logits`，三个生成案例尚无完成项，E2–E5 尚未执行。
该结果保存为 `fp16-attempt/result.json`，状态为 failed，没有进入分页对照，不能将它判为分页适配器导致的错误。

随后用同一个已校验模型、同一个提示词，单独检查普通连续缓存路径：

| 诊断 | FP16 | BF16 |
|---|---|---|
| 非有限权重数量 | 0 | 0 |
| 首个非有限模块输出 | 第 0 层 attention | 未发现 |
| Prefill logits 有限 | 否 | 是 |
| 本机支持 BF16 | 不适用 | 是 |

据此在完整重试前修订方案，两条对照路径统一 BF16。模型原配置也标注为 bfloat16。
通过容差、输入文本、块大小和贪心回归标准保持不变。
诊断只定位到普通 FP16 attention 输出异常，尚未进一步定位其中某个具体算子的溢出。

## 3. E1–E5 正式结果

| 编号 | 实验与判据 | BF16 结果 |
|---|---|---|
| E1 | 三个提示词普通生成，logits 有限，无异常 | 3/3，通过 |
| E2 | B=4/16/32，边界与长度 127，真实 K/V 分页写入后按表读回 | 15/15，28 层 K/V 形状、dtype 和内容逐元素一致 |
| E2 错误对照 | 故意交换块表项并恢复 | 6/6 检出错误，恢复后完全一致 |
| E3 | 三种块大小×三个提示词，逐步 logits 与独立贪心 token 对照 | 9/9，通过，288 次 logits 对照最大绝对差值 0，所有 token 相同 |
| E4 | B−1/B/B+1 前缀，共享、两个不同 token 分支、CoW、回收 | 9/9，通过，18 个分支的 logits 对照差值均为 0 |
| E4 逻辑 OOM | 逐层强制共享尾块分配失败，检查原状态和内容 | 28/28 层通过，最终全部回收 |
| E5 | 64 token 输入、8 次 decode、B=16，预热并重复观察 | 两种路径各 3 次正式测量完成，空间公式核对通过 |

E3 的事前容差为 `abs(error) <= 0.02 + 0.002 * abs(reference)`，没有放宽。
288 次比较包含 prefill 预测和后续 decode 预测，不全是 decode。
零误差只对应本轮固定模型、后端、精度和输入，不保证其他场景逐位相同。

E4 中，未满的共享尾块由两个分支各复制一次，每例跨 28 层共 56 次真实 CoW；刚好位于块边界的前缀无需 CoW，各分支领取新块。
每次操作都检查所有活动块表与引用计数一致，并将分支 K/V 和 logits 与各自独立连续缓存对照，父缓存及另一分支保持不变。
OOM 检查仅覆盖每层缓存分配失败时的状态保持，不代表整个模型 forward 是事务性的，也没有耗尽整卡显存。

### E1 输出说明

三个模板化输入分别为 14/15/13 token，均生成到 32 个新 token 的上限。
DeepSeek 模板开启推理文本，短上限下样例主要是部分推理过程，尚未完成最终回答。
这不构成“答错”或“能力不及另一模型”的结论。本轮评价的是两种缓存对同一推理过程是否保持一致。

## 4. 空间与耗时观察

BF16 下，全部 28 层每个 token 的有效 K/V 大小为：

`2 × 28 × 2 × 128 × 2 = 28672 bytes = 28 KiB`

最终 T=72、B=16 时：

- 有效 K/V：`72 × 28 KiB = 1.96875 MiB`。
- 5 个已映射块、80 个槽位：`2.1875 MiB`，尾空为 `0.21875 MiB`。
- 预分配 9 块、144 个槽位：整个池为 `3.9375 MiB`，包含未映射容量 `1.75 MiB`。

每种方法预热 1 次，正式重复 3 次并交替顺序。Prefill 为三轮中位数，decode 为各轮 8 步中位数再取中位数。

| 指标 | DynamicCache | 教学分页适配器 |
|---|---:|---:|
| Prefill 中位数，ms | 40.684 | 51.043 |
| 单步 Decode 中位数，ms | 40.720 | 47.435 |
| 最终活跃分配增量，MiB | 1.96924 | 3.93799 |
| 峰值新增活跃分配，MiB | 7.88379 | 10.07129 |

活跃分配是相对常驻模型等对象所在基线的 `torch.cuda.memory_allocated` 增量，不是整卡占用或 CUDA reserved。
最终增量略高于缓存张量理论值，因为仍有单 token 张量及分配粒度影响。
模型驻留时的结束采样整卡显存为 4778MiB，这只是单次采样，不是整次实验峰值。
GPU 利用率开始/结束采样分别为 10%/35%，桌面活动未隔离。

当前分页适配器预分配了额外余量，并产生临时连续张量，耗时与单请求分配都高于连续缓存。
这组观察不能支持加速声明，也不能代表服务端吞吐、TTFT、TPOT 或原生分页 Kernel 的性能。

## 5. 与 Qwen3 实验的关系

| 项目 | Qwen3-0.6B 原实验 | DeepSeek-R1-Distill-Qwen-1.5B |
|---|---:|---:|
| 通过的模型精度 | FP16 | BF16 |
| 层数 | 28 | 28 |
| KV 头数 | 8 | 2 |
| head_dim | 128 | 128 |
| 全层每 token 有效 K/V | 112 KiB | 28 KiB |
| 真实 K/V 读回案例通过 | 15/15 | 15/15 |
| 生成一致性案例通过 | 9/9 | 9/9 |
| 分支案例通过 | 9/9 | 9/9 |

这说明较大的参数量不必对应更大的每 token KV Cache，KV 大小取决于层数、KV 头数、每头维度和元素字节数。
FP16 与 BF16 都是每元素 2 bytes，因此上述有效 K/V 字节核算可直接比较，但两个模型的耗时或回答质量不宜据此横向排名。

通用脚本修改后还完成一次 Qwen E1–E5 回归，全部通过，其结果和对应源码快照另行保留。
原始 Qwen v1 归档没有覆盖。

## 6. 文件与复现

- [DeepSeek 事前方案及修订](DEEPSEEK_PLAN.md)
- [成功结果与失败诊断归档](results/2026-10-23-deepseek-15b-win3070/README.md)
- [模型下载与 SHA256 校验](prepare_small_model.py)
- [通用 E1–E5 脚本](run_small_model.py)
- [分页缓存适配器](model_paged_cache.py)
- [独立精度诊断脚本](diagnose_deepseek_precision.py)

从仓库根目录执行：

```powershell
.\uv.ps1 run --locked python experiments/prepare_small_model.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
.\uv.ps1 run --locked python experiments/run_small_model.py --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B --dtype bfloat16 --plan experiments/DEEPSEEK_PLAN.md --output experiments/runs/deepseek-reproduce/result.json
```

输出文件必须不存在。manifest 固定模型 commit，运行前再次校验模型文件哈希。
依赖、缓存和模型均保留在仓库本地，通过 `uv.ps1` 管理，无需全局安装项目包。

官方资料：[模型页](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)、[模型配置](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/config.json)。实测数字以归档 JSON 为准。
