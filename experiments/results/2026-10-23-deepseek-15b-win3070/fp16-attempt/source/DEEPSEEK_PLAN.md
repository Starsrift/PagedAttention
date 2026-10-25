# 新增 DeepSeek 1.5B 的实验方案

日期：2026-10-23。用户已确认加入 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`。

## 目标与范围

在本机 RTX 3070 8GB 上，将同一套 E1–E5 扩展到第二个模型，检验分页缓存适配器能否适配另一种模型结构及 KV 头数。
保留 Qwen3-0.6B 原始归档，不覆盖已有结果。本次采用独立模型目录、独立运行目录和独立结果报告。

本模型是 DeepSeek-R1 的 Qwen 架构蒸馏模型，不是完整 DeepSeek-R1，也不是 DeepSeek-V3。
根据官方配置，基础架构为 Qwen2，28 层，12 个 Q 头、2 个 KV 头，hidden_size=1536，每头维度为 1536/12=128。
两条推理路径使用同一个模型、FP16 和 eager attention，batch=1。

## 与原方案的差异

- 模型：固定 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 的实际 commit，并校验权重 SHA256。
- 配置兼容：head_dim 若未显式提供，则由 hidden_size / num_attention_heads 推导，不假设和 Qwen3 有相同的 KV 头数。
- Chat template：采用 DeepSeek 自带模板，不传 Qwen3 的 `enable_thinking=False`，不宣称关闭了 DeepSeek 推理行为。
- 模板字符串分词时显式使用 `add_special_tokens=False`，避免重复添加模板已有的 BOS 标记。
- 解码仍使用事前固定的贪心回归条件，最多 32 个新 token。可能只输出部分推理文本，这不影响同模型两种缓存的正确性对照，也不构成推理能力评测。
- 依赖沿用 uv 锁定的 PyTorch 2.10.0+cu126、Transformers 4.57.6，暂不新增第三方包。

## 实验步骤与通过标准

| 编号 | 步骤 | 通过标准 |
|---|---|---|
| E1 | 三个固定提示词，加载并运行普通生成 | logits 有限，无异常，至少生成一个 token |
| E2 | B=4/16/32，长度 1、B−1、B、B+1、127，真实 K/V 分页存取 | 每层 K/V 的形状、dtype、内容完全一致；块数和尾空符合公式 |
| E2 错误对照 | 对 T>B 的配置故意交换两个块表项并恢复 | 测试发现数据错误，恢复后完全一致 |
| E3 | 三个提示词×三种块大小，逐步对照连续与分页缓存生成 | logits 满足 abs(error)≤0.02+0.002×abs(reference)，贪心 token 完全相同 |
| E4 | B−1/B/B+1 前缀，两个分支各追加不同 token | 与独立连续参考对照通过，父缓存和另一分支不变，引用正确，全部回收 |
| E4 池耗尽 | 每层一个已占用块，共享尾块申请新块失败 | 各层长度、块表、引用、空闲列表和 K/V 保持不变 |
| E5 | 64 token 输入、8 次 decode，B=16，预热1次、正式3次，交替顺序 | 保存全部耗时/显存数据，核对真实模型字节公式，不预设性能提升 |

## 评测与失败处理

沿用 [小模型基础方案](SMALL_MODEL_PLAN.md) 的正确性判据、异常保存与性能边界。
E3 如果出现不同生成 token，记录首个分歧及 top-2 logits 间隔，不更改容差来迁就结果。
E4 的 OOM 结论仅针对逐层分配失败，不代表整个 forward 的事务保证。
E5 记录桌面负载，数据只观察教学适配器，不比较两个模型的“能力”，也不把跨模型耗时差归因为分页策略。

```powershell
.\uv.ps1 run --locked python experiments/prepare_small_model.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
.\uv.ps1 run --locked python experiments/run_small_model.py --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B --plan experiments/DEEPSEEK_PLAN.md --output experiments/runs/deepseek-reproduce/result.json
```

结果填写到 `DEEPSEEK_RESULTS.md`，原始记录归档到单独日期目录。

官方依据：[模型页](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)、[配置](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/config.json)。
