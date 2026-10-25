# 小模型与分页 KV Cache 实验方案

制定日期：2026-10-23。设备：本机 RTX 3070 8GB，原生 Windows。
执行后将实际结果写入 `SMALL_MODEL_RESULTS.md`，原始 JSON、日志及模型版本记录保存在独立运行目录。

## 目标和范围

将已完成的随机张量机制验证推进到真实模型：加载 Qwen/Qwen3-0.6B，使用模型各层实际产生的 K/V，验证分页存取和分页缓存参与推理时的正确性。
本轮实现 Python/PyTorch 教学缓存适配器，attention 之前仍重建连续 K/V。暂不实现原生分页 CUDA Kernel，不把本轮结果解释为 vLLM 服务性能或论文性能复现。

## 固定条件

- 模型：`Qwen/Qwen3-0.6B`。下载前解析并记录具体 commit SHA，以该不可变版本下载和运行。
- 依赖：使用仓库 `uv.ps1`、`pyproject.toml`、`uv.lock`，固定兼容版本的 Transformers，保持现有 PyTorch 与 NumPy版本。
- 本地隔离：Python 位于 `.local/python/`，依赖位于 `.venv/`，uv 缓存位于 `.local/uv-cache/`，模型与下载缓存存放仓库内忽略目录。
- 推理：CUDA FP16、`eval()`、`inference_mode()`、batch=1、关闭 TF32、固定种子、使用相同 eager attention 后端。
- 解码：关闭 thinking，贪心 argmax 用于数值回归，固定最多 32 个新 token。贪心仅为受控验证条件，不评价模型回答质量。
- 块大小：B=4/16/32。主实验只覆盖短上下文，逐级运行以避免影响桌面使用。

## 实验顺序与通过标准

### E1：加载和普通生成

步骤：加载 tokenizer 和 FP16 模型，记录参数量、模型结构、设备、软件版本和显存；用三个固定提示词生成短回答。

通过标准：CUDA 上成功执行 prefill/decode，无 NaN/Inf、无异常，实际生成至少一个新 token。记录输出 token IDs 和文本，不使用主观回答质量作为通过条件。

### E2：真实多层 K/V 分页写入与读回

步骤：使用真实 token 输入取得模型连续缓存，保留 28 层、多 KV 头和 head_dim 的实际布局。对 B=4/16/32，选取去重后的长度 1、B−1、B、B+1、127，随机化物理块分配顺序，将真实 K/V 写入物理块池并按块表重建。

通过标准：每层 K、V 的形状、dtype、内容逐元素完全一致（`torch.equal`）；块数等于 `ceil(T/B)`，每条独立序列尾部空槽不超过 B−1。

错误对照：对长度大于 B 的样本故意交换两个块表项，必须检测到读回数据错误，恢复映射后必须再次完全一致。

### E3：分页缓存参与逐步推理

步骤：通过 Transformers Cache 接口实现真实分页缓存适配器，覆盖 prefill 批量写入和 decode 单 token 追加。对每个提示词和块大小，比较 DynamicCache 与分页缓存的 prefill logits 和每步 decode logits，独立执行贪心生成。

通过标准（执行前固定）：

1. 每步 logits 满足 `abs(error) <= 0.02 + 0.002 * abs(reference)`，不允许 NaN/Inf。
2. 贪心生成 token IDs 完全相同。
3. logits 对照以同一 token 历史为前提。如独立生成发生分歧，记录首个分歧、最大误差和 top-2 间隔，判该案例未通过，不将后续不同历史下的 logits 当成同条件比较。
4. 适配器实际报告正确的已缓存长度，不存在持久连续 K/V 影子缓存。

### E4：真实模型前缀共享、CoW 与回收

步骤：从真实 prompt 的分页缓存 fork 出两个分支，分别输入不同的后续 token。检查父序列与未写分支的 K/V 保持不变，再将每个分支的 logits 与各自独立连续缓存参考对照。覆盖 B−1、B、B+1 三类前缀长度。

通过标准：共享时物理块引用计数正确；共享未满尾块写入时先复制；分支结果符合 E3 容差；全部关闭后所有块引用归零并可重新分配。
另外设置只能容纳现有前缀的逻辑池，强制共享尾块申请失败，检查现有长度、块表、引用计数及 K/V 内容保持不变。此检查仅覆盖分配失败，不代表模型 forward 的跨层事务保证。

### E5：空间核算和受控耗时观察

步骤：计算真实模型每 token K/V 字节数，将有效 token、分页占用、池容量及尾部空槽分别记录。独立测量 DynamicCache 与分页适配器的活跃张量分配/峰值及 prefill/decode 墙钟时间，GPU 同步后计时。每种方法预热 1 次，正式重复 3 次，交替运行顺序。

评测标准：空间公式与实际分配对象匹配；所有计时保留原始样本、中位数和桌面负载。记录重建连续 K/V 的临时内存，不预设分页适配器更快或占用更少。
本轮计时只观察该教学适配器，不作为原生分页 Kernel 或服务端吞吐、TTFT/TPOT 的性能结论。

## 结果记录要求

每项记录：实验编号、输入长度、块大小、模型 SHA、依赖/脚本 SHA256、判据、观测值、通过/未通过、异常详情与原始文件路径。
主 JSON 即使失败也写入失败状态和异常信息，只有全部必要检查通过才标记总体 passed。
区分“已通过”“未通过”和“尚未执行”，不因下载失败或环境错误编造实验结果。

## 执行入口

在仓库根目录运行，具体脚本提供 `--help`：

```powershell
.\uv.ps1 run --locked python experiments/prepare_small_model.py
.\uv.ps1 run --locked python experiments/run_small_model.py --output experiments/runs/small-model/result.json
```

结果报告包含上述命令最终使用的确切参数、模型版本和复现目录。

## 官方依据

- [Qwen3-0.6B 模型说明](https://huggingface.co/Qwen/Qwen3-0.6B)
- [模型配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Transformers Cache 接口与逐步生成](https://huggingface.co/docs/transformers/main/en/cache_explanation)

API 以实际安装且锁定版本的源码为准。后续发生方案调整时，先记录原因和修订，再执行调整后的配置。
