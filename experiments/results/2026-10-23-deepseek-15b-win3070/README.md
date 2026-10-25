# DeepSeek 1.5B 本机实验归档

日期：2026-10-23。设备：RTX 3070 8GB，原生 Windows。
模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，commit `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`。

- [方案、步骤和精度修订](../../DEEPSEEK_PLAN.md)
- [结果报告](../../DEEPSEEK_RESULTS.md)
- `result.json` / `run.log`：正式 BF16 的 E1–E5，状态 passed。
- `fp16-attempt/`：首次 FP16 的 failed 结果、日志、独立精度诊断及当时的修改文件快照。
- `qwen-regression/`：通用脚本改造后的 Qwen 回归结果，E1–E5 全部 passed，以及当时修改文件的快照。
- `source/`：BF16 正式运行记录的全部源码/方案/配置哈希对应快照及诊断脚本。
- `experiment-model-manifest.json`：模型版本、文件字节数、SHA256 和官方 LFS 对照值。
- `packages.json`：安装依赖版本。
- `evidence-manifest.json`：归档与报告哈希、源码一致性验证结果。

正式结果：真实 K/V 读回 15/15、错误映射检测 6/6、生成一致性 9/9（288 次 logits 最大差值0）、分支9/9、逐层池耗尽28/28通过。
FP16 普通基线第一层 attention 出现非有限输出，BF16 普通基线及完整实验通过。两种精度的结果分开保存。

复现必须显式指定 BF16，输出到一个不存在的新文件：

```powershell
.\uv.ps1 run --locked python experiments/run_small_model.py --model-dir models/DeepSeek-R1-Distill-Qwen-1.5B --dtype bfloat16 --plan experiments/DEEPSEEK_PLAN.md --output experiments/runs/deepseek-reproduce/result.json
```

当前适配器只验证分页存储与生命周期，仍在 attention 前重建连续 K/V。耗时与分配观察不构成生产分页 Kernel 加速或服务性能结论。
