# RTX 3070 上的 Qwen3-0.6B 实验归档

2026-10-23，原生 Windows，模型 commit `c1899de289a04d12100db370d81485cdf75e47ca`。

- [事前方案：目标、步骤与评测标准](../../SMALL_MODEL_PLAN.md)
- [实测结果与解释](../../SMALL_MODEL_RESULTS.md)
- `result.json`：E1–E5 全部 passed，含逐步误差、token、分支、显存、耗时和来源哈希。
- `run.log`：本次实际执行日志。
- `packages.json`：本次已安装依赖版本。
- `experiment-model-manifest.json`：固定模型版本和文件 SHA256，权重保留在仓库本地 `models/`。
- `evidence-manifest.json`：结果、报告、方案及代码的哈希与汇总。

主要结果：真实 K/V 读回 15/15，分页与连续生成对照 9/9，183 次 logits 对照最大差值 0；分支共享/CoW/回收 9/9，逐层逻辑 OOM 28/28 通过。
当前实现以分页方式持久存储 K/V，在 attention 前重建连续张量。它比本次 DynamicCache 基线更慢且预分配更多内存，不构成原生分页 Kernel 或服务性能复现。

复现从仓库根目录执行：

```powershell
.\uv.ps1 run --locked python experiments/run_small_model.py --output experiments/runs/qwen-reproduce/result.json
```
