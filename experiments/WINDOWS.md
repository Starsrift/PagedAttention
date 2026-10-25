# Windows RTX 3070 本地实验

使用仓库根目录的 `uv.ps1` 作为 uv 入口。它只为当前命令设置环境变量，退出时恢复，不修改系统 PATH 或注册表。

| 内容 | 项目内位置 |
|---|---|
| uv 管理的 Python 3.13.16 | `.local/python/` |
| 第三方依赖与虚拟环境 | `.venv/` |
| uv 缓存 | `.local/uv-cache/` |
| 临时文件 | `.local/tmp/` |
| Python 版本 | `.python-version` |
| 依赖声明与锁定 | `pyproject.toml`、`uv.lock` |
| 原始实验结果 | `experiments/runs/`，默认不提交 |

当前锁文件针对 Windows x64、Python 3.13，使用 PyTorch 2.10.0+cu126 与 NumPy 2.2.6。PyTorch 的 CUDA 运行时版本与 `nvidia-smi` 显示的驱动支持版本不是同一概念。

2026-10-23 已通过同一 uv 环境增加 Transformers 4.57.6，完成 Qwen3-0.6B 原生 Windows 模型实验。
事前目标、步骤和标准见 [小模型实验方案](SMALL_MODEL_PLAN.md)，实测结果见 [小模型实验报告](SMALL_MODEL_RESULTS.md)。
Hugging Face 下载缓存由 `uv.ps1` 限定在 `.local/huggingface/`，模型位于 `models/Qwen3-0.6B/`。

同日新增 `DeepSeek-R1-Distill-Qwen-1.5B`：普通 FP16 基线出现非有限 logits，经诊断和事前方案修订后，BF16 的 E1–E5 全部通过。
详见 [DeepSeek 方案](DEEPSEEK_PLAN.md) 和 [实测结果](DEEPSEEK_RESULTS.md)。运行该模型必须显式指定 `--dtype bfloat16`，模型目录与 Qwen 分开。

## 安装与同步

在仓库根目录的 PowerShell 中执行：

```powershell
.\uv.ps1 python install 3.13.16 --no-bin --no-registry
.\uv.ps1 sync --locked
```

不需要激活环境，也不要向系统 Python 安装实验依赖。新增依赖使用 `.\uv.ps1 add 包名`；更新依赖后保留 `pyproject.toml` 和 `uv.lock`。运行时、虚拟环境、安装包、缓存与权重不提交。

## 运行实验

```powershell
.\uv.ps1 run --locked python experiments/run_small.py --device cpu --output experiments/runs/local/cpu.json
.\uv.ps1 run --locked python experiments/run_small.py --device cuda --output experiments/runs/local/cuda.json
.\uv.ps1 run --locked python experiments/run_windows.py --output experiments/runs/local/windows.json
```

第三条命令覆盖 FP32/FP16/BF16 寻址正确性、GPU 上的共享分支与回收、逻辑池耗尽、32MiB KV 池分配，以及教学代码与预先连续存储 K/V 的调用耗时比较。默认预热 5 次，每配置重复 5 轮，每轮 10 次调用。

`run_small.py --device cuda` 的生命周期扩展现在也在 GPU 上运行；旧的 RTX 3060 归档来自修改前脚本，其扩展仍是 CPU 测试。

## 结果边界

当前 Notebook 按块查表后使用 `torch.cat` 重建连续 K/V。它能验证分页机制，不能代表生产分页 CUDA Kernel。计时包括 Python 调度及每组调用末尾同步，连续基线已持有连续 K/V，未包含打包或扩容成本；桌面 GPU 活动也会影响计时。

逻辑 OOM 是预分配块池耗尽，不是故意耗尽整张 GPU 显存。块释放归还池，不会自动将整个池的张量交还 CUDA；删除张量后 PyTorch 还可能保留缓存。

本机没有 WSL。真实模型 vLLM serving、吞吐、TTFT 和 TPOT 属于后续 Linux/WSL 实验，不能从这里的教学计时推断。
