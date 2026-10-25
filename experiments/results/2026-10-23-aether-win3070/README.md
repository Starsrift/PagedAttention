# RTX 3070 Windows 本机实验

日期：2026-10-23，Asia/Shanghai。结果来自本机实际运行，独立于旧的 RTX 3060 Linux 归档。

本机已通过教学实现的 CPU/CUDA 数值验证、三种精度的 GPU 分支共享/CoW/回收与逻辑 OOM 检查。真实分页 Kernel、真实模型 serving、吞吐、TTFT 与 TPOT 尚未测试。

## 环境与本地隔离

| 项目 | 实测配置 |
|---|---|
| 系统 | Windows 11，10.0.26300，原生 Windows |
| CPU / 内存 | Ryzen 5 5600，6 核 12 线程；可见物理内存 15.94GiB |
| GPU | GeForce RTX 3070，8192MiB，计算能力 8.6，WDDM |
| 驱动 | 610.88 |
| Python / uv | 3.13.16 / 0.12.22 |
| PyTorch / NumPy | 2.10.0+cu126 / 2.2.6 |
| PyTorch CUDA 运行时 | 12.6，与驱动报告的 CUDA UMD 13.3 区分 |
| WSL | 未安装；`wsl --status` 返回 50 |
| CPU 线程 / TF32 | PyTorch 1 线程；补充实验关闭 TF32，FP32 matmul precision 为 highest |

Python 运行时由 uv 管理，位于 `.local/python/`；依赖位于 `.venv/`；缓存位于 `.local/uv-cache/`；临时文件位于 `.local/tmp/`。`uv.ps1` 设置这些位置并在退出时恢复调用者的环境变量。没有向全局 Python 安装项目依赖，也没有注册项目 Python 的 PATH 或 Windows 注册表项。

版本由 `.python-version`、`pyproject.toml`、`uv.lock` 固定；实际安装清单见 `packages.txt`。已验证离线 `uv sync --locked --dry-run` 不需要更改环境，`uv lock --check --offline` 和 `uv pip check` 通过。

PyTorch 大型安装包下载使用断点分段续传，安装前与官方索引的 SHA256 核对通过：`91e21e7ad572bf0136e5b7f192714f120c8abde8e128f1a0759f158951643822`。安装包、下载分段、缓存与虚拟环境均不进入 Git。

## 正确性与生命周期

| 实验 | 案例数 | 结果 / 最大绝对误差 |
|---|---:|---|
| Notebook FP32，CPU | 54 | 全部通过；2.32965e-7 |
| Notebook FP32，CUDA | 54 | 全部通过；2.10331e-7 |
| 补充 GPU FP32 | 72 | 全部通过；2.63033e-7 |
| 补充 GPU FP16 | 72 | 全部通过；1.13823e-3 |
| 补充 GPU BF16 | 72 | 全部通过；9.91668e-3 |
| 共享前缀、尾块 CoW、分支独立、全部块回收 | CPU 9；CUDA 基础 9；CUDA 三精度扩展 27 | 全部通过 |
| 共享尾块写入时逻辑池耗尽，状态保持 | CPU 基础 1；CUDA 基础 1；CUDA 三精度扩展 3 | 全部通过 |

基础数值实验覆盖 B=4/16/32、6 种长度、3 个种子。扩展实验覆盖每种精度的 B=4/16/32、长度 1/B-1/B/B+1/127/257/1024/2048、种子 0/1/2；单头维度为 64。

扩展实验先在 CPU 生成并转为目标精度，再将这些确切的量化输入转换为 CPU FP64，计算独立连续 attention 参考。因此误差反映目标精度计算相对该参考的差异，不包含输入从 FP32 量化前后的误差。比较使用 `abs(actual-reference) <= atol + rtol*abs(reference)`：FP32 为 2e-6/2e-5，FP16 为 1e-3/1e-2，BF16 为 1e-2/5e-2（atol/rtol）。

基础与扩展类别存在覆盖重叠，不代表这些案例全部互不重复。逻辑 OOM 是预分配池耗尽，未故意耗尽整张 GPU 显存。尚未覆盖空序列、多头/GQA、并发写入或长期压力测试。

## 显存与空间核算

在 CUDA 上分配 4096 个块、每块 16 token、维度 128 的 FP16 K/V 池。按 `2 × 4096 × 16 × 128 × 2` 计算，池应占 33554432 bytes，即 32MiB；`torch.cuda.memory_allocated()` 实测增量恰为 32MiB。

归还逻辑块后，张量分配保持不变；删除整个池后，活跃张量分配回到基线。这说明逻辑块释放是归还预分配池，不能把它误读为整池显存立即归还驱动。PyTorch 的 reserved 缓存与整卡 VRAM 占用不等同于这里的 live allocated 指标。

固定 64 请求的合成轨迹也已通过尾部浪费界限检查。B=16 的逻辑槽位利用率为 95.47%，每请求静态预留 512 slots 的基线为 30.91%；这不是 CUDA allocator、真实模型容量或吞吐测量。

## 教学实现调用耗时

共测 24 个配置：FP32/FP16 × B=4/16/32 × 长度 128/512/1024/2048。每种方法预热 5 次、重复 5 轮、每轮 10 次调用，轮末同步 GPU；下表为各轮每调用平均墙钟耗时的中位数，单位 ms。batch=1，单头，head_dim=64。

| FP16、B=16，长度 | Notebook 查表 + torch.cat + attention | 已持有连续 K/V 的 attention |
|---:|---:|---:|
| 128 | 0.16831 | 0.08767 |
| 512 | 0.28222 | 0.09211 |
| 1024 | 0.44390 | 0.09332 |
| 2048 | 0.74848 | 0.09865 |

这组数据体现当前 Python 查表与拼接路径的额外开销。连续基线不包含打包、增长复制与外部碎片成本，不能据此评判生产分页算法的优劣。测量包括 CPU 调度与同步组内的调用，不能作为纯 Kernel 时延或服务端 TPOT。

桌面 GPU 活动没有隔离；补充实验开始时 nvidia-smi 显示 GPU 利用率 23%。原始文件保留全部 5 轮样本，计时只作为本机教学实现的粗略观测，不作稳定性能或论文复现结论。

## 复现与归档

在仓库根目录的 PowerShell 执行：

```powershell
.\uv.ps1 python install 3.13.16 --no-bin --no-registry
.\uv.ps1 sync --locked
.\uv.ps1 run --locked python experiments/run_small.py --device cpu --output experiments/runs/local/cpu.json
.\uv.ps1 run --locked python experiments/run_small.py --device cuda --output experiments/runs/local/cuda.json
.\uv.ps1 run --locked python experiments/run_windows.py --output experiments/runs/local/windows.json
```

本目录的 `cpu.json`、`cuda.json`、`windows.json` 与对应日志是本次归档。补充实验在修复 Windows 命令输出编码处理后重新运行并通过；这里保留最终脚本对应的数据。首次导入大型 PyTorch 动态库耗时较长，随后已验证正常导入与 CUDA 可用。

本次基线提交为 `191e3f510c67498d933c99ee8d7b5e97b7eae206`，工作分支为 `aether-win3070/local-experiments`，本次脚本与环境配置修改尚未提交。`windows.json` 记录代码、Notebook、依赖配置与锁文件的 SHA256，供核对具体实验版本。

环境说明见 `../../WINDOWS.md`。后续真实分页 Kernel 与 vLLM serving 需要准备 Linux/WSL 环境；官方说明见 [vLLM GPU 安装要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。本轮证据支持教学机制与上述数值案例在本机运行正确。
