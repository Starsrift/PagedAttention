# 多端实验交接

目标：理解 PagedAttention 的实现机制，完成 RTX 3070 8GB 上可复现的小规模实验。不编辑讲义。

## 当前证据

`README.md` 和 `results/` 是 RTX 3060 Linux 服务器的历史验证，不是 RTX 3070 的结果。
CPU/CUDA 各 54 个正确性案例通过；扩展的 CoW/回收测试在 CPU 运行。
当前仓库使用 torch.cat 重建 K/V，不能用来证明生产级分页 Kernel 的速度。
真实模型、吞吐、TTFT、TPOT 尚未验证。

## Windows PC 下一步

1. 检查 Windows `nvidia-smi`、`wsl --status`、`wsl -l -v`；在 WSL2 中再次确认 GPU 可见。
2. 在 WSL Linux 文件系统中 clone 本仓库，例如 `~/projects/PagedAttention`。
3. 检查空闲显存、驱动和 Python，选择兼容的独立依赖环境。不要迁移服务器旧虚拟环境或其 `.pth` 文件。
4. 先运行 `experiments/run_small.py` 的 CPU/CUDA 正确性测试，新结果输出到 `experiments/runs/`，不要覆盖历史结果。
5. 为真实分页 Kernel 和 vLLM 创建适配的环境并固定版本；从 Qwen2.5-0.5B-Instruct、FP16、最大上下文 2048 开始，显存允许后再扩到 1.5B。
6. 扫描并发 1/2/4/8、输入长度 128/512/1024、输出 64；预热并重复测量。记录代码 commit、模型 revision、依赖、GPU、驱动、种子、参数、失败与显存。

以上配置是待验证起点，不保证所有组合在 8GB 上可运行。系统间性能对比不得直接归因于分页单一因素。

## Git 同步

开始工作先 `git pull --ff-only`；一次尽量只在一台设备上修改同一分支。
代码与经检查的小体积结果通过 commit/push 同步。虚拟环境、权重、缓存、凭据不提交。
`experiments/runs/` 默认忽略；需要归档的结果检查后复制到 `experiments/results/<日期>-<设备>/` 再提交。
当前尚未提供跨平台依赖锁文件；目标机验证后应补充实际可用的版本配置，不能把未验证的配置写成已复现环境。
