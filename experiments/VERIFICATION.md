# 本机机制验证：2026-10-23

本轮在 RTX 3070 8GB、原生 Windows 上实际重新执行，结果目录为
`experiments/runs/2026-10-23-075401-verification/`。原始 JSON 与日志留在本地，未覆盖此前归档。

## 实现对应关系

- 上游 `PagedAttention_Tutorial.ipynb`：`KVCacheAllocator` 管理块池，`SequenceState` 维护块表并追加 token，`paged_attention` 按表收集 K/V 后使用 `torch.cat` 拼接，再执行普通 attention。
- 实验扩展 `run_small.py`：`Pool` 管理 K/V 张量、空闲块列表及引用计数。`Seq.append` 用当前长度对块大小取模定位槽位，在块边界领取新块。共享尾块尚未写满时，先分配新块、复制 K/V、减少旧引用并更新块表，再写入。
- `Seq.fork` 复制块表并增加物理块引用，不复制完整前缀张量。`Seq.close` 减少引用计数，归零后归还空闲列表。
- 新增 `verify_mechanism.py`：记录实际状态变化，并在追加、分支和释放后检查块表、引用计数、K/V 内容与独立连续参考。它验证上述扩展，不是新的高性能分页算子。

## 实验及判据

| 实验 | 判据 | 本轮结果 |
|---|---|---|
| 上游 CPU FP32 | 分页 attention 与独立连续 FP64 参考对照 | 54/54，最大绝对误差 2.32965e-7 |
| 上游 CUDA FP32 | 同上，随机物理块分配 | 54/54，最大绝对误差 2.10331e-7 |
| GPU 三精度扩展 | B=4/16/32，边界和较长序列，3 个种子 | FP32/FP16/BF16 各 72/72 |
| 生命周期 | 共享前缀、独立分支、完整回收 | 基础 CPU/CUDA 各 9 例，GPU 三精度扩展共 27 例通过 |
| 逻辑池耗尽 | 共享尾块申请新块失败，原长度、映射和引用保持 | 基础 CPU/CUDA 各 1 例，GPU 三精度扩展共 3 例通过 |
| 新增逐步轨迹 | 每步检查内容及引用，保存块表快照 | CPU/CUDA 各三精度，每种保存 13 个状态 |
| 新增混合操作 | 追加、fork、close 后逐一检查所有活动序列 | 每设备每精度 450 次操作、1504 次非空序列检查，通过 |
| 新增错误注入 | 交换两项块表映射，检查测试是否发现错误 | CPU/CUDA 三精度共 6 组均发现 K/V 及 attention 输出错误，恢复后通过 |
| 显存核算 | FP16 K/V 池 4096×16×128，理论与实测分配增量一致 | 32MiB；释放逻辑块后保持，删除整个池后回到基线 |

多精度测试用量化后的相同 Q/K/V 构造 FP64 参考，按 `abs(error) <= atol + rtol * abs(reference)` 判定。
FP32/FP16/BF16 容差分别为 `2e-6/2e-5`、`1e-3/1e-2`、`1e-2/5e-2`。
GPU 扩展的最大绝对误差分别为 `2.63033e-7`、`1.13823e-3`、`9.91668e-3`。
基础与扩展覆盖存在重叠，不应累加为互不重复的实验数量。

混合操作每个设备每个精度使用种子 0/1/2，各 150 次操作，合计 321 次追加、62 次 fork、67 次 close。
本轮混合轨迹没有触发 OOM，OOM 结论来自单独的确定性池耗尽案例。
每步同时检查：引用计数等于所有活动块表引用次数、空闲块无重复且恰好对应零引用块、每条序列的 K/V 内容与独立记录一致。
这些是有限步数回归实验，不是长期压力或并发安全测试。

## 可用于汇报的真实状态变化

以下为 CUDA FP32 逐步实验中的实际物理块编号。块大小 B=4，块池共 8 块。
物理编号由空闲列表顺序决定，和 HTML 中的示意编号不同，但规则相同。

| 操作 | A 的块表 | B 的块表 | 关键状态 |
|---|---|---|---|
| A 已写入 6 token | [1, 0] | 无 | P1 和 P0 各引用 1 次 |
| fork 为 B | [1, 0] | [1, 0] | 两块引用都变为 2 |
| A 追加第 7 个 token | [1, 5] | [1, 0] | 将共享尾块 P0 复制到 P5，P1 继续共享 |
| B 追加自己的第 7 个 token | [1, 5] | [1, 0] | P0 已私有，A 的数据保持不变 |
| A 增长到 9 token | [1, 5, 2] | [1, 0] | 跨块边界领取 P2 |
| 释放 B | [1, 5, 2] | 无 | P0 回池，P1 仍由 A 使用 |
| 释放 A | 无 | 无 | 8 块全部回池，引用全部归零 |

错误注入实验交换一条长度为 6 的序列的两个块表项。引用次数本身不变，但读取顺序错误，CUDA FP32 attention 最大绝对误差增至约 `0.612694`，测试成功拒绝错误映射。
这说明仅检查块数或引用次数不够，必须同时检查内容和数值结果。

## 复现命令

从仓库根目录执行，所有命令均使用仓库 uv 入口：

```powershell
.\uv.ps1 run --frozen --no-sync python experiments/run_small.py --device cpu --output experiments/runs/reproduce/cpu.json
.\uv.ps1 run --frozen --no-sync python experiments/run_small.py --device cuda --output experiments/runs/reproduce/cuda.json
.\uv.ps1 run --frozen --no-sync python experiments/run_windows.py --output experiments/runs/reproduce/windows.json
.\uv.ps1 run --frozen --no-sync python experiments/verify_mechanism.py --device cpu --output experiments/runs/reproduce/mechanism-cpu.json
.\uv.ps1 run --frozen --no-sync python experiments/verify_mechanism.py --device cuda --output experiments/runs/reproduce/mechanism-cuda.json
```

`--no-sync` 复用本机已经安装并验证的环境。新机器先按 `WINDOWS.md` 使用锁文件同步环境。
`verify_mechanism.py` 的随机混合操作步数可用 `--steps` 指定，默认每种子 150 步。
脚本启用断言，发现错误会以非零状态退出，不会写入 `status: passed`。

## 结果边界及下一步

本轮证明指定输入和状态转换下的分页管理机制正确，测试能够发现所注入的错误。
尚未覆盖空序列 attention、多头/GQA、并发写入、真实模型 token 生成、原生分页 CUDA Kernel 或服务调度。
`run_windows.py` 的 24 组教学调用计时已重新记录，但包含 Python 和拼接开销，桌面 GPU 负载没有隔离，不能用来宣称推理加速。
槽位核算与整卡显存不同，逻辑回收与 CUDA reserved 缓存也不同。

下一步接入真实模型时，可复用独立连续参考、边界案例和生命周期断言。
先对齐真实各层 K/V 的布局与精度并检查输出，再增加原生分页后端和服务指标。
