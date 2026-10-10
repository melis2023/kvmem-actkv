# GDN 输出融合：256K IQ3 任务 2 压力对照

2026-10-01，`feat/gdn-output-fusion`，最终 128 线程内核。

两组均完整通过任务 2，每组 33 个请求，最终 prompt 为 261546 token，生成 512 token，总计 **262058/262144 token（99.97%）**。本轮开启融合后，聚合有效 prefill 提升 **3.39%**，工具轮次聚合 decode 提升 **2.09%**，最终代码 decode 提升 **3.37%**；总请求耗时减少 **31.06 秒（3.19%）**。

这是按用户要求每种模式完整跑一次的 OFF → ON 对照，没有重复或 ABBA。两组 Windows 工作集行为差异明显，因此结果应视为本轮观测，不能把全部提升严格归因于算子融合，也不足以确认稳定收益。

## 配置与负载

复用 [仓库任务 2](recommended-config-performance.md) 和 `scripts/multimodal_canary.py` 的工具回放负载：基础请求之后分 32 轮输入 `read_file` 工具结果，每轮约 8192 个新 token，最后发送 `final.py` 并生成 Python 代码。文件含固定 batch ID、checksum 和重复增量语句。模型的回答不写入后续请求历史，保证两组收到相同输入。

- RTX 5060 Ti 16 GiB 单卡，物理 GPU 1；Windows，Intel Ultra 7 255H，32 GB RAM，CUDA 13.2、MSVC 14.44。
- 模型 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`；`mmproj-Q8_0.gguf` 在同卡加载，任务不发送图片，图像上限沿用 512。
- context=262144，batch=512，ubatch 保持默认 512；retrieval budget=36864、reserve=16384、block=128。
- 主 KV=q8_0，draft KV=f16，MTP=3、state=replay；query replay=auto、policy=user。
- thinking 预算 128，每请求最多输出 512；temperature=1、top_p=.95、top_k=20、min_p=0、seed=42，penalty 与仓库任务配置一致。
- 同一个服务器二进制，仅切换 `KVMEM_GDN_OUT_FUSION=0/1`。CUDA Graphs 与原有融合保持开启，GDN dispatch trace、详细 KVMem trace 和会增加同步的 `KVMEM_PERF` 关闭。
- 每组启动独立服务器，执行一次完整负载后关闭；不设额外预热轮，不计入模型载入时间。

## 聚合速度

有效 prefill 使用全部 33 个请求的 HTTP/SSE `timings.prompt_ms` 总和，包含历史重算、检索和缓存管理；分子为 **261545 个新增输入位置**。MTP 将最后一个 prompt token 留到 `spec_generate` 处理，故分子比最终 API prompt 长度少 1。没有将重算历史重复计入新增输入量，也没有把逐请求 token/s 做算术平均。

工具轮次 decode 使用 32 个工具请求的总生成量 **3645 token** 除以总 `timings.predicted_ms`，包含思考、MTP 验证和最终代码，排除基础请求。基础请求两组均生成 196 token。未启用额外 trace，因此本表不提供剥离历史重算后的首遍 prefill 或 MTP 接受率。

| 指标 | OFF | ON | 变化 |
| --- | ---: | ---: | ---: |
| 全任务有效 prefill | 307.00 token/s | 317.41 token/s | +3.39% |
| 全任务 prefill 总耗时 | 851.93 s | 823.99 s | 少 27.94 s |
| 工具轮次聚合 decode | 33.80 token/s | 34.50 token/s | +2.09% |
| 工具轮次 decode 总耗时 | 107.85 s | 105.64 s | 少 2.21 s |
| 最终代码 decode，512 token | 29.55 token/s | 30.54 token/s | +3.37% |
| 最终请求 prefill | 29.13 s | 27.73 s | 耗时 −4.82% |
| 最终请求 TTFT | 29.67 s | 28.12 s | 少 1.55 s |
| 全部请求耗时之和 | 974.65 s | 943.59 s | 耗时 −3.19% |

总请求耗时是客户端各次请求耗时之和，包括 HTTP/SSE 与提交缓存等开销，不含载入、客户端在请求间构造或保存下一份请求的时间。

## 随上下文增长的结果

按请求的总 prompt 长度分段，各段仍按总输入/耗时和总生成/耗时聚合。下表只计工具请求，排除基础请求。128K 边界附近的一轮 prompt 为 131069，所以归入 64K–128K 段。

| Prompt 区间 | 请求数/组 | 有效 prefill OFF / ON（token/s） | 变化 | Decode OFF / ON（token/s） | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–64K | 7 | 539.64 / 542.60 | +0.55% | 37.52 / 37.40 | −0.30% |
| 64K–128K | 9 | 272.26 / 279.46 | +2.64% | 35.01 / 35.73 | +2.05% |
| 128K–192K | 8 | 274.99 / 287.60 | +4.58% | 33.30 / 34.32 | +3.05% |
| 192K–256K | 8 | 274.71 / 286.70 | +4.36% | 31.35 / 32.32 | +3.12% |

本轮长上下文阶段的差异大于 [任务 1](gdn-task1-comparison.md)；早期 0–64K 段的差异仍很小。不同阶段也涉及 KV 预算、历史重算和系统工作集变化，不能仅据这些分段推断内核随长度增加的独立收益。

## 压力与资源

66 个请求全部 HTTP 200，SSE 均完整返回 `[DONE]`，无 CUDA/OOM/检索失败日志。每组 32 个工具轮次均保留前缀缓存；最后输出达到 512 token 上限。33 对请求的输入文件哈希、回答、思考文本、finish reason 和 token 用量完全一致。

检查覆盖协议、缓存推进及输出一致性，不验证任务语义质量或被 512 token 上限截断的 Python 代码是否完整可运行。

| 采样指标 | OFF | ON | 差值 |
| --- | ---: | ---: | ---: |
| 整卡峰值显存 | 15688.91 MiB | 15660.01 MiB | −28.90 MiB |
| 最低可用显存 | 363.09 MiB | 391.99 MiB | +28.90 MiB |
| 运行阶段进程峰值 RSS | 14136.39 MiB | 17200.92 MiB | +3064.53 MiB |
| 运行阶段峰值 PrivateUsage（commit） | 29217.31 MiB | 29271.86 MiB | +54.54 MiB |

显存为原生 NVML 整卡采样，目标间隔 50 ms，两组无采样错误。99% 间隔分别小于约 68.7/70.8 ms；最大间隔 136.5/642.5 ms 均发生在载入阶段，不能排除采样未捕捉短暂峰值。RSS/PrivateUsage 用 Windows `GetProcessMemoryInfo` 每 200 ms 采样，表中排除载入。

**RSS 峰值差不能解读成融合额外分配了约 3 GiB。** 基线在 `long-tool-007` 阶段 RSS 一度降至 265.81 MiB，随后逐步增长，最终一轮峰值 9080.11 MiB；开启组同一阶段 RSS 为 15319.34–15929.69 MiB，最终一轮峰值 16599.07 MiB。两组同一阶段的 PrivateUsage 都约 21782–21784 MiB，结束时的 commit 峰值也只差约 55 MiB。这说明驻留工作集行为不同；本次未记录硬缺页计数，无法确定其对时间差的具体贡献。

GPU 运行期平均 SM 时钟为 OFF 2696.18 MHz、ON 2688.55 MHz，平均温度 75.79/76.88°C。开启组没有更高平均 GPU 时钟，但这不能证明其他系统状态一致。单次顺序测试不能提供统计置信区间。本报告保留当时实验构建的记录；测试后按用户要求改为默认开启，仍可用 `KVMEM_GDN_OUT_FUSION=0` 关闭。

## 复现与原始记录

```powershell
$env:PYTHONUTF8 = '1'
C:\Python314\python.exe scripts/test-gdn-task2.py --output artifacts/gdn-task2-new
```

默认模型和投影器位于仓库上一级 `models` 目录，可用 `--server`、`--model`、`--mmproj` 覆盖。`--resume` 重用已完整完成的组；要求原配置和服务器、补丁、canary 哈希一致。完成后重算汇总不会重新运行 GPU 负载。

本次原始记录在 `artifacts/gdn-task2-20261001/`：

- `config.json`、`summary.json`：配置与开关对比；`partial-summary.json` 保存单组完成结果。
- `baseline/`、`fused/`：各组的 `metrics.json`、`rows.json`、全部请求/响应、逐请求 trace、服务器命令与日志、`long-context.json`、NVML/RSS CSV。
- 仓库 `artifacts/gdn-task2-20261001-run.log`：编排执行日志。

引擎沿用任务 1 的同一构建，结束后重新核对服务器、补丁和 canary 哈希均未变化：

- server：`ebd205059b93055733c500cc18f30262d9e41013d6d10db60a785962edd6046a`
- GDN patch：`fc5840e51e5b94def601562509c9fe6bfa6a887980a334d0b63ee5a818c3fd18`
- canary：`c6dd34c0d57a46526992b3bc90807343e0bfe70731125b29300b416b6e287bb7`

本次新增测试编排脚本，不修改引擎或 GGUF。实现与此前正确性验证见 [gdn-output-fusion.md](gdn-output-fusion.md)。
