# GDN 输出融合：完整任务 1 对照

2026-09-30，`feat/gdn-output-fusion` 分支，最终 128 线程内核。

完整任务耗时中位数由 **30.943 秒降至 30.698 秒**，缩短 **0.79%（0.245 秒）**。长文本 prefill 与代码 decode 的提升都约 1%，属于小收益。正式请求的输入、输出、思考文本和 token 用量完全一致，未观察到峰值显存增加。本报告记录当时默认关闭的实验构建；2026-10-01 后按用户要求改为默认开启，仍可用 `KVMEM_GDN_OUT_FUSION=0` 关闭。

## 测试方法

复用仓库 [任务 1 配置](recommended-config-performance.md) 和 `scripts/multimodal_canary.py` 的请求构造、图像与检查：

1. 12000 个 `apple` 单词组成背景，要求回复 OK；每次重置前缀缓存。
2. 同一对话加入 896×896 三色图形 PNG，识别红色正方形、蓝色圆形、绿色三角形，检查背景缓存复用。
3. 继续对话，根据图片生成带无障碍标签和变色按钮的 HTML/SVG。

全部请求开启 thinking，预算 128 token，最多生成 512 token；temperature=1、top_p=.95、top_k=20、min_p=0、seed=42。第三步按原任务配置在 512 token 上限停止，两组均为 `finish_reason=length`。本次完成的是三步性能工作负载，不验证生成了完整可运行的 HTML。

同一个服务器二进制以 `KVMEM_GDN_OUT_FUSION=0/1` 切换，按 OFF → ON → ON → OFF 顺序启动四个独立服务器。每个服务器先跑一次完整预热任务，再跑两次正式任务；每种模式共四次正式任务，总计 24 个正式请求和 12 个预热请求，全部 HTTP 200。预热不计入耗时统计，正式轮次按原任务设置复用图像嵌入。

使用 RTX 5060 Ti 16 GiB 单卡（物理 GPU 1），Windows，Intel Ultra 7 255H，32 GB 内存，CUDA 13.2、MSVC 14.44。模型为 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`，投影器为 `mmproj-Q8_0.gguf`，均无需修改。

固定参数：context=262144，batch=512，ubatch 使用默认值 512；retrieval budget=36864、reserve=16384、block=128；主 KV=q8_0、MTP KV=f16，MTP=3、state=replay；query replay=auto、policy=user；图像 token 上限 1024，投影器在同卡。CUDA Graphs 和原有融合均开启，GDN dispatch trace 与详细 KVMem trace 关闭。

## 速度结果

以下为每种模式四个正式样本的中位数。Prefill 来自 HTTP `timings.prompt_ms`，decode 来自 `timings.predicted_per_second`；decode token 数包含思考 token。速度变化按吞吐比计算。

| 阶段 | 新 prefill token | 生成 token | Prefill OFF / ON（ms） | Prefill 速度变化 | Decode OFF / ON（token/s） | Decode 速度变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 长文本背景 | 12060 | 45 | 15208.72 / 15077.06 | +0.87% | 37.641 / 38.066 | +1.13% |
| 图片识别 | 813 | 80 | 1523.74 / 1521.27 | +0.16% | 48.595 / 49.201 | +1.25% |
| HTML/SVG 生成 | 47 | 512 | 374.54 / 382.93 | −2.19% | 48.967 / 49.426 | +0.94% |

图片、代码阶段的总 prompt token 分别为 12918、13045；缓存命中分别为 12105、12998。代码阶段只有 47 个新 token，prefill 中位数增加约 8.4 ms，样本区间重叠，不能据此推断稳定回退。

整任务耗时先对同一次任务的三个客户端请求耗时求和，再取中位数，包含 HTTP、图像和服务器开销，不含模型载入及预热，也不包含客户端在请求之间构造数据的时间。

| 指标 | OFF | ON | 变化 |
| --- | ---: | ---: | ---: |
| 整任务请求耗时中位数 | 30.9429 s | 30.6980 s | −0.79% |
| 四次整任务请求耗时范围 | 30.7332–31.1160 s | 30.5779–30.7231 s | — |
| 代码 decode 速度范围 | 48.513–49.424 token/s | 49.345–49.493 token/s | — |

长文本 prefill 四个 OFF 样本为 15191.67–15234.81 ms，ON 为 15047.65–15101.72 ms，本轮均略快。Decode 等指标有重叠，且只测了本机、单模型、四次任务，约 1% 的差异不足以证明普遍收益。这里比较同机同二进制开关，不与仓库此前其他平台的绝对速度作比较。

## 显存与系统内存

显存通过原生 NVML 每 50 ms 采样，四组均无采样错误，最大实际间隔小于 68 ms。Windows 进程 RSS 通过 `GetProcessMemoryInfo` 每 200 ms 采样。下表排除载入和预热，取各模式全部正式阶段的采样最大值。

| 指标 | OFF | ON | 差值 |
| --- | ---: | ---: | ---: |
| 整卡峰值显存 | 15596.91 MiB | 15596.91 MiB | 0 MiB |
| 进程峰值 RSS | 13970.01 MiB | 13979.52 MiB | +9.52 MiB |
| 进程峰值 PrivateUsage（commit） | 19881.19 MiB | 19885.56 MiB | +4.37 MiB |

单个服务器的正式阶段 RSS 峰值在 13646–13980 MiB 之间波动，包含模型映射和 Windows 工作集变化；不能把不同服务器之间的 RSS 差直接归因于融合。PrivateUsage 是提交内存，与实际驻留 RSS 不同。显存是整卡采样值，不能捕捉小于采样间隔的瞬时变化。本轮没有观察到可辨认的显存增加。

## 复现与记录

本次只增加测试编排和 Windows 原生显存/RSS 采样，未再修改引擎。服务器和补丁 SHA256 在实验结束后重新核对一致：

- server：`ebd205059b93055733c500cc18f30262d9e41013d6d10db60a785962edd6046a`
- GDN patch：`fc5840e51e5b94def601562509c9fe6bfa6a887980a334d0b63ee5a818c3fd18`
- canary：`c6dd34c0d57a46526992b3bc90807343e0bfe70731125b29300b416b6e287bb7`

```powershell
$env:PYTHONUTF8 = '1'
C:\Python314\python.exe scripts/test-gdn-task1.py --output artifacts/gdn-task1-new
```

默认模型和投影器位于仓库上一级的 `models` 目录；可用 `--server`、`--model`、`--mmproj` 指定。`--resume` 可重用已完成的组，要求原配置、服务器、补丁和 canary 哈希一致。

原始数据位于 `artifacts/gdn-task1/`：`config.json`、`rows.json`、`groups.json`、`summary.json`；四个 `round-*-*` 子目录各有服务器命令、请求/响应、日志、`vram.csv`、`rss.csv` 和图像。完整融合实现与此前正确性测试见 [gdn-output-fusion.md](gdn-output-fusion.md)。
