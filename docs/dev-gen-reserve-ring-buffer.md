# 后续开发:Ring Buffer 突破 `gen_reserve`(让生成超过 reserve 继续输出)

状态:**规划中,尚未实现**。设计已在 `docs/architecture.md`(第 49–123 行,
"Known v1 limit" + "Follow-up: ring buffer inside gen_reserve")定稿。

## 问题

KVMem 的 GPU 池分两个分区:

```
GPU 池 = budget(钉住的检索工作集) + gen_reserve(decode 余量)
```

检索完成后选中的块被 **pin 死**,decode 不许 recency-reselect 把它们换出。
新 decode token 只能占用 `gen_reserve` 分区里的空闲 slot。

当 `gen_reserve` 占满时:

```
alloc_slot() -> -1
  -> "no free GPU slot for block N"
  -> prepare_working_set 里 truncate_to(t0) + return false
  -> llama_decode(gen) failed rc=1
  -> 输出停止
```

所以**一次生成长度(含思考)不能超过 `--kvmem-gen-reserve`**
(IQ3 16384 / IQ4 12288 / CLI 默认 256)。这就是"reserve 限制"。

### 为什么当前不从选中集偷 slot

- 选中 KV 被 pin 住,换出会丢掉 query 检索到的事实。
- 16 GiB 显卡上把 `budget + gen_reserve` 加大不现实(recipe 已贴到 15.5 GiB)。
- 整个生成流式过 VRAM 会重新引入 adaptive-KV-streaming 的成本曲线。

## 方案:在 `gen_reserve` 内部做 ring buffer

不加第三池、不加显存。把 `gen_reserve` 从"硬最大长度"改成像
"本轮输出里 attention 还能看到多少"的滑动窗口。

### 实现步骤(按 architecture.md 的 "First implementation cut")

1. **Pin 时记录 `gen_start_pos`**
   - 检索 pin 工作集时,记下本轮生成的起始 `orig_pos`。
   - 落点:`prepare_working_set`(llama-memory-kvmem.cpp:2236)之前的检索 pin 路径,
     或在 store 里加一个 `gen_start_pos_` 字段。

2. **把 `alloc_slot()` 失败分支从 truncate 改成驱逐最老生成块**
   - 位置:`prepare_working_set` 的 alloc_slot 失败分支(llama-memory-kvmem.cpp:2303),
     当前是 `truncate_to(t0) + return false`。
   - 改为:找**最老的已完成生成块**,条件:
     - `orig_pos_start >= gen_start_pos`(只动本轮生成,不碰历史)
     - 块已满(完整 128 token)
     - `gpu_slot >= 0`
   - **永不**驱逐进行中的块;**永不**驱逐 pin 住的检索块(它们的 slot 不进 free 列表)。

3. **溢出该块 → 释放 → 重试**
   - 该块已被收割进 host store(decode mean-K / 打包副本)。
   - 把它 stage out,`free_slot()`,重试 `alloc_slot()`。
   - 一次只溢出一个块,粒度 = `--kvmem-block-tokens`(16 GiB recipe 上是 128)。
   - 128 decode 步相对一次 q8 块的几 MiB,PCIe 开销可忽略。

4. **溢出后的 GPU 持有**
   - 完整的选中窗口 + 最近 ~`gen_reserve` 个本轮生成 token。
   - 至少保留进行中的块(不要为了吐已完成块而把当前 128 token 也丢出 GPU)。
   - MTP 的 follower 池用同一 slot 索引,在 draft cache 上释放同一 slot。
   - GDN / recurrent 状态每 token 更新,不占这些 attention slot。

## 取舍(已定稿,不再争论)

- 硬崩消失:能超过 `gen_reserve` 继续输出,且不掉检索针(needle)。
- 但同一轮内,模型**只能看到自己最近 ~`gen_reserve` 个 token**。
  更早的自己生成溢出到 host,本轮 Flash Attention 看不到。
  一个 32K 思考 dump 仍然只 attend 自己最后的 12K/16K。
- 溢出的生成 KV 留在 host store;**下一轮检索能选到它**,本轮不能。
- 尾长保持现有 recipe reserve(IQ3 16K / IQ4 12K)——那份 VRAM 已付过钱,
  覆盖默认 4096 思考预算 + 正常回答,不需要转 ring。
  ring 建好后**不要再加 slot**。
- 同回合内看得到更早的自己的思考,是更晚一步的事(用本轮生成做 query,
  从溢出的生成块里检索),不属于本 cut。

## 验证

1. 本地无 CUDA 时只能编译 + 单测(宿主侧 `kvmem` 库)。
   本 cut 是 `prepare_working_set`(decode 最敏感路径)的改动,
   **本地无法真正跑 decode 验证"继续输出"**。
2. 需 CUDA 构建(`build-cuda13.4-dll`,Ninja, Release):
   - 把 `gen_reserve` 设小(如 128/256),跑一段超过它的思考 + 回答。
   - 确认:超过 `gen_reserve` 仍继续输出,`llama_decode(gen)` 不再 rc=1。
   - 确认:检索针没丢(pin 住的块未被换出)。
   - 确认:本轮 attention 只看到最近 `gen_reserve` 个自生成 token
     (溢出块本轮不可见)。
3. 回归:未超 reserve 的短生成不受 ring 影响(零额外成本)。

## 相关代码位置

| 位置 | 说明 |
|---|---|
| `docs/architecture.md:49-123` | 设计定稿(v1 限制 + ring buffer 方案) |
| `llama-memory-kvmem.cpp:2236` | `prepare_working_set(uint32_t n_new_tokens)` |
| `llama-memory-kvmem.cpp:2303` | alloc_slot 失败 → 当前的 truncate + return false(要改的点) |
| `llama-memory-kvmem.cpp:1659-1678` | `alloc_slot()` / `free_slot()` |
| `kvmem_store.hpp` `KvMemStoreConfig` | `select_budget` / `gen_reserve` / `block_tokens` |
| `llama-memory-kvmem.cpp:4376` | `prefill_needs_offload` 检查(prefill 侧,与本 cut 正交) |

## 与 ActKV 改动的关系

本 cut 独立于 ActKV 三模块(LRFU / 置信度预算 / 动作驱逐)。
ActKV 的置信度预算目前只能把 `select_budget` 在配置上限内降,
不能突破它;本 cut 解决的是另一个维度——**decode 自生成长度**的 reserve 限制。
