# kvmem-llama.cpp-0.17.0 × ActKV 审查与 KV 交换改造方案

> 审查对象：`I:\llama\kvmem-llama.cpp-0.17.0`（工作树未提交改动 + `kvmem/` 新文件）
> 论文：ActKV: Efficient LLM Agents through Action-Guided KV Cache Management, arXiv:2609.31395v1
> 说明：本次为**静态审查**，未编译、未运行测试（沙箱内无法在你仓库目录写构建产物，见 §6）。

---

## 0. TL;DR

1. **模块 2（LRFU 动作导向驱逐）已经落地但接入有硬伤**：keep score 的写回与 `set_retrieval_scores` 冲突，导致"同一轮第二次评分"用错信号；轮次标识 `lrfu_round_id()` 也不成立（被丢弃的 preview 也会衰减一轮）。
2. **模块 3（置信度自适应预算）是完整死代码**：`llama_kvmem_observe_confidence` 全仓库零调用；即使接线也**永不生效**——`active == ceiling` 恒成立，乘 1.75 后仍被 clamp 回原值直接 return。
3. **动作区注意力并不存在**：当前 AttnScore 是 `cos(q_sum, mean_k)`（有符号），而论文要求的是 softmax 注意力质量（≥0）且来自 action region。这既造成 KS 可负、也不满足"动作导向"的前提。
4. **本仓库的架构其实比论文更适合做"交换"**：这里的"驱逐"是降级到 CPU/NVMe（可复活），不是销毁。因此 LRFU 的正确定位是**分层/降级优先级 + 预取预测器**，而不是一个 top-k 打分器。
5. **Page-aware compaction（§7）不能照搬**：论文面向 vLLM 式共享块池（回收 = 释放物理块），而 KVMem 的回收单位是 **block slot**——`stage_out` + `free_slot` 已经把整块内存还回池子，块内 token 级空洞不占池内存（`patches/0003-kv-cache-skip-hole-purge.patch` 明确保留并掩码，llama.cpp 的 defrag 在本 pin 已废弃）。可借鉴的只有"压到**尾部**块 + 无冲突槽位规划"这一半，且必须先证伪收益。详见 `docs/actkv-kvmem-port-analysis.md` 的 G 行与本文 §4 D6。

---

## 1. 现状地图（先确认"改了什么"）

### 1.1 四层存储与交换路径

| 层 | 载体 | 代码位置 |
|---|---|---|
| GPU 槽池 | `budget + gen_reserve` 个 cell，一块一槽 | `llama-memory-kvmem.h:29-33, 519-524` |
| host 权威 | `RawKvStore raw_`：per-block mean-K(FP32 和) + 收割时拷贝的 packed GPU 格式 K/V | `raw_kv_store.hpp:3-8, 27-38` |
| pinned CPU arena | `PinnedKvTier`，slot_bytes 一槽 | `pinned_kv_tier.hpp:22-41`；`kvmem_runtime.cpp:20-29` |
| NVMe arena | `NvmeKvTier`（含 raw-K-on-SSD 的 `raw_k_nvme`） | `kvmem_runtime.cpp:30-36`；`nvme_kv_tier.hpp:45-89` |

- **GPU → 下层**：`KvMemRuntime::stage_out` (`kvmem_runtime.cpp:243-292`)——仅 Retrieval 模式才 D2H 拷贝 packed K/V，然后 CPU tier → NVMe → 只丢驻留标记；GPU slot 延后释放（`pending_gpu_frees_`）。
- **下层 → GPU**：`KvMemRuntime::stage_in` (`294-328`)；冷块不是层级拷贝，而是 `write_block_to_gpu` (`llama-memory-kvmem.cpp:4041-4112`) 从 `raw_` 重新物化。
- **顺序**：`spill_outgoing` → `seq_rm_logical` → `admit_incoming`（`llama-memory-kvmem.cpp:1787/1794-1803/1810`）。
- **关键性质**：`stage_in = 不在 working set`，`stage_out = GPU 驻留但未选中`（`kvmem_store.cpp:860-866, 883`）。**被淘汰的块永远不会从 block table / `raw_` 消失**，只是 tier 变 CPU/SSD。只有 `truncate_to` 会弹块并释放层级槽（`kvmem_store.cpp:207-227`）。

### 1.2 一轮选择的完整生命周期

```
prefill 请求线程 (tools/kvmem-multimodal-server.h:296-416)
  ├─ preview_retrieval()            llama-memory-kvmem.cpp:4485-4505
  │    ├─ harvest_flush()
  │    ├─ if (method_ == 1) score_retrieval()        :4492
  │    │     └─ store.set_retrieval_scores(scores)   :4294
  │    │        + store.accumulate_lrfu(scores, lrfu_round_id())  :4303-4305
  │    └─ runtime_->preview_reselect(mandatory) → store.pick_topk_blocks
  ├─ llama_kvmem_apply_selection(selection)          :4582
  │    ├─ runtime_->prepare_selection → KvMemStore::set_selection（diff 出 stage_in/out）
  │    └─ apply_plan_to_kv(plan)：spill_outgoing → seq_rm → admit_incoming
  └─ retrieval_pinned_ = true（选中集被 pin，decode 不得换出）
```

- 重复调用由 `commit_resident` / `commit_unchanged`（`:4524-4540`）做无操作快路径判定，判据是 **epoch 相同且 rows 相同**。
- prefill 压力路径不参与评分：`prepare_working_set` → `maybe_offload_during_prefill` → `pick_prefill_pressure_blocks`（sink + 最新尾块，`kvmem_store.cpp:485-541`）。

### 1.3 每块分数与消费者

| 字段 | 谁写 | 谁读 |
|---|---|---|
| `attn_score` | `accumulate_attn`(H2O)、`set_retrieval_scores`、**`accumulate_lrfu`（覆盖写回）** | `pick_topk_ungrouped:676`、quota leftover `:651`、`constrain_media:467` |
| `profile_score` | `accumulate_attn`（加和） | quota `take_top(profile_quota) :642` |
| `retrieval_score` | `set_retrieval_scores`（每轮覆盖） | quota `take_top(retrieval_quota) :639` |
| `lrfu_score` | `accumulate_lrfu` | 无直接读者（镜像进 `attn_score:302`） |
| `PinnedKvTier::heat_` | 未接线 | 未接线（CPU tier 实际是 plain LRU front） |

> 注意：`accumulate_attn` / `set_attn_scores` 在 adapter 里**没有调用者**，所以"H2O 热度"这条信号在生产路径上其实没喂进去；线上唯一信号就是 retrieval cosine。

### 1.4 预算

- `select_budget = 131072`（默认），`budget_blocks() = select_budget_tokens()/block_tokens`。
- `runtime_select_budget_` 只在 `set_runtime_select_budget` 里写；该校验：≤ 配置上限、必须是 `block_tokens` 正整数倍、`sink+recent ≤ budget_blocks`（`kvmem_store.cpp:32-54`）。
- GPU 物理池 = `budget + gen_reserve`（`llama-memory-kvmem.cpp:721-749`），`selection_fits:4514` / `can_append:4453` 会以 `insufficient_slots` 拒绝超出。

### 1.5 检索 query 与分数类型

- query = 最后一条 `role=user` span（`--kvmem-query-policy-user`），退化时取末尾 512 token（`kvmem-multimodal-server.h:302-306`，`llama-kvmem-server.cpp:336`）。
- 分数 = 每 (layer, kv head) 的 **cosine**：`dot/(|q||k|+1e-6)`（`llama-memory-kvmem.cpp:4274, 4286`），再对 kv head 与 layer 取均值；`--kvmem-score-agg max` 改为组内取 max。
- 即：**不是 softmax 注意力质量，也与"动作区"无关**。

### 1.6 并发与节奏

- httplib + 单推理槽（`st.mu` 锁整个请求，`llama-kvmem-server.cpp:3190`），请求串行。
- 评分/选择在请求线程同步执行，**每轮一次**（不是每 token）；decode 只做 mean-K 累加（`decode_mean_commit:3928-3956`）。

---

## 2. ActKV 逐项对照

| 论文机制 | 本仓库对应 | 状态 |
|---|---|---|
| §5.1 AAV 访问模式分析（冷启/间歇/连续） | 无 | 缺失（只有 LRFU 公式） |
| §5.2 `KS_t = λ·KS_{t-1} + hit?Attn` | `lrfu_score` + `accumulate_lrfu`（λ=0.5, p=0.9） | 已实现，接入有缺陷（§3.3/§3.4） |
| §5.2 top-p hit mask | `nth_element` 取 rank=(1-p)·N 阈值 | 已实现，但基于**有符号 cosine** |
| §5.2 按 KS 取 top-B 驱逐 | `pick_topk_ungrouped` 读 `attn_score`（LRFU 镜像） | 部分正确 |
| §6.1 置信度 `T_i = -(1/k)Σlog P`，min-pool + bottom-b% + 线性拟合 | `KvMemConfidenceMonitor` | 数学已实现 |
| §6.2 `α<0 → B'=min(B·scale_up, B_upper)`，只增不减，每压缩步检查一次 | `observe_confidence` | **未接线 + 永不生效** |
| §7.1 LSE 恢复注意力核 | 无（用 cosine 近似） | 缺失 |
| §7.2 无冲突 in-place 压缩核 | `layout_gpu_slots_by_orig_pos`（复制式重排）+ reselect 留洞 | 部分 / 粒度不同 |
| 三原语抽象（CalAttn / Eviction / Compaction） | 分散在 pick / apply / stage | 未抽象 |

---

## 3. 本地改动的问题清单

### P0-1 模块 3 全链路死代码（critical）
`llama_kvmem_observe_confidence`（`llama-kvmem-hooks.h:75`，定义 `llama-memory-kvmem.cpp:5028`）与 `llama_kvmem_select_budget_tokens`（`:76` / `:5032`）在整个仓库**零调用**；`tools/` 里没有任何 top-k logprob 聚合；`conf_pool/conf_stride/conf_bottom_permille/conf_min_pool/conf_scale_up` 也没有任何 CLI/server 参数（只能走 `set_params` 的 C API）。
**影响**：§6.1/§6.2 的贡献在集成层完全不存在。
**修复**：在生成循环接 `T_i`（`llama-kvmem-server.cpp:3632/3844` 的 `common_sampler_sample` 附近，从 sampler candidates 取 top-k logprob），并在压缩步消费 `select_budget_tokens()`；同时补 `--kvmem-conf-*` 参数。若短期不接，建议先把该模块标记为 experimental 或从默认构建中隔离，避免误判"已支持自适应预算"。

### P0-2 即使接线，预算也永不增长（critical）
`llama-memory-kvmem.cpp:4345-4362`：
```cpp
const uint32_t active  = st.select_budget_tokens();   // runtime ?: cfg_.select_budget
const uint32_t ceiling = st.config().select_budget;   // 同一个值
...
uint64_t target = min(active * 1.75, ceiling);        // == active
if (target <= active) target = active + bt;
if (target > ceiling) target = ceiling;               // 又变回 active
if (target == active) return;                         // 必然走到这里
```
`runtime_select_budget_` 全仓库**唯一**写入者就是 `observe_confidence` 自己，首轮 `active == ceiling` 恒成立。

论文 §8 的实测配置是：**从 512 entries 起步，负斜率时 ×1.75，封顶 `B_upper = 30K` entries**，即"小起步、按需增长"。本代码恰好相反——从 `select_budget = 131072` token（≈1024 块）这个**上限**起步，所以"自适应"在数学上不可能发生。

**修复**：引入真正的 `B_upper`（独立参数，或 `select_budget` 与 `budget+gen_reserve` 分层），并让**运行时预算起始值显著低于上界**（论文 512 vs 30K）；`ceiling` 与 `active` 必须来自不同来源，`active` 的初值由新参数给出而不是取配置上限。

### P0-3 轮内去重时 keep score 被丢弃（high）
`score_retrieval()` 顺序是 `set_retrieval_scores`（把 `attn_score = retrieval_score`，`kvmem_store.cpp:242-248`）→ `accumulate_lrfu`。而 `accumulate_lrfu` 在 round 去重时**提前 return（`:263-267`），跳过了 `:302` 的写回**。于是同一轮的第二次评分会让 `attn_score` 停留在一次性 cosine，而不是 KS，直接违反 `kvmem_store.hpp:470-476` 声明的幂等契约；该轮所有选择（`:650-652`、`:673-680`、`constrain_media:466-475`）都按错误信号排序。
**修复**（任选其一）：
```cpp
if (round_id != 0 && lrfu_round_seen_ && lrfu_last_round_ == round_id) {
    // 只重放写回，不衰减、不累加
    for (auto & b : blocks_) b.attn_score = b.lrfu_score;
    return;
}
```
或让 `set_retrieval_scores` 在 Action 模式下不碰 `attn_score`（更干净，见 §4 D1 的 `ks_score` 独立字段）。

### P0-4 `lrfu_round_id()` 不构成可靠的"轮"（high）
`lrfu_round_id() = (attention_epoch_ << 32) | store_n_tokens()`（`:4308-4314`）。`attention_epoch_` 会因**与评分无关**的事情自增（`:1770` 仅在有 staging 时、`:2441` `seq_rm`、`:1072` `set_replay`、`:1023` `reset_policy`、`:4050` `write_block_to_gpu`）。
- **多算一轮**：`tools/kvmem-multimodal-server.h` 的 cached-query 分支里，`:329` 先 preview（若 `selection_fits` 失败则**被丢弃、从不 apply**），随后 `:365` query prefill 追加 rows，`:377` 再 preview 一次 → rows 变了 → 视为两轮，KS 被**衰减并累加两次**，其中一次对应一个从未执行的压缩决策。
- **丢更新**：当 epoch 与 rows 都没变（正是 `commit_unchanged:4508/4526` 定义的"未变化"态），第二轮分数被静默丢弃。

**修复**：不要在评分函数里推导轮次。改为在**决策提交点**维护显式计数：
```cpp
// llama-memory-kvmem.h
uint64_t lrfu_round_ = 0;
// apply_selection() 里，plan 真正提交后
if (runtime_->store().config().action_lrfu) runtime_->store().commit_lrfu_round(++lrfu_round_);
```
并且 preview 只允许"读 KS"，只有 commit 才允许"改 KS"。

### P0-5 无按轮门控 / 无迟滞：一旦接线就是"每 token ×1.75"（high）
`should_increase_budget()` 是 pooled ring 的**持续属性**，不是一次性事件；`observe_confidence` 每次调用都可能乘 `scale_up`。若按 token 调用，4096 → 7168 → 12544 → 21952 → 36864，**4~6 个 token 就顶到上限**；`:4359` 的 `target = active + bt` 还让 `scale_up ≤ 1` 失去意义、变成每 token +1 块。论文的节奏是"每个压缩步检查一次"。
**修复**：增长只在压缩/reselect 边界执行；加冷却（例如至少间隔 N 轮或直到监控窗口完全换血）；增长后 reset 监控窗口（否则负斜率会持续触发）。

### P0-6 hit mask 建立在有符号 cosine 上，且没有"动作区"（high）
论文 `AttnScore` 是 action region 的 softmax 注意力（≥0）；本仓库是 `cos(q_sum, mean_k)`，可为负：
- KS 可被推到负值且**永不自愈**（论文的 KS 是非负注意力累积，单调不减）；
- λ=1 时 KS 退化为"带符号幅值和"，排序跟踪的是 |cos| 而非命中频率；
- 不在 `raw_` 里的块 score=0，却会被其他块的阈值判为 hit/miss；
- `grep action src/adapter` 只有注释，query 是 user span（`kvmem-multimodal-server.h:304-305`）。

**修复**：改用非负注意力质量（见 §4 D2），并对 KS 做 `max(0, ·)` 下限。

### P0-7 覆写 `attn_score` 波及未声明的消费者（medium）
`constrain_media`（`kvmem_store.cpp:426-479`，比较器 `:466-475`）按 `attn_score` 决定多模态媒体组/强制块的存活与裁剪，进而影响 `query_replay_skipped` 路径（`kvmem-multimodal-server.h:398-408`）。开启 Action 模式后，"哪些图片块在超预算时被保下来"变成 LRFU 历史值的函数，而不是本轮的检索排名。
**修复**：`ks_score` 独立字段，只在 Action 模式的选择器里读它（§4 D1）。

### P0-8 `reset_lrfu()` 无调用者 → KS 跨前缀改写残留（medium）
`truncate_cached` / `truncate_to`（`:1051-1066, 2461`）收缩尾部块时保留其旧 KS；前缀回退重放后，块 id 复用会继承历史热度。另外 `conf_monitor_` 挂在 `llama_memory_kvmem`（`llama-memory-kvmem.h:416`）上，而 store 是**每会话**的（多会话时分时复用），置信度历史会跨会话串味。
**修复**：`clear()/reset_policy()/swap_conv` 时 `reset_lrfu()`；把 monitor 放进每会话对象。

### P0-9 线程契约缺失（medium）
新路径对 `blocks_`、`runtime_select_budget_`、`conf_monitor_` 无锁；adapter 其它地方有 harvest 互斥（`:2811-2911`），导出入口读裸全局指针（`llama-kvmem-capture.cpp:14,21`）。当前服务器串行所以是**潜在**问题，但 API 未写线程约束。
**修复**：文档化"仅 decode/请求线程"，或改成"采样线程只投递，压缩步应用"。

### P0-10 参数/CLI 缺口（medium）
- 所有 `conf_*` 没有 CLI/server 参数；
- `llama-kvmem-cli.cpp:195-199` 用 `atof` 且**不 clamp**（server 用 `kvmem_cli_real(...,0,1)`）→ `--kvmem-lrfu-lambda 5` 会让 KS 指数发散；
- `llama-memory-kvmem.cpp:448-451` 把 `λ ≤ 0` 映射为 0.50，于是 **`--kvmem-lrfu-lambda 0`（论文的纯 recency）无法表达**，而帮助文本却写 0–1；
- `--kvmem-action-lrfu` 与 `--kvmem-method recency` 同用时静默失效（`score_retrieval` 仅在 `method_==1` 调用，`:4492`），无警告。
**修复**：补参数；用哨兵值（<0 / NaN）表示"未设置"；λ 校验范围；在 action_lrfu 且 method≠retrieval 时告警或直接报错。

### P0-11 GQA 重缩放是死代码，真实缺口相反（medium-low）
`g = n_head_ / n_head_kv_` 为**整除**（`:4196`），因此 `h*g+gi ≤ n_head_-1` 恒成立，`qh_i >= n_head_` 的 break 永不触发，`seen == g` 恒成立 → `:4230-4233` 的重缩放永不执行。真正的缺陷是：当 `n_head_ % n_head_kv_ != 0` 时，尾部 `n_head_ - n_head_kv_*g` 个 query head 在 mean 与 max 两条路径里都被**静默丢弃**（Alg. 3 step 8 要求聚合整组）。
**修复**：`g = (n_head_ + n_head_kv_ - 1) / n_head_kv_` 且每组按 `min(g, n_head_ - h*g)` clamp；或断言整除并在不整除时打日志。

### P0-12 静默 catch + "只改目标不改物理池"（medium-low）
`:4369-4373` 吞掉所有 `std::exception`（无日志/计数/记忆），持续性拒绝（例如低于 sink+recent 下限）会被每个 token 重试且不可见。此外抬高 `select_budget_tokens()` 只是抬高目标：物理池是 `budget + gen_reserve`，`selection_fits`/`can_append` 会以 `insufficient_slots` 拒绝，结果要么等下一次 reselect 才生效，要么吃掉你刻意留出的 `gen_reserve`。
**修复**：失败要记日志/计数并记住失败目标；明确"预算增长是否允许借用 gen_reserve"。

### P0-13 置信度监控细节（low）
- `min_window()`（`confidence_monitor.cpp:57-60`）在 ring 未填满时也对整 ring 取 min，前 `w/s` 个 pooled 点单调不增，污染 bottom-b% 拟合；论文只对满窗定义 R′。
- `ready()`（pooled ≥ `min_pool`）不是有效门限：`slope()` 还要求 `lround(b·n) ≥ 2`（`:81-83`），默认 b=0.10/min_pool=8 时实际需要 20 个 pooled 点，`hooks.h:57` 的注释有误导。
- `slope()` 每次 `observe()` 都 `vector<size_t>(n)` 分配 + 最多 512 元素 partial_sort（`:88-95`），而它每 `stride` 个 token 才可能变；这是逐 token 堆流量。缓存拟合结果即可。

### P0-14 测试盲区（low）
- 无任何测试覆盖 `observe_confidence` / `lrfu_round_id` / `score_agg`，所以 P0-1~P0-5、P0-11、P0-12 零覆盖；
- `test_lrfu_round_id_is_idempotent`（`kvmem_store_test.cpp:428-449`）直接调 `accumulate_lrfu`，**没有与 `set_retrieval_scores` 交错**，抓不到 P0-3；
- `test_lrfu_top_frac_zero_clamps` 断言的值在修复前同样是 7.0（旧 bug 是越界读），只有 checked STL/sanitizer 才会炸；
- `test_lrfu_top_frac_threshold` 的注释把 rank 说错（rank 6/10 是第 7 小），断言本身是对的。

### P0-15 命名与文档（low）
- `hooks.h:67-75` 描述的是 `T_i = -(1/k)Σ top-k log P`，导出函数却叫 `llama_kvmem_observe_confidence(float logprob)`，容易被喂成"单个采样 token 的 logprob"；
- `docs/dev-gen-reserve-ring-buffer.md:109-110` 写"ActKV 的置信度预算目前只能把 `select_budget` 在配置上限内降"——与代码相反（代码只增不减，且是 no-op），需要更正。

### P0-16 论文强制保留的"系统提示 + 用户任务描述"，本仓库没有对应保证（medium）
论文 §8 明确：**system prompt 与 user task description 对应的 KV 永远保留，不参与压缩**。本仓库只有 `sink_blocks`（默认 **1 块 = 128 token**）保护前缀，用户任务描述只有在恰好落在 query/mandatory span 时才被强制保留；一旦任务描述很长或不在末尾 user span 内，就可能被 top-k 挤掉。这属于"动作导向压缩"的前置约束（丢了任务描述，action 质量无从谈起）。
**修复**：把"system prompt span + 首个 user 任务描述 span"加入 `retrieval_mandatory()`（`llama-memory-kvmem.cpp:4464-4483`），并按 token 而不是按"1 块"计价；计入预算校验（`sink+recent ≤ budget_blocks`）。

---

## 4. 更好的 KV 交换方式（结合 ActKV 的改造方案）

### 4.1 定位差异：这里要优化的是"交换"，不是"销毁"

论文的 eviction 是**不可逆**的（被驱逐的 KV 消失）；本仓库的"驱逐"其实是**降级 + 可复活**。这带来两个比论文更有利的空间：

1. 论文的 KS 可以**直接当作分层优先级**：KS 高 → GPU 热集；中 → pinned CPU；低 → NVMe；接近 0 才考虑真淘汰（且仅在 `truncate_to` 时）。
2. 论文 §5.1 的核心洞察"action-critical 集合跨轮稳定"可以**直接变成预取预测器**：上一轮 KS 的 top 集合大概率就是本轮要 stage_in 的集合 → 把交换从关键路径移走。

现状最痛的三点是：固定 `select_budget` 决定 GPU 热集大小；reselect 留洞、压缩靠整块重排；交换纯粹被动（先评分再 stage_in）。下面 D1–D7 按"收益/改动成本"排序。

### D1（P0，必修）把 KS 变成独立字段，选择器显式读它
```cpp
// kvmem_store.hpp
double ks_score = 0.0;        // LRFU keep score，与 attn_score 解耦
bool   action_mode = false;   // cfg_.action_lrfu
```
- `pick_topk_ungrouped` / quota leftover / `constrain_media` 里的取分改成
  `auto sel_score = [&](const KvMemBlock & b){ return action_mode ? b.ks_score : b.attn_score; };`
- `set_retrieval_scores` 只写 `retrieval_score`；是否覆盖 `attn_score` 由模式决定。
- 一次性解决 P0-3、P0-7、P0-15，并让 `reset_lrfu` 只清 KS。

### D2（P0/P1）用真正的注意力质量替换 cosine（论文 §7.1）
分两档，先做便宜的：

**档 A（不改 FA）**：`raw_` 已经在 host 侧持有每块 packed K/V 与 mean-K。对"动作区 query"（见下）在 host 上算精确 `q·k`：
```
logit_j = q · k_j / sqrt(d)  (+ 位置/掩码项等价物)
mass_block = Σ_{j∈block} exp(logit_j - LSE)
```
只算出 top-B 候选块的精确质量即可，成本 O(candidate_blocks × 128 × d)。

**档 B（论文 §7.1 原意）**：让 FA 输出每 query row 的 LSE（llama.cpp 的 `ggml_flash_attn_ext` 本来就算 log-sum-exp），用 `exp(s - lse)` 从分页缓存里**恢复**每个 token 的注意力概率，再按块求和。这样无需把分页数据物化成连续内存，正是论文所谓的 recovery-based kernel；代价是要给 ggml 的 FA 加一个可选输出（会触及 `llama.cpp/ggml/src/ggml-cuda/fattn.cu`，记得同步 `patches/`）。

**动作区 query 的定义**（当前完全缺失，必须补）：不要把 last-user span 当 action region。可选口径，按可行性排序：
1. 上一轮 assistant 输出里 **tool-call / action 段**的 token（若有解析）；
2. 上一轮 assistant 输出的**尾部 N token**（最接近下一个 action 的生成位置）；
3. 生成中每 token 的 query 累积（decode 侧 `q_sum_` 已经在维护，`decode_mean_commit:3928-3956`）——即"用正在生成的 action 去检索"，语义上最贴论文，但要处理 pin 后的窗口。
另外把 `hit` 判定与 KS 增量都换成这个非负质量，并对 KS 加 `max(0,·)`（同时修 P0-6）。

### D3（P0）轮次只在提交时推进
见 P0-4 的修复代码。配套规则：
- `preview_*` 只能**读** KS；
- 只有 `apply_selection` 成功提交后才 `+1` 轮并 `accumulate_lrfu`；
- 被丢弃/失败的 preview 不再消耗一轮（修掉 multimodal-server `:329` 的幽灵轮）。

### D4（P1）KS 驱动分层 + 滞回，替换二值 top-k
```cpp
// 每轮提交后，按 KS 排序划两条阈值带
if (ks >= ks_gpu_in)      want_tier = GPU;    // 进热集
else if (ks >= ks_cpu_in) want_tier = CPU;    // pinned host 备选
else                      want_tier = SSD;
```
- 加**滞回**（`ks_gpu_in > ks_gpu_out`）避免同一块反复 in/out（现在 CPU tier 里的 `heat_` 就是为这个设计的，但未接线，见 `pinned_kv_tier.hpp:152-156,185-193,237-258`；建议让 LRFU 直接替代 `HeatAware`）。
- 顺带把 `update_mode = Step` 那条从未被读的分支接上（`kvmem_store.hpp:193,266`），长 tool 轨迹里可以按 step 而非 interval 更新。

### D5（P1）用 KS 稳定性做预取 —— 把"交换"从关键路径移走
论文 §5.1 已证明集合稳定；本仓库已有异步 NVMe 预取骨架（`kvmem_runtime.cpp:143-170` 的 `std::async`）。
```
上一轮 KS top-K 命中集合 S_prev
本轮 prefill/decode 期间：对 S_prev \ 当前驻留 异步 stage_in（NVMe→pinned→GPU）
评分完成后：只补差集（真实 stage_in 数量通常远小于 top-K）
```
收益：把 PCIe/NVMe 延迟与 prefill 重叠，交换次数不变但关键路径变短；命中率本身可以作为一个新的可观测指标（见 §5）。

### D6（P2，需先证伪收益）论文 §7.2 的有限移植：尾部压缩 + 无冲突槽位规划
**先纠正一个容易搞错的前提**：论文的压缩收益来自 vLLM 式共享块池——token 级驱逐留下空洞，空洞所在物理块无法归还调度器，所以"把保留项压到尾部块"能**真正释放物理块**。KVMem 不是这个内存模型：

- 回收单位是 block slot，`stage_out` + `free_slot` 已把**整块**归还池子（`docs/architecture.md:29-32`）；
- 块内 token 级空洞不占池内存，且是刻意保留并被 mask 掉的（`patches/0003-kv-cache-skip-hole-purge.patch`；本 pin 的 llama.cpp 已废弃 `defrag_thold`）；
- 现有 `layout_gpu_slots_by_orig_pos`（`llama-memory-kvmem.cpp:1923-2254`）已经在做整块重排压缩。

所以 §7.2 **不能带来"同样显存装更多块"**（我上一版报告此处判断有误，已按 `docs/actkv-kvmem-port-analysis.md` 的 G 行更正）。仍可能值得借鉴的是它的两个具体机制：
1. **压到尾部而非头部**：论文观察到驱逐后保留项在已分配页的尾部更密，于是把保留项集中到**最后 X 个逻辑块**，用尾部区域内的空洞槽位承接区外保留项——源/目标槽位互斥，因此**一趟拷贝、无需辅助 buffer**；
2. **槽位规划与数据搬移分离**（先算 `(src, dst)` 配对，再做单次并行拷贝）。

在 KVMem 里对应的收益只能是**缩短窗口内空洞跨度**（页表连续性/掩码开销），而不是省显存。落地前必须先量化：统计一次真实 reselect 的"保留 cell / 占用 cell"，空洞比例低于阈值就不做。

### D7（P2，论文没有，但这个仓库需要）交换成本感知的预算分配
论文的 §6.2 只用置信度决定"总数"。在真实交换系统里还应该看**代价**：
```
ΔBudget 的收益 ≈ KS 质量增益  − 交换成本(PCIe/SSD 带宽 × 字节 + 延迟)
```
即：只在"补进这些块带来的注意力质量 > 一次 swap 开销"时扩预算；对 SSD 层块要求更高的 KS 门槛。这条最贴合你问的"更好的 KV 交换方式"，因为它把论文的"驱逐策略"升级成了"带成本模型的分层放置策略"。

---

## 5. 验证计划（建议按顺序）

1. **宿主机单测**（新代码零成本回归，先跑）：
   ```bat
   call env-msvc.bat
   cmake -S . -B build-kvmem-host -G Ninja -DKVMEM_ENABLE_NVME=OFF -DCMAKE_BUILD_TYPE=Release
   ninja -C build-kvmem-host kvmem_store_test confidence_monitor_test
   ctest --test-dir build-kvmem-host -R "kvmem_store_test|confidence_monitor_test" --output-on-failure
   ```
2. **必须新增的用例**（对应上面的缺陷）：
   - 交错 `set_retrieval_scores` 与 `accumulate_lrfu` 两次同轮 → 断言第二轮的选择信号仍是 KS（抓 P0-3）；
   - 模拟"preview 被丢弃 + rows 变化" → 断言 KS 只更新一次（抓 P0-4）；
   - `observe_confidence` 连续调用 N 次 → 断言预算至多按轮变化、不超过 `B_upper`（抓 P0-5/P0-2）；
   - `score_agg=max` 的合成 GQA 测试，含 `n_head_ % n_head_kv_ != 0`（抓 P0-11）。
3. **端到端 A/B**（需要 CUDA 构建）：`--kvmem-action-lrfu` on/off，固定 seed；
   - 质量：`scripts/multi_needle.py`、`scripts/needle_recall.py`、`tests/mtp-kv-test.cpp`；
   - 交换开销：`retr_` 计数（`n_stage_in`、`n_move`、`n_raw`、`laid_out`，`retr_perf_print:4247`）与 `scripts/nvme_spill.py`；
   - 新增指标：每轮 `stage_in/out` 字节与**预取命中率**（D5）、KS 集合跨轮 Jaccard 稳定性（论文 §5.1 的 AAV 可视化）。
4. **已知红测**：`ctest-out2.log`（2026-10-01）显示 14/15 通过，`kvmem-session-cache-lifecycle-test` 失败（`test_symlinks_are_not_followed` 期望 `skipped==2`、实际 1）。与 ActKV 无关，但 CI 是红的，建议先修或标记。该日志**早于**本次改动（且不含 `confidence_monitor_test`），所以新代码尚未被任何自动化验证过。

---

## 6. 风险与注意事项

- **不要直接改 `llama.cpp/` 子模块就完事**：本轮改动没触碰子模块，但 D2-档 B（FA 输出 LSE）会；届时必须同步 `patches/llama-kvmem-current.patch`（`patches/README.md`：累积补丁基于 `v0.5.0 7fe450e`，`scripts/apply-patches.sh` 会检查"已应用的树"）。
- **`--kvmem-method recency` 下 action_lrfu 无效**，需要在 CLI/文档明确依赖关系（P0-10）。
- **`gen_reserve` 是硬限制**：D4 的预算增长若借用 `gen_reserve`，会与 `docs/dev-gen-reserve-ring-buffer.md` 的 ring buffer 计划冲突，建议在 D7 的成本模型里显式区分"语义预算"与"生成长度余量"。
- **本次未做编译/运行验证**：我在工作区（`I:\dsh\html`）建了一个只读引用源码的 host 构建工程，但沙箱内 `cmake` 的编译器探测会挂住，因此 §5.1 的命令需要你在本机执行。
- 报告中的"未接线/零调用"结论基于 `I:\llama\kvmem-llama.cpp-0.17.0` 全树 grep（含 `llama.cpp/` 子模块），可复现：
  `grep -rn "llama_kvmem_observe_confidence\|set_runtime_select_budget\|reset_lrfu"`。
```
