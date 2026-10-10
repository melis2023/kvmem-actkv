# 开发文档:ActKV 动作导向 KV 交换改造

状态:**设计定稿,待实现**。P0(P0-1~P0-4)为必修,其余按序推进。
关联文档:
- `docs/kvmem-actkv-review-2026-10-10.md` —— 本地改动审查 + 问题清单(P0-1~P0-16)
- `docs/dev-gen-reserve-ring-buffer.md` —— decode 自生成长度的 ring buffer cut,**与本文档正交**
- `docs/actkv-kvmem-port-analysis.md` —— ActKV→KVMem 可移植性矩阵(A–H),含 §7 的否决结论与本项目口径
- `docs/architecture.md:49-123` —— v1 限制与 ring buffer 设计

论文:ActKV: Efficient LLM Agents through Action-Guided KV Cache Management(arXiv:2609.31395v1)

---

## 1. 目标与非目标

### 1.1 目标

| 编号 | 目标 | 解决的问题 |
|---|---|---|
| G1 | keep score(KS)独立成字段,选择器显式读它 | 修复"同轮第二次评分用错信号"与多模态媒体裁剪被污染(P0-3 / P0-7) |
| G2 | 轮次只在**压缩决策提交**时推进 | 幽灵轮/丢更新(P0-4);preview 不再有副作用 |
| G3 | 用**非负动作区注意力质量**替换 cosine | KS 可为负、与论文语义不符(P0-6);补上 §5.1 的动作区前提 |
| G4 | KS 驱动 GPU / pinned CPU / NVMe 分层 + 滞回 | 从"二值 top-k"升级为"分层放置";接线已存在但未使用的 CPU tier heat 策略 |
| G5 | KS 的跨轮稳定性驱动**异步预取** | 把 PCIe/NVMe 延迟移出关键路径(论文 §5.1 洞察的系统化) |
| G6 | 窗口内空洞整理(论文 §7.2 的**有限**移植) | 先证伪收益再动:`stage_out`+`free_slot` 已归还整块 slot,块内空洞不占池内存。仅"压缩到尾部块 + 无冲突槽位规划"可能对 `layout_gpu_slots_by_orig_pos` 有用;优先级最低 |
| G7 | 真正的自适应预算:`B_upper` + 每轮门控 + 冷却 + 可观测 | 模块 3 当前是死代码且恒为 no-op(P0-1 / P0-2 / P0-5) |

### 1.2 非目标

- 不改 Flash Attention 内核(**P0/P1 阶段**;G3 的"论文原意档"需要改 `ggml-cuda/fattn.cu`,单独立项)。
- 不改 ring buffer cut(`docs/dev-gen-reserve-ring-buffer.md`),两者共享 `gen_reserve` 但语义不同。
- 不新增第三池、不突破 recipe 的 VRAM 预算(`docs/recommended-config-performance.md`)。
- 不实现 `KvMemRetrievalMethod::PerToken/SubBlockMeanK/DeltaNet`(当前仅枚举,无实现)。

---

## 2. 设计

### 2.1 数据模型(G1)

```cpp
// kvmem/include/kvmem/kvmem_store.hpp  struct KvMemBlock
double   ks_score = 0.0;    // LRFU keep score(仅 action 模式有意义)
uint64_t ks_round = 0;      // 该 KS 最后更新的轮号(调试/断言用)

// KvMemStoreConfig
bool  action_lrfu   = false;
float lrfu_lambda   = 0.50f;   // λ
float lrfu_top_frac = 0.90f;   // top-p
float ks_hit_floor  = 0.0f;    // 质量下限(非负注意力后通常为 0)
```

统一取分入口,避免再出现"某个比较器忘了换信号":

```cpp
// KvMemStore 私有
inline double KvMemStore::sel_score(const KvMemBlock & b) const {
    return cfg_.action_lrfu ? b.ks_score : b.attn_score;
}
```

改造点(全部改为 `sel_score`):
- `pick_topk_ungrouped` 的 leftover `take_top` 与主比较器(`kvmem_store.cpp:650-652, 673-680`)
- `constrain_media` 比较器(`kvmem_store.cpp:466-475`)
- quota 选择器:检索/画像配额继续用 `retrieval_score`/`profile_score`,**补充**一个 KS 配额(可选)
- `set_retrieval_scores` 在 `action_lrfu` 下**不再**覆写 `attn_score`

### 2.2 轮次只在提交时推进(G2)

```cpp
// llama-memory-kvmem.h
uint64_t lrfu_round_ = 0;             // 单调计数,只在提交时 ++

// KvMemStore:preview 只读,commit 才写
void KvMemStore::commit_lrfu_round(const std::vector<double> & scores,
                                   uint64_t round_id);   // = 旧的 accumulate_lrfu,去掉内置去重
```

调用序列(唯一合法路径):

```
preview_retrieval():  score_retrieval() 只算 scores,写入 store 的 pending_scores_     [只读 KS]
apply_selection():    ... apply_plan_to_kv(plan) 成功提交后
                      store.commit_lrfu_round(pending_scores_, ++lrfu_round_);        [写 KS]
```

规则:
- **被丢弃的 preview 不消耗轮次**(修掉 `tools/kvmem-multimodal-server.h:329` 幽灵轮)。
- `attention_epoch_` / `store_n_tokens()` 与轮次解耦。
- 删除 `lrfu_round_id()`(`llama-memory-kvmem.cpp:4308-4314`)。
- 若同一轮确需重算 scores(例如 query span 变化),用**显式**的 `lrfu_round_++` 表示决策变化,而不是让 rows 变化隐式触发。

### 2.3 动作区注意力(G3)

**阶段 A(先做,不改 FA)**:host 侧精确打分,只对候选块。
```
对动作区 query 集合 Q_a 与块 b 的 packed K(已在 raw_ 中):
  logit(q,k) = q·k / sqrt(d)
  mass(b)    = Σ_{q∈Q_a} Σ_{k∈b} exp(logit - LSE_q)     // LSE_q 用该 query 的完整 key 集合估计
```
复杂度 O(|Q_a| × candidate_blocks × 128 × d),候选块取上一轮 KS top-(B·2) 即可。
KS 增量与 hit 判定都改用 `mass`,并对 KS 加 `max(0, ·)`。

**阶段 B(论文 §7.1 原意,单独立项)**:FA 已产 log-sum-exp,用 `exp(s - lse)` 从分页缓存恢复注意力概率,按块求和,无需物化分页数据。需要给 ggml 的 `ggml_flash_attn_ext` 加可选 LSE 输出,并**同步 `patches/llama-kvmem-current.patch`**。

**动作区 query 的口径**(按可行性排序,先用 2):
1. 上一轮 assistant 输出中 tool-call / action 段 token(需要解析);
2. 上一轮 assistant 输出的**尾部 N token**(默认 N = `block_tokens`,可配);
3. 边生成边累积的 `q_sum_`(需处理 pin 后窗口语义)。
> 当前 `query_policy_user` 把 query 设成 last-user span(`tools/kvmem-multimodal-server.h:302-306`),这是**观测**而非动作,不能直接当作 ActKV 的 action region。

### 2.4 分层与滞回(G4)

```cpp
struct KvMemTierPolicy {
    double ks_gpu_in   = 0.30;   // 进入 GPU 热集
    double ks_gpu_out  = 0.20;   // 退出 GPU(hysteresis)
    double ks_cpu_in   = 0.05;   // 进入 pinned CPU
    double ks_cpu_out  = 0.02;
    uint32_t min_dwell_rounds = 2;
};
```
- 每轮提交后按 KS 划带;`min_dwell_rounds` 防止抖动。
- 接线已有的 `PinnedKvTier` heat 策略(`pinned_kv_tier.hpp:152-156,185-193,237-258`;当前 adapter 从不 `record_selected`/`begin_selection_epoch`)。
- 冷到 `ks < ks_ssd_out` 且非 sink/recent/mandatory → 允许在 `truncate_to` 时释放层级槽。

### 2.5 预取(G5)

```
每轮提交后记录 S_prev = KS top-K 块集合(K = 当前 budget)
下次 prefill 开始前:对 S_prev \ 当前驻留 异步 stage_in(NVMe→pinned→GPU)
评分完成后:只补真实 stage_in 差集
```
复用 `kvmem_runtime.cpp:143-170` 的 `std::async` 预取骨架;新增指标"预取命中率"。
注意:预取不得跨越 `seq_rm`/reselect 边界面,且失败必须可回退为同步路径。

### 2.6 自适应预算(G7)

```cpp
// 真实上界与运行值必须来自不同来源
uint32_t b_upper = params.budget_upper_tokens;   // 新增参数,默认 = select_budget
uint32_t active  = params.budget_initial_tokens; // 新增参数,默认 = select_budget(向后兼容)
```
- 论文 §8 实测配置:**启动 512 entries,×1.75,封顶 `B_upper = 30K` entries**。即 `budget_initial` 必须**显著小于** `b_upper`,而不是等于它——当前代码 `active == ceiling` 恒成立,这正是模块 3 恒为 no-op 的根因。
- 增长公式保持论文原样:`B' = min(B × scale_up, B_upper)`。
- **只在压缩步(每轮)执行一次**;带 `cooldown_rounds`(默认 3)与"窗口换血"条件。
- 增长后 **reset 监控窗口**,并记录 `last_growth_round`。
- 删除 `target = active + bt` 这条"总要前进"的兜底(它让 `scale_up ≤ 1` 失效)。
- 失败(`std::invalid_argument`)必须打日志 + 计数 + 记忆失败目标,不再静默吞。
- 文档化:预算增长**是否允许**借用 `gen_reserve`(与 ring buffer cut 的边界)。

### 2.7 尾部压缩 + 槽位规划(G6,论文 §7.2 的**有限**移植)

**前提修正(必读)**:论文 §7.2 的收益来自 vLLM 式共享块池——空洞使物理块无法归还调度器,把保留项压到尾部块可**真正释放块**。KVMem 的回收单位是 block slot(`stage_out`+`free_slot` 已归还整块,`docs/architecture.md:29-32`),块内 token 级空洞不占池内存且被刻意保留+掩码(`patches/0003-kv-cache-skip-hole-purge.patch`)。因此 §7.2 **不提供显存收益**,只可能在页表连续性/掩码开销上有边际收益。

若度量后仍要做,采用论文的两个机制:
```
1) 目标区域 = 已占用逻辑块的尾部 X 块(驱逐后保留项在尾部更密)
2) 配对:目标区域内的 evicted slot ←→ 区域外的 retained slot(源/目标互斥)
3) 一趟并行拷贝,无辅助 KV buffer
```
- **先度量再决定**:统计一次真实 reselect 的"保留 cell / 占用 cell";空洞比例低于阈值(建议 20%)则关闭该项。
- **不要**做"压到头部"的 `prefix_sum` 变体:那会移动整个窗口的 bake 位置,增加 `remap_count`,触发 raw-K 重建。
- 保留现有 raw-K 漂移刷新策略(`immutable_refresh_*`);压缩不得增加 `remap_count`。

---

## 3. 任务拆分

| # | 任务 | 主要改动点 | 依赖 | 验收 |
|---|---|---|---|---|
| T1 | KS 独立字段 + `sel_score()` | `kvmem_store.hpp/.cpp`,`llama-memory-kvmem.cpp:4294-4305` | — | 同轮两次 preview 得到相同 selection;`constrain_media` 不再受 KS 影响 |
| T2 | 轮次提交化 | `llama-memory-kvmem.h/.cpp`(`apply_selection`),删除 `lrfu_round_id` | T1 | 幽灵 preview 不再衰减 KS;新增单测覆盖"丢弃的 preview" |
| T3 | `reset_lrfu` 接线 + monitor 移入会话 | `truncate_cached:1051-1066`,`reset_policy:1023`,`swap_conv:1589` | T1 | 前缀回退后 KS 归零;会话切换不共享置信度历史 |
| T4 | 参数与 CLI | `tools/llama-kvmem-cli.cpp:195-207`,`tools/llama-kvmem-server.cpp:2491-2503`,`llama-kvmem-hooks.h:40-58` | — | λ 越界报错;λ=0 可表达;`conf_*` 有 flag;action_lrfu + method=recency 告警 |
| T5 | GQA 聚合修正 | `llama-memory-kvmem.cpp:4196, 4230-4233` | — | `g=ceil` + 每组 clamp;`n_head_%n_head_kv_≠0` 有单测 |
| T6 | 动作区 query + 精确打分(阶段 A) | 新增 host 打分函数;`score_retrieval()` 替换/并行 | T1 | 质量非负;needle 任务上 KS 排序与人工标注 action 块一致 |
| T7 | 分层 + 滞回 | `pinned_kv_tier.hpp` 接线,`kvmem_runtime.cpp:243-292` | T1,T6 | 同一块在 N 轮内不反复 in/out(计数为 0) |
| T8 | 自适应预算 | `llama-memory-kvmem.cpp:4336-4374`,hooks/cli | T4 | 预算至多每轮变一次、不超 `B_upper`;失败有日志 |
| T9 | 预取 | `kvmem_runtime.cpp:143-170`,`apply_plan_to_kv` | T7 | 预取命中率 > 0 且不增加端到端时延 |
| T10 | 尾部压缩(可选) | `layout_gpu_slots_by_orig_pos` 路径 | T7,需先出度量 | 空洞比例 > 20% 才开工;KV 内容 bit 级一致(dump 对比);不增加 `remap_count` |
| T11 | 保护 system prompt + 用户任务描述 | `retrieval_mandatory()`(`llama-memory-kvmem.cpp:4464-4483`),`sink_blocks` 计价 | T1 | 论文 §8 要求这两段永不参与压缩;超预算时仍强制保留,并计入 `sink+recent ≤ budget_blocks` 校验 |

建议顺序:`T1 → T2 → T3 → T4/T5(可并行) → T8 → T6 → T11 → T7 → T9 → T10(需先证伪空洞收益)`。

---

## 4. 验收标准(可测)

1. **确定性**:同一请求、同一输入,连续两次 `preview_retrieval()` 返回**完全相同**的 block 列表(当前不满足,见审查 P0-3)。
2. **一轮一更新**:一次请求内 KS 更新次数 ≤ 实际提交的压缩决策数;丢弃的 preview 计数为 0。
3. **非负性**:所有 `ks_score ≥ 0`;λ=1 时 KS 排序等价于"命中次数"排序而非幅值排序。
4. **预算**:`observe_confidence` 连续调用 10⁴ 次,预算变化次数 ≤ 轮数,且 ≤ `B_upper`。
5. **分层稳定**:连续 5 轮中,同一块的 tier 变化次数 ≤ 2(滞回生效)。
6. **压缩正确性**:压缩前后对同一 selection 做 attention dump 对比,保留块的 K/V 逐字节一致。
7. **无回归**:现有 15 项 ctest 全绿(含修复 `kvmem-session-cache-lifecycle-test`)。

---

## 5. 测试计划

### 5.1 宿主机(无 GPU)

```bat
call env-msvc.bat
cmake -S . -B build-kvmem-host -G Ninja -DKVMEM_ENABLE_NVME=OFF -DCMAKE_BUILD_TYPE=Release
ninja -C build-kvmem-host kvmem_store_test confidence_monitor_test
ctest --test-dir build-kvmem-host -R "kvmem_store_test|confidence_monitor_test" --output-on-failure
```

必须新增的用例:
- `set_retrieval_scores` 与 `commit_lrfu_round` 交错两次 → 第二轮 selection 仍按 KS(抓 P0-3);
- 模拟"preview 丢弃 + rows 变化" → KS 只更新一次(抓 P0-4);
- `observe_confidence` 连打 → 预算变化次数与 `B_upper` 断言(抓 P0-2/P0-5);
- `score_agg=max` 的合成 GQA,含 `n_head_ % n_head_kv_ != 0`(抓 P0-11);
- 分层滞回:构造 KS 在阈值附近抖动的序列,断言 tier 不抖(抓 P0-7/T7)。

### 5.2 端到端(需 CUDA 构建)

- 质量:`scripts/multi_needle.py`、`scripts/needle_recall.py`、`tests/mtp-kv-test.cpp`
- 交换开销:`retr_` 计数(`n_stage_in`/`n_move`/`n_raw`/`laid_out`,`retr_perf_print:4247`)、`scripts/nvme_spill.py`
- 新增指标:预取命中率、KS 集合跨轮 Jaccard 稳定性(论文 §5.1 AAV 的可视化等价物)
- A/B 开关:`--kvmem-action-lrfu` on/off,固定 seed

---

## 6. 风险与回滚

| 风险 | 说明 | 缓解 |
|---|---|---|
| 预取污染内存 | 预取块挤占 `gen_reserve` | 预取目标只进 pinned CPU,不直接进 GPU;上限可配 |
| 预算增长吃掉 gen_reserve | 与 ring buffer cut 冲突 | D7 成本模型显式区分语义预算与生成长度余量 |
| FA 改动难回滚 | 阶段 B 需改子模块 | 单独立项 + `patches/` 同步;阶段 A 不碰子模块 |
| KS 冷启动 | 新块 KS=0 永不被选 | sink/recent 已兜底;可选"新块 N 轮保底" |
| 压缩引入拷贝开销 | in-place 搬移的带宽 | 先做 run 级分段统计,量化收益后再替换整块重排 |

回滚:所有新行为均由 `--kvmem-action-lrfu` 与新增参数控制,**默认关闭**;默认路径必须与当前行为逐字节一致。

---

## 7. 代码位置索引

| 位置 | 说明 |
|---|---|
| `kvmem/include/kvmem/kvmem_store.hpp:36-78` | `KvMemBlock` 分数字段(T1 加 `ks_score`) |
| `kvmem/include/kvmem/kvmem_store.hpp:333-344` | LRFU 配置 |
| `kvmem/src/host/kvmem_store.cpp:242-313` | `set_retrieval_scores` / `accumulate_lrfu` / `reset_lrfu`(T1/T2) |
| `kvmem/src/host/kvmem_store.cpp:466-475` | `constrain_media` 比较器(P0-7) |
| `kvmem/src/host/kvmem_store.cpp:547-700` | `pick_topk_ungrouped`(选择分数) |
| `kvmem/src/host/confidence_monitor.{hpp,cpp}` | 置信度监控(T8) |
| `src/adapter/llama-memory-kvmem.cpp:4191-4314` | `score_retrieval` / `lrfu_round_id`(T1/T2/T5/T6) |
| `src/adapter/llama-memory-kvmem.cpp:4336-4374` | `observe_confidence`(T8) |
| `src/adapter/llama-memory-kvmem.cpp:4485-4505` | `preview_retrieval`(只读化) |
| `src/adapter/llama-memory-kvmem.cpp:4582-4701` | `apply_selection`(提交点,T2) |
| `src/adapter/llama-memory-kvmem.cpp:1923-2254` | `layout_gpu_slots_by_orig_pos`(T10) |
| `kvmem/src/host/kvmem_runtime.cpp:143-170` | 异步 NVMe 预取骨架(T9) |
| `kvmem/src/host/kvmem_runtime.cpp:243-292` | `stage_out` 分层(T7) |
| `kvmem/include/kvmem/pinned_kv_tier.hpp:152-156,185-193,237-258` | CPU tier heat(未接线,T7) |
| `tools/kvmem-multimodal-server.h:296-416` | 一轮选择/应用序列(T2 幽灵轮) |
| `docs/kvmem-actkv-review-2026-10-10.md` | 问题清单 P0-1~P0-15 |
