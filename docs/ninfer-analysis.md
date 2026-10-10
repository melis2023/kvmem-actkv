# What NInfer can teach KVMem about decode speed

Reference: [Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco)
(local clone: `I:\llama\ninfer`).

That repository is **not** a llama.cpp fork. It is a self-contained engine with
its own operator library, its own KV encodings and its own weight formats, built
for exactly our target: ~16 GB of VRAM and Qwen3.8-27B GSQ-RCO. So the question
is not "can we port it" but "which of its ideas transfer to llama.cpp/KVMem".

## The headline: decode is weight-bandwidth bound, and we are already there

`docs/gguf.md` states it outright:

> The lead comes from reading **31% fewer weight bytes per token**.

GSQ-RCO IQ3_S (3.5 bpw, 10.95 GiB) versus the official artifact (Q4/Q5 groups,
15.9 GiB): 31% fewer bytes, and decode gets 22–49% faster depending on card.

It is worth checking what that means for our card, because the arithmetic closes:

| | RTX 5060 Ti (ours) | RTX 5090 (NInfer's) |
| --- | --- | --- |
| memory bandwidth | ~448 GB/s | ~1792 GB/s |
| weights (IQ3_S) | 11.76 GB | 11.76 GB |
| **theoretical no-speculation ceiling** | **38.1 tok/s** | 152 tok/s |
| measured, no speculation | 26.9 tok/s | 107.5 tok/s |
| efficiency vs ceiling | **71%** | **71%** |
| measured, MTP | 65.2 tok/s (n_max 4) | 221 tok/s (3 drafts) |
| MTP amplification | **2.42x** | 2.06x |

Our engine reaches **the same 71% of the bandwidth ceiling** as NInfer does on a
5090, and our MTP amplification is slightly higher. The gap to NInfer's numbers
is bandwidth, not implementation quality.

**Conclusion: on this card, with this model, decode is already close to the
physical limit.** No amount of further micro-optimisation moves a
memory-bound loop past the bandwidth wall. The remaining levers are all
bandwidth levers: fewer weight bytes per token, or a faster card.

## Why NInfer's KV work does not transfer here

NInfer puts real effort into compact KV (`rk8v4` 408 → `rk4v4-e8` 280 →
`rk2v4-e8` 216 bytes per token per KV head). That pays off on models where KV
dominates.

Qwen3.8-27B is a **hybrid** model, which cuts the other way:

```
qwen35.attention.head_count_kv      = 4
qwen35.attention.key_length         = 256
qwen35.attention.value_length       = 256
qwen35.full_attention_interval      = 4     <- only every 4th layer is full attention
n_layer                             = 64
```

Only roughly `64 / 4 = 16` layers hold a KV cache at all, and each is small
(4 heads × 256 dims × K and V). So KV traffic is a small share of a decode step,
and halving it with `-ctk q4_0 -ctv q4_0` buys very little throughput. It may
still be worth it to *fit a longer context into 16 GB* — that is a capacity
question, not a speed one.

(Consistent with this: a 65,536-context run with a 28K prompt failed on this
card, while 4,096 and 8,192 contexts are comfortable.)

## What is genuinely transferable, ranked

1. **Draft count 4–5 (done).** NInfer reports "MTP currently supports up to 15
   drafts, but the fastest is usually 3–5". Our own sweep on this exact model
   peaks at 4–5 (see `docs/mtp-tuning.md`). Independent confirmation of the
   same optimum — no further work needed.

2. **MTP attention window — worth considering for long contexts.** NInfer offers
   `--mtp-attention-window N` so the draft head attends to only the first 64 keys
   plus the last N, "avoiding draft reads growing with history". Our MTP
   measurements are all short-context; if long-context decode matters, capping
   the draft's attention is the mechanism that keeps draft cost flat. This lives
   in llama.cpp's draft construction, not in KVMem, so it is a submodule change.

3. **Adaptive MTP — measured as not worth it, even by NInfer.** `--adaptive-mtp`
   varies the verified width per round from measured survival rate. NInfer's own
   numbers: short input 200 vs 204 tok/s fixed, 8K 150 vs 163 — i.e. it lost.
   It also costs one CUDA graph per width. Skip it.

4. **Whole-program CUDA compilation.** NInfer builds core and ops without
   relocatable device code so ptxas can inline across loops ("about a fifth
   fewer kernels needing a stack frame"). Cheap to try on llama.cpp, but the
   reported benefit is code-size, not a decode throughput number.

5. **N-gram drafter — blocked.** NInfer layers n-gram copy drafts on top of the
   neural drafter at zero VRAM cost. llama.cpp does have n-gram speculation
   (`COMMON_SPECULATIVE_TYPE_NGRAM_*`), but `tools/kvmem-spec.cpp` hardcodes
   `p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP }`, so KVMem
   cannot reach it today. Enabling it would mean extending the KVMem spec
   options, and it would still not stack with MTP.

## What does not transfer

These are the bulk of NInfer's advantage and all of them presuppose its own
runtime:

* E8 lattice KV encodings (`rk4v4-e8`, `rk2v4-e8`) — a private KV format.
* Ternary / unified Linear templates, `t2_g128_fp16` weights, W4A4 NVFP4 paths.
* Per-card device profiles: NInfer *measures* each operation at each width on
  your GPU and picks the winner, with a saved calibration. llama.cpp has no
  equivalent routing layer; importing the idea means building one.
* DFlash2 / DSPark drafters — different drafters entirely, not MTP.

## Bottom line

For decode on this hardware there is no large win left to port. Effort is better
spent on either

* **capacity** — smaller KV types and a smarter tier policy to fit longer
  contexts into 16 GB (KVMem's actual remit), or
* **prefill**, which NInfer's own tables show is far from the bandwidth wall
  (their prompt-kernel work reports 19–30% wins), and which our tree already
  documents as the real remaining gap.
