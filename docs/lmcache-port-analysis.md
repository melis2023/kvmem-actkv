# LMCache → KVMem port analysis

| Field | Value |
|---|---|
| **Title** | Which LMCache mechanisms transfer to this llama.cpp KVMem tree, and which are already here |
| **Date** | 2026-10-01 |
| **Status** | Applied (transfer/launch-count changes). No end-to-end speed claim. |
| **Source analysed** | [`LMCache/LMCache`](https://github.com/LMCache/LMCache) `dev`, commit `8d7d2c4` |
| **Target** | this tree, `VERSION` 0.17.0, llama.cpp pin `7fe450e19` (v0.5.0) |
| **Build** | `build-cuda13.4-dll`, MSVC 14.44 + CUDA 13.4, `CMAKE_CUDA_ARCHITECTURES=120a-real` |

Read `docs/prefill-harvest-optimization.md` and `docs/retrieval-stagein-optimization.md`
first: both describe work that is **already landed in this tree**. This document only
covers what those two plans do not, and is explicit about the parts of LMCache that do
not transfer at all.

---

## 1. What LMCache is, and what is actually portable

LMCache is a **KV-cache management layer for vLLM/SGLang**: it moves KV out of GPU memory
into a CPU / disk / remote hierarchy and reuses it across requests. Its value to a
different engine is *not* the hierarchy — KVMem already has one (slot pool + `RawKvStore`
+ `KvMemStore` tiers). What transfers is the **mechanics of moving KV bytes across the
PCIe boundary without stalling the engine**.

| # | LMCache mechanism | Where | Portable? | Verdict for this tree |
|---|---|---|---|---|
| A | `use_layerwise`: generator-driven per-layer `retrieve_layer` / `store_layer`, lookahead 2 | `lmcache/integration/vllm/vllm_v1_adapter.py:842-853`, `lmcache/v1/cache_engine.py:1106-1123`, `753-777` | Yes (idea) / No (code) | Idea already present at a coarser grain: KVMem harvests per ubatch, not per layer. Per-layer is the wrong grain here — §7 of the dev notes explains why an `eval_callback` per layer costs 192 graph splits. |
| B | Dual stream + **one-directional** `store_stream.wait_stream(current_stream)`, `sync` only at the batch boundary (load = last request, store = first request) | `lmcache/v1/gpu_connector/gpu_connectors.py:1118-1119, 1269-1326, 1401-1440` | Yes | **Already here in spirit**: `CaptureD2hPipe` harvest stream + `snap_be` / `compute_done` events (`src/adapter/llama-memory-kvmem.cpp:2994-3115`). This pass finished two spots the pattern had not reached (§3.2, §3.5). |
| C | One kernel for all layers: `grid = (tokens, layers, k_or_v)`, scattered straight into the paged layout | `csrc/cuda/mem_kernels.cu:472-516, 664-670` | Yes | KVMem's equivalent is `copy_bytes` (`src/adapter/llama-kvmem-stagein.cu:233-258`) driven by `kvmem_d2d_batched` / `kvmem_stageout_submit`. **This pass added the missing consumer** (§3.1). |
| D | Fused multi-chunk launch, params by value, `MAX_FUSED_TRANSFER_CHUNKS = 16` | `csrc/cuda/mem_kernels.cuh:17-33`, `mem_kernels.cu:429-464, 775-834` | Yes | Adopted as the shape for the batched mean-K submit (§3.1). |
| E | `cudaMemcpyBatchAsync` with runtime probe + fallback | `csrc/cuda/mp_mem_kernels.cu:748-761, 777-906` | Yes, CUDA ≥ 12.8 | **Not applied.** The only place it would help is the block host-mirror copy, whose destination is a pageable `std::vector` (`kvmem/src/host/kvmem_runtime.cpp:252,317`) — batching pageable copies does not reduce the host-visible cost. Revisit only if that buffer becomes `cudaMallocHost`. |
| F | 2 MiB huge pages + NUMA bind + `first_touch`; pin chunk granularity | `csrc/cuda/mem_alloc.cpp:49-132`, `mem_kernels.cu:1366-1399` | Linux only | Not applicable on this Windows build (`KVMEM_ENABLE_NVME=OFF`, `cudaMallocHost` whole buffers). |
| G | CacheGen / TurboQuant KV compression (2.6×–5.0× serialized) | `lmcache/storage_backend/serde/cachegen_*.py`, `docs/design/v1/distributed/serde/turboquant.md:228-234` | Yes in principle | **Rejected.** It trades KV quality for transfer volume. KVMem already has the orthogonal control (`--kv-dtype`, `--spec-kv-dtype`) and `docs/mtp-kv-controlled-experiment-2026-09-13.md` shows the naive "smaller KV is faster" instinct is wrong on this hardware. |
| H | Non-prefix reuse (CacheBlend): recompute top-k of the layers whose cached K differs most | `lmcache/v1/compute/blend/blender.py:60-121` | Algorithm yes, hooks no | Out of v1 scope; needs per-layer forward callbacks KVMem deliberately avoids. |
| I | Chunk + prefix-hash keys; concurrent disk read pool; `O_DIRECT` | `lmcache/v1/token_database.py:334-365`, `local_disk_backend.py:508-553, 778-831` | Yes | NVMe tier is POSIX-only and forced off on Windows, so none of it reaches this build. |
| J | Pinned staging pooled by `(shape, dtype)`, borrowed/returned | `lmcache/v1/multiprocess/transfer_context/async_engine_driven.py:104-157` | Yes | Already here: the two-slot `CaptureD2hPipe` and the 32 MiB stage-in slab. |

**The mechanism worth naming**: LMCache's store path *never* host-synchronizes per layer.
It establishes a one-directional stream dependency and pays a host fence **only where the
host actually consumes the bytes**, or at the batch boundary chosen for the request. Every
item this pass changed is that rule applied to a loop that still had a per-layer fence.

---

## 2. What this tree already had (do not re-litigate)

| Already present | Evidence |
|---|---|
| Harvest fence split into `compute_done` (host-wait, for the `set_input` race) + `snap_be` (compute stream waits for the D2D) | `llama-memory-kvmem.cpp:2985-3115`; `docs/prefill-harvest-optimization.md` PR 1 |
| Adapter harvest worker thread; pack + NVMe off the compute thread | `harvest_worker_start` / `harvest_loop`, `llama-memory-kvmem.cpp:2758-2874`; same doc, PR 3 |
| FP16 raw-K without the F32 round trip | `bytes_to_f16_token_major`, `llama-memory-kvmem.cpp:3275-3299`; same doc, PR 2 |
| Fused gather/scatter kernel + batched D2D for slot-pool layout | `copy_bytes` / `kvmem_d2d_batched`, `llama-kvmem-stagein.cu:233-258, 967-1000`; `docs/retrieval-stagein-optimization.md` PR 1-2 |
| V slab batching (`harvest_v_pending_`, 2 pinned slots + events) | `harvest_gpu_v_flush_slab`, `llama-memory-kvmem.cpp:3470-3498` |
| GPU-side dequant → RoPE → Hadamard → requantize straight into the cache layout | `kvmem_stagein_enqueue_k` / `kvmem_stagein_flush`, `llama-kvmem-stagein.cu:634-755, 778-820`. This is the "dequantize directly into the attention layout" gap the LMCache survey flags as *missing upstream* — KVMem already does it for cold stage-in. |
| MTP harvest is a no-op on the per-step path (no blocking `tensor_get`) | `llama-memory-kvmem-mtp.cpp:368-374` |

**Therefore the remaining prefill gap is engine-side, not KVMem-side.** The 4087-node ggml
graph, IQ3 GEMM and `ggml_flash_attn_ext` are what separate KVMem prefill from stock
(`docs/retrieval-stagein-optimization.md`, "Prefill 2× is not this plan"). Nothing in
LMCache changes that, and the LMCache survey found no decode-side mechanism at all: LMCache
has no per-token store optimisation because vLLM's decode step is not where it lives.

---

## 3. Changes applied in this pass

### 3.1 Batched multi-layer mean-K submit — [`D`]

`decode_mean_add_range()` runs **once per decode token** and launched one `meank_add_*`
kernel per attention layer: 64 launches per token on a dense 27B, each with its own
`cudaGetLastError`. The layers differ only in `(tensor, strides, layer index)`, exactly the
shape LMCache folds with `grid.y = layer_id`.

Added `kvmem_meank_add_batch()` + `meank_add_batch<<<(d, layers)>>>`
(`llama-kvmem-stagein.cu`, `KvMemMeankOp` in `llama-kvmem-stagein.h`). One ops-array upload
plus one launch replaces n launches. Layers the batch cannot take (host-resident tensors,
unsupported element types) are collected and still go through the old per-layer / host
fallback, so a partial batch degrades instead of dropping layers.

`decode_mean_zero_acc()` now calls `kvmem_meank_zero_all()` — one `cudaMemsetAsync` over
the whole accumulator instead of n_layer memsets every time the window crosses a block.

### 3.2 One fence per decode-mean flush, not one per layer — [`B`]

`decode_mean_flush()` called `kvmem_stagein_sync()` **inside** its layer loop after each
`kvmem_meank_d2h`. All the copies are on one stream and nothing reads them until the loop
below, so the fences were pure host round trips: 64 of them per block boundary on a dense
27B. The function now enqueues every layer's D2H, fences once, then consumes.

Same rule removed the mid-sequence sync in `decode_mean_add_host_layer()`:
`kvmem_meank_zero()` is enqueued behind the D2H on the same stream, so it cannot run before
that copy has read the accumulator.

### 3.3 Deferred stage-in slab fence — [`B`]

`kvmem_stagein_flush()` host-waited on its own H2D (`cudaEventSynchronize(g_st.h2d_ev)`)
immediately after issuing it. The dequant / RoPE / Hadamard / quantize chain that follows
reads the same `g_st.dev_q` on the same stream, so stream order already carries that
dependency. The host only needs the fence before it refills `g_st.pin[0]`.

The event is now recorded and the fence moved to `slab_pin_release()`, called from
`slab_reserve()` right before the next `memcpy` into the pin, and from
`kvmem_stagein_gpu_free()` before the buffers are released. The host can therefore pack the
following slab while the previous 32 MiB H2D is still in flight.

### 3.4 `score_retrieval()` — no heap traffic in the hot loop

`docs/retrieval-stagein-optimization.md` flags this as the no-risk first step and it had
not been done: the inner GQA loop heap-allocated `std::vector<float> qh(n_embd_head_)` per
`block × layer × kv-head` — **320k allocations** for a 2500-block pool with 16 layers and
8 kv heads. The grouped-query sum and its `1/(q_count * g)` factor do not depend on the
block, so they are folded once per `(layer, kv head)` before the block loop. The arithmetic
is unchanged: `invq` is folded into `qh`, and it cancels out of `dot / (|q| |k|)`.

### 3.5 Not changed, and why

* **The harvest `compute_done` host-wait stays.** Removing it is the first thing that looks
  worth doing, and it is not: the wait is what orders ubatch N+1's `set_input`, which runs
  on `cudaStreamPerThread` (`ggml-cuda.cu:790-791`). Replacing it with
  `cudaStreamWaitEvent(cudaStreamPerThread, compute_done)` is safe but gains nothing,
  because `ggml_backend_cuda_buffer_set_tensor` ends with its own
  `cudaStreamSynchronize(cudaStreamPerThread)` — the host blocks in the same place either
  way. The only real fix is to move `set_input` onto the compute stream, which both KVMem
  plans list as an explicit non-goal.
* **`cudaMemcpyBatchAsync`** — see `E` above.
* **KV compression** — see `G` above.
* **Per-layer `use_layerwise` harvest** — see `A` above.

---

## 4. How to verify

Unit and integration tests are unchanged and should stay green:

```bash
cmd //c ctest-msvc.bat                       # 13 tests; kvmem-mtp-kv-test shows "Not Run" without an MTP GGUF
bash scripts/kvmem-admission-check.sh        # if the Lua/POSIX shell is available
```

The four changes are all exercised by the existing traces, so an A/B is a matter of
watching counters rather than adding them:

```bash
export KVMEM_PERF=1
```

* §3.1 / §3.2 → `KVMEM_DECODE_MEAN n_tok=... n_flush=... n_gpu=...` must be byte-identical
  to the pre-change run for the same prompt, and `KVMEM_DECODE_MEAN flush ... layers=...
  gpu=... host=... rms=...` must match on `layers`, `n`, `pos0` and `rms`. `rms` is the
  sensitive one: it is computed from the first flushed layer's accumulator, so a wrong
  batch shape changes it immediately.
* §3.3 → `KVMEM_RETR_SUM ... set_ms=...` is the bucket the slab fence lives in. `copy_ms`
  and `rope_ms` must not grow.
* §3.4 → `KVMEM_RETR_SUM ... score_ms=...`.

**No end-to-end number is claimed.** `models/` is empty in this checkout, so no identity
canary, needle test or tok/s measurement was run for this change. Expected magnitudes, for
calibration only: §3.1 and §3.2 remove ~n_layer host round trips per decode token and per
block boundary; §3.4 removes ~2n_blk·n_layer·n_head_kv heap operations per retrieval;
§3.3 removes one PCIe round trip per 32 MiB stage-in slab. None of these is a claim about
decode tok/s — the decode bottleneck in this tree is weight bandwidth and MTP acceptance,
not KVMem bookkeeping.

---

## 5. Build and test status (this pass)

Toolchain: MSVC 14.44.35207, Windows SDK 10.0.26100.0, CUDA 13.4, Ninja,
`CMAKE_CUDA_ARCHITECTURES=120a-real`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `KVMEM_ENABLE_NVME=OFF`.

**Build: clean.** `ninja` reached `[109/109]`, 45 link steps, 0 errors, 36 executables in
`build-cuda13.4-dll/bin/`. `llama-kvmem-server.exe` mtime `20:53:15` is newer than every
source touched by this change (latest `20:51:07`), so the binaries do contain §3.

Two build-side notes worth keeping:

* `llama.cpp/` was an empty submodule in this source package. It was restored by cloning the
  sibling checkout and detaching at `7fe450e19305b828c199d602c23a8337aaa1f03b` (llama.cpp
  v0.5.0), then running `./scripts/apply-patches.sh`.
* `cmd //c ninja-msvc.bat` always ends with `PROGRAM BLOCKED BY SECURITY POLICY ... reg.exe`.
  That is the sandbox blocking `reg.exe` *after* the build has finished; it is not a build
  failure. Judge the run by `[N/N]`, the error count and the `.exe` mtimes, not the exit
  status of the `cmd` wrapper.

**Tests: 14/15 pass.** The one failure is `kvmem-session-cache-lifecycle-test`
(`test_symlinks_are_not_followed`, `assertEqual(stats["skipped"], 2)` → got 1).

It is **not** attributable to this change. That test drives
`tests/session-cache-lifecycle-worker.cpp` → `tools/kvmem-session-cache-dir.h`, and the
worker links **only** `worker.cpp.obj` — no adapter object, no `llama-memory-kvmem.cpp.obj`,
no `llama-kvmem-stagein.cu.obj`. None of the four files touched by this pass is in its
translation units. Treat it as pre-existing and environment-dependent (Windows reparse-point
handling in the session-cache sweeper); it needs a separate investigation with a model-free
A/B against an unpatched tree.

Everything else is green, including `kvmem-session-transfer-test` (130 s) and the two GDN
tests that cover the fused-output path.
