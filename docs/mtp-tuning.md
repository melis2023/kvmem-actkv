# MTP (speculative decode) tuning — measured

Hardware / software this was measured on:

* GPU: **NVIDIA GeForce RTX 5060 Ti, 16311 MiB** (< 16 GB usable)
* Model: `I:\model\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (11.5 GB, IQ3_S)
* Build: `build-cuda13.4-dll` (MSVC 14.44 + CUDA 13.4, `-DGGML_CUDA_FA_ALL_QUANTS=ON`)
* Method: `llama-kvmem-cli`, `-n 128 -c 4096 -ngl 99`, synthetic ~810-word prompt,
  one warm-up run discarded, one measured run per cell, reading `KVMEM_GEN_WALL toks=`.

Reproduce with `scripts/bench-mtp-sweep.py`.

## The trap: do not load two models

Two things multiply into a bad default configuration:

1. `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` (11.2 GB) contains **no** MTP metadata
   (no `qwen35.nextn_predict_layers`), so on its own it cannot draft.
2. `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (11.5 GB) **does** contain it
   (`qwen35.nextn_predict_layers = 1`, weights `blk.64.nextn.*`).

So the natural-looking setup — target + `--spec-draft-model <the -mtp file>` —
loads **22.7 GB of weights into 16 GB of VRAM**. Whatever does not fit gets
offloaded, and the draft model is then touched **every decode step**, so every
step pays PCIe traffic for it. Speculation cannot win that race.

**Load the `-mtp.gguf` alone.** llama.cpp detects `nextn` in the target model and
builds the MTP draft context against it, with no second copy:

```
llama_model_loader: - kv  52:   qwen35.nextn_predict_layers u32 = 1
common_speculative_init_result: creating MTP draft context against the target model ''
```

The empty `''` confirms no sidecar model was loaded.

## Measured sweep — `--spec-draft-n-max`

| `--spec-draft-n-max` | tok/s | vs no MTP |
| --- | --- | --- |
| off | 26.92 | — |
| 1 | 37.07 | +37.7% |
| 2 | 41.72 | +55.0% |
| 3 (old default) | 51.25 | +90.4% |
| **4 (new default)** | **65.32 – 65.62** | **+143%** |
| 5 | 65.90 – 66.38 | +147% |
| 6 | 65.94 | +145% |
| 8 | 67.00 | +149% |

The knee is at **n_max = 4**. Steps beyond it add under 4% while every additional
draft token costs another draft+verify forward even when rejected.

## Measured sweep — `--spec-draft-p-min` (keep it at 0)

| p_min | n_max=4 | n_max=5 |
| --- | --- | --- |
| 0.0 | **65.32** | **65.90** |
| 0.5 | 62.89 | 57.87 |
| 0.75 | 57.34 | 35.84 |
| 0.9 | 52.23 | 52.20 |

Raising `p_min` is **monotonically worse** here. It drops low-probability drafts,
but it also drops drafts that would have been accepted often enough to pay for
themselves. Leave it at the default `0.0`.

## Measured sweep — `--spec-kv-dtype` (MTP draft K/V type)

Three runs each, same prompt, `n_max=4`:

| `--spec-kv-dtype` | tok/s |
| --- | --- |
| default (inherits target q8_0) | 58.59 / 58.85 / 57.93 |
| `f16` | 58.95 / 59.98 / **60.48** |

`f16` is **~2.4% faster** (median 58.6 → 60.0). Real but small, and it costs 2x
the VRAM per draft KV element — affordable here because the draft context is
tiny compared to the target, but it is the first thing to drop if you are
chasing VRAM.

**The two entry points default differently**, which is easy to trip over:

| entry point | default MTP K/V type |
| --- | --- |
| `llama-kvmem-server` | **f16** (`st.spec_cache_type = GGML_TYPE_F16`) |
| `llama-kvmem-cli` | inherits target (q8_0) — `kvmem_spec_opts::draft_type = GGML_TYPE_COUNT` |

So `--spec-kv-dtype f16` is a no-op on the server and a small win on the CLI.

## Minimum working command

`--spec-type draft-mtp` is the **only required** knob — everything else already
defaults well:

* `n_max` defaults to 4 since this change (see below),
* the server already defaults `--spec-kv-dtype` to f16; add it explicitly only
  for the CLI.

Server:

```
llama-kvmem-server -m <the -mtp.gguf> --spec-type draft-mtp
```

CLI:

```
llama-kvmem-cli -m <the -mtp.gguf> --spec-type draft-mtp --spec-kv-dtype f16
```

Being explicit costs nothing and survives a future default change, so keeping
`--spec-draft-n-max 4` in your command line is fine.

**Do not add `--spec-draft-model`.** That loads a second ~11.5 GB copy of the
weights and is exactly the mistake described at the top of this file.

## Change applied

Default `n_max` raised to 4 in three places (all under `tools/`, KVMem's own
code, so no submodule patch is needed):

* `tools/llama-kvmem-server.cpp` — `spec_n_max` 3 → 4
* `tools/llama-kvmem-cli.cpp` — `spec_n_max` 2 → 4
* `tools/kvmem-spec.h` — `kvmem_spec_opts::n_max` 2 → 4

Help text updated to match. Verified after rebuild: with no `--spec-draft-n-max`
given, the same prompt yields **64.24 tok/s**, matching an explicit `n_max=4`.

## Caveats

* Tuned for **this model + this GPU**. `n_max` optimum tracks acceptance rate and
  draft cost, so re-run the sweep after changing either.
* Numbers are short-prompt (810 words), `-n 128`, no KVMem (`--kvmem` off) and
  `n_parallel = 1`. A short prompt measures a different acceptance rate than a
  long one; compare the previous section's 45.85 tok/s (98-word prompt) against
  64.24 tok/s (810-word prompt) at identical settings to see how large that is.
  Batch/parallel serving will land somewhere else again.
* Repeat the sweep under `--kvmem` and a long context before trusting these
  numbers for production serving.
