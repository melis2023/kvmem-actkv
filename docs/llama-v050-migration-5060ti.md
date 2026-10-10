# llama.cpp v0.5.0 migration and RTX 5060 Ti Task 1 comparison

2026-09-28 local test. The starting point was local KVMem `master`
`95a2155b75f41eada1398c86b19dcf9703581b30`. Its llama.cpp pin was
`b81c99b479d4c24e5eeca10de99032ebd343ef8f` (llama.cpp 0.3.0). The
migrated pin is the official llama.cpp `v0.5.0` tag,
`7fe450e19305b828c199d602c23a8337aaa1f03b`.

The cumulative KVMem patch was rebased to the new pin. The RDNA2 fix remains
separate. The old `GGML_CUDA_FA_ALL_QUANTS` switch was changed to upstream's
`GGML_CUDA_FA_QUANTS=all`. The server target also links `kvmem`, which is
required for the session-memory symbol in this master revision. Both patch
files applied cleanly to a fresh v0.5.0 export, including the reverse checks.
The migrated Linux CUDA server built and served a Qwen3.5-0.8B smoke request.
The native Windows CUDA 13.2.86 build also completed all 423 compilation and
link steps, including `llama-kvmem-server.exe` and its test executables.
Native Windows runtime tests were not run in this comparison.

## Benchmark setup

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GiB, physical GPU index 1. The existing
  Windows server was stopped for the test and restored afterward; its
  `http://127.0.0.1:18200/health` endpoint returned HTTP 200.
- System: Ubuntu 22.04 under WSL2, CUDA Toolkit 13.2.86, Release build,
  `120a-real` CUDA target. Both versions were built from the same KVMem master
  base and with the corresponding all-quant Flash Attention option.
- Model: `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`; projector:
  `mmproj-Q8_0.gguf`. Both were the same local files for each run.
- Workload: the repository's `scripts/multimodal_canary.py` Task 1 preset:
  12,000-word context, 896×896 tricolor image and HTML/SVG request. KVMem
  query replay `auto`, query policy `user`, MTP 3 with replay state, GPU KV
  `q8_0`, draft KV `f16`, 36,864-token budget, 16,384-token reserve, 262,144
  context, batch 512, image cap 1,024 tokens and thinking budget 128.
- One warmup plus two measured runs per version. The exact launcher and raw
  logs are in the sibling `benchmark-llama-v050` directory. All nine requests
  per version returned successfully.

## Results

Means below use the two measured runs. Speed and prefill timing come from the
server's response `timings`; wall time comes from the client. The code step
generated exactly 512 tokens on each run for both versions.

| Metric | b81 / llama.cpp 0.3.0 | v0.5.0 | Difference |
|---|---:|---:|---:|
| Complete Task 1 wall time per run | 31.04 s | 30.01 s | −3.31% |
| Long-text prefill, 12,060 prompt tokens | 14.70 s | 14.65 s | −0.33% |
| Image-step wall time | 3.81 s | 3.36 s | −11.85% |
| Code-step wall time | 10.77 s | 10.66 s | −1.05% |
| Code-step decode throughput | 50.90 token/s | 52.02 token/s | +2.20% |
| Peak GPU memory | 15,591 MiB | 15,599 MiB | +8 MiB |
| Peak runtime process RSS | 4,327 MiB | 4,366 MiB | +39 MiB |

The image descriptions correctly named the red square, blue circle and green
triangle in both versions. The v0.5.0 responses used fewer tokens in the
long-text step (45 versus 57) and image step (80 versus 94), so their wall-time
improvements include an output-length effect. The code throughput comparison
uses equal 512-token outputs and is the cleaner decode comparison. Both code
responses ended at the benchmark's 512-token limit; this run does not establish
that either version produced a complete HTML page. No swap stop occurred.

This is a two-run result on one GPU. Differences near 1–3% should be treated
as indicative, not a general performance claim across models or hardware.
