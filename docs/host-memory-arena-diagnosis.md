# Host-memory RSS growth: diagnosis and mitigation

## Symptom

During the 33-round, 256K-context multimodal canary
(`scripts/multimodal_canary.py --long-context-benchmark`), the server process
RSS grew from about 13 GiB to more than 47.8 GiB without falling. On hosts
with less RAM, this can end in OOM.

## Cause

Each decode step allocates and releases a roughly 10 MiB scratch buffer in
`llama_memory_kvmem_mtp::init_batch → llama_batch_allocr::ubatch_add`.
The observed RSS growth comes from these frequent allocations interacting
with glibc arena retention and fragmentation. The effect depends on CPU count,
glibc version, and allocation timing: the same workload held around 13.5 GiB
on one WSL2 host but exceeded 47.8 GiB on a 32-vCPU Linux host.

The growth occurred with both `--kvmem-mtp-state snapshots` and `replay`,
so it was not specific to the GDN state-management mode.

## Mitigation

The shared IQ3/IQ4 Linux launcher, `scripts/start-server.py`, now defaults
`MALLOC_ARENA_MAX` to `2` for the child server process. An explicitly set
value takes precedence. Capping glibc arenas kept the same 33-round canary
at about 21 GiB peak RSS, including the resident model and retrieval buffers,
with zero measured swap usage.

The launcher compares this setting when deciding whether an existing server
can be reused. A server started before this change, or with a different cap,
must be restarted with `--restart` for the new value to take effect. Starting
the server binary directly requires setting the environment variable yourself.

This is an allocator-level mitigation. Reusing or right-sizing the per-step
scratch allocation is a separate source-level change.

## Reproduction

- Workload: `scripts/multimodal_canary.py --long-context-benchmark`, 33 requests
  and 262058 of 262144 context tokens.
- Model: Qwen3.8-27B IQ3_S-mtp; retrieval budget 36864 and reserve 16384.
- Before: `peak_runtime_rss_mib` exceeded the equivalent of 47.8 GiB.
- With `MALLOC_ARENA_MAX=2`: `peak_runtime_rss_mib` was about 20965 MiB and
  `peak_process_swap_mib` was 0.

The allocation-site gdb backtrace was retained by the original PR author
and is available on request.
