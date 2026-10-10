#!/usr/bin/env python3
"""Sweep the KVMem MTP (speculative decode) knobs and report decode tok/s.

Scans --spec-draft-n-max against --spec-draft-p-min and prints a table plus the
best combination.

Usage:
    python scripts/bench-mtp-sweep.py -m <model.gguf> [options]

Example:
    python scripts/bench-mtp-sweep.py -m I:\\models\\qwen3-30b-iq3.gguf \\
        --kvmem -c 16384 -ngl 99 -n 256

Notes:
  * Uses llama-kvmem-cli and parses its "KVMEM_PERF ... gen_toks=" line.
  * The first run of each cell is a warm-up and is discarded: both CUDA graph
    capture and KVMem pool occupancy settle during the first tokens, so an
    unwarmed measurement reads far lower than steady state.
  * Stop any running llama-kvmem-server before benchmarking; it will otherwise
    contend for GPU memory and skew everything.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile

DEFAULT_EXE = r"I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll\bin\llama-kvmem-cli.exe"

# Authoritative throughput line is KVMEM_GEN_WALL ("n=48 ms=... toks=39.64").
# The KVMEM_PERF line reports gen_n=1 gen_toks=0 on this path and is useless
# for measurement, but keep it as a fallback / cross-check.
WALL_RE = re.compile(
    r"KVMEM_GEN_WALL\b.*?\bn=(\d+)\s+ms=([\d.]+)\s+toks=([\d.]+)",
    re.DOTALL,
)
PERF_RE = re.compile(
    r"KVMEM_PERF\b.*?\bgen_n=(\d+)\s+gen_ms=([\d.]+)\s+gen_toks=([\d.]+)",
    re.DOTALL,
)

# A prompt long enough that KVMem actually engages its tier machinery. Repeated
# but varied text, so the retrieval path has something to distinguish.
BASE_PROMPT = (
    "The following is a long technical discussion about distributed key-value "
    "caching for large language model inference. "
)


def build_prompt(target_tokens):
    tail = (
        " We examine block sizes, tier promotion between GPU memory, host RAM "
        "and NVMe, harvest pipelines for KV capture, retrieval scoring by "
        "mean-key probes, and the trade-offs each of these introduces for "
        "decode throughput and prefill latency."
    )
    paras = []
    n = 0
    i = 0
    while n < target_tokens:
        para = "Section " + str(i) + ": " + BASE_PROMPT + tail
        paras.append(para)
        n += len(para.split())
        i += 1
    return "\n\n".join(paras)


def run_once(exe, args, timeout):
    proc = subprocess.run(
        [exe] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = WALL_RE.search(blob)
    if m:
        n_tok, ms, tps = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if n_tok > 0 and ms > 0.0 and tps > 0.0:
            return tps, None
    # Fallback: the PERF line, only if it actually reports a non-zero sample.
    m = PERF_RE.search(blob)
    if m and int(m.group(1)) > 0 and float(m.group(3)) > 0.0:
        return float(m.group(3)), None
    tail = "\n".join(blob.strip().splitlines()[-6:])
    return None, "no KVMEM_GEN_WALL/KVMEM_PERF in output. tail:\n" + tail


def sweep_cell(exe, base, n_max, p_min, repeats, timeout):
    args = list(base)
    if n_max is None:
        args += ["--spec-type", "none"]
    else:
        args += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(n_max),
            "--spec-draft-p-min", str(p_min),
        ]
    # Warm-up (discarded): settles CUDA graph capture and KVMem pool state.
    run_once(exe, args, timeout)
    samples = []
    for _ in range(repeats):
        tps, err = run_once(exe, args, timeout)
        if tps is not None:
            samples.append(tps)
    if not samples:
        return None, err or "all runs failed"
    return statistics.median(samples), None


def main():
    ap = argparse.ArgumentParser(description="Sweep KVMem MTP parameters")
    ap.add_argument("-m", "--model", required=True, help="path to GGUF model")
    ap.add_argument("--exe", default=DEFAULT_EXE, help="llama-kvmem-cli path")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--ctx-size", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--ubatch-size", type=int, default=512)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--prompt-tokens", type=int, default=1200,
                    help="approximate prompt length to generate")
    ap.add_argument("--prompt-file", default=None,
                    help="use this prompt file instead of a synthetic one")
    ap.add_argument("--n-max", default="0,1,2,3,4,5,6,8",
                    help="comma list; 0 means --spec-type none (baseline)")
    ap.add_argument("--p-min", default="0.0,0.5,0.75,0.9", help="comma list")
    ap.add_argument("--repeats", type=int, default=2,
                    help="measured runs per cell (warm-up excluded)")
    ap.add_argument("--kvmem", action="store_true", help="enable KVMem slot pool")
    ap.add_argument("--extra", default="", help="extra raw args appended verbatim")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-run timeout in seconds")
    args = ap.parse_args()

    if not os.path.isfile(args.exe):
        sys.exit("cli not found: %s (build first)" % args.exe)

    if args.prompt_file:
        prompt_file = args.prompt_file
    else:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                          encoding="utf-8")
        tmp.write(build_prompt(args.prompt_tokens))
        tmp.close()
        prompt_file = tmp.name

    base = [
        "-m", args.model,
        "-f", prompt_file,
        "-n", str(args.n_predict),
        "-c", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ngl", str(args.n_gpu_layers),
    ]
    if args.kvmem:
        base.append("--kvmem")
    if args.extra:
        base += args.extra.split()

    n_max_list = [int(x) for x in args.n_max.split(",")]
    p_min_list = [float(x) for x in args.p_min.split(",")]

    print("model      : %s" % args.model)
    print("kvmem      : %s" % ("on" if args.kvmem else "off"))
    print("n_predict  : %d   ctx: %d" % (args.n_predict, args.ctx_size))
    print("repeats    : %d (plus 1 discarded warm-up)" % args.repeats)
    print()

    results = []
    baseline = None
    for n_max in n_max_list:
        if n_max == 0:
            tps, err = sweep_cell(args.exe, base, None, 0.0, args.repeats, args.timeout)
            if tps is None:
                print("baseline (no MTP) : FAILED -- %s" % err)
            else:
                baseline = tps
                print("baseline (no MTP) : %8.2f tok/s" % tps)
            print()
            continue

        for p_min in p_min_list:
            tps, err = sweep_cell(args.exe, base, n_max, p_min, args.repeats,
                                  args.timeout)
            label = "n_max=%-2d p_min=%-4s" % (n_max, p_min)
            if tps is None:
                print("%s : FAILED -- %s" % (label, err))
                continue
            gain = ""
            if baseline:
                gain = "   (%+.1f%% vs baseline)" % ((tps / baseline - 1.0) * 100.0)
            print("%s : %8.2f tok/s%s" % (label, tps, gain))
            results.append((tps, n_max, p_min))
        print()

    if not results:
        sys.exit("no successful measurements")

    results.sort(reverse=True)
    print("=" * 62)
    print("ranking (decode tok/s):")
    for tps, n_max, p_min in results:
        print("  %8.2f   n_max=%-2d p_min=%-4s" % (tps, n_max, p_min))
    best = results[0]
    print()
    print("best: --spec-draft-n-max %d --spec-draft-p-min %s   (%.2f tok/s)"
          % (best[1], best[2], best[0]))
    if baseline:
        print("      vs no-MTP baseline %.2f tok/s  ->  %+.1f%%"
              % (baseline, (best[0] / baseline - 1.0) * 100.0))


if __name__ == "__main__":
    main()
