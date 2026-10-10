#!/usr/bin/env python3
"""Sweep KVMem n-gram self-speculation and compare against draft-mtp / off.

Unlike MTP, n-gram needs no draft model and no extra VRAM: the draft is a
verbatim copy of an earlier token run found in the target's own history. The
knobs are --spec-ngram-simple-size-n (lookup length) and --spec-ngram-simple-size-m (draft
length).

HARD CONSTRAINT (llama.cpp common/ngram-map.cpp, common_ngram_simple_draft):
    copy_max = min(size_m, cur_len - match_pos - size_n)
    if (copy_max < size_n) -> return {}      # drafts nothing, ever
so size_m must be >= size_n or n-gram silently does nothing (n_drafted == 0).

Usage:
    python scripts/bench-ngram-sweep.py -m <model.gguf> [options]

Runs are serial by design: one model instance at a time on a 16 GB card.
"""

import argparse
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "bench-mtp-sweep.py")

DEFAULT_EXE = r"I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll\bin\llama-kvmem-cli.exe"


def load_sweep():
    spec = importlib.util.spec_from_file_location("bench_mtp_sweep", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(exe, base, extra, timeout, diag=False):
    """Run once; return (tok_s, n_drafted, n_accept)."""
    env = None
    if diag:
        env = dict(os.environ, KVMEM_TRACE="1")
    proc = subprocess.run([exe] + base + extra, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    tps = None
    m = load_sweep().WALL_RE.search(blob)
    if m and int(m.group(1)) > 0 and float(m.group(2)) > 0.0:
        tps = float(m.group(3))
    drafted = accept = -1
    m2 = load_sweep().re.search(r"spec_stats\b.*?n_drafted=(\d+)\s+n_accept=(\d+)", blob)
    if m2:
        drafted, accept = int(m2.group(1)), int(m2.group(2))
    return tps, drafted, accept


def cell(exe, base, extra, repeats, timeout):
    # warm-up discarded (CUDA graph capture + KVMem pool settle)
    measure(exe, base, extra, timeout)
    samples, last = [], (-1, -1)
    for _ in range(repeats):
        tps, d, a = measure(exe, base, extra, timeout)
        if tps is not None:
            samples.append(tps)
        last = (d, a)
    if not samples:
        return None, last
    return statistics.median(samples), last


def main():
    ap = argparse.ArgumentParser(description="Sweep KVMem n-gram speculation")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--ubatch-size", type=int, default=512)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--prompt-tokens", type=int, default=810)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--size-n", default="12")
    ap.add_argument("--size-m", default="16,24,32,48")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if not os.path.isfile(args.exe):
        sys.exit("cli not found: %s" % args.exe)

    sweep = load_sweep()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(sweep.build_prompt(args.prompt_tokens))
    tmp.close()

    base = ["-m", args.model, "-f", tmp.name, "-n", str(args.n_predict),
            "-c", str(args.ctx_size), "-b", str(args.batch_size),
            "-ub", str(args.ubatch_size), "-ngl", str(args.n_gpu_layers)]

    rows = []
    off, _ = cell(args.exe, base, ["--spec-type", "none"], args.repeats, args.timeout)
    rows.append(("off (no spec)", off, "baseline"))
    if off:
        print("off            %6.2f tok/s  (baseline)" % off, flush=True)

    for sn in args.size_n.split(","):
        for sm in args.size_m.split(","):
            label = "ngram n=%s m=%s" % (sn, sm)
            tps, (d, a) = cell(args.exe, base,
                               ["--spec-type", "ngram-simple",
                                "--spec-ngram-simple-size-n", sn,
                                "--spec-ngram-simple-size-m", sm],
                               args.repeats, args.timeout)
            rows.append((label, tps, "drafted=%d accept=%d" % (d, a)))
            if tps:
                print("%-14s %6.2f tok/s  (%.2fx)  drafted=%d accept=%d"
                      % (label, tps, tps / off if off else 0.0, d, a), flush=True)
            else:
                print("%-14s FAILED" % label, flush=True)

    # n-gram first, MTP as fallback
    tps, (d, a) = cell(args.exe, base,
                       ["--spec-type", "ngram-simple,draft-mtp",
                        "--spec-ngram-simple-size-n", args.size_n.split(",")[0],
                        "--spec-ngram-simple-size-m", args.size_m.split(",")[0],
                        "--spec-draft-n-max", "4"],
                       args.repeats, args.timeout)
    rows.append(("ngram+mtp", tps, "drafted=%d accept=%d" % (d, a)))
    if tps:
        print("%-14s %6.2f tok/s  (%.2fx)  drafted=%d accept=%d"
              % ("ngram+mtp", tps, tps / off if off else 0.0, d, a), flush=True)

    # MTP alone for reference
    tps, (d, a) = cell(args.exe, base,
                       ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
                       args.repeats, args.timeout)
    rows.append(("draft-mtp n=4", tps, "drafted=%d accept=%d" % (d, a)))
    if tps:
        print("%-14s %6.2f tok/s  (%.2fx)  drafted=%d accept=%d"
              % ("draft-mtp n=4", tps, tps / off if off else 0.0, d, a), flush=True)

    print("\n=== summary ===")
    for label, tps, note in rows:
        if tps is None:
            print("%-16s FAILED  %s" % (label, note))
        else:
            print("%-16s %6.2f tok/s  %s" % (label, tps, note))


if __name__ == "__main__":
    main()
