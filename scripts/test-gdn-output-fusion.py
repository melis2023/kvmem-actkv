#!/usr/bin/env python3
"""Compare optional GDN output fusion with the current two-kernel path over HTTP."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextlib.contextmanager
def server(args, mode, label, correctness=False):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = args.output / f"{label}-{mode}.log"
    command = [str(args.server), "-m", str(args.model), "--device", "CUDA0", "--split-mode", "none",
               "-c", "4096", "-b", "1024" if correctness else "512", "-ub", "256" if correctness else "512",
               "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-kv-dtype", "f16",
               "--kvmem-mtp-state", args.mtp_state, "--kvmem-method", "retrieval" if correctness else "recency",
               "--kvmem-budget", "1024" if correctness else "4096", "--no-ui",
               "--host", "127.0.0.1", "--port", str(port)]
    if correctness:
        command += ["--kvmem-conversations", "2", "--kvmem-session-ram-gb", "1", "--kvmem-trace", "--verbosity", "4"]
    else:
        command += ["--no-kvmem-trace"]
    environment = dict(os.environ, KVMEM_GDN_OUT_FUSION="0",
                       KVMEM_GDN_OUT_FUSION_TRACE="1" if correctness else "0")
    if mode == "fused":
        environment.pop("KVMEM_GDN_OUT_FUSION", None)
    environment.pop("KVMEM_PERF", None)
    environment.pop("KVMEM_TRACE", None)
    if correctness:
        environment["KVMEM_TRACE"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=environment,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
                try:
                    urllib.request.urlopen(url + "/health", timeout=1).close()
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"server startup timeout; see {log_path}")
                    time.sleep(0.3)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def request(url, messages, reset=True, count=64, conversation="", stream=False):
    body = {"model": "kvmem", "messages": messages, "cache_reset": reset, "max_tokens": count,
            "temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0, "seed": 1, "repeat_penalty": 1,
            "presence_penalty": 0, "frequency_penalty": 0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    if conversation:
        body["kvmem"] = {"conversation_id": conversation}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    response = urllib.request.urlopen(req, timeout=180)
    if stream:
        return response
    with response:
        return json.load(response)


def user(text):
    return [{"role": "user", "content": text}]


def correctness(args):
    results = {}
    for mode in ("baseline", "fused"):
        records = {}
        with server(args, mode, "correctness", correctness=True) as url:
            cases = {
                "english": "List integers from 1 to 40, separated by spaces.",
                "chinese": "请用中文说明前缀缓存如何加速连续对话，列出四个要点。",
                "code": "Write a Python function that implements an LRU cache with capacity 3. Include get and put.",
                "long": "Cache notes:\n" + "Keep recent tokens. Code: cache.push(token); reuse(prefix).\n" * 170 +
                        "\nSummarize the notes in five numbered items.",
            }
            for name, text in cases.items():
                response = request(url, user(text), conversation=name)
                records[name] = response
                print(f"correctness {mode} {name}: prompt={response['timings']['prompt_n']} "
                      f"generated={response['timings']['predicted_n']}", flush=True)
            # Distinct long prefixes keep inferred matching from merging two short conversations.
            text_a = "Conversation A notes: preserve the integer ordering.\n" * 20 + cases["english"]
            text_b = "会话 B 的主题是缓存管理，请保留这些中文背景。\n" * 15 + cases["chinese"]
            first = request(url, user(text_a), count=32, conversation="A")
            messages = user(text_a) + [first["choices"][0]["message"],
                        {"role": "user", "content": "Now list the same integers in reverse order."}]
            request(url, user(text_b), count=32, conversation="B")
            records["swapped_continuation"] = request(url, messages, reset=False, count=32, conversation="A")
            if records["swapped_continuation"]["timings"]["cache_n"] < 32:
                raise RuntimeError("continuation did not exercise prefix reuse")
            # Regenerate the last turn from its retained pre-generation checkpoint.
            records["rewind"] = request(url, messages, reset=False, count=32, conversation="A")
            if records["rewind"]["timings"]["cache_n"] <= 0:
                raise RuntimeError("rewind did not exercise checkpoint reuse")
            with request(url, user("List integers from 1 to 300."), count=200, conversation="cancel", stream=True) as stream:
                for line in stream:
                    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                        chunk = json.loads(line[6:])
                        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                            break
            time.sleep(0.5)
            records["after_cancel"] = request(url, user(cases["code"]), conversation="cancel")
        log = (args.output / f"correctness-{mode}.log").read_text(encoding="utf-8", errors="replace")
        records["spec_stats"] = re.findall(r"spec_stats n_gen=(\d+) n_drafted=(\d+) n_accept=(\d+) n_restore=(\d+)", log)
        records["fusion_hits"] = log.count("KVMEM_GDN_OUT_FUSION name=")
        if mode == "fused" and records["fusion_hits"] == 0:
            raise RuntimeError("GDN output fusion did not run")
        if not any(int(row[1]) > int(row[2]) for row in records["spec_stats"]):
            raise RuntimeError("test did not exercise rejected drafts")
        if not re.search(r"store_select .*switched=1 restaged=1", log):
            raise RuntimeError("test did not restore an inactive conversation")
        if not any(0 < int(row[0]) < 32 for row in records["spec_stats"]):
            raise RuntimeError("test did not cancel during speculative generation")
        results[mode] = records
        save(args.output / "correctness.json", results)
    for name in results["baseline"]:
        if name in ("spec_stats", "fusion_hits"):
            continue
        a, b = results["baseline"][name], results["fused"][name]
        if a["choices"][0]["message"] != b["choices"][0]["message"] or a["usage"] != b["usage"]:
            raise RuntimeError(f"baseline/fused result mismatch: {name}")
    print("correctness: matching outputs and usage; cache restore, rewind, cancellation and rejected drafts passed", flush=True)


def benchmark(args):
    prompts = {
        "short": "Explain prefix caching in detail with ten numbered items.",
        "medium": "Cache notes:\n" + "Keep recent tokens. Code: cache.push(token); reuse(prefix).\n" * 24 +
                  "\nExplain prefix caching in detail with ten numbered items.",
        "long": "Cache notes:\n" + "Keep recent tokens. Code: cache.push(token); reuse(prefix).\n" * 106 +
                "\nExplain prefix caching in detail with ten numbered items.",
    }
    rows = []
    for round_id in range(args.rounds):
        modes = ("baseline", "fused") if round_id % 2 == 0 else ("fused", "baseline")
        for mode in modes:
            with server(args, mode, f"bench-{round_id}") as url:
                for text in prompts.values():
                    request(url, user(text), count=96)
                for repeat in range(args.repeats):
                    names = list(prompts)
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    for name in order:
                        response = request(url, user(prompts[name]), count=96)
                        timings = response["timings"]
                        if timings["cache_n"] != 0:
                            raise RuntimeError("benchmark unexpectedly reused a prefix")
                        record = {"round": round_id, "mode": mode, "repeat": repeat, "case": name,
                                  "timings": timings, "message": response["choices"][0]["message"], "usage": response["usage"]}
                        rows.append(record)
                        save(args.output / "benchmark.json", rows)
                        print(f"bench {round_id} {mode} {name}: prefill={timings['prompt_ms']:.2f} ms "
                              f"decode={timings['predicted_per_second']:.2f} tok/s", flush=True)
    summary = []
    for name in prompts:
        cases = {mode: [row for row in rows if row["mode"] == mode and row["case"] == name] for mode in ("baseline", "fused")}
        reference = cases["baseline"][0]
        if any(row["message"] != reference["message"] or row["usage"] != reference["usage"] for records in cases.values() for row in records):
            raise RuntimeError(f"benchmark outputs were not identical: {name}")
        timings = {mode: {field: statistics.median(row["timings"][field] for row in records)
                           for field in ("prompt_ms", "predicted_per_second", "predicted_ms", "predicted_n", "prompt_n")}
                   for mode, records in cases.items()}
        item = {"case": name, "baseline": timings["baseline"], "fused": timings["fused"],
                "prefill_gain": timings["baseline"]["prompt_ms"] / timings["fused"]["prompt_ms"] - 1,
                "decode_gain": timings["fused"]["predicted_per_second"] / timings["baseline"]["predicted_per_second"] - 1}
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    save(args.output / "summary.json", summary)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=root / "build-win/bin/llama-kvmem-server.exe")
    parser.add_argument("--model", type=Path, default=root.parent / "models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
    parser.add_argument("--output", type=Path, default=root / "artifacts/gdn-output-fusion")
    parser.add_argument("--phase", choices=("correctness", "benchmark", "all"), default="all")
    parser.add_argument("--mtp-state", choices=("replay", "snapshots"), default="replay")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.server, args.model, args.output = args.server.resolve(), args.model.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "config.json", {"server": str(args.server), "model": str(args.model),
                                      "patch_sha256": hashlib.sha256((root / "patches/gdn-output-fusion.patch").read_bytes()).hexdigest(),
                                      "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rounds": args.rounds,
                                      "repeats": args.repeats, "mtp_state": args.mtp_state})
    if args.phase in ("correctness", "all"):
        correctness(args)
    if args.phase in ("benchmark", "all"):
        benchmark(args)


if __name__ == "__main__":
    main()
