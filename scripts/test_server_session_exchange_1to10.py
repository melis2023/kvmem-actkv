"""Run the 10/15/20 GiB session exchange scenario at 1:10 on a real server.

The test owns only the server it starts. Stop other workloads on the selected
GPU before running it. The companion transfer-test --scale-1to10 forces the
exchange route with bounded RAM; this server run checks model/KV integration.
"""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid


GIB = 1 << 30
CODES = {"A": "ALPHA-731", "B": "BETA-118"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="CUDA GPU UUID, e.g. GPU-... for RTX 5060 Ti")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=2, help="additional A/B restores after the first pair")
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--keep-cache", action="store_true", help="retain up to 2 GiB of private snapshots")
    args = parser.parse_args()
    if args.cycles < 0 or args.ctx_size < 16384:
        parser.error("cycles must be nonnegative and ctx-size at least 16384")
    server, model = args.server.resolve(strict=True), args.model.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ("cache-" + uuid.uuid4().hex[:12])
    cache.mkdir()
    log_path = out / "server.log"
    checks, turns = [], []
    histories = {name: [{"role": "system", "content": "Remember each channel's code. Answer with the code only."}]
                 for name in CODES}
    next_line = dict.fromkeys(CODES, 0)

    def check(label, condition):
        checks.append({"name": label, "pass": bool(condition)})
        print(("PASS " if condition else "FAIL ") + label, flush=True)
        if not condition:
            raise AssertionError(label)

    def call(base, path, body=None):
        request = urllib.request.Request(base + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def counters(base):
        status, data = call(base, "/slots")
        check("/slots available", status == 200)
        return data[0]["kvmem"]["conversations"]

    def turn(base, name, content, label):
        history = histories[name]
        history.append({"role": "user", "content": content})
        status, result = call(base, "/v1/chat/completions", {
            "messages": history, "max_tokens": 16, "temperature": 0,
            "reasoning_effort": "none", "kvmem": {"conversation_id": name},
        })
        check(label + " HTTP 200", status == 200)
        answer = result["choices"][0]["message"]["content"] or ""
        check(label + " isolated answer", CODES[name] in answer and
              all(code not in answer for other, code in CODES.items() if other != name))
        history.append({"role": "assistant", "content": answer})
        stats = counters(base)
        check(label + " quota and session count", stats["disk_bytes"] <= stats["disk_bytes_max"]
              and stats["count"] <= 3)
        turns.append({"label": label, "name": name, "answer": answer,
                      "usage": result["usage"], "counters": stats})
        return turns[-1]

    def lines(name, count):
        start = next_line[name]
        next_line[name] += count
        return "\n".join(f"Channel {name} note {i:05d}: routing entry {i} is stable and requires no action."
                         for i in range(start, start + count))

    def grow(base, name, target_gib):
        target = int(target_gib * GIB)
        previous_bytes = previous_lines = 0
        for attempt in range(9):
            if attempt == 0:
                count = 240
            else:
                delta = measured - previous_bytes
                slope = delta / previous_lines if previous_lines and delta > 0 else measured / next_line[name]
                count = max(8, min(150, int(0.72 * (target - measured) / max(1, slope))))
            previous_bytes, previous_lines = (measured if attempt else 0), count
            note = lines(name, count)
            item = turn(base, name, note + f"\nThe {name} code is {CODES[name]}. What is the {name} code?",
                        f"grow {name} {attempt}")
            measured = item["counters"]["bytes"]
            if attempt:
                check(f"grow {name} reuses prefix {attempt}",
                      item["usage"]["prompt_cache_hit_tokens"] > 1024)
            check(f"grow {name} stays within context {attempt}",
                  item["usage"]["prompt_tokens"] < args.ctx_size - 512)
            if measured >= target * 0.94:
                check(f"{name} near {target_gib:g} GiB", measured <= target * 1.06)
                return measured
        raise AssertionError(f"{name} did not reach {target_gib:g} GiB after 9 turns")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
               "-dev", "CUDA0", "-ngl", str(args.gpu_layers), "-c", str(args.ctx_size),
               "-n", "16", "--reasoning-effort", "none", "--temp", "0", "--no-ui",
               "--kv-dtype", "q8_0", "-fa", "on", "--kvmem-budget", "2048",
               "--kvmem-gen-reserve", "512", "--spec-type", "draft-mtp",
               "--spec-draft-n-max", "3", "--kvmem-mtp-state", "replay",
               "--kvmem-conversations", "3", "--kvmem-session-ram-gb", "1.2",
               "--kvmem-session-nvme-gb", "2", "--kvmem-session-cache-dir", str(cache)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", KVMEM_TRACE="1")
    report = {"scale": "1:10", "a_target_bytes": GIB, "b_target_bytes": int(1.5 * GIB),
              "ram_soft_bytes": int(1.2 * GIB), "disk_quota_bytes": 2 * GIB,
              "gpu": args.gpu, "command": command, "checks": checks, "turns": turns}
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"server exited; see {log_path}")
                    try:
                        if call(base, "/health")[0] == 200:
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("server did not become ready")
                    time.sleep(0.5)
                a_bytes = grow(base, "A", 1.0)
                b_bytes = grow(base, "B", 1.5)
                check("active B exceeds RAM soft cap", b_bytes > int(1.2 * GIB))
                check("A snapshot is on disk", turns[-1]["counters"]["disk_bytes"] > 0)
                check("no cache evictions while building", turns[-1]["counters"]["evictions"] == 0)
                for cycle in range(args.cycles + 1):
                    for name in ("A", "B"):
                        item = turn(base, name, f"Recall the {name} code from this conversation. Code only.",
                                    f"restore {cycle} {name}")
                        check(f"restore {cycle} {name} reuses KV",
                              item["usage"]["prompt_cache_hit_tokens"] > 1024)
                        check(f"restore {cycle} {name} preserves both sessions",
                              item["counters"]["evictions"] == 0 and item["counters"]["count"] == 3)
                        if name == "B":
                            check(f"restore {cycle} B active over soft cap",
                                  item["counters"]["bytes"] > item["counters"]["bytes_max"])
                trace = log_path.read_text(encoding="utf-8", errors="replace")
                routes = re.findall(r"session_transfer target=\d+ path=(\w+)", trace)
                check("real service spills and restores", "session_spill" in trace and "session_restore" in trace)
                check("disk constrained switch uses target-file credit",
                      "ram_first" in routes or "exchange" in routes)
                check("no unexpected cache-clear fallback", "clearing completed cache" not in trace)
                report.update(a_measured_bytes=a_bytes, b_measured_bytes=b_bytes, routes=routes)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.keep_cache and cache.resolve().is_relative_to(out):
            for run in cache.glob("run-*"):
                if run.is_symlink() or not run.resolve().is_relative_to(cache):
                    continue
                for file in run.iterdir():
                    if file.is_file() and not file.is_symlink() and re.fullmatch(r"\d+-\d+\.(kv|tmp)", file.name):
                        file.unlink()
                try:
                    run.rmdir()
                except OSError:
                    pass
            try:
                cache.rmdir()
            except OSError:
                pass
    print(f"1:10 real-server stability test passed: {len(checks)} checks; report: {out / 'result.json'}")


if __name__ == "__main__":
    main()
