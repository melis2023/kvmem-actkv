"""Exercise three ~5 GiB IQ3 K8/V4 sessions with a 10 GiB NVMe quota.

The hot phase switches A/B/C twice and checks answer isolation, KV reuse and
quota accounting. A fresh server then prefills each exact first-restore prompt
from zero, so hot and cold HTTP wall times can be compared fairly.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid


GIB = 1 << 30
CODES = {"A": "ALPHA-731", "B": "BETA-118", "C": "GAMMA-492"}


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_ready(process, base, log_path):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited ({process.returncode}); see {log_path}")
        try:
            if request(base, "/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server startup; see {log_path}")


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-gib", type=float, default=4.82)
    parser.add_argument("--disk-gib", type=float, default=10.0)
    parser.add_argument("--ram-gib", type=float, default=5.0)
    parser.add_argument("--ctx-size", type=int, default=262144)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    if not (0 < args.target_gib < args.ram_gib and
            2 * args.target_gib < args.disk_gib and args.cycles >= 1):
        parser.error("target must fit RAM and two inactive sessions must fit disk")
    server, model = args.server.resolve(strict=True), args.model.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ("cache-" + uuid.uuid4().hex[:12])
    cache.mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", KVMEM_TRACE="1")
    report = {"model": str(model), "gpu": args.gpu, "kv_k": "q8_0", "kv_v": "q4_0",
              "target_gib": args.target_gib, "ram_gib": args.ram_gib,
              "disk_gib": args.disk_gib, "ctx_size": args.ctx_size,
              "checks": [], "turns": [], "cold": []}
    histories = {name: [{"role": "system", "content":
                 "Remember each channel's secret code. Answer with only that channel's code."}]
                 for name in CODES}
    next_line = dict.fromkeys(CODES, 0)
    first_restore = {}
    first_messages = {}

    def check(label, condition):
        report["checks"].append({"name": label, "pass": bool(condition)})
        print(("PASS " if condition else "FAIL ") + label, flush=True)
        if not condition:
            raise AssertionError(label)

    def command(port, directory):
        return [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
                "-dev", "CUDA0", "-ngl", "99", "-c", str(args.ctx_size),
                "-n", "16", "--reasoning-effort", "none", "--temp", "0", "--no-ui",
                "-ctk", "q8_0", "-ctv", "q4_0", "-fa", "on",
                "--kvmem-budget", "2048", "--kvmem-gen-reserve", "512",
                "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                "--kvmem-mtp-state", "replay", "--kvmem-conversations", "3",
                "--kvmem-session-ram-gb", str(args.ram_gib),
                "--kvmem-session-nvme-gb", str(args.disk_gib),
                "--kvmem-session-cache-dir", str(directory)]

    def session_stats(base):
        status, data = request(base, "/slots")
        check("/slots available", status == 200)
        return data[0]["kvmem"]["conversations"]

    def turn(base, name, content, label):
        history = histories[name]
        history.append({"role": "user", "content": content})
        if label.startswith("restore 0 "):
            first_messages[name] = copy.deepcopy(history)
        start = time.perf_counter()
        status, response = request(base, "/v1/chat/completions", {
            "messages": history, "max_tokens": 16, "temperature": 0,
            "reasoning_effort": "none", "kvmem": {"conversation_id": name},
        })
        wall_ms = (time.perf_counter() - start) * 1000
        check(label + " HTTP 200", status == 200)
        answer = response["choices"][0]["message"]["content"] or ""
        check(label + " isolated answer", CODES[name] in answer and
              all(code not in answer for other, code in CODES.items() if other != name))
        history.append({"role": "assistant", "content": answer})
        stats = session_stats(base)
        check(label + " quota/count/disk errors",
              stats["disk_bytes"] <= stats["disk_bytes_max"] and
              stats["disk_bytes_max"] == int(args.disk_gib * GIB) and
              stats["count"] <= 3 and stats["disk_errors"] == 0)
        item = {"label": label, "name": name, "answer": answer,
                "wall_ms": round(wall_ms, 2), "usage": response["usage"], "counters": stats}
        report["turns"].append(item)
        print(f"{label}: RAM={stats['bytes']/GIB:.3f} GiB "
              f"disk={stats['disk_bytes']/GIB:.3f} GiB "
              f"tokens={item['usage']['prompt_tokens']} wall={wall_ms/1000:.2f}s", flush=True)
        return item

    def grow(base, name):
        target = args.target_gib * GIB
        measured = previous_bytes = previous_lines = 0
        for attempt in range(16):
            if attempt == 0:
                count = 240
            else:
                gain = measured - previous_bytes
                slope = gain / previous_lines if gain > 0 else measured / next_line[name]
                count = max(10, min(1600, int(0.78 * (target - measured) / max(1, slope))))
            previous_bytes, previous_lines = measured, count
            start = next_line[name]
            next_line[name] += count
            note = "\n".join(
                f"Channel {name} note {i:05d}: routing entry {i} is stable and requires no action."
                for i in range(start, start + count))
            item = turn(base, name,
                        note + f"\nThe {name} code is {CODES[name]}. What is the {name} code?",
                        f"grow {name} {attempt}")
            measured = item["counters"]["bytes"]
            if attempt:
                check(f"grow {name} prefix reuse {attempt}",
                      item["usage"]["prompt_cache_hit_tokens"] > 1024)
            check(f"grow {name} context {attempt}",
                  item["usage"]["prompt_tokens"] < args.ctx_size - 512)
            if measured >= target * 0.975:
                check(f"{name} near {args.target_gib:g} GiB",
                      measured <= target * 1.03)
                return measured
        raise AssertionError(f"{name} did not reach {args.target_gib} GiB")

    def launch_and_run(port, directory, log_path, callback):
        base = f"http://127.0.0.1:{port}"
        cmd = command(port, directory)
        with log_path.open("wb") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                wait_ready(process, base, log_path)
                return callback(base)
            finally:
                stop(process)

    try:
        port = free_port()
        report["command"] = command(port, cache)

        def hot_phase(base):
            report["measured_bytes"] = {name: grow(base, name) for name in CODES}
            stats = session_stats(base)
            check("three sessions retained", stats["count"] == 3)
            check("two inactive sessions on NVMe",
                  stats["disk_bytes"] > 2 * args.target_gib * 0.93 * GIB)
            for cycle in range(args.cycles):
                for name in CODES:
                    label = f"restore {cycle} {name}"
                    item = turn(base, name, f"Recall the {name} code. Code only.", label)
                    check(label + " KV reuse",
                          item["usage"]["prompt_cache_hit_tokens"] > 10000)
                    check(label + " all sessions retained",
                          item["counters"]["count"] == 3 and
                          item["counters"]["disk_bytes"] >
                          2 * args.target_gib * 0.93 * GIB)
                    if cycle == 0:
                        first_restore[name] = item

        log_path = out / "hot-server.log"
        launch_and_run(port, cache, log_path, hot_phase)
        trace = log_path.read_text(encoding="utf-8", errors="replace")
        restores = [float(value) for value in re.findall(
            r"session_restore id=\d+ rows=\d+ ms=([0-9.]+)", trace)]
        routes = re.findall(r"session_transfer target=\d+ path=(\w+)", trace)
        evicted_rows = [int(value) for value in re.findall(
            r"store_evict id=\d+ rows=(\d+)", trace)]
        check(f"{args.cycles * 3} real NVMe restores logged",
              len(restores) == args.cycles * 3)
        check("real disk transfers logged", "session_spill" in trace and bool(routes))
        check("no populated session evicted", all(rows == 0 for rows in evicted_rows))
        check("no cache-clear fallback", "clearing completed cache" not in trace)
        report["restore_ms"] = restores
        report["routes"] = routes
        report["evicted_rows"] = evicted_rows
        for name, messages in first_messages.items():
            (out / f"cold-messages-{name}.json").write_text(
                json.dumps(messages, ensure_ascii=False), encoding="utf-8")

        for name in CODES:
            cold_cache = out / ("cold-cache-" + name + "-" + uuid.uuid4().hex[:8])
            cold_cache.mkdir()
            port = free_port()

            def cold_phase(base):
                start = time.perf_counter()
                status, response = request(base, "/v1/chat/completions", {
                    "messages": first_messages[name], "max_tokens": 16,
                    "temperature": 0, "reasoning_effort": "none",
                    "kvmem": {"conversation_id": name},
                })
                wall_ms = (time.perf_counter() - start) * 1000
                check(f"cold {name} HTTP 200", status == 200)
                answer = response["choices"][0]["message"]["content"] or ""
                usage = response["usage"]
                check(f"cold {name} isolated answer", CODES[name] in answer and
                      all(code not in answer for other, code in CODES.items() if other != name))
                check(f"cold {name} exact tokens and zero hits",
                      usage["prompt_tokens"] == first_restore[name]["usage"]["prompt_tokens"] and
                      usage["prompt_cache_hit_tokens"] == 0)
                result = {"name": name, "prompt_tokens": usage["prompt_tokens"],
                          "cache_hit_tokens": usage["prompt_cache_hit_tokens"],
                          "wall_ms": round(wall_ms, 2),
                          "hot_wall_ms": first_restore[name]["wall_ms"],
                          "speedup": round(wall_ms / first_restore[name]["wall_ms"], 3)}
                report["cold"].append(result)
                print(f"cold {name}: {wall_ms/1000:.2f}s vs hot "
                      f"{result['hot_wall_ms']/1000:.2f}s; {result['speedup']:.2f}x", flush=True)

            cold_log = out / f"cold-{name}.log"
            try:
                launch_and_run(port, cold_cache, cold_log, cold_phase)
                matches = re.findall(r"prompt eval time =\s*([0-9.]+) ms",
                                     cold_log.read_text(encoding="utf-8", errors="replace"))
                check(f"cold {name} prefill timing logged", len(matches) == 1)
                report["cold"][-1]["prompt_eval_ms"] = float(matches[0])
            finally:
                shutil.rmtree(cold_cache)
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        shutil.rmtree(cache)
    print(f"Three-session 5 GiB test passed: {len(report['checks'])} checks; "
          f"report: {out / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
