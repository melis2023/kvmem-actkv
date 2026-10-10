"""Real-model session API, concurrent-request and shared-cache restart regression.

Uses only the supplied model/GPU and temporary servers owned by this invocation.
No downloads. Unlike the portable lifecycle test, every cache here is produced
and restored by llama-kvmem-server while answering real HTTP requests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--cycles", type=int, default=8)
    args = parser.parse_args()
    args.server = args.server.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ("cache-" + uuid.uuid4().hex)
    cache.mkdir()
    sentinel = cache / "user-file"
    sentinel.write_text("keep")
    report = {"checks": [], "turns": [], "commands": []}
    servers = []
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", KVMEM_TRACE="1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    codes = {"A": "ALPHA-731", "B": "BETA-118", "C": "GAMMA-905", "D": "DELTA-426"}

    def check(name, ok):
        report["checks"].append({"name": name, "pass": bool(ok)})
        print(("PASS " if ok else "FAIL ") + name, flush=True)
        if not ok:
            raise AssertionError(name)

    def request(server, path, body=None, stream=False):
        req = urllib.request.Request(server["base"] + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=180) as response:
                raw = response.read()
                return response.status, raw.decode() if stream else json.loads(raw)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def launch(label):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        command = [str(args.server), "-m", str(args.model), "--host", "127.0.0.1", "--port", str(port),
                   "-dev", "CUDA0", "-ngl", "99", "-c", "16384", "-n", "32", "--no-ui",
                   "--reasoning-effort", "none", "--temp", "0", "--threads-http", "4", "-fa", "on",
                   "--kv-dtype", "q8_0", "--kvmem-budget", "2048", "--kvmem-gen-reserve", "512",
                   "--spec-type", "draft-mtp" if args.mtp else "none", "--kvmem-mtp-state", "snapshots",
                   "--kvmem-conversations", "8", "--kvmem-session-ram-gb", "0.05",
                   "--kvmem-session-nvme-gb", "2", "--kvmem-session-cache-dir", str(cache)]
        log_path = out / (label + ".log")
        log = log_path.open("wb")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        server = {"process": process, "log": log, "log_path": log_path, "base": f"http://127.0.0.1:{port}"}
        servers.append(server)
        report["commands"].append(command)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode}: {log_path}")
            try:
                if request(server, "/health")[0] == 200:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    match = re.search(r"session disk cache=(.+) quota=", text)
                    if not match:
                        raise RuntimeError("missing cache directory diagnostic")
                    server["run"] = Path(match.group(1))
                    return server
            except OSError:
                pass
            time.sleep(.25)
        raise TimeoutError(f"server startup: {log_path}")

    def kill(server):
        if server["process"].poll() is None:
            server["process"].kill()
        server["process"].wait(timeout=30)
        server["log"].close()

    def stats(server):
        return request(server, "/slots")[1][0]["kvmem"]["conversations"]

    def history(name):
        notes = "\n".join(f"Channel {name} note {i:03d}: routing entry {i} is stable and requires no action."
                          for i in range(180))
        return [{"role": "system", "content": "Read the notes and answer with the code only."},
                {"role": "user", "content": notes + f"\nThe {name} code is {codes[name]}. What is the {name} code? Code only."}]

    def turn(server, histories, name, label, client_id=None, stream=False, responses=False, reuse=None):
        messages = histories[name]
        if messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": f"Repeat the {name} code exactly, nothing else."})
        body = {"messages": messages, "max_tokens": 32, "temperature": 0, "reasoning_effort": "none"}
        if client_id is not None:
            body["kvmem"] = {"conversation_id": client_id}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        path = "/v1/chat/completions"
        if responses:
            path = "/v1/responses"
            body["input"] = body.pop("messages")
            body["max_output_tokens"] = body.pop("max_tokens")
            body["reasoning"] = {"effort": "none"}
        started = time.perf_counter()
        status, result = request(server, path, body, stream)
        check(label + " HTTP 200", status == 200)
        if stream:
            events = [json.loads(line[6:]) for line in result.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            if responses:
                completed = [event["response"] for event in events if event.get("type") == "response.completed"]
                check(label + " stream completed once", len(completed) == 1)
                result = completed[0]
            else:
                check(label + " stream DONE", result.count("data: [DONE]") == 1)
                result = {"usage": next(event["usage"] for event in events if event.get("usage")),
                          "choices": [{"message": {"content": "".join(
                              choice.get("delta", {}).get("content") or "" for event in events
                              for choice in event.get("choices", []))}}]}
        if responses:
            answer = "".join(part.get("text", "") for item in result["output"]
                             if item.get("type") == "message" for part in item.get("content", [])
                             if part.get("type") == "output_text")
            usage = result["usage"]
            hit = usage["input_tokens_details"]["cached_tokens"]
        else:
            answer = result["choices"][0]["message"]["content"] or ""
            usage = result["usage"]
            hit = usage["prompt_cache_hit_tokens"]
            check(label + " token accounting", usage["prompt_tokens"] == hit + usage["prompt_cache_miss_tokens"])
        check(label + " answer isolation", codes[name] in answer and
              all(code not in answer for other, code in codes.items() if other != name))
        if reuse is not None:
            check(label + (" prefix hit" if reuse else " cold miss"), hit > 1024 if reuse else hit == 0)
        messages.append({"role": "assistant", "content": answer})
        counters = stats(server)
        check(label + " quota and health", counters["count"] <= 8 and counters["disk_bytes"] <= 2 * 2**30
              and counters["disk_errors"] == 0 and request(server, "/health")[0] == 200)
        report["turns"].append({"label": label, "answer": answer, "usage": usage, "counters": counters,
                                "wall_ms": round((time.perf_counter() - started) * 1000, 2)})
        return hit

    try:
        first = launch("first")
        histories = {name: history(name) for name in codes}
        for name in "ABC":
            turn(first, histories, name, "initial " + name, client_id=name, reuse=False)
        check("idle sessions spilled", stats(first)["disk_bytes"] > 0)
        turn(first, histories, "A", "chat streaming restore", client_id="A", stream=True, reuse=True)
        for value in (42, "", "x" * 129, "not printable\n", "中文", "unknown-id"):
            turn(first, histories, "B", "ID fallback " + repr(value)[:30], client_id=value, reuse=True)
        turn(first, histories, "C", "Responses restore", responses=True, reuse=True)
        turn(first, histories, "A", "Responses streaming restore", responses=True, stream=True, reuse=True)

        # A bound id must not make an unrelated prompt use the old conversation.
        turn(first, histories, "D", "reused ID with different prompt", client_id="A", reuse=False)
        turn(first, histories, "A", "original A survives ID reassignment", reuse=True)

        before = stats(first)
        status, _ = request(first, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "overflow " * 24000}], "max_tokens": 32})
        check("oversized disk-mode request rejected", status == 400)
        after = stats(first)
        check("rejected request leaves all cache counters unchanged", before == after)

        # All requests are concurrent at HTTP level; the server owns one decode slot.
        for cycle in range(args.cycles):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(turn, first, histories, name, f"concurrent {cycle} {name}", reuse=True)
                           for name in "ABC"]
                for future in futures:
                    future.result()
        check("no session evicted during repeated switches", stats(first)["evictions"] == 0)

        # Disconnect mid-stream. Other parked sessions must remain available.
        cancel = {"messages": histories["A"] + [{"role": "user", "content": "Count from 1 to 1000, writing every number."}],
                  "max_tokens": 512, "stream": True, "temperature": 0, "reasoning_effort": "none"}
        req = urllib.request.Request(first["base"] + "/v1/chat/completions", data=json.dumps(cancel).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=180) as response:
            for raw in response:
                if raw.startswith(b"data: {"):
                    event = json.loads(raw[6:])
                    if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                        break
            else:
                raise AssertionError("no generated token before cancellation")
        turn(first, histories, "B", "B after streaming cancellation", reuse=True)
        turn(first, histories, "A", "A after streaming cancellation")

        second = launch("second-shared-root")
        check("second service preserves first run", first["run"].is_dir())
        peers = {name: history(name) for name in "CD"}
        for name in "CD":
            turn(second, peers, name, "peer initial " + name, reuse=False)
        turn(first, histories, "C", "first remains usable with shared root", reuse=True)
        old_run = first["run"]
        orphan_bytes = sum(path.stat().st_size for path in old_run.iterdir() if path.suffix in (".kv", ".tmp"))
        check("real server has orphanable disk payload", orphan_bytes > 0)
        peer_files = {path.name: path.stat().st_size for path in second["run"].glob("*.kv")}
        check("peer has live disk payload", bool(peer_files))
        kill(first)
        check("forced termination leaves cache", old_run.exists())
        restarted = launch("restart")
        check("restart removes dead server directory", not old_run.exists())
        check("restart preserves peer snapshots", peer_files ==
              {path.name: path.stat().st_size for path in second["run"].glob("*.kv")})
        text = restarted["log_path"].read_text(encoding="utf-8", errors="replace")
        check("cleanup diagnostic matches actual bytes and live peer",
              f"removed_runs=1 removed_bytes={orphan_bytes} active_runs=1" in text)
        turn(second, peers, "C", "peer restores after cleanup", reuse=True)
        fresh = {"A": history("A")}
        turn(restarted, fresh, "A", "restart recomputes rather than loading stale runtime state", reuse=False)
        check("user file survives all service startups", sentinel.read_text() == "keep")
        check("all test services healthy", all(request(server, "/health")[0] == 200 for server in (second, restarted)))
    finally:
        for server in servers:
            kill(server)
        (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Every process using this unique test root has been joined. This path is
        # created by this invocation, never supplied as an arbitrary cache root.
        if cache.parent == out and cache.name.startswith("cache-") and not cache.is_symlink():
            shutil.rmtree(cache)


if __name__ == "__main__":
    main()
