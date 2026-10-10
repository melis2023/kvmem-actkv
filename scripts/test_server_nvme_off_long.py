"""Check the default single-session path with a long IQ3 K8/V4 prompt.

Use cold-messages-A.json from test_server_three_session_5g.py for a matched
prefill comparison. No conversation or NVMe session-cache flags are passed.
"""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request


def request(base, path, body=None):
    req = urllib.request.Request(base + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--messages", type=Path, required=True)
    p.add_argument("--expected-tokens", type=int, required=True)
    p.add_argument("--expected-code", default="ALPHA-731")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    server, model = args.server.resolve(strict=True), args.model.resolve(strict=True)
    messages = json.loads(args.messages.read_text(encoding="utf-8"))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(server), "-m", str(model), "--host", "127.0.0.1",
               "--port", str(port), "-dev", "CUDA0", "-ngl", "99",
               "-c", "262144", "-n", "16", "--reasoning-effort", "none",
               "--temp", "0", "--no-ui", "-ctk", "q8_0", "-ctv", "q4_0",
               "-fa", "on", "--kvmem-budget", "2048",
               "--kvmem-gen-reserve", "512", "--spec-type", "draft-mtp",
               "--spec-draft-n-max", "3", "--kvmem-mtp-state", "replay"]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", KVMEM_TRACE="1")
    report = {"command": command, "checks": [], "gpu": args.gpu,
              "expected_tokens": args.expected_tokens}

    def check(label, ok):
        report["checks"].append({"name": label, "pass": bool(ok)})
        print(("PASS " if ok else "FAIL ") + label, flush=True)
        if not ok:
            raise AssertionError(label)

    log_path = out / "server.log"
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
                        request(base, "/health")
                        break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("server did not become ready")
                    time.sleep(0.5)
                props = request(base, "/props")
                slots = request(base, "/slots")
                check("default single slot", props["total_slots"] == 1)
                check("no conversation capability advertised",
                      "conversations" not in props.get("kvmem", {}))
                check("no multi-session slot counters", "kvmem" not in slots[0])
                start = time.perf_counter()
                first = request(base, "/v1/chat/completions", {
                    "messages": messages, "max_tokens": 16, "temperature": 0,
                    "reasoning_effort": "none"})
                first_wall_ms = (time.perf_counter() - start) * 1000
                first_answer = first["choices"][0]["message"]["content"] or ""
                first_usage = first["usage"]
                check("cold answer correct", args.expected_code in first_answer)
                check("cold exact tokens and zero hits",
                      first_usage["prompt_tokens"] == args.expected_tokens and
                      first_usage["prompt_cache_hit_tokens"] == 0)
                print(f"default cold: {first_usage['prompt_tokens']} tokens, "
                      f"{first_wall_ms/1000:.2f}s", flush=True)
                followup = messages + [{"role": "assistant", "content": first_answer},
                    {"role": "user", "content": "Recall the A code. Code only."}]
                start = time.perf_counter()
                second = request(base, "/v1/chat/completions", {
                    "messages": followup, "max_tokens": 16, "temperature": 0,
                    "reasoning_effort": "none"})
                second_wall_ms = (time.perf_counter() - start) * 1000
                second_answer = second["choices"][0]["message"]["content"] or ""
                check("same-session follow-up answer correct", args.expected_code in second_answer)
                check("same-session prefix reuse",
                      second["usage"]["prompt_cache_hit_tokens"] > 100000)
                report.update(first_wall_ms=round(first_wall_ms, 2),
                              first_usage=first_usage,
                              second_wall_ms=round(second_wall_ms, 2),
                              second_usage=second["usage"])
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        trace = log_path.read_text(encoding="utf-8", errors="replace")
        check("no session disk transfer", "session_transfer" not in trace and
              "session_spill" not in trace and "session_restore" not in trace)
        check("no disk cache initialized", "session disk cache=" not in trace)
        evals = [float(value) for value in re.findall(
            r"prompt eval time =\s*([0-9.]+) ms", trace)]
        check("both prompt eval timings logged", len(evals) == 2)
        report["prompt_eval_ms"] = evals
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Default single-session path passed: {len(report['checks'])} checks", flush=True)


if __name__ == "__main__":
    main()
