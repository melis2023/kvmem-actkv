#!/usr/bin/env python3
"""Check whether n-gram self-speculation actually drafts under a given server config.

Starts llama-kvmem-server with KVMEM_TRACE=1, sends two requests (one highly
repetitive, one ordinary), then greps the log for KVMEM_TRACE spec_stats.
n_drafted == 0 means n-gram is silently doing nothing.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE = r"I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll\bin\llama-kvmem-server.exe"
MODEL = os.environ.get("NGC_MODEL", r"I:\model\Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf")
TEMPLATE = r"I:\templates\chat_template.jinja"
PORT = 8080
LOG = r"C:\Users\zhuan\AppData\Local\Temp\ngram_effect.log"
API_KEY = "sk-123"

# extra args from the user's launcher, everything after the model
BASE = [
    "-m", MODEL,
    "-lm", "none",
    "--jinja",
    "--chat-template-file", TEMPLATE,
    "--host", "127.0.0.1", "--port", str(PORT),
    "-c", "262144", "-n", "16384", "-b", "512", "-ngl", "99",
    "--kvmem-budget", "32768", "--kvmem-gen-reserve", "16384",
    "--kvmem-gpu-ratio", "0.8", "--kvmem-block-tokens", "32",
    "--kv-dtype", "q8_0",
]

REQUESTS = [
    ("repetitive", "Please repeat the following line five times, one per line:\n"
                   "KVMem keeps the KV cache on the GPU and spills cold blocks to RAM.\n"),
    ("ordinary", "用三句话解释一下什么是量子纠缠，不要用列表。"),
]


def ready():
    url = "http://127.0.0.1:%d/health" % PORT
    for _ in range(180):
        try:
            with opener().open(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def opener():
    # The sandbox exports http_proxy/https_proxy; urllib would route 127.0.0.1
    # through it and get 502. Bypass all proxies for the loopback server.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def chat(prompt, max_tokens=256):
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + API_KEY})
    t0 = time.time()
    with opener().open(req, timeout=600) as r:
        data = json.loads(r.read().decode())
    return data, time.time() - t0


def run(label, spec_args):
    print("\n" + "=" * 72)
    print("CONFIG %s: %s" % (label, " ".join(spec_args)))
    print("=" * 72)
    args = [EXE] + ([] if label.endswith("no-kvmem") else BASE) + spec_args + [
        "--enable-thinking", "--reasoning-effort", "medium", "--reasoning-budget", "8192",
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
        "--presence-penalty", "0.0", "--frequency-penalty", "0.0", "--repeat-penalty", "1.0",
        "--api-key", API_KEY,
    ]
    env = dict(os.environ)
    env["KVMEM_TRACE"] = "1"
    log = open(LOG, "wb")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        if not ready():
            print("SERVER DID NOT START; tail of log follows")
            proc.kill()
            log.close()
            print(open(LOG, "rb").read().decode("utf-8", "replace")[-3000:])
            return None
        print("server ready")
        for name, prompt in REQUESTS:
            try:
                data, dt = chat(prompt)
            except Exception as e:
                print("  %-11s REQUEST FAILED: %r" % (name, e))
                continue
            usage = data.get("usage", {})
            gen = usage.get("completion_tokens", 0)
            print("  %-11s gen=%-4d wall=%.2fs  -> %.2f tok/s"
                  % (name, gen, dt, gen / dt if dt > 0 else 0.0))
            txt = data["choices"][0]["message"].get("content") or ""
            print("             %s" % txt[:110].replace("\n", " | "))
        time.sleep(1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    text = open(LOG, "rb").read().decode("utf-8", "replace")
    stats = [ln for ln in text.splitlines() if "spec_stats" in ln]
    print("  --- KVMEM_TRACE spec_stats ---")
    if not stats:
        print("  (none found)")
        return None
    for ln in stats:
        print("  " + ln.strip())
    last = stats[-1]
    fields = dict(kv.split("=") for kv in last.split() if "=" in kv and kv.split("=")[0] in
                  ("n_gen", "n_drafted", "n_accept", "accept_pct"))
    return fields


def main():
    specs = sys.argv[1:] if len(sys.argv) > 1 else []
    if not specs:
        specs = ["ngram"]
    for s in specs:
        if s.startswith("ngram:"):
            n, m = s.split(":")[1], s.split(":")[2]
            run("ngram-simple %s/%s" % (n, m), ["--spec-type", "ngram-simple",
                                                "--spec-ngram-simple-size-n", n,
                                                "--spec-ngram-simple-size-m", m,
                                                "--spec-kv-dtype", "f16"])
        elif s == "ngram":
            run("ngram-simple 8/8", ["--spec-type", "ngram-simple",
                                     "--spec-ngram-simple-size-n", "8",
                                     "--spec-ngram-simple-size-m", "8",
                                     "--spec-kv-dtype", "f16"])
        elif s == "ngramnk":
            run("ngram 8/8 no-kvmem", ["-m", MODEL, "-lm", "none", "--jinja",
                                       "--chat-template-file", TEMPLATE,
                                       "--host", "127.0.0.1", "--port", str(PORT),
                                       "-c", "32768", "-n", "16384", "-b", "512", "-ngl", "99",
                                       "--kv-dtype", "q8_0", "--no-kvmem",
                                       "--spec-type", "ngram-simple",
                                       "--spec-ngram-simple-size-n", "8",
                                       "--spec-ngram-simple-size-m", "8"])
        elif s == "ngmtp":
            # n-gram first, MTP as fallback; server defaults to mtp_state=replay
            run("ngram+mtp replay(default)", ["--spec-type", "ngram-simple,draft-mtp",
                                              "--spec-ngram-simple-size-n", "8",
                                              "--spec-ngram-simple-size-m", "8",
                                              "--spec-draft-n-max", "4",
                                              "--spec-kv-dtype", "f16"])
        elif s == "ngmtp-snap":
            run("ngram+mtp snapshots", ["--spec-type", "ngram-simple,draft-mtp",
                                        "--spec-ngram-simple-size-n", "8",
                                        "--spec-ngram-simple-size-m", "8",
                                        "--spec-draft-n-max", "4",
                                        "--spec-kv-dtype", "f16",
                                        "--kvmem-mtp-state", "snapshots"])
        elif s == "mtp":
            run("draft-mtp only", ["--spec-type", "draft-mtp",
                                   "--spec-draft-n-max", "4", "--spec-kv-dtype", "f16"])
        elif s == "none":
            run("spec off", ["--spec-type", "none"])
        elif s == "mtp":
            run("draft-mtp n=4", ["--spec-type", "draft-mtp",
                                  "--spec-draft-n-max", "4", "--spec-kv-dtype", "f16"])
        else:
            print("unknown config %r" % s)


if __name__ == "__main__":
    main()
