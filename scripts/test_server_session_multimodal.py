"""Verify image-session isolation and NVMe restoration with a supplied vision model."""
import argparse
import base64
import copy
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import time
import urllib.request
import uuid
import zlib


def image_url(rgb):
    # Deterministic test fixture, with no imaging-library dependency.
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    pixels = (b"\0" + bytes(rgb) * 512) * 512
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 512, 512, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ("cache-" + uuid.uuid4().hex)
    cache.mkdir()
    checks, turns = [], []
    colors = {"A": ("RED", (255, 0, 0)), "B": ("BLUE", (0, 0, 255)), "C": ("GREEN", (0, 255, 0))}
    histories = {}
    for name, (_, color) in colors.items():
        notes = "\n".join(f"Channel {name} routing note {i}: this entry has no bearing on the attached picture."
                          for i in range(180))
        histories[name] = [{"role": "system", "content": "Answer questions about the attached image with one color word only."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(color)}},
                {"type": "text", "text": notes + "\nWhat is the dominant color of the attached image? One color word only."}]}]
    original_a = copy.deepcopy(histories["A"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(args.server.resolve(strict=True)), "-m", str(args.model.resolve(strict=True)),
               "--mmproj", str(args.mmproj.resolve(strict=True)), "--image-min-tokens", "1024", "--image-max-tokens", "1024",
               "--host", "127.0.0.1", "--port", str(port), "-dev", "CUDA0", "-ngl", "99", "-c", "16384",
               "--reasoning-effort", "none", "--temp", "0", "--no-ui", "-ctk", "q8_0", "-ctv", "q4_0",
               "-fa", "on", "--kvmem-budget", "2048", "--kvmem-gen-reserve", "512",
               "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--kvmem-mtp-state", "snapshots",
               "--kvmem-conversations", "5", "--kvmem-session-ram-gb", "0.2", "--kvmem-session-nvme-gb", "3",
               "--kvmem-session-cache-dir", str(cache)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", KVMEM_TRACE="1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=300) as response:
            return json.load(response)

    def check(name, ok):
        checks.append({"name": name, "pass": bool(ok)})
        print(("PASS " if ok else "FAIL ") + name, flush=True)
        if not ok:
            raise AssertionError(name)

    def turn(messages, expected, label, client_id=None, reuse=None):
        body = {"messages": messages, "max_tokens": 24, "temperature": 0, "reasoning_effort": "none"}
        if client_id is not None:
            body["kvmem"] = {"conversation_id": client_id}
        response = request("/v1/chat/completions", body)
        answer = response["choices"][0]["message"]["content"] or ""
        words = set(re.findall(r"[A-Z]+", answer.upper()))
        check(label + " correct image color", expected in words and not (words & ({"RED", "BLUE", "GREEN", "YELLOW"} - {expected})))
        hit = response["usage"]["prompt_cache_hit_tokens"]
        if reuse is not None:
            check(label + (" prefix hit" if reuse else " cold miss"), hit > 1024 if reuse else hit < 512)
        messages.append({"role": "assistant", "content": answer})
        counters = request("/slots")[0]["kvmem"]["conversations"]
        check(label + " disk accounting", counters["disk_bytes"] <= 3 * 2**30 and counters["count"] <= 5 and counters["disk_errors"] == 0)
        turns.append({"label": label, "answer": answer, "usage": response["usage"], "counters": counters})

    process = None
    log_path = out / "server.log"
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited; see {log_path}")
                try:
                    request("/health")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server startup")
                    time.sleep(.5)
            for name, (color, _) in colors.items():
                turn(histories[name], color, "initial " + name, client_id=name, reuse=False)
            for cycle in range(2):
                for name, (color, _) in colors.items():
                    histories[name].append({"role": "user", "content": "Recall the dominant color of my attached image. One word only."})
                    turn(histories[name], color, f"restore {cycle} {name}", client_id=name, reuse=True)
            changed = copy.deepcopy(original_a)
            changed[1]["content"][0]["image_url"]["url"] = image_url((255, 255, 0))
            turn(changed, "YELLOW", "same text and ID with changed image", client_id="A", reuse=False)
            histories["A"].append({"role": "user", "content": "Recall the dominant color of my attached image. One word only."})
            turn(histories["A"], "RED", "original image history survives", reuse=True)
            check("no image session evicted", request("/slots")[0]["kvmem"]["conversations"]["evictions"] == 0)
            trace = log_path.read_text(encoding="utf-8", errors="replace")
            check("at least six real disk restores", len(re.findall(r"session_restore id=", trace)) >= 6)
            check("vision encoder and MTP exercised", "vision" in trace.lower() and "mtp_follow" in trace)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=30)
        (out / "result.json").write_text(json.dumps({"command": command, "checks": checks, "turns": turns}, indent=2), encoding="utf-8")
        if cache.parent == out and cache.name.startswith("cache-") and not cache.is_symlink():
            shutil.rmtree(cache)


if __name__ == "__main__":
    main()
