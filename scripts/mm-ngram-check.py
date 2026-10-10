"""End-to-end check: n-gram-only speculation + vision (mmproj) must not error.

Reproduces the reported failure:
    KVMEM_TRACE multimodal_error error=MTP carry checkpoint failed

Starts llama-kvmem-server with --spec-type ngram-simple (no draft context), sends
one chat completion carrying a small PNG, and greps the server log for errors.
Runs serially: one GPU process at a time.
"""
import base64, json, os, struct, subprocess, sys, time, urllib.request, zlib

EXE = r"I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll\bin\llama-kvmem-server.exe"
MODEL = r"I:/model/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
MMPROJ = r"I:/model/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf"
PORT = 8097
LOG = r"C:\Users\zhuan\WorkBuddy\2026-10-01-19-57-21\.workbuddy\tmp\mm-ngram-server.log"


def make_png(w=224, h=224):
    """Minimal RGB PNG: a red/green checkerboard, no external deps."""
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0
        for x in range(w):
            c = (220, 60, 60) if ((x // 28) + (y // 28)) % 2 == 0 else (60, 160, 90)
            raw += bytes(c)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def wait_ready(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = urllib.request.urlopen(
                urllib.request.Request("http://127.0.0.1:%d/v1/models" % PORT,
                                       headers={"Authorization": "Bearer sk-123"}), timeout=3)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    spec = sys.argv[1:] or ["ngram-simple"]
    args = [EXE, "-m", MODEL, "--mmproj", MMPROJ, "--no-mmproj-offload",
            "--image-max-tokens", "512", "-lm", "none", "--jinja",
            "--host", "127.0.0.1", "--port", str(PORT),
            "-n", "512", "-c", "8192", "-b", "512", "-ngl", "99",
            "--kvmem-budget", "16384", "--kvmem-gen-reserve", "8192",
            "--kvmem-block-tokens", "32", "--kv-dtype", "q8_0",
            "--spec-type", ",".join(spec)]
    if "draft-mtp" not in ",".join(spec):
        args += ["--spec-ngram-simple-size-n", "8", "--spec-ngram-simple-size-m", "8"]
    else:
        args += ["--spec-draft-n-max", "4", "--spec-kv-dtype", "f16"]
    args += ["--api-key", "sk-123"]

    logf = open(LOG, "wb")
    p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
    try:
        if not wait_ready():
            print("SERVER DID NOT START")
            return 2
        png = make_png()
        b64 = base64.b64encode(png).decode()
        body = {
            "model": "kvmem",
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": "What color are the squares? One sentence."},
            ]}],
            "max_tokens": 48,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % PORT,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer sk-123"})
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            out = json.loads(resp.read().decode())
            text = out["choices"][0]["message"]["content"]
            print("SPEC=%s  OK in %.1fs" % (",".join(spec), time.time() - t0))
            print("  reply: %s" % text.strip()[:160])
        except Exception as e:
            print("SPEC=%s  REQUEST FAILED: %r" % (",".join(spec), e))
        time.sleep(2)
    finally:
        p.terminate()
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()
        logf.close()

    blob = open(LOG, "rb").read().decode("utf-8", "replace")
    bad = [l for l in blob.splitlines()
           if "multimodal_error" in l or "carry checkpoint" in l
           or "gdn_sync fail" in l or "KVMEM_STARTUP_ERROR" in l
           or "GDN replay begin failed" in l]
    print("  error lines: %d" % len(bad))
    for l in bad[:6]:
        print("   ! " + l.strip()[:200])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
