#!/usr/bin/env python3
"""Replay the patch stack on pinned files and compare with the working tree."""

from pathlib import Path
import argparse
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--crlf", action="store_true", help="check Windows checkout and mixed patch line endings")
args = parser.parse_args()
patches = [root / "patches" / name for name in (
    "llama-kvmem-current.patch", "cuda-graph-decode.patch",
    "gdn-output-fusion.patch", "0005-hip-rdna2-quantized-kv-fa-vec.patch")]
paths = set()
for patch in patches:
    paths.update(re.findall(r"^\+\+\+ b/(.+)$", patch.read_text(encoding="utf-8"), re.MULTILINE))
(root / "artifacts").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="gdn-patch-check-", dir=root / "artifacts") as temporary:
    checkout = Path(temporary)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    for path in paths:
        result = subprocess.run(["git", "-C", str(root / "llama.cpp"), "show", "HEAD:" + path], capture_output=True)
        if result.returncode == 0:
            destination = checkout / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = result.stdout.replace(b"\r\n", b"\n")
            destination.write_bytes(data.replace(b"\n", b"\r\n") if args.crlf else data)
    for patch in patches:
        normalized = checkout / (patch.name + ".lf")
        normalized.write_bytes(patch.read_bytes() if args.crlf else patch.read_bytes().replace(b"\r\n", b"\n"))
        subprocess.run(["git", "-C", str(checkout), "apply", "--ignore-space-change", "--check", str(normalized)], check=True)
        subprocess.run(["git", "-C", str(checkout), "apply", "--ignore-space-change", str(normalized)], check=True)
    for path in paths:
        expected = (checkout / path).read_bytes().replace(b"\r\n", b"\n")
        actual = (root / "llama.cpp" / path).read_bytes().replace(b"\r\n", b"\n")
        if expected != actual:
            raise RuntimeError("patch replay differs: " + path)
    subprocess.run(["git", "-C", str(checkout), "apply", "--ignore-space-change", "--reverse", "--check", str(patches[2])], check=True)
print(f"PASS: fresh {'CRLF' if args.crlf else 'LF'} patch replay matches all {len(paths)} affected files")
