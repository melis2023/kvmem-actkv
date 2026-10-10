#!/usr/bin/env python3
"""Measure prefill + decode throughput of a running llama-kvmem-server.

Turn 1: a haystack of --target-tokens (measures prefill).
Turn 2: a long answer at the same operating point (measures decode, i.e. the
        attention window that KVMem actually kept for this context).

Run it twice -- small and large -- to see what the sparse window costs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from niah_gkamradt import Server, assemble, build_needles  # noqa: E402

INTRO = "Read the following warehouse log carefully.\n\n=== WAREHOUSE LOG ===\n"
OUTRO = "\n=== END OF LOG ===\n\nReply with the single word ACK."
QUESTION = ("Explain in detail, step by step, how the warehouse log is "
            "organised and what information each entry contains.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18200")
    ap.add_argument("--target-tokens", type=int, default=250000)
    ap.add_argument("--chars-per-token", type=float, default=3.682)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--out", default="speed.json")
    a = ap.parse_args()

    srv = Server(a.base_url, 14400.0)
    if srv.health() is None:
        print("FATAL: server not healthy")
        return 2

    hay = assemble(build_needles(1), a.target_tokens, a.chars_per_token,
                   INTRO, OUTRO)
    msgs = [{"role": "user", "content": hay}]

    j1 = srv.chat(msgs, 1, a.reasoning_effort)
    t1, u1 = j1.get("timings") or {}, j1.get("usage") or {}
    prefill = {"prompt_tokens": u1.get("prompt_tokens"),
               "prompt_ms": t1.get("prompt_ms"),
               "prompt_per_second": t1.get("prompt_per_second"),
               "wall_s": j1["_wall_s"]}
    print("PREFILL  prompt_tokens=%s  prompt_ms=%s  tok/s=%s  wall=%.1fs" % (
        prefill["prompt_tokens"], prefill["prompt_ms"],
        ("%.1f" % prefill["prompt_per_second"]) if prefill["prompt_per_second"] else "-",
        prefill["wall_s"]), flush=True)

    msgs.append({"role": "assistant",
                 "content": j1["choices"][0]["message"].get("content") or "ACK"})
    msgs.append({"role": "user", "content": QUESTION})
    j2 = srv.chat(msgs, a.gen, a.reasoning_effort)
    t2, u2 = j2.get("timings") or {}, j2.get("usage") or {}
    decode = {"predicted_n": u2.get("completion_tokens"),
              "prompt_n": u2.get("prompt_tokens"),
              "predicted_ms": t2.get("predicted_ms"),
              "predicted_per_second": t2.get("predicted_per_second"),
              "prompt_ms": t2.get("prompt_ms"),
              "prompt_per_second": t2.get("prompt_per_second"),
              "wall_s": j2["_wall_s"]}
    print("DECODE   predicted_n=%s  predicted_ms=%s  tok/s=%s  "
          "(turn prompt_n=%s prompt_ms=%s wall=%.1fs)" % (
        decode["predicted_n"], decode["predicted_ms"],
        ("%.1f" % decode["predicted_per_second"]) if decode["predicted_per_second"] else "-",
        decode["prompt_n"], decode["prompt_ms"], decode["wall_s"]), flush=True)

    Path(a.out).write_text(json.dumps(
        {"target_tokens": a.target_tokens, "prefill": prefill, "decode": decode},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
