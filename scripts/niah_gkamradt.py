#!/usr/bin/env python3
"""gkamradt-style needle-in-a-haystack (NIAH) against a running llama-kvmem-server.

Method (Greg Kamradt's "Needle In A Haystack"):
  * build one long haystack of a target token count,
  * bury N unique needles at N different fractional depths (0% .. 100%),
  * ask one question per needle in the same conversation,
  * score by exact value match, then report recall per depth.

Two stages are supported on purpose:
  --stage probe  : small haystack, validates the pipeline end to end,
  --stage full   : the real run (e.g. 250k tokens).

The 250k prefill happens exactly once (turn 1); every later question is a
normal incremental turn, which is also the path where KVMem re-selects its
working set from the 250k stored blocks.

Usage:
  python niah_gkamradt.py --stage probe --target-tokens 8000 --needles 3
  python niah_gkamradt.py --stage full  --target-tokens 250000 --needles 12
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# deterministic filler: lexically varied, never byte-identical paragraphs
# --------------------------------------------------------------------------
NOUNS = ["pallet", "crate", "manifest", "forklift", "trolley", "drum", "cable",
         "gasket", "valve", "bearing", "panel", "hose", "bracket", "spool"]
ADJS = ["overcast", "humid", "brisk", "still", "grey", "damp", "clear", "hazy"]
VERBS = ["restacked", "inspected", "relabelled", "weighed", "sealed", "moved",
         "counted", "photographed"]
PLACES = ["north yard", "dock 4", "east aisle", "cold store", "loading bay",
          "south ramp", "mezzanine", "hazmat cage"]

FILLER_TMPL = (
    "Log {n:06d}. At {hh:02d}:{mm:02d} the crew {verb} every {noun} in the "
    "{place} while the {adj} weather held. The shift sheet for entry {n:06d} "
    "records {k} units and no exceptions; radio traffic covered {noun} "
    "rotation and the {adj} forecast. Inventory against printout {n:06d} "
    "matched, and nothing in this paragraph is a credential, password, badge, "
    "dock code, city, or access code."
)


def filler_block(i0: int, count: int) -> str:
    out = []
    for i in range(i0, i0 + count):
        out.append(FILLER_TMPL.format(
            n=i, hh=(i * 7) % 24, mm=(i * 13) % 60,
            verb=VERBS[i % len(VERBS)], noun=NOUNS[i % len(NOUNS)],
            place=PLACES[(i // 3) % len(PLACES)], adj=ADJS[(i // 5) % len(ADJS)],
            k=100 + (i * 37) % 900))
    return "\n".join(out)


CODE_WORDS = ["BLUEBIRD", "ORCHID", "CEDAR", "MAPLE", "PINE", "FALCON",
              "AMBER", "QUARTZ", "LANTERN", "COPPER", "WILLOW", "SABLE",
              "TOPAZ", "IVORY", "ONYX", "JASPER", "MARLIN", "NIMBUS",
              "OBSIDIAN", "PEBBLE", "RAVEN", "SAFFRON", "THISTLE", "UMBER"]


@dataclass
class Needle:
    idx: int
    depth_pct: float
    code: str
    sentence: str
    question: str
    char_pos: int = 0


@dataclass
class Result:
    idx: int
    depth_pct: float
    code: str
    found: bool
    answer: str
    prompt_tokens: int
    predicted_tokens: int
    wall_s: float


def build_needles(n: int) -> list[Needle]:
    if n > len(CODE_WORDS):
        raise SystemExit(
            f"need {n} needles but only {len(CODE_WORDS)} unique label words: "
            "extend CODE_WORDS (duplicate labels make the questions ambiguous)")
    needles = []
    for i in range(n):
        depth = 0.0 if n == 1 else 100.0 * i / (n - 1)
        word = CODE_WORDS[i % len(CODE_WORDS)]
        code = f"{word}-{100 + i * 7}"
        label = f"{word.lower()} checkpoint"
        needles.append(Needle(
            idx=i, depth_pct=depth, code=code,
            sentence=(f"IMPORTANT NOTE: the access code for the {label} today "
                      f"is {code}."),
            question=(f"What is the access code for the {label}? "
                      f"Answer with the code only."),
        ))
    return needles


class Server:
    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict | None:
        try:
            r = requests.get(f"{self.base}/health", timeout=10)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def chat(self, messages, max_tokens, reasoning_effort, temperature=0.0,
             seed=1234) -> dict:
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            "cache_prompt": True,
        }
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        t0 = time.time()
        r = requests.post(f"{self.base}/v1/chat/completions", json=body,
                          timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        j["_wall_s"] = time.time() - t0
        return j


def assemble(needles: list[Needle], target_tokens: int, chars_per_token: float,
             intro: str, outro: str) -> str:
    """Build the haystack so the prompt is ~target_tokens.

    This server build has no /tokenize route, so sizing is char-based and the
    exact size is read back from usage.prompt_tokens after the prefill turn.
    """
    target_chars = int(target_tokens * chars_per_token)
    needle_chars = sum(len(nd.sentence) + 2 for nd in needles)
    body_chars = max(2000, target_chars - len(intro) - len(outro) - needle_chars)

    paras: list[str] = []
    total = 0
    i = 0
    while total < body_chars:
        p = filler_block(i, 50)
        paras.append(p)
        total += len(p) + 1
        i += 50

    # needle position = fractional depth of the character stream
    cum: list[int] = []
    acc = 0
    for p in paras:
        cum.append(acc)
        acc += len(p) + 1
    total_chars = acc

    at: dict[int, list[Needle]] = {}
    for nd in needles:
        want = nd.depth_pct / 100.0 * total_chars
        idx = 0
        while idx + 1 < len(cum) and cum[idx + 1] <= want:
            idx += 1
        at.setdefault(idx, []).append(nd)

    parts = [intro]
    for idx, p in enumerate(paras):
        if idx in at:
            for nd in sorted(at[idx], key=lambda x: x.idx):
                nd.char_pos = sum(len(x) for x in parts)
                parts.append("\n" + nd.sentence + "\n")
        parts.append(p + "\n")
    parts.append(outro)
    return "".join(parts)


def normalize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18200")
    ap.add_argument("--stage", default="full", choices=["probe", "full"])
    ap.add_argument("--target-tokens", type=int, default=250000)
    ap.add_argument("--needles", type=int, default=12)
    ap.add_argument("--reasoning-effort", default="none",
                    help="'none' disables thinking for the recall questions")
    ap.add_argument("--answer-max-tokens", type=int, default=64)
    ap.add_argument("--prefill-max-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=14400.0)
    ap.add_argument("--out", default="niah-result.json")
    ap.add_argument("--skip-prefill-turn", action="store_true")
    ap.add_argument("--chars-per-token", type=float, default=3.9,
                    help="sizing fallback; calibrate it from a probe run")
    ap.add_argument("--decode-probe", type=int, default=0,
                    help="after the needles, generate N tokens and report decode tok/s")
    args = ap.parse_args()

    srv = Server(args.base_url, args.timeout)
    h = srv.health()
    if h is None:
        print("FATAL: server not healthy at", args.base_url)
        return 2
    print("server health:", json.dumps(h), flush=True)

    needles = build_needles(args.needles)
    intro = ("Read the following warehouse log carefully. Several operational "
             "facts are buried in it. After the log, and in later messages, "
             "you will be asked about them.\n\n=== WAREHOUSE LOG ===\n")
    outro = ("\n=== END OF LOG ===\n\nReply with the single word ACK.")

    t0 = time.time()
    hay = assemble(needles, args.target_tokens, args.chars_per_token, intro, outro)
    est_tokens = int(len(hay) / args.chars_per_token)
    hay_tokens = est_tokens
    print(f"haystack built: {len(hay)} chars ~= {est_tokens} tokens "
          f"(estimate at {args.chars_per_token} chars/token, "
          f"{time.time() - t0:.1f}s)", flush=True)
    for nd in needles:
        print(f"  needle {nd.idx:2d} depth {nd.depth_pct:6.2f}%  {nd.code}",
              flush=True)

    messages = [{"role": "user", "content": hay}]

    prefill_s = None
    prefill_tok_s = None
    if not args.skip_prefill_turn:
        print("turn 1: prefill + ACK ...", flush=True)
        j = srv.chat(messages, args.prefill_max_tokens, args.reasoning_effort)
        prefill_s = j["_wall_s"]
        content = j["choices"][0]["message"].get("content") or ""
        usage = j.get("usage") or {}
        tm = j.get("timings") or {}
        hay_tokens = usage.get("prompt_tokens") or est_tokens
        prefill_tok_s = tm.get("prompt_per_second")
        print(f"  prefill done in {prefill_s:.1f}s "
              f"prompt_tokens={hay_tokens} "
              f"prompt_per_s={prefill_tok_s}", flush=True)
        messages.append({"role": "assistant", "content": content or "ACK"})

    results: list[Result] = []
    for nd in needles:
        messages.append({"role": "user", "content": nd.question})
        j = srv.chat(messages, args.answer_max_tokens, args.reasoning_effort)
        msg = j["choices"][0]["message"]
        answer = (msg.get("content") or "")
        usage = j.get("usage") or {}
        tm = j.get("timings") or {}
        found = normalize(nd.code) in normalize(answer)
        results.append(Result(
            idx=nd.idx, depth_pct=nd.depth_pct, code=nd.code, found=found,
            answer=answer.strip()[:400],
            prompt_tokens=usage.get("prompt_tokens") or 0,
            predicted_tokens=usage.get("completion_tokens") or 0,
            wall_s=j["_wall_s"]))
        messages.append({"role": "assistant", "content": answer})
        print(f"  depth {nd.depth_pct:6.2f}%  {nd.code:14s} "
              f"{'HIT ' if found else 'MISS'}  prompt={usage.get('prompt_tokens')}"
              f"  gen={usage.get('completion_tokens')}  {j['_wall_s']:.1f}s",
              flush=True)
        if not found:
            print(f"        answer: {answer.strip()[:160]!r}", flush=True)

    hit = sum(1 for r in results if r.found)

    decode = None
    if args.decode_probe:
        messages.append({"role": "user", "content":
                         "Now explain in detail, step by step, how this "
                         "warehouse log is organised and what each numbered "
                         "entry contains."})
        jd = srv.chat(messages, args.decode_probe, args.reasoning_effort)
        td, ud = jd.get("timings") or {}, jd.get("usage") or {}
        decode = {"predicted_n": ud.get("completion_tokens"),
                  "predicted_ms": td.get("predicted_ms"),
                  "predicted_per_second": td.get("predicted_per_second"),
                  "prompt_n": ud.get("prompt_tokens"),
                  "prompt_ms": td.get("prompt_ms"),
                  "prompt_per_second": td.get("prompt_per_second"),
                  "wall_s": jd["_wall_s"]}
        print(f"DECODE PROBE predicted_n={decode['predicted_n']} "
              f"tok/s={decode['predicted_per_second']} "
              f"wall={decode['wall_s']:.1f}s", flush=True)
    summary = {
        "stage": args.stage,
        "target_tokens": args.target_tokens,
        "haystack_tokens": hay_tokens,
        "needles": len(needles),
        "hits": hit,
        "recall": hit / max(1, len(needles)),
        "prefill_s": prefill_s,
        "prefill_tok_s": prefill_tok_s,
        "decode": decode,
        "reasoning_effort": args.reasoning_effort,
        "results": [asdict(r) for r in results],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRECALL {hit}/{len(needles)} = {hit / max(1, len(needles)):.1%} "
          f"@ {hay_tokens} tokens  -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
