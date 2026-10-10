#!/usr/bin/env python3
"""Multi-turn, realistic-content memory test at ~256k context (ActKV ON vs OFF).

Design
------
Turn 1 dumps a synthetic but realistic multi-module repository (~240k tokens).
Then ~40 consecutive turns:
  * haystack questions  -- ask about a detail buried deep in the dumped code
                           (16 needles: error codes, constants, queue/table
                           names, feature flags, cron, bucket, class name, ...)
  * intro turns         -- the user states a new fact in passing
                           ("note the staging tenant id is t-9f42c")
  * memory questions    -- ask about a fact introduced several turns EARLIER,
                           i.e. recall from the live conversation, not the dump
  * filler turns        -- ordinary requests that churn the context
Scoring is exact-literal match on the expected value.

Everything is deterministic, so the ON and OFF runs use a byte-identical prompt
stream; only the server flags differ.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from niah_gkamradt import Server, normalize  # noqa: E402

# --------------------------------------------------------------------------
# realistic needle content: (line to plant, question, expected)
# --------------------------------------------------------------------------
HAY_NEEDLES: list[tuple[str, str, str]] = [
    ('READ_TIMEOUT_MS = 8412  # tuned against eu-west p99; do not lower',
     "What is the read timeout, in milliseconds, of the ingest client?",
     "8412"),
    ('raise ConfigError("ERR-4471: tenant id missing in shard map")',
     "Which error code is raised when a tenant id is missing from the shard map?",
     "ERR-4471"),
    ('INGEST_QUEUE = "ingest-v3-eu-west"',
     "Which queue name does the ingest worker consume from?",
     "ingest-v3-eu-west"),
    ('FEATURE_SHARD_REBALANCE_V2 = True',
     "What is the feature flag that turns on the second-generation shard rebalancer?",
     "FEATURE_SHARD_REBALANCE_V2"),
    ('SALT_PREFIX = "sk_live_9f42c"',
     "What salt prefix is used when deriving tenant API keys?",
     "sk_live_9f42c"),
    ('# TODO(alina.kovacs): drop the legacy fanout path after 2026-Q3',
     "Which engineer owns the TODO about dropping the legacy fanout path?",
     "alina.kovacs"),
    ('SCHEDULE = "*/7 3 * * 1-5"  # weekday rebalance sweep',
     "What cron expression drives the weekday rebalance sweep?",
     "*/7 3 * * 1-5"),
    ('TABLE = "tenant_shard_map_v7"',
     "Which database table holds the tenant shard map?",
     "tenant_shard_map_v7"),
    ('SIGNAL = "SIG_REBALANCE_DONE"',
     "What signal name marks the end of a rebalance run?",
     "SIG_REBALANCE_DONE"),
    ('METRIC = "ingest.rebalance.lag_seconds"',
     "What metric reports rebalance lag?",
     "ingest.rebalance.lag_seconds"),
    ('ARCHIVE_BUCKET = "s3://acme-ingest-archive-2026"',
     "Which S3 bucket receives the ingest archive?",
     "acme-ingest-archive-2026"),
    ('DRAFT_MODEL = "qwen3-8b-instruct-rc2"',
     "Which model id is configured for the drafting endpoint?",
     "qwen3-8b-instruct-rc2"),
    ('class ShardRebalanceCoordinatorV3:',
     "What is the class name of the third-generation rebalance coordinator?",
     "ShardRebalanceCoordinatorV3"),
    ('CACHE_TTL_SECONDS = 5417  # tenant metadata cache',
     "How many seconds is the tenant metadata cache valid for?",
     "5417"),
    ('INGEST_TOPIC = "ingest.events.v9"',
     "What is the name of the ingest topic?",
     "ingest.events.v9"),
    ('BACKOFF_PROFILE = "exp-jitter-8"',
     "Which backoff profile does the retry helper use?",
     "exp-jitter-8"),
]

# facts stated in conversation, asked several turns later
TURN_FACTS: list[tuple[str, str, str]] = [
    ("Before we go on, note for later: the staging tenant id is t-9f42c.",
     "What staging tenant id did I mention earlier in this conversation?",
     "t-9f42c"),
    ("Let's call the new service ledger-sync for now.",
     "What did we decide to name the new service?",
     "ledger-sync"),
    ("The deadline we agreed on is 2026-11-20.",
     "What deadline did we agree on earlier?",
     "2026-11-20"),
    ("I'll push this on branch fix/rebalance-lag.",
     "Which branch did I say I would push this on?",
     "fix/rebalance-lag"),
    ("The customer account id is ACCT-88213.",
     "What is the customer account id I gave you?",
     "ACCT-88213"),
    ("We settled on Postgres 17.2 for the new cluster.",
     "Which Postgres version did we settle on?",
     "17.2"),
    ("The canary group is canary-eu-4.",
     "What is the canary group called?",
     "canary-eu-4"),
    ("Track the follow-up under ticket INFRA-2291.",
     "What ticket number is the follow-up tracked under?",
     "INFRA-2291"),
]

FILLERS = [
    "In one sentence, what does the ingest worker do?",
    "Refactor the retry helper so jitter is applied per attempt.",
    "What would you change first to reduce rebalance lag?",
    "Sketch the call path from the HTTP handler to the shard map.",
    "Which module would you unit-test first, and why?",
    "Summarise the failure modes you see in the rebalance path.",
    "Propose a rollback plan for a bad shard map migration.",
    "What metrics would you add before shipping this?",
]

# --------------------------------------------------------------------------
# synthetic repository dump
# --------------------------------------------------------------------------
PKGS = ["ingest", "shard", "ledger", "control", "telemetry", "auth", "jobs",
        "gateway", "policy", "archive"]
MODS = ["client", "worker", "coordinator", "planner", "store", "router",
        "handler", "registry", "watcher", "budget", "signer", "reconciler"]
NOUNS = ["shard", "tenant", "partition", "replica", "cursor", "lease", "batch",
         "offset", "epoch", "checkpoint", "quota", "lane"]


def _ident(i: int, w: str) -> str:
    return f"{w}_{i:04d}"


TMPL = '''# {pkg}/{mod}.py
"""{pkg}.{mod} -- {title}

Owned by the platform team. See docs/architecture/{pkg}.md for the
rollout notes; the {noun} path is the one that historically regressed.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

LOG = logging.getLogger(__name__)

DEFAULT_{noun}_LIMIT = {lim}
{CONST}

@dataclass(frozen=True)
class {Cls}Config:
    """Tunables for {mod}; changing these changes the {noun} budget."""
    max_inflight: int = {inflight}
    {noun}_limit: int = {lim}
    window_ms: int = {win}
    strict: bool = {strict}


class {Cls}:
    """{title}.

    The class keeps a small in-memory view of the {noun} so that repeated
    lookups do not hit the store; the view is rebuilt whenever a lease
    expires or the {noun} epoch advances.
    """

    def __init__(self, cfg: {Cls}Config) -> None:
        self._cfg = cfg
        self._cache: dict[str, int] = {{}}
        self._lease_until = 0.0
        self._closed = False

    def _expired(self, now: float) -> bool:
        return now >= self._lease_until

    async def _reload(self, keys: Iterable[str]) -> int:
        """Refresh the local view; returns the number of refreshed keys."""
        refreshed = 0
        for key in keys:
            if not key:
                LOG.debug("skipping empty %(noun)s key in %(mod)s", {{"noun": "{noun}", "mod": "{mod}"}})
                continue
            self._cache[key] = len(key) + {lim}
            refreshed += 1
        self._lease_until = time.monotonic() + self._cfg.window_ms / 1000.0
        return refreshed

    async def handle(self, batch: Iterable[dict]) -> int:
        """Process one batch of {noun} records.

        Raises TimeoutError when the {noun} lease cannot be renewed inside
        the configured window; callers are expected to back off and retry.
        """
        if self._closed:
            raise RuntimeError("{mod} closed")
        now = time.monotonic()
        if self._expired(now):
            await self._reload([str(i) for i in range({reload})])
        done = 0
        inflight = 0
        for record in batch:
            key = str(record.get("key", ""))
            value = self._cache.get(key, {lim})
            if value > self._cfg.{noun}_limit and self._cfg.strict:
                LOG.warning("%(noun)s {nounexpr} over limit in {mod}", {{"noun": "{noun}"}})
                continue
            inflight += 1
            if inflight >= self._cfg.max_inflight:
                await asyncio.sleep(0)
                inflight = 0
            done += 1
        return done

    def close(self) -> None:
        """Drop the cached {noun} view. Idempotent."""
        self._cache.clear()
        self._closed = True


class {Cls}Observer:
    """Read-only companion used by the {pkg} health endpoint."""

    def __init__(self, owner: "{Cls}") -> None:
        self._owner = owner

    def snapshot(self) -> dict:
        return {{
            "cached": len(self._owner._cache),
            "closed": self._owner._closed,
            "window_ms": self._owner._cfg.window_ms,
        }}


def {fn}(items: Iterable[int], limit: int = {lim}) -> Iterator[int]:
    """Filter {noun} ids, newest first; used by the {pkg} reconciler."""
    seen = 0
    for item in sorted(items, reverse=True):
        if item < 0:
            continue
        if seen >= limit:
            break
        seen += 1
        yield item


def {fn}_summary(items: Iterable[int]) -> dict:
    values = list({fn}(items))
    return {{
        "count": len(values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
    }}
'''


def gen_file(i: int, needle_line: str | None) -> str:
    pkg = PKGS[i % len(PKGS)]
    mod = MODS[(i // len(PKGS)) % len(MODS)]
    noun = NOUNS[(i * 7) % len(NOUNS)]
    cls = "".join(p.capitalize() for p in (pkg, mod)) + _ident(i, "")
    const = (needle_line + "\n" if needle_line else
             f"{_ident(i, 'LIMIT').upper()} = {(i * 13) % 97 + 3}\n")
    body = TMPL.format(
        pkg=pkg, mod=mod, noun=noun, title=f"{pkg} {mod} maintenance",
        Cls=cls, CONST=const, lim=(i * 17) % 90 + 8,
        inflight=(i * 5) % 16 + 1, win=(i * 29) % 400 + 50,
        strict="True" if i % 3 else "False", reload=(i % 11) + 2,
        fn=_ident(i, "filter"), nounexpr='{key}',
    )
    return body


def build_dump(target_chars: int, needles: list[tuple[str, str, str]]) -> tuple[str, list[int]]:
    """Return the dump text and the file index each needle was planted in."""
    avg_len = max(1, len(gen_file(0, None)))
    n_files = max(len(needles), int(target_chars / avg_len) + 1)
    plant: dict[int, str] = {}
    for k, (line, _, _) in enumerate(needles):
        idx = min(n_files - 1, int(round(k / max(1, len(needles) - 1) * (n_files - 1))))
        while idx in plant:            # never stack two needles on one file
            idx = (idx + 1) % n_files
        plant[idx] = line
    parts, planted_at = [], []
    for i in range(n_files):
        parts.append(gen_file(i, plant.get(i)))
        if i in plant:
            planted_at.append(i)
    text = "\n".join(parts)
    return text, planted_at


def build_schedule() -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    events += [("hay", n) for n in HAY_NEEDLES]
    events += [("mem", f) for f in TURN_FACTS]
    events += [("filler", t) for t in FILLERS]
    random.Random(11).shuffle(events)
    # place each fact's intro 6 turns before its memory question
    for fact in reversed(TURN_FACTS):
        idx = next(i for i, e in enumerate(events) if e[0] == "mem" and e[1] is fact)
        events.insert(max(1, idx - 6), ("intro", fact))
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18200")
    ap.add_argument("--target-tokens", type=int, default=240000)
    ap.add_argument("--calibrate-chars", type=int, default=6000)
    ap.add_argument("--answer-max-tokens", type=int, default=64)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--timeout", type=float, default=14400.0)
    ap.add_argument("--out", default="dialog.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    srv = Server(a.base_url, a.timeout)
    if srv.health() is None:
        print("FATAL: server not healthy at", a.base_url)
        return 2

    # ---- calibrate chars/token on representative generated text ----
    sample = "".join(gen_file(i, None) for i in range(3))
    if a.dry_run:
        cpt = 3.0
    else:
        j = srv.chat([{"role": "user", "content": sample}], 1, a.reasoning_effort)
        n = (j.get("usage") or {}).get("prompt_tokens") or 1
        cpt = len(sample) / max(1, n - 8)          # -8 for template overhead
    print(f"calibration: {len(sample)} chars -> {cpt:.3f} chars/token", flush=True)

    dump, planted = build_dump(int(a.target_tokens * cpt), HAY_NEEDLES)
    intro = ("You are reviewing a repository. Below is the full source dump of "
             "the service. Read it; I will ask follow-up questions in the next "
             "messages.\n\n=== REPOSITORY DUMP ===\n")
    outro = "\n=== END OF DUMP ===\n\nReply with the single word ACK."
    messages = [{"role": "user", "content": intro + dump + outro}]

    print(f"dump: {len(dump)} chars, {len(planted)} needles planted, "
          f"schedule: {len(build_schedule())} turns", flush=True)

    if a.dry_run:
        print("dry run: not sending")
        return 0

    t0 = time.time()
    j1 = srv.chat(messages, 1, a.reasoning_effort)
    u1, t1 = j1.get("usage") or {}, j1.get("timings") or {}
    dump_tokens = u1.get("prompt_tokens")
    print(f"turn 0 (dump): {dump_tokens} prompt tokens in {j1['_wall_s']:.1f}s "
          f"({t1.get('prompt_per_second'):.0f} tok/s)", flush=True)
    messages.append({"role": "assistant",
                     "content": j1["choices"][0]["message"].get("content") or "ACK"})

    rows, results = [], []
    for tno, (kind, payload) in enumerate(build_schedule(), start=1):
        if kind == "hay":
            line, question, expected = payload            # type: ignore[misc]
            text, typ = question, "haystack"
        elif kind == "mem":
            _, question, expected = payload               # type: ignore[misc]
            text, typ = question, "conversation"
        elif kind == "intro":
            text, typ, expected = payload[0], "intro", ""  # type: ignore[index]
        else:
            text, typ, expected = payload, "filler", ""    # type: ignore[assignment]

        messages.append({"role": "user", "content": text})
        j = srv.chat(messages, a.answer_max_tokens, a.reasoning_effort)
        ans = (j["choices"][0]["message"].get("content") or "")
        messages.append({"role": "assistant", "content": ans})
        u = j.get("usage") or {}

        hit = None
        if typ in ("haystack", "conversation"):
            hit = normalize(expected) in normalize(ans)
        row = {"turn": tno, "type": typ, "question": text[:120],
               "expected": expected, "answer": ans.strip()[:200],
               "hit": hit, "prompt_tokens": u.get("prompt_tokens"),
               "gen": u.get("completion_tokens"), "wall_s": j["_wall_s"],
               "finish_reason": j["choices"][0].get("finish_reason"),
               "stop_type": j.get("stop_type")}
        rows.append(row)
        if hit is not None:
            results.append(row)
        flag = "" if hit is None else ("HIT " if hit else "MISS")
        if hit is not None:
            print(f"  t{tno:02d} {typ:12s} {flag} {expected!r}"
                  f"{'' if hit else '  <- ' + ans.strip()[:70]!r}", flush=True)
        elif tno % 10 == 0:
            print(f"  t{tno:02d} {typ:12s} ...", flush=True)

    def recall(t: str) -> tuple[int, int]:
        sel = [r for r in results if r["type"] == t]
        return sum(1 for r in sel if r["hit"]), len(sel)

    hh, hn = recall("haystack")
    mh, mn = recall("conversation")
    summary = {
        "label": a.label,
        "dump_tokens": dump_tokens,
        "final_prompt_tokens": rows[-1]["prompt_tokens"] if rows else None,
        "turns": len(rows),
        "haystack": {"hit": hh, "n": hn},
        "conversation": {"hit": mh, "n": mn},
        "overall_hit": hh + mh, "overall_n": hn + mn,
        "recall": (hh + mh) / max(1, hn + mn),
        "prefill_s": j1["_wall_s"],
        "prefill_tok_s": t1.get("prompt_per_second"),
        "reasoning_effort": a.reasoning_effort,
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== {a.label}: haystack {hh}/{hn}, conversation {mh}/{mn}, "
          f"overall {hh+mh}/{hn+mn} = {summary['recall']:.1%}  "
          f"(dump {dump_tokens} tok, final {summary['final_prompt_tokens']} tok, "
          f"{time.time()-t0:.0f}s) -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
