#!/usr/bin/env python3
"""Render a gkamradt-NIAH result JSON into a markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def bar(frac: float, width: int = 20) -> str:
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--config", default="")
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    res = sorted(d["results"], key=lambda r: r["depth_pct"])
    lines: list[str] = []
    A = lines.append

    A(f"# Needle-in-a-Haystack (gkamradt method) — {args.label or d['stage']}")
    A("")
    A(f"- 上下文(prompt tokens, 实测): **{d['haystack_tokens']}**")
    A(f"- 针数: **{d['needles']}**  命中: **{d['hits']}**  召回率: "
      f"**{d['recall']:.1%}**")
    if d.get("prefill_s"):
        A(f"- 首次 prefill 墙钟: **{d['prefill_s']:.1f} s** "
          f"(≈{d['haystack_tokens'] / max(0.001, d['prefill_s']):.0f} tok/s)")
    A(f"- 提问时 reasoning_effort: `{d.get('reasoning_effort')}`")
    if args.config:
        A(f"- 服务端配置: {args.config}")
    A("")
    A("## 逐深度结果")
    A("")
    A("| 深度 | 针(期望值) | 结果 | 该轮 prompt | 生成 | 墙钟 | 热力 |")
    A("|---:|---|---|---:|---:|---:|---|")
    for r in res:
        A(f"| {r['depth_pct']:.2f}% | `{r['code']}` | "
          f"{'✅ HIT' if r['found'] else '❌ MISS'} | {r['prompt_tokens']} | "
          f"{r['predicted_tokens']} | {r['wall_s']:.1f}s | "
          f"`{bar(1.0 if r['found'] else 0.0)}` |")
    A("")
    A("## 深度 → 召回")
    A("")
    A("```")
    for r in res:
        A(f"{r['depth_pct']:6.2f}%  {'HIT ' if r['found'] else 'MISS'}  "
          f"{r['code']}")
    A("```")
    A("")
    miss = [r for r in res if not r["found"]]
    A("## 未命中")
    A("")
    if not miss:
        A("无——全部深度均命中。")
    else:
        for r in miss:
            A(f"- 深度 **{r['depth_pct']:.2f}%** (`{r['code']}`): "
              f"回答 `{r['answer'][:200]}`")
    A("")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
