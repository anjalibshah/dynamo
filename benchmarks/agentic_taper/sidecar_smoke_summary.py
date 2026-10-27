# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize sidecar_smoke.sh arms: interactive on-deadline share, background
throughput, and the regulation metrics. Usage: sidecar_smoke_summary.py OUT_ROOT"""

import json
import re
import sys
from pathlib import Path

TTFT_MS, PER_TOKEN_MS = 500.0, 18.48


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def metrics(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"dynamo_component_decode_regulation_(\w+?)(?:_total)?\{.*\} (\S+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main(root: Path) -> None:
    for arm in ("off", "observe", "class_blind"):
        d = root / arm
        if not (d / "victims.jsonl").exists():
            continue
        vs = [json.loads(ln) for ln in open(d / "victims.jsonl")]
        ok = [v for v in vs if v.get("ok") and v.get("t_done")]
        e2e, hit = [], 0
        for v in ok:
            ms = (v["t_done"] - v["t_arrival"]) * 1e3
            n = len(v.get("itls_ms") or []) + 1
            e2e.append(ms)
            hit += ms <= TTFT_MS + PER_TOKEN_MS * (n - 1)
        bg = [json.loads(ln) for ln in open(d / "bg.jsonl")] if (d / "bg.jsonl").exists() else []
        bg_ok = [b for b in bg if b.get("status") == 200]
        span = (max(b["t_done"] for b in bg) - min(b["t_start"] for b in bg)) if bg else 0
        toks = sum(b.get("completion_tokens") or 0 for b in bg_ok)
        m = metrics(d / "metrics.txt")
        print(f"== {arm}")
        print(f"  interactive: {len(ok)}/{len(vs)} ok, on deadline {hit / max(len(ok), 1):.1%}, "
              f"e2e p50 {pct(e2e, .5):.0f} ms p90 {pct(e2e, .9):.0f} ms")
        print(f"  background: {len(bg_ok)}/{len(bg)} ok, {toks / max(span, 1):.0f} output tok/s")
        if m:
            reg = m.get("regulated", 0)
            print(f"  regulation: enabled={m.get('enabled')} bias={m.get('step_time_bias', 0):.2f} "
                  f"steps={m.get('steps', 0):.0f} engaged={m.get('engaged_steps', 0):.0f} "
                  f"interactive={m.get('interactive', 0):.0f} regulated={reg:.0f} "
                  f"skipped={m.get('skipped', 0):.0f} ({m.get('skipped', 0) / max(reg, 1):.1%})")
        errs = [b.get("error") for b in bg if b.get("status") != 200][:2]
        if errs:
            print(f"  background errors (first 2): {errs}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
