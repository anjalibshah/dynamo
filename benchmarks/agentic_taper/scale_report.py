# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-worker headroom check for TAPER sibling placement (NEXT_STEPS 7.3g rethink, phase 1).

Run on a stock-kv, multi-worker Harbor run (``harbor_ab.sh diag`` with
WORKERS=N). Answers the go/no-go question for placement:

  1. Co-location: does KV routing put a task's subagents on the parent's
     worker? (request trace ``worker.decode_worker_id``)
  2. Imbalance: while that happens, is decode context uneven across workers,
     i.e. is there idle capacity a placement policy could use? (fpm.jsonl)

    python3 scale_report.py --trace run/trace.jsonl --fpm run/fpm.jsonl \\
        [--budget 300000] [--slo-ms 14.85]

Stdlib only, so it runs on the login node against /data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def _pct(v: list[float], f: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")


def _trace(path: str) -> list[dict]:
    out = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        r = r.get("event", r)
        if r.get("event_type") != "request_end":
            continue
        m, c = r["request"], r.get("agent_context") or {}
        w = (m.get("worker") or {}).get("decode_worker_id")
        if m.get("request_received_ms") is None or m.get("total_time_ms") is None:
            continue
        s = float(m["request_received_ms"])
        out.append({"session": c.get("session_id") or "", "parent": c.get("parent_session_id"),
                    "start": s, "end": s + float(m["total_time_ms"]), "worker": w})
    return out


def colocation(reqs: list[dict]) -> None:
    trunk = defaultdict(list)
    for r in reqs:
        if r["session"] and not r["parent"] and not r["session"].startswith("victim-"):
            trunk[r["session"]].append(r)
    for rs in trunk.values():
        rs.sort(key=lambda r: r["start"])
    kids = [r for r in reqs if r["parent"] and r["worker"] is not None]
    same = known = 0
    groups = defaultdict(list)  # (parent, 30 s window) -> child workers
    for k in kids:
        prior = [t for t in trunk.get(k["parent"], []) if t["start"] <= k["start"] and t["worker"] is not None]
        if prior:
            known += 1
            same += prior[-1]["worker"] == k["worker"]
        groups[(k["parent"], int(k["start"] // 30000))].append(k["worker"])
    workers = {r["worker"] for r in reqs if r["worker"] is not None}
    print(f"workers seen in trace: {len(workers)}  ids: {sorted(workers)[:8]}")
    if known:
        print(f"child requests on the parent's latest worker: {same}/{known} = {same / known:.1%}"
              f"  (random placement would give ~{1 / max(1, len(workers)):.0%})")
    multi = [g for g in groups.values() if len(g) > 1]
    if multi:
        one = sum(len(set(g)) == 1 for g in multi)
        print(f"sibling bursts (same parent, 30 s window, >1 request): {len(multi)}; "
              f"all on one worker: {one} ({one / len(multi):.0%}); distinct workers p50="
              f"{_pct([len(set(g)) for g in multi], .5):.0f}")


def imbalance(path: str, budget: float, slo_ms: float, skip_s: float) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print("no FPM rows")
        return
    t0 = rows[0]["t"] + skip_s
    bins: dict[int, dict[str, dict]] = defaultdict(dict)  # 1 s bin -> worker -> latest row
    steps = defaultdict(list)
    for r in rows:
        if r["t"] < t0:
            continue
        bins[int(r["t"])][r["worker"]] = r
        steps[r["worker"]].append(r["wall_ms"])
    nw = len(steps)
    print(f"\nFPM workers: {nw}  ids: {sorted(steps)[:8]}")
    for w in sorted(steps):
        s = steps[w]
        print(f"  worker {w}: steps={len(s):7d} step ms p50={_pct(s, .5):5.1f} p95={_pct(s, .95):6.1f} "
              f">SLO={sum(x > slo_ms for x in s) / len(s):5.1%}")
    full = [b for b in bins.values() if len(b) == nw]
    if not full:
        print("no 1 s bins with readings from every worker")
        return
    ctx = [[r["decode_kv_tok"] for r in b.values()] for b in full]
    spread = [max(c) - min(c) for c in ctx]
    ratio = [max(c) / (sum(c) / len(c)) for c in ctx if sum(c) > 0]
    hot = [c for c in ctx if max(c) > budget]
    usable = [c for c in hot if min(c) < budget / 2]
    print(f"\n1 s bins with all {nw} workers: {len(full)}")
    print(f"  decode context spread (max-min) p50={_pct(spread, .5) / 1e3:.0f}k p95={_pct(spread, .95) / 1e3:.0f}k;"
          f" max/mean p50={_pct(ratio, .5):.2f} p95={_pct(ratio, .95):.2f}")
    print(f"  bins with some worker over budget ({budget / 1e3:.0f}k): {len(hot) / len(full):.1%}")
    print(f"  ... of those, another worker under half budget (placement headroom): "
          f"{(len(usable) / len(hot)) if hot else 0:.1%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--fpm", required=True)
    p.add_argument("--budget", type=float, default=300000)
    p.add_argument("--slo-ms", type=float, default=14.85)
    p.add_argument("--skip-first-s", type=float, default=300)
    a = p.parse_args()
    colocation(_trace(a.trace))
    imbalance(a.fpm, a.budget, a.slo_ms, a.skip_first_s)


if __name__ == "__main__":
    main()
