# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fan-out and latency summary from Dynamo request traces (DYN_REQUEST_TRACE).

Answers two pilot questions before a benchmark run:
1. Does the agent fan out? TAPER only gates child-session requests, and only
   ones that overlap a sibling can be deferred. Reports the share of child
   requests and how many run concurrently per parent.
2. Unloaded latency for the SLO rule (NEXT_STEPS 7.3d): per-request ITL and
   per-task duration, measured on an otherwise idle stack.

    python3 trace_fanout.py /path/to/trace_dir_or_file [...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
from collections import defaultdict


def _files(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                out += [os.path.join(root, n) for n in names
                        if n.endswith((".jsonl", ".jsonl.gz"))]
        else:
            out.append(p)
    return sorted(out)


def _records(paths: list[str]):
    for path in _files(paths):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                # The jsonl sink wraps each record: {"timestamp": ..., "event": {...}}.
                if isinstance(r.get("event"), dict):
                    r = r["event"]
                if r.get("event_type") == "request_end" and r.get("request"):
                    yield r


def _pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def _max_overlap(intervals: list[tuple[float, float]]) -> int:
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals],
                    key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    a = p.parse_args()

    reqs = []
    for r in _records(a.paths):
        m, ctx = r["request"], r.get("agent_context") or {}
        start = m.get("request_received_ms")
        total = m.get("total_time_ms")
        if start is None or total is None:
            continue
        ttft, ntok = m.get("ttft_ms"), m.get("output_tokens") or 0
        itl = (total - ttft) / (ntok - 1) if ttft is not None and ntok > 1 else None
        reqs.append({"session": ctx.get("session_id"), "parent": ctx.get("parent_session_id"),
                     "start": float(start), "end": float(start) + float(total),
                     "ttft": ttft, "itl": itl})
    if not reqs:
        raise SystemExit("no request_end records with timing found")

    with_ctx = [r for r in reqs if r["session"]]
    child = [r for r in with_ctx if r["parent"]]
    print(f"requests: {len(reqs)}  with agent identity: {len(with_ctx)}  "
          f"child-session (gateable): {len(child)} ({len(child) / max(1, len(with_ctx)):.1%})")

    # Resolve every session to its root task by following parent links.
    parent_of = {r["session"]: r["parent"] for r in with_ctx if r["parent"]}

    def root(s):
        seen = set()
        while s in parent_of and s not in seen:
            seen.add(s)
            s = parent_of[s]
        return s

    by_root = defaultdict(list)
    for r in with_ctx:
        by_root[root(r["session"])].append(r)
    subagents = [len({r["session"] for r in rs} - {t}) for t, rs in by_root.items()]
    print(f"root tasks: {len(by_root)}  subagent sessions per task: "
          f"p50={_pct(subagents, 0.5):.0f} max={max(subagents)}  "
          f"tasks with any subagent: {sum(1 for n in subagents if n)}")

    # Sibling concurrency: child requests of the same immediate parent (the
    # grouping taper_router uses) that overlap in time.
    by_parent = defaultdict(list)
    for r in child:
        by_parent[r["parent"]].append((r["start"], r["end"]))
    if by_parent:
        peaks = [_max_overlap(iv) for iv in by_parent.values()]
        overlapping = sum(
            1 for iv in by_parent.values() for i, (s, e) in enumerate(iv)
            if any(j != i and s2 < e and s < e2 for j, (s2, e2) in enumerate(iv)))
        print(f"parents with children: {len(by_parent)}  peak concurrent siblings: "
              f"p50={_pct(peaks, 0.5):.0f} p95={_pct(peaks, 0.95):.0f} max={max(peaks)}")
        print(f"child requests overlapping a sibling (deferrable): {overlapping} "
              f"({overlapping / len(child):.1%} of child requests)")
    else:
        print("no child-session requests: TAPER would have nothing to gate")

    itls = [r["itl"] for r in reqs if r["itl"] is not None]
    ttfts = [r["ttft"] for r in reqs if r["ttft"] is not None]
    durs = [(max(x["end"] for x in rs) - min(x["start"] for x in rs)) / 1000.0
            for rs in by_root.values()]
    print(f"latency: ITL p50={_pct(itls, 0.5):.2f} ms p95={_pct(itls, 0.95):.2f} ms  "
          f"TTFT p50={_pct(ttfts, 0.5):.0f} ms  task duration p50="
          f"{statistics.median(durs) if durs else 0:.1f} s")


if __name__ == "__main__":
    main()
