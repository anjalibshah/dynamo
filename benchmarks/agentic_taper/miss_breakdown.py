# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why do interactive requests miss their target? Zero-GPU check from existing runs.

Work-conserving priority (engine scheduling priority instead of router holds)
reorders the engine's queue and prefill: it can fix misses caused by waiting
for the first token (TTFT), but not misses caused by every decode step being
slow because the batch is large (ITL), unless the batch is also bounded. This
splits each run's interactive misses, in the same T80 window and end-to-end
target as sweep_report.py, into:

  ttft      first token later than the TTFT budget, decode part on target
  itl       first token on time, decode part over budget
  both      both over budget

The decode budget is factor x anchor x (output tokens - 1); the TTFT budget is
--ttft-ms. Also reports interactive TTFT / ITL quantiles and agent TTFT by role
(main session vs subagent).

    python3 miss_breakdown.py --suite=-mm --root /data/$USER/harbor_ab-mm \\
        --sweep s1 --sweep s2

Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

from sweep_report import _job_dir, _ts


def _q(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def window(job: str, n_tasks: int) -> tuple[float, float] | None:
    starts, ends = [], []
    for p in glob.glob(f"{job}/*__*/result.json"):
        a = json.load(open(p)).get("agent_execution") or {}
        if a.get("started_at"):
            starts.append(_ts(a["started_at"]))
        if a.get("finished_at"):
            ends.append(_ts(a["finished_at"]))
    k = math.ceil(0.8 * n_tasks)
    ends.sort()
    return (min(starts), ends[k - 1]) if starts and len(ends) >= k else None


def breakdown(root: str, run: str, suite: str, n_tasks: int, factor: float, anchor: float,
              ttft_ms: float) -> dict:
    w = window(_job_dir(root, run, suite), n_tasks)
    if w is None:
        return {"error": "fewer than 80% of tasks finished"}
    t0, t80 = w
    cats = {"ok": 0, "ttft": 0, "itl": 0, "both": 0}
    v_ttft, v_itl, a_ttft = [], [], {"main": [], "sub": []}
    for line in open(os.path.join(root, run, "trace.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        r = r.get("event", r)
        if r.get("event_type") != "request_end":
            continue
        m, c = r["request"], r.get("agent_context") or {}
        s, tot, ttft = m.get("request_received_ms"), m.get("total_time_ms"), m.get("ttft_ms")
        if s is None or tot is None or ttft is None or not (t0 <= float(s) <= t80):
            continue
        sess = c.get("session_id") or ""
        out = m.get("output_tokens") or 0
        if sess.startswith("victim-"):
            dec_budget = factor * anchor * max(out - 1, 0)
            late_first = ttft > ttft_ms
            slow_decode = (tot - ttft) > dec_budget
            if tot <= ttft_ms + dec_budget:
                cats["ok"] += 1
            else:
                cats["both" if late_first and slow_decode else "ttft" if late_first else "itl"] += 1
            v_ttft.append(ttft)
            if out > 1:
                v_itl.append((tot - ttft) / (out - 1))
        elif sess:
            a_ttft["sub" if c.get("parent_session_id") else "main"].append(ttft)
    n = sum(cats.values())
    return {"n": n, "cats": cats, "v_ttft": v_ttft, "v_itl": v_itl, "a_ttft": a_ttft}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="", help="SUITE of the runs; write --suite=-mm (with =)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--n-tasks", type=int, default=64)
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--anchor-ms", type=float, default=6.16)
    ap.add_argument("--ttft-ms", type=float, default=500.0)
    a = ap.parse_args()
    root = a.root or f"harbor_ab{a.suite}"
    runs = list(a.runs)
    for tag in a.sweep:
        for d in sorted(glob.glob(f"{root}/sw{tag}-*")):
            name = os.path.basename(d)
            if re.match(rf"^sw{re.escape(tag)}-[a-z0-9]+-n\d+$", name) and name not in runs:
                runs.append(name)

    print(f"interactive target: TTFT <= {a.ttft_ms:g} ms, decode <= {a.factor:g} x {a.anchor_ms:g} ms/token\n")
    print(f"{'run':<20} {'n':>5} {'on':>6} | {'miss: ttft':>10} {'itl':>6} {'both':>6} | "
          f"{'TTFT p50/p90 ms':>17} {'ITL p50/p90 ms':>15} | {'agent TTFT p50/p95 s main':>25} {'sub':>11}")
    for run in runs:
        b = breakdown(root, run, a.suite, a.n_tasks, a.factor, a.anchor_ms, a.ttft_ms)
        if "error" in b:
            print(f"{run:<20} {b['error']}")
            continue
        n, c = b["n"], b["cats"]
        pct = lambda k: f"{c[k] / n:.0%}" if n else "-"
        at = lambda k: f"{_q(b['a_ttft'][k], .5) / 1e3:.1f}/{_q(b['a_ttft'][k], .95) / 1e3:.1f}"
        print(f"{run:<20} {n:>5} {pct('ok'):>6} | {pct('ttft'):>10} {pct('itl'):>6} {pct('both'):>6} | "
              f"{_q(b['v_ttft'], .5):>8.0f}/{_q(b['v_ttft'], .9):<8.0f} "
              f"{_q(b['v_itl'], .5):>7.1f}/{_q(b['v_itl'], .9):<7.1f} | {at('main'):>25} {at('sub'):>11}")


if __name__ == "__main__":
    main()
