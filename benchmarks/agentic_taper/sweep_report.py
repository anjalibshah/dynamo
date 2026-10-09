# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a load sweep (NEXT_STEPS 7.3i, pre-registered).

Per run (sw<tag>-<arm>-n<agents>):
  window      first agent start .. time the ceil(0.8 * N_TASKS)-th task's agent
              finished (T80) -- excludes the straggler tail
  throughput  ceil(0.8 * N_TASKS) / window, tasks per hour (primary)
  steps/min   agent LLM requests started in the window per minute (secondary)
  interactive share of interactive requests started in the window that meet the
              end-to-end deadline TTFT bound + factor x anchor x (tokens - 1)
  solved      Harbor reward 1.0 count (exceptions count as unsolved)

Frontier test: for each arm, sort runs by interactive share and linearly
interpolate throughput at each level in --levels (no extrapolation). The ctx
arm wins a sweep if, at every level both arms bracket (at least two such
levels), its interpolated throughput >= --min-ratio x kv's.

    python3 sweep_report.py --sweep s1 [--sweep s2] [--root /data/anjshah/harbor_ab]

Stdlib only; reads <root>/<run>/trace.jsonl and the Harbor job directory
(<repo>/jobs/harbor-ab-<run> or <root>/harbor-ab-<run>).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
from collections import defaultdict


def _ts(s: str) -> float:
    d = dt.datetime.fromisoformat(s.replace("Z", ""))
    return d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000.0  # Harbor stamps are UTC


def _job_dir(root: str, run: str, suite: str = "") -> str:
    name = f"harbor-ab{suite}-{run}"
    for d in (os.path.join(root, name),
              os.path.join(os.path.dirname(os.path.abspath(root)), "jobs", name)):
        if os.path.isdir(d):
            return d
    raise SystemExit(f"no Harbor job dir for {run}")


def score_run(root: str, run: str, n_tasks: int, factor: float, anchor: float, ttft_ms: float,
              suite: str = "") -> dict:
    job = _job_dir(root, run, suite)
    starts, ends = [], []
    for p in glob.glob(f"{job}/*__*/result.json"):
        a = json.load(open(p)).get("agent_execution") or {}
        if a.get("started_at"):
            starts.append(_ts(a["started_at"]))
        if a.get("finished_at"):
            ends.append(_ts(a["finished_at"]))
    k = math.ceil(0.8 * n_tasks)
    ends.sort()
    if not starts or len(ends) < k:
        return {"run": run, "error": f"only {len(ends)} agent completions (< {k})"}
    t0, t80 = min(starts), ends[k - 1]
    ev = list(json.load(open(f"{job}/result.json"))["stats"]["evals"].values())[0]
    solved = len(ev["reward_stats"]["reward"].get("1.0", []))

    agent_steps = vic = vic_ok = 0
    for line in open(os.path.join(root, run, "trace.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        r = r.get("event", r)
        if r.get("event_type") != "request_end":
            continue
        m, c = r["request"], r.get("agent_context") or {}
        s, tot = m.get("request_received_ms"), m.get("total_time_ms")
        if s is None or tot is None or not (t0 <= float(s) <= t80):
            continue
        sess = c.get("session_id") or ""
        if sess.startswith("victim-"):
            vic += 1
            out = m.get("output_tokens") or 0
            vic_ok += float(tot) <= ttft_ms + factor * anchor * max(out - 1, 0)
        elif sess:
            agent_steps += 1
    win_h = (t80 - t0) / 3.6e6
    return {"run": run, "window_min": win_h * 60, "throughput": k / win_h,
            "steps_per_min": agent_steps / (win_h * 60), "interactive": vic_ok / vic if vic else float("nan"),
            "n_interactive": vic, "solved": solved}


def interp(points: list[tuple[float, float]], level: float) -> float | None:
    """Throughput at an interactive level from (interactive, throughput) points; no extrapolation."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= level <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (level - x0) / (x1 - x0)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="append", required=True)
    p.add_argument("--suite", default="", help="SUITE used for the runs; write --suite=-mm (with =) for MiniMax-M2")
    p.add_argument("--root", default=None, help="results dir (default harbor_ab<suite>)")
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--anchor-ms", type=float, default=4.95, help="unloaded interactive ITL")
    p.add_argument("--ttft-ms", type=float, default=500.0)
    p.add_argument("--levels", default="0.75,0.80,0.85")
    p.add_argument("--min-ratio", type=float, default=1.10)
    a = p.parse_args()
    levels = [float(x) for x in a.levels.split(",")]
    a.root = a.root or f"harbor_ab{a.suite}"

    verdicts, solved = [], defaultdict(lambda: [0, 0])
    for tag in a.sweep:
        runs = sorted({os.path.basename(d) for d in glob.glob(os.path.join(a.root, f"sw{tag}-*"))},
                      key=lambda r: (r.split("-")[1], int(r.rsplit("-n", 1)[1])))
        print(f"\n### sweep {tag}  (interactive: e2e, factor {a.factor:g}, anchor {a.anchor_ms} ms; "
              f"window: first agent start .. T{int(round(80))} of {a.n_tasks} tasks)")
        print(f"{'run':22s} {'window':>7s} {'tasks/h':>8s} {'steps/min':>9s} {'interactive':>11s} {'n_int':>6s} {'solved':>6s}")
        pts = defaultdict(list)
        for run in runs:
            m = re.match(r"sw.+?-(kv|ctx)-n(\d+)$", run)
            if not m:
                continue
            r = score_run(a.root, run, a.n_tasks, a.factor, a.anchor_ms, a.ttft_ms, a.suite)
            if "error" in r:
                print(f"{run:22s} {r['error']}")
                continue
            print(f"{run:22s} {r['window_min']:6.1f}m {r['throughput']:8.1f} {r['steps_per_min']:9.1f} "
                  f"{r['interactive']:11.3f} {r['n_interactive']:6d} {r['solved']:4d}/{a.n_tasks}")
            pts[m.group(1)].append((r["interactive"], r["throughput"]))
            solved[m.group(1)][0] += r["solved"]
            solved[m.group(1)][1] += a.n_tasks
        covered = wins = 0
        for lv in levels:
            kv, ctx = interp(pts["kv"], lv), interp(pts["ctx"], lv)
            if kv is None or ctx is None:
                print(f"  level {lv:.2f}: not bracketed by both arms (kv={kv}, ctx={ctx})")
                continue
            covered += 1
            ratio = ctx / kv
            wins += ratio >= a.min_ratio
            print(f"  level {lv:.2f}: kv {kv:6.1f} tasks/h, ctx {ctx:6.1f} tasks/h, ratio {ratio:.2f}")
        ok = covered >= 2 and wins == covered
        verdicts.append(ok)
        print(f"  sweep {tag}: {'ctx WINS' if ok else 'no win'} ({wins}/{covered} covered levels at >= {a.min_ratio:g}x)")

    print("\n### pooled solved rate: " + ", ".join(
        f"{arm} {s}/{n} = {s / n:.3f}" for arm, (s, n) in sorted(solved.items()) if n))
    if len(verdicts) > 1:
        print(f"### overall: {'ctx WINS in every sweep' if all(verdicts) else 'no consistent win'} "
              f"({sum(verdicts)}/{len(verdicts)} sweeps)")


if __name__ == "__main__":
    main()
