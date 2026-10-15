# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the engine-level A/B (ENGINE_AB_RUNBOOK.md) against its pre-registered rule.

Runs sw<tag>-<arm>-n<agents> with arm kv (stock), prio (vLLM priority
scheduling) or cb (class-blind decode regulation). Per run, with
sweep_report.py's definitions (window first agent start .. T80):

  interactive  share of interactive requests meeting the end-to-end deadline
  throughput   ceil(0.8 * N_TASKS) / window, tasks per hour
  task p50/p90 agent wall time per task (Harbor agent_execution), all tasks
  solved       Harbor reward 1.0 count

Pre-registered rule (defaults below), at every agent count with all three arms:

  R1  cb interactive >= 0.80
  R2  cb interactive - prio interactive >= 0.30 (existing priority does not
      already deliver it)
  R3  cb task p90 / kv task p90 <= 1.40
  R4  cb throughput / kv throughput >= 0.90
  R5  cb ran regulated: not failed open, median step-time error <= 15%,
      decision time p99 <= 1000 us (from the last taper.engine log line)

PASS if R1-R5 hold at every agent count. --noise-sweeps prints the kv-vs-kv
spread between earlier sweeps at the same agent count, to read R3/R4 against.

    python3 engine_ab_report.py --sweep e1 --suite=-mm --n-tasks 64 --root <results> \\
        [--noise-sweeps s1 s2 --noise-root /data/$USER/harbor_ab-mm]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

from sweep_report import _job_dir, _ts, score_run


def task_times(root: str, run: str, suite: str) -> list[float]:
    """Agent wall time (min) of every task with a finished agent."""
    out = []
    for p in glob.glob(f"{_job_dir(root, run, suite)}/*__*/result.json"):
        a = json.load(open(p)).get("agent_execution") or {}
        if a.get("started_at") and a.get("finished_at"):
            out.append((_ts(a["finished_at"]) - _ts(a["started_at"])) / 60000.0)
    return sorted(out)


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def engine_health(root: str, run: str) -> dict:
    """The last taper.engine summary line of a cb run's stack.log."""
    path = os.path.join(root, run, "stack.log")
    h = {"disabled": False, "pred_err": None, "decide_p99_us": None, "line": ""}
    if not os.path.exists(path):
        return h
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if "taper.engine disabled" in line:
            h["disabled"] = True
        if "taper.engine steps=" in line:
            h["line"] = line[line.index("steps="):].strip()
    m = re.search(r"pred_err_p50=([\d.]+)%", h["line"])
    if m:
        h["pred_err"] = float(m.group(1)) / 100
    m = re.search(r"decide_p50/p99=\d+/(\d+)us", h["line"])
    if m:
        h["decide_p99_us"] = float(m.group(1))
    return h


def score(root: str, run: str, a) -> dict:
    r = score_run(root, run, a.n_tasks, a.factor, a.anchor_ms, a.ttft_ms, a.suite)
    if "error" not in r:
        t = task_times(root, run, a.suite)
        r["task_p50"], r["task_p90"] = pct(t, 0.5), pct(t, 0.9)
    return r


def runs_of(root: str, tag: str) -> dict:
    """(arm, agents) -> run name."""
    out = {}
    for d in glob.glob(os.path.join(root, f"sw{tag}-*")):
        m = re.match(r"sw.+?-(kv|prio|cb)-n(\d+)$", os.path.basename(d))
        if m:
            out[(m.group(1), int(m.group(2)))] = os.path.basename(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="append", required=True)
    p.add_argument("--suite", default="")
    p.add_argument("--root", default=None)
    p.add_argument("--noise-sweeps", nargs="*", default=[])
    p.add_argument("--noise-root", default=None, help="results dir of the noise sweeps (default --root)")
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--anchor-ms", type=float, default=4.95)
    p.add_argument("--ttft-ms", type=float, default=500.0)
    p.add_argument("--min-interactive", type=float, default=0.80)
    p.add_argument("--min-over-prio", type=float, default=0.30)
    p.add_argument("--max-task-p90-ratio", type=float, default=1.40)
    p.add_argument("--min-throughput-ratio", type=float, default=0.90)
    p.add_argument("--max-pred-err", type=float, default=0.15)
    p.add_argument("--max-decide-p99-us", type=float, default=1000.0)
    a = p.parse_args()
    a.root = a.root or f"harbor_ab{a.suite}"
    a.noise_root = a.noise_root or a.root

    verdicts = []
    for tag in a.sweep:
        runs = runs_of(a.root, tag)
        print(f"\n### sweep {tag}")
        print(f"{'run':22s} {'interactive':>11s} {'tasks/h':>8s} {'task p50/p90 min':>17s} {'solved':>6s}")
        res = {}
        for key in sorted(runs, key=lambda k: (k[1], k[0])):
            r = res[key] = score(a.root, runs[key], a)
            if "error" in r:
                print(f"{runs[key]:22s} {r['error']}")
                continue
            print(f"{runs[key]:22s} {r['interactive']:11.3f} {r['throughput']:8.1f} "
                  f"{r['task_p50']:8.1f}/{r['task_p90']:<8.1f} {r['solved']:4d}/{a.n_tasks}")
        for n in sorted({k[1] for k in runs}):
            kv, prio, cb = (res.get((arm, n)) for arm in ("kv", "prio", "cb"))
            if not all(x and "error" not in x for x in (kv, prio, cb)):
                print(f"  n{n}: needs scored kv, prio and cb runs")
                verdicts.append(False)
                continue
            h = engine_health(a.root, runs[("cb", n)])
            checks = [
                ("R1 cb interactive", cb["interactive"], cb["interactive"] >= a.min_interactive),
                ("R2 cb - prio interactive", cb["interactive"] - prio["interactive"],
                 cb["interactive"] - prio["interactive"] >= a.min_over_prio),
                ("R3 task p90 cb/kv", cb["task_p90"] / kv["task_p90"],
                 cb["task_p90"] / kv["task_p90"] <= a.max_task_p90_ratio),
                ("R4 throughput cb/kv", cb["throughput"] / kv["throughput"],
                 cb["throughput"] / kv["throughput"] >= a.min_throughput_ratio),
            ]
            r5 = (not h["disabled"] and h["pred_err"] is not None
                  and h["pred_err"] <= a.max_pred_err and h["decide_p99_us"] is not None
                  and h["decide_p99_us"] <= a.max_decide_p99_us)
            print(f"  n{n}:")
            for name, v, ok in checks:
                print(f"    {name:26s} {v:6.3f}  {'pass' if ok else 'FAIL'}")
            print(f"    {'R5 engine health':26s} {'pass' if r5 else 'FAIL'}  {h['line'] or 'no taper.engine line'}"
                  + ("  [failed open]" if h["disabled"] else ""))
            verdicts.append(all(ok for _, _, ok in checks) and r5)

    if a.noise_sweeps:
        print("\n### kv-vs-kv spread between noise sweeps (read R3/R4 against it)")
        kv = defaultdict(dict)
        for tag in a.noise_sweeps:
            for d in glob.glob(os.path.join(a.noise_root, f"sw{tag}-kv-n*")):
                run = os.path.basename(d)
                r = score(a.noise_root, run, a)
                if "error" not in r:
                    kv[int(run.rsplit("-n", 1)[1])][tag] = r
        for n, by_tag in sorted(kv.items()):
            rs = list(by_tag.values())
            if len(rs) < 2:
                continue
            p90 = [r["task_p90"] for r in rs]
            thr = [r["throughput"] for r in rs]
            print(f"  n{n}: task p90 max/min {max(p90) / min(p90):.2f}, "
                  f"throughput min/max {min(thr) / max(thr):.2f} over {len(rs)} sweeps")

    print(f"\n### verdict: {'PASS' if verdicts and all(verdicts) else 'FAIL'} "
          f"({sum(verdicts)}/{len(verdicts)} agent counts pass)")


if __name__ == "__main__":
    main()
