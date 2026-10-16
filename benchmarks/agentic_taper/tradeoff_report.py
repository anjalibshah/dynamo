# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the trade-off experiment (ENGINE_AB_RUNBOOK.md, "Trade-off experiment").

Pools runs sw<tag>-<arm>-n<agents> over every --sweep, so e2's replicates
join e1's. Arms: kv (stock), prio (vLLM priority scheduling), cb (class-blind
at the default interactive target), cbt<m> (target x m), cbk<K> (DYN_TAPER_K).

Each regulated arm is one point on the curve, compared with the mean of all
kv runs at the same agent count:

  own target   interactive share meeting the arm's own deadline
               (500 ms + 3 * m * 6.16 ms * (tokens - 1); m = 1 except cbt<m>;
               6.16 ms is MiniMax-M2's unloaded interactive ITL p50, NEXT_STEPS 7.3j)
  nominal      interactive share meeting the default deadline (m = 1)
  task p90     agent task wall time p90 / kv's
  throughput   tasks per hour to T80 / kv's

Pre-registered rule: an arm is an acceptable operating point if
  A1  own-target interactive >= 0.80
  A2  own-target interactive - prio interactive at the same deadline >= 0.30
  A3  task p90 / kv <= 1.40
  A4  throughput / kv >= 0.90
  A5  engine health as engine_ab_report.py R5
with every metric the mean over that arm's runs.

    python3 tradeoff_report.py --sweep e1 --sweep e2 --suite=-mm --n-tasks 64 \\
        --root $REPO/harbor_ab-mm --agents 24
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
from collections import defaultdict

from engine_ab_report import engine_health, score
from sweep_report import score_run

ARM = re.compile(r"sw.+?-(kv|prio|cb|cbt[\d.]+|cbk\d+)-n(\d+)$")


def target_mult(arm: str) -> float:
    return float(arm[3:]) if arm.startswith("cbt") else 1.0


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="append", required=True)
    p.add_argument("--suite", default="")
    p.add_argument("--root", default=None)
    p.add_argument("--agents", type=int, default=24)
    p.add_argument("--n-tasks", type=int, default=64)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--anchor-ms", type=float, default=6.16,
                   help="unloaded interactive ITL p50, ms (MiniMax-M2 TP4 H200: 6.16)")
    p.add_argument("--ttft-ms", type=float, default=500.0)
    p.add_argument("--min-interactive", type=float, default=0.80)
    p.add_argument("--min-over-prio", type=float, default=0.30)
    p.add_argument("--max-task-p90-ratio", type=float, default=1.40)
    p.add_argument("--min-throughput-ratio", type=float, default=0.90)
    p.add_argument("--max-pred-err", type=float, default=0.15)
    p.add_argument("--max-decide-p99-us", type=float, default=1000.0)
    a = p.parse_args()
    a.root = a.root or f"harbor_ab{a.suite}"

    runs = defaultdict(list)  # arm -> run names at --agents
    for tag in a.sweep:
        for d in sorted(glob.glob(os.path.join(a.root, f"sw{tag}-*"))):
            m = ARM.match(os.path.basename(d))
            if m and int(m.group(2)) == a.agents:
                runs[m.group(1)].append(os.path.basename(d))

    res = {}
    print(f"### runs at {a.agents} agents")
    print(f"{'run':24s} {'nominal':>8s} {'tasks/h':>8s} {'task p50/p90 min':>17s}")
    for arm in sorted(runs):
        for run in runs[arm]:
            r = res[run] = score(a.root, run, a)
            if "error" in r:
                print(f"{run:24s} {r['error']}")
                continue
            print(f"{run:24s} {r['interactive']:8.3f} {r['throughput']:8.1f} "
                  f"{r['task_p50']:8.1f}/{r['task_p90']:<8.1f}")

    def ok_runs(arm):
        return [r for r in runs.get(arm, []) if "error" not in res[r]]

    kv, prio = ok_runs("kv"), ok_runs("prio")
    if not kv or not prio:
        print("\nneeds scored kv and prio runs at this agent count")
        return
    kv_thr = mean([res[r]["throughput"] for r in kv])
    kv_p90 = mean([res[r]["task_p90"] for r in kv])
    print(f"\nbaseline kv: {len(kv)} runs, {kv_thr:.1f} tasks/h, task p90 {kv_p90:.1f} min")
    if len(kv) > 1:
        thr = [res[r]["throughput"] for r in kv]
        p90 = [res[r]["task_p90"] for r in kv]
        print(f"  kv-vs-kv spread: task p90 max/min {max(p90) / min(p90):.2f}, "
              f"throughput min/max {min(thr) / max(thr):.2f}")
    cb = ok_runs("cb")
    if len(cb) > 1:
        thr = [res[r]["throughput"] for r in cb]
        p90 = [res[r]["task_p90"] for r in cb]
        print(f"  cb-vs-cb spread: task p90 max/min {max(p90) / min(p90):.2f}, "
              f"throughput min/max {min(thr) / max(thr):.2f}")

    print("\n### curve (means over each arm's runs)")
    print(f"{'arm':8s} {'runs':>4s} {'own tgt':>7s} {'prio@tgt':>8s} {'nominal':>7s} "
          f"{'p90/kv':>6s} {'thr/kv':>6s}  skip  verdict")
    found = []
    for arm in sorted((x for x in runs if x.startswith("cb")), key=lambda x: (x != "cb", x)):
        rs = ok_runs(arm)
        if not rs:
            continue
        f = a.factor * target_mult(arm)

        def at(run):
            return score_run(a.root, run, a.n_tasks, f, a.anchor_ms, a.ttft_ms, a.suite)["interactive"]

        own = mean([at(r) for r in rs])
        prio_own = mean([at(r) for r in prio])
        nominal = mean([res[r]["interactive"] for r in rs])
        p90 = mean([res[r]["task_p90"] for r in rs]) / kv_p90
        thr = mean([res[r]["throughput"] for r in rs]) / kv_thr
        hs = [engine_health(a.root, r) for r in rs]
        healthy = all(not h["disabled"] and h["pred_err"] is not None
                      and h["pred_err"] <= a.max_pred_err and h["decide_p99_us"] is not None
                      and h["decide_p99_us"] <= a.max_decide_p99_us for h in hs)
        skip = re.search(r"skip_share~(\S+)", hs[-1]["line"])
        checks = {
            "A1": own >= a.min_interactive,
            "A2": own - prio_own >= a.min_over_prio,
            "A3": p90 <= a.max_task_p90_ratio,
            "A4": thr >= a.min_throughput_ratio,
            "A5": healthy,
        }
        failed = [k for k, v in checks.items() if not v]
        if not failed:
            found.append(arm)
        print(f"{arm:8s} {len(rs):4d} {own:7.3f} {prio_own:8.3f} {nominal:7.3f} "
              f"{p90:6.2f} {thr:6.2f}  {skip.group(1) if skip else '-':5s} "
              f"{'OK' if not failed else 'fails ' + ','.join(failed)}")

    print(f"\n### acceptable operating points: {', '.join(found) if found else 'none'}")


if __name__ == "__main__":
    main()
