# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-level scoring for Agentic TAPER milestone 1 (NEXT_STEPS 7.3m, pre-registered).

Reference run (stock kv at low load, same tasks): d_ref(t) = the task's Harbor
agent_execution duration. Tasks that raised an exception in the reference run
are excluded everywhere. Task objective D(t) = factor x d_ref(t) (1.5x, the
"developer at the terminal" profile; 2x and 3x reported as secondary).

Per run (sw<tag>-<arm>-n<agents>):
  window      first agent start .. the ceil(0.8 * N_TASKS)-th agent completion (T80)
  goodput G   eligible tasks finished inside the window, without exception and
              within D(t), per hour (primary)
  attain@k    share of eligible tasks finished without exception within k x d_ref
              (whole run; an exception or a missing trial is a miss)
  p90 slow    90th percentile of duration / d_ref over tasks finished without
              exception
  interactive e2e deadline attainment in the window (as sweep_report.py)
  solved      Harbor reward 1.0 count

Win rule (7.3m, restated 2026-10-13), per cell (rep x load): (a) taper G >=
1.10 x eager; (b) taper G >= 0.95 x the best fixed policy (eager/cap1/cap2);
(c) taper interactive >= 0.80 and p90 slowdown <= the best fixed policy's. WIN
iff all cells pass, pooled solved within 5 points of eager, and no cap k has
G >= 1.10 x eager and >= 0.95 x taper in every cell (then a fixed cap suffices).

    python3 task_report.py --suite=-mm --root /data/$USER/harbor_ab-mm \\
        --ref m1ref-kv-n8 --sweep m1a --sweep m1b

Stdlib only.
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

from sweep_report import _job_dir, _ts, score_run

BASELINES = ("eager", "cap1", "cap2")


def trials(job: str) -> dict[str, dict]:
    """{task: {start, end, dur_s, exc}} from a Harbor job directory."""
    out = {}
    for p in glob.glob(f"{job}/*__*/result.json"):
        d = json.load(open(p))
        a = d.get("agent_execution") or {}
        task = os.path.basename(os.path.dirname(p)).rsplit("__", 1)[0]
        s, e = a.get("started_at"), a.get("finished_at")
        out[task] = {
            "trial": os.path.basename(os.path.dirname(p)),
            "start": _ts(s) if s else None,
            "end": _ts(e) if e else None,
            "dur_s": (dt.datetime.fromisoformat(e.replace("Z", "")) -
                      dt.datetime.fromisoformat(s.replace("Z", ""))).total_seconds() if s and e else None,
            "exc": bool(d.get("exception_info") or d.get("exception")),
        }
    return out


def solved_trials(job: str) -> set[str]:
    try:
        ev = list(json.load(open(f"{job}/result.json"))["stats"]["evals"].values())[0]
    except (OSError, KeyError, IndexError):
        return set()
    return set(ev["reward_stats"]["reward"].get("1.0", []))


def score(root: str, suite: str, run: str, ref: dict[str, float], factors: list[float],
          n_tasks: int, anchor: float, ttft_ms: float) -> dict:
    job = _job_dir(root, run, suite)
    ts = trials(job)
    k80 = math.ceil(0.8 * n_tasks)
    starts = [t["start"] for t in ts.values() if t["start"] is not None]
    ends = sorted(t["end"] for t in ts.values() if t["end"] is not None)
    if not starts or len(ends) < k80:
        return {"run": run, "error": f"only {len(ends)} agent completions (< {k80})"}
    t0, t80 = min(starts), ends[k80 - 1]
    win_h = (t80 - t0) / 3.6e6

    def ok(task: str, k: float, in_window: bool) -> bool:
        t = ts.get(task)
        if t is None or t["exc"] or t["dur_s"] is None:
            return False
        if in_window and not t["end"] <= t80:
            return False
        return t["dur_s"] <= k * ref[task]

    primary = factors[0]
    slow = sorted(ts[t]["dur_s"] / ref[t] for t in ref
                  if t in ts and not ts[t]["exc"] and ts[t]["dur_s"] is not None)
    inter = score_run(root, run, n_tasks, 3.0, anchor, ttft_ms, suite)
    solved_names = solved_trials(job)
    return {
        "run": run,
        "window_min": win_h * 60,
        "goodput": sum(ok(t, primary, True) for t in ref) / win_h,
        "tasks_per_h": k80 / win_h,
        "attain": {k: sum(ok(t, k, False) for t in ref) / len(ref) for k in factors},
        "p50_slow": slow[len(slow) // 2] if slow else float("nan"),
        "p90_slow": slow[max(0, math.ceil(0.9 * len(slow)) - 1)] if slow else float("nan"),
        "interactive": inter.get("interactive", float("nan")),
        "solved": sum(ts[t]["trial"] in solved_names for t in ts),
        "trials": len(ts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="", help="SUITE of the runs; write --suite=-mm (with =)")
    ap.add_argument("--root", default=None, help="results dir (default harbor_ab<suite>)")
    ap.add_argument("--ref", required=True, help="reference run name (stock kv, low load)")
    ap.add_argument("--sweep", action="append", default=[], help="sweep tag; repeat per rep")
    ap.add_argument("--runs", nargs="*", default=[], help="explicit run names (no win check)")
    ap.add_argument("--factors", default="1.5,2,3", help="task objective factors; first is primary")
    ap.add_argument("--n-tasks", type=int, default=64)
    ap.add_argument("--anchor-ms", type=float, default=6.16, help="unloaded interactive ITL")
    ap.add_argument("--ttft-ms", type=float, default=500.0)
    ap.add_argument("--min-ratio", type=float, default=1.10)
    ap.add_argument("--within", type=float, default=0.95, help="(b): share of the best fixed policy's G")
    ap.add_argument("--min-interactive", type=float, default=0.80)
    ap.add_argument("--json", help="also write all scores here")
    args = ap.parse_args()
    root = args.root or f"harbor_ab{args.suite}"
    factors = [float(x) for x in args.factors.split(",")]

    ref_trials = trials(_job_dir(root, args.ref, args.suite))
    ref = {t: v["dur_s"] for t, v in ref_trials.items() if not v["exc"] and v["dur_s"]}
    print(f"reference {args.ref}: {len(ref)} eligible tasks "
          f"({len(ref_trials) - len(ref)} excluded: exception or no duration); "
          f"primary objective D(t) = {factors[0]:g} x d_ref(t)\n")

    runs = list(args.runs)
    for tag in args.sweep:
        pat = re.compile(rf"^sw{re.escape(tag)}-([a-z0-9]+)-n(\d+)$")
        for d in sorted(set(glob.glob(f"{root}/sw{tag}-*")) |
                        set(glob.glob(f"{os.path.dirname(os.path.abspath(root))}/jobs/harbor-ab{args.suite}-sw{tag}-*"))):
            name = os.path.basename(d).removeprefix(f"harbor-ab{args.suite}-")
            if pat.match(name) and name not in runs:
                runs.append(name)

    hdr = (f"{'run':<22} {'win_min':>7} {'G/h':>6} {'tasks/h':>7} " +
           " ".join(f"{'att@' + format(k, 'g'):>7}" for k in factors) +
           f" {'p50sl':>6} {'p90sl':>6} {'inter':>6} {'solved':>7}")
    print(hdr)
    scores = {}
    for run in runs:
        s = score(root, args.suite, run, ref, factors, args.n_tasks, args.anchor_ms, args.ttft_ms)
        scores[run] = s
        if "error" in s:
            print(f"{run:<22} {s['error']}")
            continue
        print(f"{run:<22} {s['window_min']:>7.1f} {s['goodput']:>6.1f} {s['tasks_per_h']:>7.1f} " +
              " ".join(f"{s['attain'][k]:>7.1%}" for k in factors) +
              f" {s['p50_slow']:>6.2f} {s['p90_slow']:>6.2f} {s['interactive']:>6.1%} "
              f"{s['solved']:>3}/{s['trials']:<3}")

    if args.sweep:
        print("\nwin rule (7.3m, restated 2026-10-13):")
        cells = defaultdict(dict)  # (tag, n) -> arm -> score
        for run, s in scores.items():
            m = re.match(r"^sw(.+)-([a-z0-9]+)-n(\d+)$", run)
            if m and "error" not in s:
                cells[(m.group(1), int(m.group(3)))][m.group(2)] = s
        ok = lambda c: "ok" if c else "FAIL"
        verdicts, cap_ok = [], {b: [] for b in BASELINES if b != "eager"}
        for (tag, n), arms in sorted(cells.items()):
            if "taper" not in arms or not all(b in arms for b in BASELINES):
                print(f"  rep {tag} @{n}: incomplete ({sorted(arms)})")
                verdicts.append(False)
                continue
            g = {a: arms[a]["goodput"] for a in ("taper",) + BASELINES}
            t = arms["taper"]
            best = max(BASELINES, key=lambda b: g[b])
            a_ok = g["taper"] >= args.min_ratio * g["eager"]
            b_ok = g["taper"] >= args.within * g[best]
            i_ok = t["interactive"] >= args.min_interactive
            s_ok = t["p90_slow"] <= arms[best]["p90_slow"]
            ratio = lambda x, y: x / y if y else float("inf")
            print(f"  rep {tag} @{n}: G taper {g['taper']:.1f} eager {g['eager']:.1f} "
                  f"cap1 {g['cap1']:.1f} cap2 {g['cap2']:.1f}")
            print(f"     (a) vs eager {ratio(g['taper'], g['eager']):.2f}x [{ok(a_ok)}]  "
                  f"(b) vs best fixed ({best}) {ratio(g['taper'], g[best]):.2f}x [{ok(b_ok)}]  "
                  f"(c) interactive {t['interactive']:.1%} [{ok(i_ok)}], p90 slowdown "
                  f"{t['p90_slow']:.2f} vs {arms[best]['p90_slow']:.2f} [{ok(s_ok)}]")
            verdicts.append(a_ok and b_ok and i_ok and s_ok)
            for cap in cap_ok:
                cap_ok[cap].append(g[cap] >= args.min_ratio * g["eager"]
                                   and g[cap] >= args.within * g["taper"])

        def rate(arm: str) -> float:
            ss = [s for r, s in scores.items() if f"-{arm}-n" in r and "error" not in s]
            return sum(s["solved"] for s in ss) / max(1, sum(s["trials"] for s in ss))

        solved_ok = rate("taper") >= rate("eager") - 0.05
        print(f"  pooled solved: taper {rate('taper'):.1%} vs eager {rate('eager'):.1%} [{ok(solved_ok)}]")
        dominating = [c for c, v in cap_ok.items() if v and len(v) == len(cells) and all(v)]
        print(f"  fixed cap that makes TAPER unnecessary: {', '.join(dominating) or 'none'}")
        win = bool(verdicts) and all(verdicts) and solved_ok and not dominating
        print(f"\nVERDICT: {'WIN' if win else 'NO WIN'} ({sum(verdicts)}/{len(verdicts)} cells pass)"
              + (f"; a fixed per-task cap ({', '.join(dominating)}) captures the benefit"
                 if dominating else ""))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(scores, f, indent=1)


if __name__ == "__main__":
    main()
