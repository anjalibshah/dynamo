# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Could a history-based predictor reach decode_sim's pred0.5 accuracy?

decode_sim.py's predX policies need each session's total decode work
predicted at its start with log-error sigma ~0.5 or better. The cheapest
real predictor is history: the work the same task took in an earlier run.
Traces carry no task names, so a task is keyed by the input tokens of its
main session's first request (the same prompt gives the same count); keys
that repeat within a run are dropped as ambiguous.

For each ordered pair of runs (A predicts B) it prints the share of B's
tasks matched in A and the log-error sigma of predicting each matched
task's total decode work (main + subagents), and of its main session's
alone, by A's. "no info" is the sigma of predicting every task by the
geometric mean of B's tasks: a predictor must beat it to carry any signal.

    python3 history_check.py --root /data/$USER/harbor_ab-mm \\
        --runs sws1-kv-n24 sws1-kv-n28 sws2-kv-n24 sws2-kv-n28
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
from collections import Counter, defaultdict

from decode_sim import load_trace


def tasks(path: str, interactive_prefix: str) -> dict:
    """Key -> (task decode work, main session decode work, subagent sessions)."""
    reqs = load_trace(path, interactive_prefix)
    first, work, main, subs = {}, defaultdict(int), defaultdict(int), defaultdict(set)
    for r in sorted(reqs, key=lambda r: r.arr):
        if r.interactive:
            continue
        work[r.task] += r.need
        if r.role == "main":
            main[r.task] += r.need
            first.setdefault(r.task, r.inp)
        else:
            subs[r.task].add(r.session)
    n = Counter(first.values())
    return {k: (work[t], main[t], len(subs[t])) for t, k in first.items() if n[k] == 1}


def sigma(pairs) -> float:
    errs = [math.log(max(1, b) / max(1, a)) for a, b in pairs]
    return statistics.pstdev(errs) if len(errs) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--interactive-prefix", default="victim-")
    a = ap.parse_args()
    runs = {r: tasks(os.path.join(a.root, r, "trace.jsonl"), a.interactive_prefix)
            for r in a.runs}
    for r, ts in runs.items():
        w = [max(1, v[0]) for v in ts.values()]
        g = math.exp(statistics.fmean(math.log(x) for x in w)) if w else 1
        print(f"{r}: {len(ts)} uniquely keyed tasks, task work p10/p50/p90 "
              f"{sorted(w)[len(w) // 10] if w else 0}/{statistics.median(w) if w else 0:.0f}/"
              f"{sorted(w)[9 * len(w) // 10] if w else 0} tokens, "
              f"no-info sigma {sigma([(g, x) for x in w]):.2f}")
    print("A -> B: matched, sigma task work, sigma main session work, "
          "subagent count equal")
    for ra, ta in runs.items():
        for rb, tb in runs.items():
            if ra == rb:
                continue
            keys = ta.keys() & tb.keys()
            if not keys:
                print(f"  {ra} -> {rb}: no tasks matched")
                continue
            print(f"  {ra} -> {rb}: {len(keys)}/{len(tb)}, "
                  f"{sigma([(ta[k][0], tb[k][0]) for k in keys]):.2f}, "
                  f"{sigma([(ta[k][1], tb[k][1]) for k in keys]):.2f}, "
                  f"{sum(ta[k][2] == tb[k][2] for k in keys) / len(keys):.0%}")


if __name__ == "__main__":
    main()
