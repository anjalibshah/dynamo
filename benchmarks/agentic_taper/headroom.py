# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Go/no-go headroom check for Agentic TAPER milestone 1, from existing runs (no GPU).

The milestone-1 gate can only act on *opportunistic* requests: ones that start
while their task (root session plus subagent sessions) already has a request in
flight. This reports, per run (one serving copy):

  opportunistic share   of agent requests, and of in-flight context-time
                        (sum of (input + output tokens) x duration), the proxy
                        for their share of the decode step's KV context L
  child share           requests from subagent sessions (for comparison with
                        earlier fan-out numbers)
  frontier width        per task, peak concurrent requests (p50 / p90 / max)
  upper bound           step-time reduction if every opportunistic request were
                        removed: b x mean in-flight opportunistic requests
                        + c x mean in-flight opportunistic context (ms), from
                        T = a + b*n + c*L (--coeffs, fit_step_model.py)
  slack (with fpm.jsonl) share of loaded decode steps with any slack below tau,
                        and with room for one median opportunistic request

Proposed go rule (fix before looking): opportunistic context share >= 20% AND
upper-bound reduction >= 15% of tau_agent.

    python3 headroom.py --coeffs a,b,c sws1-kv-n28=/data/$USER/harbor_ab-mm/sws1-kv-n28 ...

Each run dir needs trace.jsonl; fpm.jsonl is optional. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

from trace_fanout import _records


def _pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def load_requests(path: str, victim_prefix: str) -> list[dict]:
    out = []
    for r in _records([path]):
        m, ctx = r["request"], r.get("agent_context") or {}
        start, total = m.get("request_received_ms"), m.get("total_time_ms")
        sess = ctx.get("session_id") or ""
        if start is None or total is None or not sess or sess.startswith(victim_prefix):
            continue
        out.append({"session": sess, "parent": ctx.get("parent_session_id"),
                    "start": float(start), "end": float(start) + float(total),
                    "tokens": (m.get("input_tokens") or 0) + (m.get("output_tokens") or 0)})
    return out


def analyze(reqs: list[dict], min_active: int) -> dict:
    parent_of = {r["session"]: r["parent"] for r in reqs if r["parent"]}

    def root(s: str) -> str:
        seen = set()
        while s in parent_of and s not in seen:
            seen.add(s)
            s = parent_of[s]
        return s

    tasks = defaultdict(list)
    for r in reqs:
        r["task"] = root(r["session"])
        tasks[r["task"]].append(r)

    # Loaded window: while >= min_active tasks are between first start and last end.
    events = sorted(e for rs in tasks.values()
                    for e in ((min(r["start"] for r in rs), 1), (max(r["end"] for r in rs), -1)))
    windows, active, opened = [], 0, None
    for t, d in events:
        active += d
        if active >= min_active and opened is None:
            opened = t
        elif active < min_active and opened is not None:
            windows.append((opened, t))
            opened = None
    if opened is not None:
        windows.append((opened, events[-1][0]))

    def clip(r: dict) -> float:
        return sum(max(0.0, min(r["end"], b) - max(r["start"], a)) for a, b in windows)

    widths = []
    for rs in tasks.values():
        rs.sort(key=lambda r: r["start"])
        for i, r in enumerate(rs):
            r["opp"] = any(o["end"] > r["start"] for o in rs[:i])
        pts = sorted([(r["start"], 1) for r in rs] + [(r["end"], -1) for r in rs],
                     key=lambda x: (x[0], x[1]))
        w = peak = 0
        for _, d in pts:
            w += d
            peak = max(peak, w)
        widths.append(peak)

    span_s = sum(b - a for a, b in windows) / 1000.0
    loaded = [r for r in reqs if any(a <= r["start"] < b for a, b in windows)]
    ctx_time = sum(r["tokens"] * clip(r) for r in reqs) / 1000.0
    opp_ctx_time = sum(r["tokens"] * clip(r) for r in reqs if r["opp"]) / 1000.0
    opp_req_time = sum(clip(r) for r in reqs if r["opp"]) / 1000.0
    opp_loaded = [r for r in loaded if r["opp"]]
    return {
        "tasks": len(tasks),
        "requests": len(loaded),
        "loaded_min": span_s / 60,
        "windows": windows,
        "opp_share": len(opp_loaded) / len(loaded) if loaded else float("nan"),
        "child_share": sum(bool(r["parent"]) for r in loaded) / len(loaded) if loaded else float("nan"),
        "opp_ctx_share": opp_ctx_time / ctx_time if ctx_time else float("nan"),
        "mean_inflight_ctx": ctx_time / span_s if span_s else float("nan"),
        "mean_opp_n": opp_req_time / span_s if span_s else float("nan"),
        "mean_opp_ctx": opp_ctx_time / span_s if span_s else float("nan"),
        "median_opp_tokens": statistics.median(r["tokens"] for r in opp_loaded) if opp_loaded else 0,
        "width_p50": _pct(widths, 0.5), "width_p90": _pct(widths, 0.9),
        "width_max": max(widths) if widths else 0,
    }


def slack(fpm_path: str, windows: list, tau: float, opp_cost_ms: float) -> dict | None:
    if not os.path.exists(fpm_path):
        return None
    walls = []
    with open(fpm_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r["t"] * 1000.0
            if r.get("n_decode", 0) >= 1 and any(a <= t < b for a, b in windows):
                walls.append(r["wall_ms"])
    if not walls:
        return None
    return {"steps": len(walls), "p50_ms": _pct(walls, 0.5),
            "any_slack": sum(w < tau for w in walls) / len(walls),
            "room_for_one": sum(tau - w >= opp_cost_ms for w in walls) / len(walls)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=run_dir (containing trace.jsonl [, fpm.jsonl])")
    ap.add_argument("--coeffs", default=None, help="a,b,c from fit_step_model.py (ms, ms/req, ms/token)")
    ap.add_argument("--tau-ms", type=float, default=21.63, help="agent per-token target")
    ap.add_argument("--min-active-tasks", type=int, default=0,
                    help="loaded window: at least this many tasks in progress (e.g. 0.75 x agents)")
    ap.add_argument("--victim-prefix", default="victim-")
    ap.add_argument("--min-ctx-share", type=float, default=0.20)
    ap.add_argument("--min-reduction-frac", type=float, default=0.15)
    args = ap.parse_args()
    coeffs = [float(x) for x in args.coeffs.split(",")] if args.coeffs else None

    for spec in args.runs:
        label, _, d = spec.partition("=")
        d = d or label
        reqs = load_requests(os.path.join(d, "trace.jsonl"), args.victim_prefix)
        if not reqs:
            print(f"{label}: no agent requests")
            continue
        s = analyze(reqs, args.min_active_tasks)
        print(f"\n{label}: {s['tasks']} tasks, {s['requests']} agent requests in "
              f"{s['loaded_min']:.0f} loaded min")
        print(f"  opportunistic: {s['opp_share']:.1%} of requests, {s['opp_ctx_share']:.1%} of "
              f"in-flight context-time   (child-session requests: {s['child_share']:.1%})")
        print(f"  task frontier width: p50={s['width_p50']} p90={s['width_p90']} max={s['width_max']}"
              f"   mean in-flight: context {s['mean_inflight_ctx'] / 1e3:.0f}k tok, "
              f"opportunistic {s['mean_opp_n']:.2f} req / {s['mean_opp_ctx'] / 1e3:.0f}k tok")
        if not coeffs:
            continue
        a, b, c = coeffs
        ub = b * s["mean_opp_n"] + c * s["mean_opp_ctx"]
        opp_cost = b + c * s["median_opp_tokens"]
        print(f"  upper bound, removing all opportunistic work: -{ub:.2f} ms/step "
              f"({ub / args.tau_ms:.0%} of tau {args.tau_ms:g} ms); "
              f"median opportunistic request costs {opp_cost:.2f} ms/step")
        sl = slack(os.path.join(d, "fpm.jsonl"), s["windows"], args.tau_ms, opp_cost)
        if sl:
            print(f"  decode steps (loaded): p50 {sl['p50_ms']:.1f} ms; below tau {sl['any_slack']:.0%}; "
                  f"room for one median opportunistic request {sl['room_for_one']:.0%}")
        go = s["opp_ctx_share"] >= args.min_ctx_share and ub >= args.min_reduction_frac * args.tau_ms
        print(f"  headroom: {'GO' if go else 'NO-GO'} (needs context share >= {args.min_ctx_share:.0%} "
              f"and upper bound >= {args.min_reduction_frac:.0%} of tau)")


if __name__ == "__main__":
    main()
