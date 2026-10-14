# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-level decode simulation of Agentic TAPER milestone 2 (M2.2), from recorded runs.

The per-step FPM records hold only step totals, so the policy cannot be
replayed on them. Instead this re-runs the decode phase of a recorded run on a
virtual engine: each request starts decoding at its recorded arrival + TTFT
(prefill held fixed; Check 0 found first-token delay is not the problem), needs
its recorded output tokens, and its context is input + tokens generated so far.
Agent requests are closed-loop by default: each waits for what it waited for
in the recording (the session's previous request; a subagent's first request
waits for its parent; a main-session request waits for the subagent requests
that finished since its previous one) plus the recorded gap (tool and think
time), so a slower reply delays the rest of its task. Interactive requests keep
their recorded arrivals. --open replays all arrivals as recorded.
Every step takes T(n, L) = a + b*n + c*L ms over the requests it decodes
(times exp(N(0, noise)) to stress the predictor). Policies:

  stock        every active request decodes every step (vLLM eager)
  class_blind  HyGen/BROS-style: interactive always, agents by fit
  role_aware   ours: plus main session / join stragglers protected

The policies are the engine's own ``TaperPolicy`` (components/src/dynamo/vllm/
taper_engine.py), loaded by file path. Prefill steps are absent and the
dependencies are inferred from timing, so this checks skip rates and the size
of the effect; it is not a result.

    python3 decode_sim.py --root /data/$USER/harbor_ab-mm --sweep s1 --sweep s2 \\
        --agents 24 28 [--noise 0.1] [--k 4]

Per run and policy: interactive on-target share (same rule as miss_breakdown.py)
and ITL p50/p90; main-session and subagent decode slowdown p50/p90 vs the stock
simulation; task completion slowdown p90; steps engaged; agent decode steps
skipped; longest skip run; GPU busy time vs stock (the throughput cost). The
stock row also shows the measured interactive ITL p50, to check the simulation.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "..", "..", "components", "src", "dynamo", "vllm", "taper_engine.py")
POLICIES = ("stock", "class_blind", "role_aware")


def load_engine(path: str):
    spec = importlib.util.spec_from_file_location("taper_engine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["taper_engine"] = mod  # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


def _q(v: list, q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


class Req:
    __slots__ = ("i", "arr", "end", "start", "ttft", "need", "ctx", "interactive", "task",
                 "role", "session", "deps", "gap", "left", "done_ms", "skip_run",
                 "max_skip", "measured_itl")

    def __init__(self, i, arr, end, ttft, need, ctx, interactive, task, role, session,
                 measured_itl):
        self.i, self.arr, self.end, self.ttft, self.need, self.ctx = i, arr, end, ttft, need, ctx
        self.start = arr + ttft  # decode start, recorded
        self.interactive, self.task, self.role, self.session = interactive, task, role, session
        self.measured_itl = measured_itl
        self.deps, self.gap = [], 0.0
        self.left, self.done_ms, self.skip_run, self.max_skip = need, None, 0, 0


def load_trace(path: str, interactive_prefix: str) -> list[Req]:
    reqs = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        r = r.get("event", r)
        if r.get("event_type") != "request_end":
            continue
        m, c = r.get("request") or {}, r.get("agent_context") or {}
        s, tot, ttft = m.get("request_received_ms"), m.get("total_time_ms"), m.get("ttft_ms")
        out = m.get("output_tokens") or 0
        if s is None or ttft is None or tot is None or out < 2:
            continue  # no decode phase to simulate
        sid = c.get("session_id") or ""
        interactive = not sid or sid.startswith(interactive_prefix)
        parent = c.get("parent_session_id")
        task = None if interactive else (parent or sid)
        role = "" if interactive else ("main" if task == sid else "sub")
        reqs.append(Req(len(reqs), float(s), float(s) + float(tot), float(ttft), out - 1,
                        int(m.get("input_tokens") or 0) + 1, interactive, task, role, sid,
                        (float(tot) - float(ttft)) / (out - 1)))
    t0 = min((r.arr for r in reqs), default=0.0)
    for r in reqs:
        r.arr, r.end, r.start = r.arr - t0, r.end - t0, r.start - t0
    infer_deps(reqs)
    return reqs


def infer_deps(reqs: list[Req]) -> None:
    """What each agent request waited for in the recording, and the gap after it."""
    by_session = defaultdict(list)
    for r in reqs:
        if not r.interactive:
            by_session[r.session].append(r)
    for rs in by_session.values():
        rs.sort(key=lambda r: r.arr)
    subs_of = defaultdict(list)
    for sid, rs in by_session.items():
        if rs[0].role == "sub":
            subs_of[rs[0].task].append(sid)

    def last_done_before(rs, t, after=-math.inf):
        best = None
        for r in rs:
            if after < r.end <= t and (best is None or r.end > best.end):
                best = r
        return best

    for rs in by_session.values():
        for j, r in enumerate(rs):
            deps = []
            prev = last_done_before(rs[:j], r.arr)
            if prev is not None:
                deps.append(prev)
            if r.role == "sub" and j == 0:
                parent = last_done_before(by_session.get(r.task, []), r.arr)
                if parent is not None:
                    deps.append(parent)
            if r.role == "main":
                since = prev.end if prev is not None else -math.inf
                for sub in subs_of.get(r.task, []):
                    d = last_done_before(by_session[sub], r.arr, since)
                    if d is not None:
                        deps.append(d)
            r.deps = [d.i for d in deps]
            r.gap = r.arr - max((d.end for d in deps), default=r.arr)


def simulate(reqs: list[Req], policy: str, eng, cfg, noise: float, seed: int,
             closed: bool = True) -> dict:
    for r in reqs:
        r.left, r.done_ms, r.skip_run, r.max_skip = r.need, None, 0, 0
    waiting = {r.i: len(r.deps) for r in reqs} if closed else {}
    dependents = defaultdict(list)
    if closed:
        for r in reqs:
            for d in r.deps:
                dependents[d].append(r)
    base_ctx = {r.i: r.ctx for r in reqs}
    rng = random.Random(seed)
    a, b, c = cfg.coeffs
    pol = None
    if policy != "stock":
        cfg.mode = policy
        pol = eng.TaperPolicy(cfg)
    # released requests: (decode start, index); closed-loop ones join when deps finish
    released = [(r.start, r.i) for r in reqs if not waiting.get(r.i)]
    heapq.heapify(released)
    latest_dep = defaultdict(float)
    active: dict[int, Req] = {}
    ctx = dict(base_ctx)
    t = busy = 0.0
    steps = engaged = agent_slots = agent_skips = 0
    sim_starts = {}
    while released or active:
        if not active and released[0][0] > t:
            t = released[0][0]
        while released and released[0][0] <= t:
            st, i = heapq.heappop(released)
            active[i] = reqs[i]
            sim_starts[i] = st
        skip = set()
        if pol is not None:
            cands = [eng.Cand(req_id=r.i, ctx=ctx[r.i], interactive=r.interactive, task=r.task,
                              role=r.role, decoded=r.need - r.left + 1) for r in active.values()]
            d = pol.decide(cands)
            skip, engaged = d.skip, engaged + d.engaged
        n = L = 0
        for i in active:
            if i not in skip:
                n, L = n + 1, L + ctx[i]
        pred = max(0.0, a + b * n + c * L)
        dt = pred * (math.exp(rng.gauss(0.0, noise)) if noise else 1.0)
        if pol is not None and pred > 0:
            pol.observe(pred * pol.bias, dt)
        t += dt
        busy += dt
        steps += 1
        for i in list(active):
            r = active[i]
            if not r.interactive:
                agent_slots += 1
            if i in skip:
                r.skip_run += 1
                r.max_skip = max(r.max_skip, r.skip_run)
                agent_skips += 1
                continue
            r.skip_run = 0
            r.left -= 1
            ctx[i] += 1
            if r.left == 0:
                r.done_ms = t
                del active[i]
                for x in dependents.get(i, ()):
                    latest_dep[x.i] = max(latest_dep[x.i], t)
                    waiting[x.i] -= 1
                    if waiting[x.i] == 0:
                        heapq.heappush(released, (latest_dep[x.i] + x.gap + x.ttft, x.i))
    return {
        "done": {r.i: r.done_ms for r in reqs},
        "starts": sim_starts,
        "busy_ms": busy, "steps": steps, "engaged": engaged,
        "agent_slots": agent_slots, "agent_skips": agent_skips,
        "max_skip": max((r.max_skip for r in reqs), default=0),
    }


def score(reqs: list[Req], sim: dict, stock: dict, factor: float, anchor: float,
          ttft_ms: float) -> dict:
    inter_ok, inter_itl, slow = [], [], {"main": [], "sub": []}
    starts = sim["starts"]
    task_end, task_end0, task_start = {}, {}, {}
    for r in reqs:
        dur = sim["done"][r.i] - starts[r.i]
        if r.interactive:
            inter_itl.append(dur / r.need)
            inter_ok.append(r.ttft + dur <= ttft_ms + factor * anchor * r.need)
            continue
        dur0 = stock["done"][r.i] - stock["starts"][r.i]
        slow[r.role].append(dur / dur0 if dur0 > 0 else 1.0)
        task_start[r.task] = min(task_start.get(r.task, r.arr), r.arr)
        task_end[r.task] = max(task_end.get(r.task, 0.0), sim["done"][r.i])
        task_end0[r.task] = max(task_end0.get(r.task, 0.0), stock["done"][r.i])
    task_slow = [(task_end[k] - task_start[k]) / max(1e-9, task_end0[k] - task_start[k])
                 for k in task_end]
    return {
        "n_inter": len(inter_ok),
        "inter_on": sum(inter_ok) / len(inter_ok) if inter_ok else float("nan"),
        "itl50": _q(inter_itl, 0.5), "itl90": _q(inter_itl, 0.9),
        "main50": _q(slow["main"], 0.5), "main90": _q(slow["main"], 0.9),
        "sub50": _q(slow["sub"], 0.5), "sub90": _q(slow["sub"], 0.9),
        "task90": _q(task_slow, 0.9),
        "engaged": sim["engaged"] / max(1, sim["steps"]),
        "skipped": sim["agent_skips"] / max(1, sim["agent_slots"]),
        "max_skip": sim["max_skip"],
        "busy": sim["busy_ms"] / stock["busy_ms"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="harbor_ab-mm")
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--arm", default="kv", help="recorded arm to take arrivals from (stock)")
    ap.add_argument("--agents", type=int, nargs="*", default=[], help="only these loads")
    ap.add_argument("--runs", nargs="*", default=[], help="explicit run names")
    ap.add_argument("--coeffs", default="6.4013,0.227066,0.000016055")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--max-opp-ctx", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="lognormal sigma on step time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--open", action="store_true", help="replay recorded arrivals (no deps)")
    ap.add_argument("--interactive-prefix", default="victim-")
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--anchor-ms", type=float, default=6.16)
    ap.add_argument("--ttft-ms", type=float, default=500.0)
    ap.add_argument("--engine", default=ENGINE, help="path to taper_engine.py")
    ap.add_argument("--json", help="also write all scores here")
    a = ap.parse_args()

    eng = load_engine(a.engine)
    runs = list(a.runs)
    for tag in a.sweep:
        for d in sorted(glob.glob(f"{a.root}/sw{tag}-{a.arm}-n*")):
            name = os.path.basename(d)
            m = re.match(rf"^sw{re.escape(tag)}-{re.escape(a.arm)}-n(\d+)$", name)
            if m and (not a.agents or int(m.group(1)) in a.agents) and name not in runs:
                runs.append(name)

    hdr = (f"{'run':<18} {'policy':<11} {'inter_on':>8} {'ITL p50/p90':>13} "
           f"{'main sl p50/p90':>15} {'sub sl p50/p90':>14} {'task p90':>8} "
           f"{'engaged':>7} {'skipped':>7} {'maxrun':>6} {'busy':>6}")
    print(f"T(n,L) = {a.coeffs} ms, K={a.k}, rho={a.rho}, noise={a.noise}, "
          f"{'open' if a.open else 'closed'} loop; "
          f"slowdowns vs the stock simulation; busy = GPU time vs stock\n")
    print(hdr)
    out = {}
    for run in runs:
        path = os.path.join(a.root, run, "trace.jsonl")
        if not os.path.exists(path):
            print(f"{run:<18} no trace.jsonl")
            continue
        reqs = load_trace(path, a.interactive_prefix)
        if not reqs:
            print(f"{run:<18} no requests with a decode phase")
            continue
        cfg = eng.TaperConfig(coeffs=tuple(float(x) for x in a.coeffs.split(",")),
                              k=a.k, rho=a.rho, max_opp_ctx=a.max_opp_ctx,
                              interactive_prefix=a.interactive_prefix)
        sims = {p: simulate(reqs, p, eng, cfg, a.noise, a.seed, not a.open) for p in POLICIES}
        measured = _q([r.measured_itl for r in reqs if r.interactive], 0.5)
        out[run] = {}
        for p in POLICIES:
            s = score(reqs, sims[p], sims["stock"], a.factor, a.anchor_ms, a.ttft_ms)
            out[run][p] = s
            print(f"{run:<18} {p:<11} {s['inter_on']:>8.1%} "
                  f"{s['itl50']:>6.1f}/{s['itl90']:<6.1f} "
                  f"{s['main50']:>7.2f}/{s['main90']:<7.2f} {s['sub50']:>6.2f}/{s['sub90']:<7.2f} "
                  f"{s['task90']:>8.2f} {s['engaged']:>7.1%} {s['skipped']:>7.1%} "
                  f"{s['max_skip']:>6} {s['busy']:>6.2f}")
        print(f"{'':<18} measured interactive ITL p50 {measured:.1f} ms "
              f"({sum(r.interactive for r in reqs)} interactive, "
              f"{sum(not r.interactive for r in reqs)} agent requests, "
              f"{sum(bool(r.deps) for r in reqs) / max(1, sum(not r.interactive for r in reqs)):.0%}"
              f" with inferred dependencies, median gap "
              f"{_q([r.gap for r in reqs if r.deps], 0.5) / 1e3:.1f} s)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
