# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-level simulation of Agentic TAPER milestone 2 (M2.2), from recorded runs.

The per-step FPM records hold only step totals, so the policy cannot be
replayed on them. Instead this re-runs a recorded stock run on a virtual
engine, one step at a time:

  prefill   each request arrives with its new prompt tokens (input minus
            cached_tokens when recorded, else minus the session's previous
            input + output: prefix reuse), queues FCFS, and is prefilled in
            chunks that share the step's token budget with the decodes; its
            first token comes when its prefill completes
  decode    it then needs its recorded output tokens; its context is input +
            tokens generated so far
  step      T(n, L) = a + b*n + c*L ms over the decodes it runs, plus q + p*P
            when it carries P > 0 prefill tokens (q, p, and the token budget
            fitted from the run's own fpm.jsonl), times exp(N(0, noise))
  closed    agent requests wait for what they waited for in the recording (the
  loop      session's previous request; a subagent's first request waits for
            its parent; a main-session request waits for the subagent requests
            that finished since its previous one) plus the recorded gap, so a
            slower reply delays the rest of its task; interactive requests keep
            their recorded arrivals (--open: everything keeps them)

Policies are the engine's own ``TaperPolicy`` (components/src/dynamo/vllm/
taper_engine.py), loaded by file path, run at each --rhos value:

  stock        every active request decodes every step (vLLM eager)
  class_blind  HyGen/BROS-style: interactive always, agents by fit
  role_aware   ours: plus the task critical path protected
  oracle       role_aware with perfect knowledge (simulation only): per task,
               protect the request whose session has the most decode work left
               (this request's remaining tokens + the session's later
               requests), and demote the least such work first. An upper bound
               on what task-aware selection can add over class_blind here

Scoring uses the T80 window (start .. the ceil(0.8 * tasks)-th task completion)
of each simulation, as sweep_report.py does for real runs. The fidelity line
compares the stock simulation with the recording in its own T80 window; do not
trust policy rows until they agree. The matched comparison then gives, per
interactive level, each policy's best task p90 slowdown among the rho values
that reach the level: role-aware must beat class-blind there (the stop rule).

    python3 decode_sim.py --root /data/$USER/harbor_ab-mm --sweep s1 --sweep s2 \\
        --agents 24 28 [--rhos 0.4,0.6,0.8,1.0] [--noise 0.15]
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
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "..", "..", "components", "src", "dynamo", "vllm", "taper_engine.py")


def load_engine(path: str):
    spec = importlib.util.spec_from_file_location("taper_engine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["taper_engine"] = mod  # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


def _q(v: list, q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class Req:
    __slots__ = ("i", "arr", "end", "ttft", "need", "inp", "new", "future", "interactive",
                 "task",
                 "role", "session", "deps", "gap")

    def __init__(self, i, arr, end, ttft, need, inp, interactive, task, role, session):
        self.i, self.arr, self.end, self.ttft, self.need, self.inp = i, arr, end, ttft, need, inp
        self.interactive, self.task, self.role, self.session = interactive, task, role, session
        self.new = inp
        self.future = 0  # decode tokens of the session's later requests
        self.deps, self.gap = [], 0.0


def load_trace(path: str, interactive_prefix: str) -> list[Req]:
    reqs, cached = [], {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        r = r.get("event", r)
        if r.get("event_type") != "request_end":
            continue
        m, c = r.get("request") or {}, r.get("agent_context") or {}
        s, tot, ttft = m.get("request_received_ms"), m.get("total_time_ms"), m.get("ttft_ms")
        if s is None or tot is None:
            continue
        sid = c.get("session_id") or ""
        interactive = not sid or sid.startswith(interactive_prefix)
        parent = c.get("parent_session_id")
        task = None if interactive else (parent or sid)
        role = "" if interactive else ("main" if task == sid else "sub")
        out = int(m.get("output_tokens") or 0)
        req = Req(len(reqs), float(s), float(s) + float(tot),
                  float(ttft) if ttft is not None else float(tot), max(0, out - 1),
                  int(m.get("input_tokens") or 0), interactive, task, role, sid)
        if m.get("cached_tokens") is not None:
            cached[req.i] = int(m["cached_tokens"])
        reqs.append(req)
    t0 = min((r.arr for r in reqs), default=0.0)
    for r in reqs:
        r.arr, r.end = r.arr - t0, r.end - t0
    new_prompt_tokens(reqs, cached)
    by_session = defaultdict(list)
    for r in reqs:
        if not r.interactive:
            by_session[r.session].append(r)
    for rs in by_session.values():
        rs.sort(key=lambda r: r.arr)
        acc = 0
        for r in reversed(rs):
            r.future = acc
            acc += r.need
    infer_deps(reqs)
    return reqs


def new_prompt_tokens(reqs: list[Req], cached: dict) -> None:
    """Prompt tokens to prefill: recorded cache hits, else prefix reuse within a session."""
    by_session = defaultdict(list)
    for r in reqs:
        by_session[r.session or f"anon-{r.i}"].append(r)
    for rs in by_session.values():
        rs.sort(key=lambda r: r.arr)
        prev = None
        for r in rs:
            if r.i in cached:
                r.new = r.inp - cached[r.i]
            elif prev is not None and r.inp >= prev.inp + prev.need + 1:
                r.new = r.inp - (prev.inp + prev.need + 1)
            else:
                r.new = r.inp
            r.new = max(1, r.new)
            prev = r


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


def fit_prefill(path: str, coeffs: tuple, max_ms: float = 1000.0) -> dict | None:
    """q, p (ms per prefill token) and the step token budget from a run's fpm.jsonl.

    Residual of mixed steps over the decode model, regressed on prefill tokens."""
    if not os.path.exists(path):
        return None
    a, b, c = coeffs
    xs, ys, budget = [], [], 0
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("prefill_tok") is None:
            continue
        budget = max(budget, int(r["prefill_tok"]) + int(r.get("n_decode") or 0))
        if r.get("n_prefill") and 0 < r["wall_ms"] <= max_ms:
            xs.append(float(r["prefill_tok"]))
            ys.append(r["wall_ms"] - (a + b * r["n_decode"] + c * r["decode_kv_tok"]))
    if len(xs) < 20:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    q = my - p * mx
    return {"q": max(0.0, q), "p": max(0.0, p), "budget": budget, "n": len(xs)}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def oracle_policy(eng, cfg):
    class OraclePolicy(eng.TaperPolicy):
        """Protects, per task, the request with the most session work left."""

        remaining: dict = {}

        def _protected(self, by_task: dict) -> set:
            return {max(cs, key=lambda c: (self.remaining[c.req_id], c.req_id)).req_id
                    for cs in by_task.values()}

        def _demote_key(self, c) -> tuple:
            return (self.remaining[c.req_id], c.req_id)

    cfg.mode = "role_aware"
    return OraclePolicy(cfg)


def simulate(reqs: list[Req], policy: str, eng, cfg, budget: int, noise: float, seed: int,
             closed: bool = True) -> dict:
    rng = random.Random(seed)
    a, b, c = cfg.coeffs
    q, p = cfg.prefill_coeffs
    pol = None
    if policy == "oracle":
        pol = oracle_policy(eng, cfg)
    elif policy != "stock":
        cfg.mode = policy
        pol = eng.TaperPolicy(cfg)
    waiting = {r.i: len(r.deps) for r in reqs} if closed else {}
    dependents = defaultdict(list)
    if closed:
        for r in reqs:
            for d in r.deps:
                dependents[d].append(r)
    released = [(r.arr, r.i) for r in reqs if not waiting.get(r.i)]
    heapq.heapify(released)
    latest_dep = defaultdict(float)
    arrive, first, done = {}, {}, {}
    left = {r.i: r.need for r in reqs}
    ctx = {r.i: r.inp + 1 for r in reqs}
    skip_run = defaultdict(int)
    prefill_q: deque = deque()  # [index, remaining new tokens]
    active: dict[int, Req] = {}
    t = busy = 0.0
    steps = engaged = agent_slots = agent_skips = demoted = max_skip = 0

    def finish(i: int) -> None:
        done[i] = t
        for x in dependents.get(i, ()):
            latest_dep[x.i] = max(latest_dep[x.i], t)
            waiting[x.i] -= 1
            if waiting[x.i] == 0:
                heapq.heappush(released, (latest_dep[x.i] + x.gap, x.i))

    while released or active or prefill_q:
        if not active and not prefill_q and released[0][0] > t:
            t = released[0][0]
        while released and released[0][0] <= t:
            at, i = heapq.heappop(released)
            arrive[i] = at
            prefill_q.append([i, reqs[i].new])
        skip = set()
        pending_prefill = sum(x[1] for x in prefill_q)
        if pol is not None and active:
            cands = [eng.Cand(req_id=r.i, ctx=ctx[r.i], interactive=r.interactive,
                              task=r.task, role=r.role, decoded=r.need - left[r.i] + 1)
                     for r in active.values()]
            if policy == "oracle":
                pol.remaining = {i: left[i] + reqs[i].future for i in active}
            d = pol.decide(cands, min(pending_prefill, budget))
            skip = d.skip
            engaged += d.engaged
            demoted += d.n_demoted
        n = L = 0
        for i in active:
            if i not in skip:
                n, L = n + 1, L + ctx[i]
        room = max(0, budget - n)
        P = min(pending_prefill, room)
        pred = max(0.0, a + b * n + c * L) + ((q + p * P) if P else 0.0)
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
                skip_run[i] += 1
                max_skip = max(max_skip, skip_run[i])
                agent_skips += 1
                continue
            skip_run[i] = 0
            left[i] -= 1
            ctx[i] += 1
            if left[i] == 0:
                del active[i]
                finish(i)
        while P > 0 and prefill_q:  # FCFS chunked prefill
            head = prefill_q[0]
            take = min(P, head[1])
            head[1] -= take
            P -= take
            if head[1] == 0:
                prefill_q.popleft()
                i = head[0]
                first[i] = t
                if left[i] == 0:
                    finish(i)
                else:
                    active[i] = reqs[i]
    return {"arrive": arrive, "first": first, "done": done, "busy_ms": busy, "steps": steps,
            "engaged": engaged, "demoted": demoted, "agent_slots": agent_slots,
            "agent_skips": agent_skips, "max_skip": max_skip}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def t80(reqs: list[Req], done: dict) -> float:
    ends = defaultdict(float)
    for r in reqs:
        if r.task is not None and r.i in done:
            ends[r.task] = max(ends[r.task], done[r.i])
    e = sorted(ends.values())
    return e[max(0, math.ceil(0.8 * len(e)) - 1)] if e else math.inf


def interactive_stats(reqs, arrive, first, done, window, factor, anchor, ttft_ms) -> dict:
    ok, itl, ttft = [], [], []
    for r in reqs:
        if not r.interactive or r.i not in done or arrive[r.i] > window:
            continue
        tf = first[r.i] - arrive[r.i]
        dec = done[r.i] - first[r.i]
        ttft.append(tf)
        ok.append(tf + dec <= ttft_ms + factor * anchor * r.need)
        if r.need:
            itl.append(dec / r.need)
    return {"on": sum(ok) / len(ok) if ok else float("nan"), "n": len(ok),
            "itl50": _q(itl, 0.5), "itl90": _q(itl, 0.9), "ttft90": _q(ttft, 0.9)}


def measured(reqs: list[Req], factor, anchor, ttft_ms) -> dict:
    arrive = {r.i: r.arr for r in reqs}
    first = {r.i: r.arr + r.ttft for r in reqs}
    done = {r.i: r.end for r in reqs}
    return interactive_stats(reqs, arrive, first, done, t80(reqs, done), factor, anchor, ttft_ms)


def score(reqs, sim, stock, factor, anchor, ttft_ms) -> dict:
    window = t80(reqs, sim["done"])
    s = interactive_stats(reqs, sim["arrive"], sim["first"], sim["done"], window,
                          factor, anchor, ttft_ms)
    slow = {"main": [], "sub": []}
    task = defaultdict(lambda: [math.inf, 0.0, math.inf, 0.0])
    for r in reqs:
        if r.interactive or r.i not in sim["done"] or r.i not in stock["done"]:
            continue
        d = sim["done"][r.i] - sim["arrive"][r.i]
        d0 = stock["done"][r.i] - stock["arrive"][r.i]
        slow[r.role].append(d / d0 if d0 > 0 else 1.0)
        k = task[r.task]
        k[0] = min(k[0], sim["arrive"][r.i])
        k[1] = max(k[1], sim["done"][r.i])
        k[2] = min(k[2], stock["arrive"][r.i])
        k[3] = max(k[3], stock["done"][r.i])
    task_slow = [(k[1] - k[0]) / max(1e-9, k[3] - k[2]) for k in task.values()]
    s.update({
        "main50": _q(slow["main"], 0.5), "main90": _q(slow["main"], 0.9),
        "sub50": _q(slow["sub"], 0.5), "sub90": _q(slow["sub"], 0.9),
        "task90": _q(task_slow, 0.9),
        "t80": window / t80(reqs, stock["done"]),
        "engaged": sim["engaged"] / max(1, sim["steps"]),
        "skipped": sim["agent_skips"] / max(1, sim["agent_slots"]),
        "demoted": sim["demoted"] / max(1, sim["steps"]),
        "max_skip": sim["max_skip"],
        "busy": sim["busy_ms"] / stock["busy_ms"],
    })
    return s


def matched(rows: dict, levels: list[float]) -> list[str]:
    """Per interactive level: each policy's best task p90 among rhos reaching it."""
    out = []
    pols = sorted({p for p, _ in rows if p != "stock"},
                  key=lambda p: ("class_blind", "role_aware", "oracle").index(p))
    for lv in levels:
        best = {}
        for pol in pols:
            ok = [(s["task90"], rho) for (p, rho), s in rows.items()
                  if p == pol and s["on"] >= lv]
            best[pol] = min(ok) if ok else None
        top = {pol: max((s["on"] for (p, _), s in rows.items() if p == pol), default=0.0)
               for pol in pols}
        parts = []
        for pol in pols:
            x = best[pol]
            parts.append(f"{pol} {x[0]:.2f} (rho {x[1]:g})" if x
                         else f"{pol} unreached (max {top[pol]:.1%})")
        cb = best.get("class_blind")
        ratios = [f"{pol}/CB {best[pol][0] / cb[0]:.2f}" for pol in pols
                  if pol != "class_blind" and cb and best[pol]]
        out.append(f"inter >= {lv:.0%}: task p90 " + ", ".join(parts)
                   + ("  " + " ".join(ratios) if ratios else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="harbor_ab-mm")
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--arm", default="kv", help="recorded arm to take arrivals from (stock)")
    ap.add_argument("--agents", type=int, nargs="*", default=[], help="only these loads")
    ap.add_argument("--runs", nargs="*", default=[], help="explicit run names")
    ap.add_argument("--coeffs", default="6.4013,0.227066,0.000016055")
    ap.add_argument("--prefill-coeffs", default="",
                    help="q,p (ms, ms/token); default: fit from the run's fpm.jsonl")
    ap.add_argument("--budget", type=int, default=0,
                    help="step token budget; default: from fpm.jsonl, else 8192")
    ap.add_argument("--rhos", default="0.6,0.8,1.0")
    ap.add_argument("--policies", default="class_blind,role_aware,oracle")
    ap.add_argument("--levels", default="0.5,0.8,0.9,0.95")
    ap.add_argument("--k", type=int, default=4)
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
    coeffs = tuple(float(x) for x in a.coeffs.split(","))
    rhos = [float(x) for x in a.rhos.split(",")]
    policies = [p for p in a.policies.split(",") if p]
    levels = [float(x) for x in a.levels.split(",")]
    runs = list(a.runs)
    for tag in a.sweep:
        for d in sorted(glob.glob(f"{a.root}/sw{tag}-{a.arm}-n*")):
            name = os.path.basename(d)
            m = re.match(rf"^sw{re.escape(tag)}-{re.escape(a.arm)}-n(\d+)$", name)
            if m and (not a.agents or int(m.group(1)) in a.agents) and name not in runs:
                runs.append(name)

    print(f"T(n,L) = {a.coeffs} ms, K={a.k}, noise={a.noise}, "
          f"{'open' if a.open else 'closed'} loop; interactive scored in each simulation's "
          f"T80 window; slowdowns and t80 vs the stock simulation; busy = GPU time vs stock\n")
    hdr = (f"{'policy':<15} {'inter_on':>8} {'ITL p50/p90':>13} {'TTFT p90':>8} "
           f"{'main sl p50/p90':>15} {'sub sl p50/p90':>14} {'task p90':>8} {'t80':>5} "
           f"{'engaged':>7} {'skipped':>7} {'dem/st':>6} {'maxrun':>6} {'busy':>5}")
    out = {}
    for run in runs:
        path = os.path.join(a.root, run, "trace.jsonl")
        if not os.path.exists(path):
            print(f"{run}: no trace.jsonl")
            continue
        reqs = load_trace(path, a.interactive_prefix)
        if not reqs:
            print(f"{run}: no requests")
            continue
        fit = fit_prefill(os.path.join(a.root, run, "fpm.jsonl"), coeffs)
        if a.prefill_coeffs:
            pc = tuple(float(x) for x in a.prefill_coeffs.split(","))
            src = "given"
        elif fit:
            pc, src = (fit["q"], fit["p"]), f"fit on {fit['n']} mixed steps"
        else:
            pc, src = (0.0, 0.0), "none (no fpm.jsonl): prefill is free"
        budget = a.budget or (fit["budget"] if fit else 8192)
        cfg = eng.TaperConfig(coeffs=coeffs, prefill_coeffs=pc, k=a.k,
                              max_opp_ctx=a.max_opp_ctx,
                              interactive_prefix=a.interactive_prefix)
        closed = not a.open
        stock = simulate(reqs, "stock", eng, cfg, budget, a.noise, a.seed, closed)
        meas = measured(reqs, a.factor, a.anchor_ms, a.ttft_ms)
        st = score(reqs, stock, stock, a.factor, a.anchor_ms, a.ttft_ms)
        print(f"== {run}: {sum(r.interactive for r in reqs)} interactive, "
              f"{sum(not r.interactive for r in reqs)} agent requests "
              f"({sum(bool(r.deps) for r in reqs) / max(1, sum(not r.interactive for r in reqs)):.0%}"
              f" with dependencies); prefill q={pc[0]:.2f} ms p={pc[1] * 1e3:.3f} ms/1k tok "
              f"({src}), budget {budget} tok")
        print(f"   fidelity, recorded vs stock sim (T80): inter_on {meas['on']:.1%} vs "
              f"{st['on']:.1%}, ITL p50 {meas['itl50']:.1f} vs {st['itl50']:.1f} ms, "
              f"p90 {meas['itl90']:.1f} vs {st['itl90']:.1f}, TTFT p90 {meas['ttft90']:.0f} vs "
              f"{st['ttft90']:.0f} ms")
        print("   " + hdr)
        rows = {("stock", 0.0): st}
        for pol in policies:
            for rho in rhos:
                cfg.rho = rho
                sim = simulate(reqs, pol, eng, cfg, budget, a.noise, a.seed, closed)
                rows[(pol, rho)] = score(reqs, sim, stock, a.factor, a.anchor_ms, a.ttft_ms)
        for (pol, rho), s in rows.items():
            label = pol if pol == "stock" else f"{pol[:11]} {rho:g}"
            print(f"   {label:<15} {s['on']:>8.1%} {s['itl50']:>6.1f}/{s['itl90']:<6.1f} "
                  f"{s['ttft90']:>8.0f} {s['main50']:>7.2f}/{s['main90']:<7.2f} "
                  f"{s['sub50']:>6.2f}/{s['sub90']:<7.2f} {s['task90']:>8.2f} {s['t80']:>5.2f} "
                  f"{s['engaged']:>7.1%} {s['skipped']:>7.1%} {s['demoted']:>6.2f} "
                  f"{s['max_skip']:>6} {s['busy']:>5.2f}")
        for line in matched(rows, levels):
            print("   " + line)
        print()
        out[run] = {"fidelity": {"recorded": meas, "stock_sim": st},
                    "rows": {f"{p}@{r:g}": s for (p, r), s in rows.items()}}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
