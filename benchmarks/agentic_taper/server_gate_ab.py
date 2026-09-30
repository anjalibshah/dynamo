# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B the server-side taper_router gate against the stock KV router.

Unlike experiment.py, which gates on the client (arms A2/A3 via permit_gate),
this sends one fixed trace eagerly (Arm.A1: every request dispatched at its
arrival time) at whatever stack is listening, so the only thing that differs
between runs is the server: `run_smoketest_1gpu.sh kv` vs `... taper`. Same
seed => identical trace, so records pair by request_id across runs.

    python3 server_gate_ab.py run --label kv    --base-url http://127.0.0.1:8100 --model M --out kv.jsonl
    python3 server_gate_ab.py run --label taper --base-url http://127.0.0.1:8100 --model M --out taper.jsonl
    python3 server_gate_ab.py compare kv.jsonl taper.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict

from replay_client import (Arm, HttpFrontend, ReplayEngine, apply_server_metrics,
                           parse_frontend_metrics)
from trace_gen import WorkloadConfig, generate


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def _mean_itl(r: dict) -> float:
    itls = r.get("itls_ms") or []
    return sum(itls) / len(itls) if itls else 0.0


def _steady(records: list[dict], skip_first_s: float) -> list[dict]:
    # The trace opens with a burst into a cold engine and an empty prefix
    # cache, before the gate has any FPM reading; every arm's slowest victims
    # land in the first ~0.6 s. Drop that window to compare steady state.
    if not skip_first_s or not records:
        return records
    t0 = min(r["t_arrival"] for r in records)
    return [r for r in records if r["t_arrival"] - t0 >= skip_first_s]


def _summary(records: list[dict]) -> dict:
    out = {"n": len(records), "failed": sum(1 for r in records if not r["ok"])}
    for role in ("victim", "root", "branch", "join"):
        rs = [r for r in records if r["role"] == role and r["ok"]]
        if not rs:
            continue
        ttft = [r["ttft_ms"] for r in rs if r["ttft_ms"] > 0]
        itl = [_mean_itl(r) for r in rs if r.get("itls_ms")]
        out[role] = {
            "n": len(rs),
            "ttft_p50": _pct(ttft, 0.50), "ttft_p95": _pct(ttft, 0.95),
            "ttft_p99": _pct(ttft, 0.99),
            "itl_p50": _pct(itl, 0.50), "itl_p95": _pct(itl, 0.95),
            "itl_p99": _pct(itl, 0.99),
        }
    return out


def _print_summary(label: str, s: dict) -> None:
    print(f"[{label}] {s['n']} requests, {s['failed']} failed")
    for role in ("victim", "root", "branch", "join"):
        if role in s:
            v = s[role]
            print(f"  {role:<7} n={v['n']:<4} "
                  f"ttft p50/p95/p99={v['ttft_p50']:.1f}/{v['ttft_p95']:.1f}/{v['ttft_p99']:.1f} ms  "
                  f"itl p50/p95/p99={v['itl_p50']:.2f}/{v['itl_p95']:.2f}/{v['itl_p99']:.2f} ms")


async def _run(a: argparse.Namespace) -> None:
    cfg = WorkloadConfig(n_tasks=a.n_tasks, victim_frac=a.victim_frac,
                         fanout_k=a.fanout_k, burst_multiplier=a.burst,
                         shared_prefix_blocks=a.prefix_blocks)
    rows = [json.loads(r.to_json()) for r in generate(cfg, a.seed)]
    frontend = HttpFrontend(a.base_url, a.model, max_conns=a.max_concurrency)
    eng = ReplayEngine(rows, Arm.A1, a.model, frontend,
                       max_concurrency=a.max_concurrency,
                       words_per_block=a.words_per_block)
    t0 = time.monotonic()
    records = await eng.run()
    wall = time.monotonic() - t0
    await frontend.close()
    errors = Counter(r.error for r in records if not r.ok)
    for msg, n in errors.most_common(3):
        print(f"FAILED x{n}: {msg}")
    if a.frontend_log:
        await asyncio.sleep(1.0)
        matched = apply_server_metrics(records, parse_frontend_metrics(a.frontend_log))
        print(f"server-metrics matched {matched}/{len(records)} requests")
        if matched < 0.9 * len(records):
            # Unmatched requests didn't complete normally on the server (e.g. the
            # stack wasn't ready and each stream ended with an error chunk, which
            # the client still counts as ok). Fail loudly instead of writing a
            # plausible-looking result.
            raise SystemExit(
                f"ERROR: only {matched}/{len(records)} requests completed on the "
                f"server; not writing {a.out}. Check {a.frontend_log}.")
    dicts = [asdict(r) | {"label": a.label} for r in records]
    with open(a.out, "w") as f:
        f.write(json.dumps({"_meta": vars(a) | {"wall_s": wall}}) + "\n")
        for d in dicts:
            f.write(json.dumps(d) + "\n")
    print(f"wall {wall:.1f}s, wrote {len(dicts)} records to {a.out}")
    _print_summary(a.label, _summary(dicts))


def _load(path: str) -> tuple[dict, list[dict]]:
    meta, recs = {}, []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            else:
                recs.append(d)
    return meta, recs


def _compare(a: argparse.Namespace) -> None:
    (ma, ra), (mb, rb) = _load(a.a), _load(a.b)
    la, lb = ma.get("label", a.a), mb.get("label", a.b)
    for m in ("seed", "n_tasks", "fanout_k", "burst", "victim_frac", "prefix_blocks"):
        if ma.get(m) != mb.get(m):
            print(f"WARNING: runs differ in {m}: {ma.get(m)} vs {mb.get(m)} (not paired)")
    if a.skip_first_s:
        ra, rb = _steady(ra, a.skip_first_s), _steady(rb, a.skip_first_s)
        print(f"(steady state: excluding requests that arrived in the first {a.skip_first_s:g}s)")
    sa, sb = _summary(ra), _summary(rb)
    _print_summary(la, sa)
    _print_summary(lb, sb)
    if "victim" in sa and "victim" in sb:
        print(f"\nvictim tail, {la} -> {lb}:")
        for k in ("ttft_p95", "ttft_p99", "itl_p95", "itl_p99"):
            x, y = sa["victim"][k], sb["victim"][k]
            pct = (y - x) / x * 100 if x else 0.0
            print(f"  {k:<9} {x:8.2f} -> {y:8.2f}  ({pct:+.1f}%)")
    # Paired per-victim ITL delta: same request_id under the same trace.
    va = {r["request_id"]: r for r in ra if r["role"] == "victim" and r["ok"]}
    vb = {r["request_id"]: r for r in rb if r["role"] == "victim" and r["ok"]}
    deltas = [_mean_itl(vb[k]) - _mean_itl(va[k]) for k in va.keys() & vb.keys()
              if va[k].get("itls_ms") and vb[k].get("itls_ms")]
    if deltas:
        print(f"  paired victim mean-ITL delta ({lb} - {la}), n={len(deltas)}: "
              f"p50={_pct(deltas, 0.5):+.2f} ms  p95={_pct(deltas, 0.95):+.2f} ms")
    for lbl, m in ((la, ma), (lb, mb)):
        print(f"  {lbl} wall time {m.get('wall_s', 0):.1f}s")


def _table(a: argparse.Namespace) -> None:
    """Median across seeds per config label, steady state; (min-max) spread."""
    runs: dict[str, list[tuple[dict, dict]]] = {}
    for path in a.files:
        meta, recs = _load(path)
        runs.setdefault(meta.get("label", path), []).append(
            (meta, _summary(_steady(recs, a.skip_first_s))))
    cols = [("victim", "itl_p50"), ("victim", "itl_p95"), ("victim", "ttft_p50"),
            ("victim", "ttft_p95"), ("root", "ttft_p50"), ("branch", "ttft_p50")]
    head = ["config", "runs"] + [f"{r}.{k}" for r, k in cols] + ["wall_s"]
    print(f"steady state (skip first {a.skip_first_s:g}s); median across runs, [min-max]")
    print(" | ".join(head))
    for label, rs in runs.items():
        cells = [label, str(len(rs))]
        for role, key in cols:
            vals = [s[role][key] for _, s in rs if role in s]
            cells.append(f"{statistics.median(vals):.1f} [{min(vals):.1f}-{max(vals):.1f}]"
                         if vals else "-")
        walls = [m.get("wall_s", 0.0) for m, _ in rs]
        cells.append(f"{statistics.median(walls):.1f}")
        print(" | ".join(cells))


def _tasks(records: list[dict], skip_first_s: float) -> tuple[list[dict], list[dict]]:
    """Split steady-state tasks into (victims, agentic).

    victim: {"ok", "ttft", "itl"}; agentic: {"ok", "ttj_s"} where ttj_s is the
    task's time-to-join: root arrival -> join completion. A task is kept only
    if its first request arrived after the skip window.
    """
    if not records:
        return [], []
    t0 = min(r["t_arrival"] for r in records)
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    victims, agentic = [], []
    for reqs in by_task.values():
        if min(r["t_arrival"] for r in reqs) - t0 < skip_first_s:
            continue
        roles = {r["role"]: r for r in reqs}
        if "victim" in roles:
            v = roles["victim"]
            victims.append({"ok": v["ok"], "ttft": v["ttft_ms"], "itl": _mean_itl(v)})
        elif "root" in roles and "join" in roles:
            agentic.append({"ok": all(r["ok"] for r in reqs),
                            "ttj_s": roles["join"]["t_done"] - roles["root"]["t_arrival"]})
    return victims, agentic


def _goodput(a: argparse.Namespace) -> None:
    """Task-level goodput: fraction of tasks meeting their SLO, median across seeds.

    Victim tasks: TTFT <= --victim-ttft-ms AND mean ITL <= the ITL SLO.
    Agentic tasks: time-to-join (root arrival -> join done) <= the task
    deadline. Unlike replay_client.task_goodput (mean ITL only), this charges
    the gate for the queueing delay it imposes on deferred branches and joins.
    """
    runs: dict[str, list[tuple[list[dict], list[dict], list[dict]]]] = {}
    for path in a.files:
        meta, recs = _load(path)
        v, g = _tasks(recs, a.skip_first_s)
        runs.setdefault(meta.get("label", path), []).append((v, g, recs))

    def med(vals: list[float]) -> str:
        return (f"{statistics.median(vals):.2f} [{min(vals):.2f}-{max(vals):.2f}]"
                if vals else "-")

    itl_slos = [float(x) for x in a.victim_itl_ms.split(",")]
    deadlines = [float(x) for x in a.task_deadline_s.split(",")]
    print(f"steady state (skip first {a.skip_first_s:g}s); median across seeds [min-max]\n")

    print("agentic time-to-join (s), p50 / p95:")
    for label, rs in runs.items():
        p50 = [_pct([t["ttj_s"] for t in g], 0.5) for _, g, _ in rs if g]
        p95 = [_pct([t["ttj_s"] for t in g], 0.95) for _, g, _ in rs if g]
        print(f"  {label:<14} {med(p50)} / {med(p95)}")

    print(f"\nvictim goodput (TTFT <= {a.victim_ttft_ms:g} ms and mean ITL <= SLO):")
    print("  " + " | ".join(["config"] + [f"ITL<={s:g}ms" for s in itl_slos]))
    for label, rs in runs.items():
        cells = []
        for slo in itl_slos:
            fr = [sum(1 for t in v if t["ok"] and t["ttft"] <= a.victim_ttft_ms
                      and 0 < t["itl"] <= slo) / len(v) for v, _, _ in rs if v]
            cells.append(med(fr))
        print("  " + " | ".join([label] + cells))

    print("\nagentic task goodput (time-to-join <= deadline):")
    print("  " + " | ".join(["config"] + [f"<={d:g}s" for d in deadlines]))
    for label, rs in runs.items():
        cells = []
        for d in deadlines:
            fr = [sum(1 for t in g if t["ok"] and t["ttj_s"] <= d) / len(g)
                  for _, g, _ in rs if g]
            cells.append(med(fr))
        print("  " + " | ".join([label] + cells))

    s_itl, s_dl = a.primary_itl_ms, a.primary_deadline_s
    print(f"\noverall task goodput, all tasks (victim: TTFT<={a.victim_ttft_ms:g}ms & "
          f"ITL<={s_itl:g}ms; agentic: time-to-join<={s_dl:g}s), and ITL-only "
          f"task goodput as in replay_client.task_goodput (ITL<={s_itl:g}ms):")
    for label, rs in runs.items():
        overall, itl_only = [], []
        for v, g, recs in rs:
            good = (sum(1 for t in v if t["ok"] and t["ttft"] <= a.victim_ttft_ms
                        and 0 < t["itl"] <= s_itl)
                    + sum(1 for t in g if t["ok"] and t["ttj_s"] <= s_dl))
            if v or g:
                overall.append(good / (len(v) + len(g)))
            steady = {r["task_id"] for r in _steady(recs, a.skip_first_s)}
            by_task: dict[str, list[dict]] = {}
            for r in recs:
                if r["task_id"] in steady:
                    by_task.setdefault(r["task_id"], []).append(r)
            if by_task:
                ok = 0
                for reqs in by_task.values():
                    itls = [x for r in reqs for x in (r.get("itls_ms") or [])]
                    if all(r["ok"] for r in reqs) and (not itls or statistics.mean(itls) <= s_itl):
                        ok += 1
                itl_only.append(ok / len(by_task))
        print(f"  {label:<14} overall {med(overall)}   ITL-only {med(itl_only)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True, help="e.g. kv | taper")
    r.add_argument("--base-url", required=True,
                   help="e.g. http://127.0.0.1:8100 (no /v1; path is /v1/completions)")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-tasks", type=int, default=120)
    r.add_argument("--victim-frac", type=float, default=0.6)
    r.add_argument("--fanout-k", type=int, default=5)
    r.add_argument("--burst", type=float, default=3.0)
    r.add_argument("--prefix-blocks", type=int, default=8)
    r.add_argument("--words-per-block", type=int, default=32)
    r.add_argument("--max-concurrency", type=int, default=128)
    r.add_argument("--frontend-log", default=None,
                   help="frontend stdout log; if given, use server-measured TTFT/ITL")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--skip-first-s", type=float, default=0.0,
                   help="exclude requests arriving in the first N seconds (startup burst)")
    t = sub.add_parser("table", help="median-across-seeds table for a sweep")
    t.add_argument("files", nargs="+")
    t.add_argument("--skip-first-s", type=float, default=1.0)
    gp = sub.add_parser("goodput", help="task-level goodput across SLOs for a sweep")
    gp.add_argument("files", nargs="+")
    gp.add_argument("--skip-first-s", type=float, default=1.0)
    gp.add_argument("--victim-ttft-ms", type=float, default=500.0)
    gp.add_argument("--victim-itl-ms", default="10,15,25,50",
                    help="comma-separated victim ITL SLOs (ms)")
    gp.add_argument("--task-deadline-s", default="5,10,15,30",
                    help="comma-separated agentic time-to-join deadlines (s)")
    gp.add_argument("--primary-itl-ms", type=float, default=15.0)
    gp.add_argument("--primary-deadline-s", type=float, default=10.0)
    a = p.parse_args()
    if a.cmd == "run":
        asyncio.run(_run(a))
    elif a.cmd == "compare":
        _compare(a)
    elif a.cmd == "table":
        _table(a)
    else:
        _goodput(a)


if __name__ == "__main__":
    main()
